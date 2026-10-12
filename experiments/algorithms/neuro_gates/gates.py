"""Go/no-go gates for SNN/NEAT directions inside the ORIGINAL v3 hover environment.

Nothing here changes the robot's interface. The controller is the original TeacherPID, which reads the 11-D
reading and commands gimbal alpha/beta and motor speed within LOW/HIGH through the 2 ms path with action
smoothing. Only the PID's gains, and in the proprio gate the attitude it is fed, are varied.

* proprio: is the PID limited by its 0.13 s-lagged attitude? It is fed oracle attitudes (lower lag,
  wingbeat rocking removed), and its gains are retuned on the same grid for every attitude source.
* adapt: do in-flight plant changes need adaptation? A PID retuned for each change is compared with one
  robust PID.

    python -m experiments.algorithms.neuro_gates.gates proprio --scratch runs/neuro_gates/scratch \
        --output artifacts/neuro_gates/proprio.json
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..topology_hover.environment import case as make_case
from ..robust_control.environment import ResearchEnv

# kp (with ki) and kd multipliers of the selected PID. The stable window is narrow: with the 0.13 s sensor,
# kd x0.5 falls and kd x2 tilts about 25 deg (lag sweep, 2026-10-12), so kd is scanned finely around 1.
GRID = [(sp, sd) for sp in (.5, 1., 1.5, 2., 3.) for sd in (.7, .85, 1., 1.2, 1.5)]
INFOS = ('env', 'truth_lp30', 'truth_ma', 'truth_derocked')
CHANGES = ('none', 'thrust_loss', 'servo_weak', 'payload', 'delay_jump')
CHANGE_T = 4.
HANDOVER_T = 1.5  # oracle attitudes reach the PID only after the takeoff transient; metrics start at 2 s
SPLITS = dict(tune=2_010_000_000, evaluate=2_020_000_000)


class Oracle:
    """Truth-based roll/pitch fed to the PID only, never to the observation.

    This is an upper bound on what an attitude estimator could deliver. Kinds:
      raw:      true angles;
      lp30:     true angles through a 30 ms low-pass;
      ma:       causal average over the true wingbeat period;
      derocked: local regression on the true crank phase, about zero lag.
    An optional suffix '@tau' adds a first-order lag of tau seconds, for example 'derocked@0.03'.
    A 'thrust_' prefix uses the thrust unit's attitude instead of the IMU site on the base, for example
    'thrust_derocked@0.03'. The case's reading delay is applied as for the sensor."""
    WINDOW = 40

    def __init__(self, env, kind):
        m = env.sim.model
        kind, _, tau = kind.partition('@')
        self.thrust = m.body('thrust_unit').id if kind.startswith('thrust_') else None
        kind = kind.removeprefix('thrust_')
        self.env, self.kind, self.tau = env, kind, float(tau) if tau else 0.
        self.site = m.site('history_imu').id
        self.crank_q = int(m.joint('carrier_2_joint').qposadr[0])
        self.crank_v = int(m.joint('carrier_2_joint').dofadr[0])
        self.history = deque(maxlen=64)
        self.lp = self.lagged = None
        self.queue = deque()
        self.update()

    def angles(self):
        d = self.env.sim.data  # same Euler convention as the virtual AHRS in sensor_step.cpp
        r = d.site_xmat[self.site] if self.thrust is None else d.xmat[self.thrust]
        return np.array([math.atan2(r[7], r[8]), math.asin(min(1., max(-1., -r[6])))])

    def update(self):
        d = self.env.sim.data
        x = self.angles()
        self.history.append((float(d.time), float(d.qpos[self.crank_q]), x))
        if self.kind == 'lp30':
            self.lp = x if self.lp is None else self.lp+(1-math.exp(-DT/.03))*(x-self.lp)
            value = self.lp
        elif self.kind == 'ma':
            n = 2*math.pi/max(abs(float(d.qvel[self.crank_v])), 1.)/DT  # samples per wingbeat
            whole = int(n)
            items = [h[2] for h in self.history]
            value = (np.mean(items, axis=0) if len(items) <= whole else
                     (np.sum(items[-whole:], axis=0)+(n-whole)*items[-whole-1])/n)
        elif self.kind == 'derocked':
            items = list(self.history)[-self.WINDOW:]
            if len(items) < 16 or abs(items[-1][1]-items[0][1]) < 3*math.pi:
                value = x  # spin-up: crank harmonics are collinear with the intercept until 1.5 turns fit
            else:
                t = np.array([h[0] for h in items])-items[-1][0]
                phi = np.array([h[1] for h in items])
                design = np.column_stack([np.ones_like(t), t, t*t]+[f(k*phi) for k in (1, 2, 3)
                                                                   for f in (np.cos, np.sin)])
                coef = np.linalg.lstsq(design, np.array([h[2] for h in items]), rcond=None)[0]
                value = coef[0]
        else:
            value = x
        if self.tau:
            value = np.asarray(value, dtype=float)
            self.lagged = value if self.lagged is None else self.lagged+(1-math.exp(-DT/self.tau))*(value-self.lagged)
            value = self.lagged
        self.queue.append(np.asarray(value, dtype=float))
        while len(self.queue) > self.env.case.get('delay_steps', 0)+1:
            self.queue.popleft()


def apply_change(env, change):
    m = env.sim.model
    if change == 'thrust_loss':  # speed-target gain: about 19 % lift loss at fixed command
        env.motor_gain *= .9
    elif change == 'servo_weak':  # both gimbal position servos lose 60 % of stiffness and damping
        for i in range(2):
            m.actuator_gainprm[i, 0] *= .4
            m.actuator_biasprm[i, 1] *= .4
            m.actuator_biasprm[i, 2] *= .4
    elif change == 'payload':  # lower body (7 g + 1 g) mass and inertia +25 %, i.e. +2 g payload
        for name in ('base_link', 'pitch_link'):
            b = m.body(name).id
            m.body_mass[b] *= 1.25
            m.body_inertia[b] *= 1.25
        mujoco.mj_setConst(m, mujoco.MjData(m))  # scratch data: the flying state is untouched
    elif change == 'delay_jump':  # whole reading delay jumps to 10 ms
        env.case['delay_steps'] = 5
    elif change != 'none':
        raise ValueError(change)


def run(env, c, gains, info='env', change='none'):
    env.configure(c)
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, gains
    teacher = env.teacher
    teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    oracle = None
    if info != 'env':
        oracle = Oracle(env, info.split('_', 1)[1])
        original = teacher.step
        handed = [False]

        def step(reading):
            if env.air.steps*DT < HANDOVER_T:
                return original(reading)
            r = np.array(reading, dtype=float)
            r[:2] = oracle.queue[0]
            if not handed[0]:  # no derivative kick at the switch
                teacher.last_rpy = r[:3].copy()
                handed[0] = True
            return original(r)
        teacher.step = step
    changed = change == 'none'
    while True:
        if not changed and env.air.steps*DT >= CHANGE_T-1e-9:
            apply_change(env, change)
            changed = True
        _, _, terminated, truncated, out = env.step(np.zeros(3))
        if oracle is not None:
            oracle.update()
        if terminated or truncated:
            break
    m = out['episode']
    s = np.asarray(env.samples)
    t, tilt = s[:, 0], np.degrees(s[:, 1])
    after = t >= CHANGE_T
    return dict(seed=c['seed'], condition=c['condition'], gains=list(gains), info=info, change=change,
                termination=m['termination'], success=m['success'], elapsed=m['elapsed'],
                tilt_rms_deg=m['tilt_rms_deg'], height_rmse_m=m['height_rmse_m'], return_sum=m['return_sum'],
                tilt_peak_deg=float(tilt[t >= 2.].max()) if (t >= 2.).any() else float('nan'),
                tilt_rms_after_change_deg=float(np.sqrt(np.mean(tilt[after]**2))) if after.any() else float('nan'))


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='pid', task='hover')


def task(args):
    return run(_ENV, *args)


def pool(scratch, workers):
    source = Path(scratch)/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def cases(split, condition, count, seconds, pulse_scale=1.):
    base = SPLITS[split]+dict(nominal=0, stress=100_000, stress2x=200_000)[condition]
    out = []
    for i in range(count):
        c = make_case(base+i, seconds, 'stress' if condition == 'stress2x' else condition, i % 8)
        if condition == 'stress2x':
            c.update(pulse_force=2*c['pulse_force'], pulse_torque=2*c['pulse_torque'])
        out.append(c)
    return out


def score(rows, key='tilt_rms_deg'):
    """Survivors first, then the mean continuous tilt over the survivors (lower is better)."""
    flown = [r[key] for r in rows if r['termination'] == 'time_limit']
    return [len(flown), -float(np.mean(flown)) if flown else -1e9]


def summarize(rows, group, keys=('tilt_rms_deg', 'tilt_peak_deg', 'height_rmse_m', 'tilt_rms_after_change_deg')):
    out = {}
    for r in rows:
        out.setdefault(group(r), []).append(r)
    table = {}
    for name, items in sorted(out.items()):
        flown = [r for r in items if r['termination'] == 'time_limit']
        entry = dict(episodes=len(items), completed=len(flown), success=int(sum(r['success'] for r in items)))
        for k in keys:
            v = [r[k] for r in flown if r[k] == r[k]]
            entry[k] = float(np.mean(v)) if v else None
        table[name] = entry
    return table


def paired(rows, a, b, group, key='tilt_rms_deg'):
    index = {}
    for r in rows:
        index[(group(r), r['seed'])] = r
    diffs = [index[(a, s)][key]-index[(b, s)][key] for (g, s) in index if g == a and (b, s) in index
             and index[(a, s)]['termination'] == index[(b, s)]['termination'] == 'time_limit']
    return dict(mean=float(np.mean(diffs)) if diffs else None, n=len(diffs),
                lower=int(np.sum(np.array(diffs) < 0)))


def proprio(executor, count_tune=8, count_eval=16):
    tune = cases('tune', 'stress', count_tune, 8.)
    tasks = [(c, g, info) for info in INFOS for g in GRID for c in tune]
    rows = list(executor.map(task, tasks, chunksize=1))
    by = {}
    for r in rows:
        by.setdefault((r['info'], tuple(r['gains'])), []).append(r)
    grid = {f'{i}|{g[0]:g},{g[1]:g}': score(v) for (i, g), v in by.items()}
    best = {info: max(GRID, key=lambda g: score(by[(info, g)])) for info in INFOS}
    specs = {('env', (1., 1.)): ['original_pid']}  # the retuned env PID may coincide with the original gains
    for info in INFOS:
        specs.setdefault((info, tuple(best[info])), []).append(f'{info}_tuned')
    evaluation = [c for cond in ('nominal', 'stress', 'stress2x') for c in cases('evaluate', cond, count_eval, 10.)]
    tasks = [(c, g, info) for (info, g) in specs for c in evaluation]
    erows = []
    for r in executor.map(task, tasks, chunksize=1):
        for name in specs[(r['info'], tuple(r['gains']))]:
            erows.append(dict(r, controller=name))
    group = lambda r: f"{r['controller']}|{r['condition']}"
    tuned = [f'{info}_tuned' for info in INFOS]
    result = dict(stage='proprio', grid=grid, best={k: list(v) for k, v in best.items()},
                  summary=summarize(erows, group),
                  paired={f'{name}-original_pid|{cond}': {k: paired(erows, f'{name}|{cond}', f'original_pid|{cond}',
                                                                     group, k)
                                                          for k in ('tilt_rms_deg', 'tilt_peak_deg', 'height_rmse_m')}
                          for name in tuned for cond in ('nominal', 'stress', 'stress2x')},
                  tuning_rows=rows, episodes=erows)
    return result


def adapt(executor, count_tune=8, count_eval=16):
    tune = cases('tune', 'stress', count_tune, 10.)
    tasks = [(c, g, 'env', ch) for ch in CHANGES for g in GRID for c in tune]
    rows = list(executor.map(task, tasks, chunksize=1))
    by = {}
    for r in rows:
        by.setdefault((r['change'], tuple(r['gains'])), []).append(r)
    key = 'tilt_rms_after_change_deg'
    oracle = {ch: max(GRID, key=lambda g: score(by[(ch, g)], key)) for ch in CHANGES}
    robust = max(GRID, key=lambda g: (sum(score(by[(ch, g)], key)[0] for ch in CHANGES),
                                      float(np.mean([score(by[(ch, g)], key)[1] for ch in CHANGES]))))
    evaluation = cases('evaluate', 'stress', count_eval, 10.)
    specs = {}
    for ch in CHANGES:
        for name, g in (('original_pid', (1., 1.)), ('robust_pid', robust), ('oracle_pid', oracle[ch])):
            specs.setdefault((ch, tuple(g)), []).append(name)
    tasks = [(c, g, 'env', ch) for (ch, g) in specs for c in evaluation]
    erows = []
    for r in executor.map(task, tasks, chunksize=1):
        for name in specs[(r['change'], tuple(r['gains']))]:
            erows.append(dict(r, controller=name))
    group = lambda r: f"{r['controller']}|{r['change']}"
    result = dict(stage='adapt', changes=list(CHANGES), change_time_s=CHANGE_T,
                  grid={f'{ch}|{g[0]:g},{g[1]:g}': score(v, key) for (ch, g), v in by.items()},
                  oracle={k: list(v) for k, v in oracle.items()}, robust=list(robust),
                  summary=summarize(erows, group),
                  paired={f'{a}-oracle_pid|{ch}': {k: paired(erows, f'{a}|{ch}', f'oracle_pid|{ch}', group, k)
                                                   for k in (key, 'tilt_peak_deg', 'height_rmse_m')}
                          for a in ('original_pid', 'robust_pid') for ch in CHANGES},
                  tuning_rows=rows, episodes=erows)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['proprio', 'adapt'])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--count-tune', type=int, default=8)
    p.add_argument('--count-eval', type=int, default=16)
    a = p.parse_args()
    with pool(a.scratch, a.workers) as executor:
        result = (proprio if a.stage == 'proprio' else adapt)(executor, a.count_tune, a.count_eval)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: result[k] for k in ('best', 'robust', 'oracle', 'summary') if k in result}, indent=1))


if __name__ == '__main__':
    main()
