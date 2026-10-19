"""Plan v6 (8-direction manoeuvres): nod and dash in 8 horizontal directions instead of the pitch plane only.

Direction convention: phi in the heading frame at the command onset, 0 = forward (heading), 90 = left, 180 = back,
270 = right; the commanded attitude direction is (pitch, roll) = A*(SP*cos phi, SR*sin phi), with the signs SP, SR
that make the body accelerate along phi (measured by `signs`, written to SIGNS below).

Tasks (same windows, durations and penalty constants as bench.py; the old tasks are the phi = 0 / 180 special case
except that the nod goal is now signed):
  nod8   from T0, tilt toward phi by >= 30 deg (projection of the derocked (pitch, roll) on the direction) held 0.1 s,
         then recover to a total tilt < 5 deg. Score = t30 + t_recover + 0.03*height loss + 0.5*drift + crash.
  dash8  from T0, travel DASH_M along phi and stop. Score = t_reach + t_settle + 2*overshoot + 0.03*height loss
         + 0.5*cross-track + crash.
Classical opponent: a 2-D set-point schedule along phi (go A for n beats, brake B opposite for m beats), tuned on the
same budget, on top of the reflex PID and the hover pulse law (bench.Classical generalised).

bench.py is not modified; this module reuses its pool / worker environment and constants.

    python -m experiments.algorithms.spike_steer.bench8 signs --workers 8
    python -m experiments.algorithms.spike_steer.bench8 pilot --workers 30 --output runs/spike_steer/bench8_pilot
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ReferencePID
from ..topology_hover.environment import case as make_case
from ..neuro_gates import gates
from . import bench as B
from . import pilot_v3 as V
from . import pilot_v6 as Y

DIRS = tuple(range(0, 360, 45))
TASK_DIRS = dict(nod=(0, 90, 180, 270), dash=DIRS)    # user decision: nod 4 directions (pitch and roll axes), dash 8
SIGNS = dict(SP=1., SR=1.)              # set by `signs` (see SIGNS_FILE); attitude -> motion direction
SIGNS_FILE = Path('artifacts/spike_steer/plan_v6/signs.json')
NOD_DEG = 30.
BASES = dict(tune=2_097_000_000, train=2_097_200_000, val=2_097_260_000)
SOURCE = Path('runs/neuro_gates/scratch/source')


def load_signs():
    if SIGNS_FILE.exists():
        SIGNS.update(json.loads(SIGNS_FILE.read_text())['signs'])


def att_dir(phi_deg):
    """Unit attitude direction (roll, pitch) for motion direction phi."""
    p = math.radians(phi_deg)
    return np.array([SIGNS['SR']*math.sin(p), SIGNS['SP']*math.cos(p)])


def cases8(task, base, n, offset=0, dirs=None):
    """Direction-stratified cases: every block of len(dirs) has each direction once, rotated by block so that the
    stress stratum (i % 8) and the direction are crossed. dirs defaults to TASK_DIRS[task]."""
    dirs = TASK_DIRS[task] if dirs is None else dirs
    out = []
    for i in range(n):
        c = make_case(base+offset+i, B.SECONDS[task], 'stress', i % 8)
        c['dir_deg'] = float(dirs[(i+i//8) % len(dirs)])
        out.append(c)
    return out


def features8(reading, cmd4):
    """20 IMU features as bench.features without the 2 command slots + 8 rectified direction-command channels."""
    x = np.asarray(reading, dtype=float)[list(Y.IDX)]/Y.SCALE
    c = np.asarray(cmd4, float)
    return np.r_[np.maximum(x, 0.), np.maximum(-x, 0.), 1., np.maximum(c, 0.), np.maximum(-c, 0.)]


class Schedule2:
    """bench.Schedule along a direction: go (magnitude go_ref along the task direction, <= go_beats, ends early when the
    goal signal drops if gated), brake (brake_ref, a signed magnitude along the same direction), then 0. `vec` (roll,
    pitch) overrides the direction (used by `signs`)."""

    def __init__(self, q):
        self.q = dict(q)
        self.u = np.array(q['vec'], float) if 'vec' in q else None

    def reset(self, c):
        self.t_go, self.t_end = None, None
        if 'vec' not in self.q:
            self.u = att_dir(c['dir_deg'])

    def ref(self, rel, cmd_on):
        q = self.q
        if rel < 0:
            return np.zeros(2), False
        if self.t_go is None:
            self.t_go = rel
        if self.t_end is None:
            if rel-self.t_go < q['go_beats']*B.BEAT and (cmd_on or not q['gated']):
                return q['go_ref']*self.u, True
            self.t_end = rel
        if rel-self.t_end < q['brake_beats']*B.BEAT:
            return q['brake_ref']*self.u, True
        return np.zeros(2), False


class Classical2:
    def __init__(self, law=None, schedule=None):
        self.law, self.schedule = law, Schedule2(schedule) if schedule else None

    def reset(self, c):
        self.steer = V.SteerV3(None, self.law, False) if self.law else None
        if self.steer:
            self.steer.reset()
        if self.schedule:
            self.schedule.reset(c)

    def act(self, env, now, t, x, cmd, rel):
        pulse = False
        if self.steer:
            pulse, _ = self.steer.step(env, now, t)
        ref, active = (self.schedule.ref(rel, bool(np.any(cmd))) if self.schedule else (np.zeros(2), False))
        return (pulse and not active), ref


def run_episode8(env, task, c, policy, trace=False):
    """bench.run_episode for nod8 / dash8. policy.act returns (fire, ref) with ref = (roll, pitch) set-point."""
    env.configure(c)
    env.reset(seed=c['seed'])
    env.upset_axes = {}
    p, (sp, sd) = env.p, B.PID_GAINS
    env.teacher = ReferencePID(dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']]), env.air.last_rpy)
    env._teacher_step = -1
    policy.reset(c)
    d = env.sim.data
    oracle = gates.Oracle(env, 'derocked')
    T0, window = B.T0, B.WINDOW[task]
    phi = math.radians(c['dir_deg'])
    u_att = att_dir(c['dir_deg'])
    rows, pos0, fwd, left, goal_done, goal_t, pulses = [], None, None, None, False, None, 0
    hold = int(round(B.HOLD_S/DT))
    big = 0
    while True:
        t, now = env.air.steps*DT, float(d.time)
        rel = t-T0
        if pos0 is None and t >= T0:
            pos, _, _ = env.air.env._state()
            pos0 = pos.copy()
            yaw = float(env.last_reading[2])
            fwd = np.array([math.cos(yaw), math.sin(yaw)])
            left = np.array([-fwd[1], fwd[0]])
        cmd = np.zeros(4)
        if rel >= 0 and not goal_done:
            k = 0 if task == 'nod' else 2
            cmd[k], cmd[k+1] = math.cos(phi), math.sin(phi)
        x = features8(env.last_reading, cmd)
        fire, ref = policy.act(env, now, t, x, cmd, rel)
        ref = np.asarray(ref, float)
        if ref.ndim == 0:
            ref = np.array([0., float(ref)])
        ref = np.clip(ref, -B.REF_MAX, B.REF_MAX)
        env.teacher.rpy_reference = ref
        command = env.teacher_command()
        if fire and t >= gates.HANDOVER_T:
            command[2] = HIGH[2]
            pulses += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pos, vel, R = env.air.env._state()
        roll_true, pitch_true = (float(v) for v in oracle.queue[-1][:2])
        tnow = env.air.steps*DT
        along = float(np.degrees(np.array([roll_true, pitch_true])@u_att))
        # columns: 0 t, 1 pitch, 2 roll, 3-5 pos, 6-8 vel, 9 tilt, 10-11 ref (roll, pitch), 12 fire, 13 tilt along phi (deg)
        rows.append([tnow, pitch_true, roll_true, *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2])))), *ref, float(fire), along])
        if task == 'nod' and not goal_done and rel >= 0:
            big = big+1 if along >= NOD_DEG else 0
            if big >= hold:
                goal_done, goal_t = True, tnow-T0
        if task == 'dash' and not goal_done and pos0 is not None:
            dvec = math.cos(phi)*fwd+math.sin(phi)*left
            if float((pos[:2]-pos0[:2])@dvec) >= B.DASH_M['value']:
                goal_done, goal_t = True, tnow-T0
        if term or trunc:
            break
    return score8(task, c, out['episode'], np.array(rows), pos0, fwd, left, pulses, goal_t, trace)


def score8(task, c, ep, a, pos0, fwd, left, pulses, goal_t, trace=False):
    """Scores of an 8-direction episode from its per-tick rows `a` (shared by run_episode8 and cache8)."""
    T0, window = B.T0, B.WINDOW[task]
    phi = math.radians(c['dir_deg'])
    hold = int(round(B.HOLD_S/DT))
    flown = ep['termination'] == 'time_limit'
    res = dict(seed=c['seed'], task=task, dir_deg=c['dir_deg'], termination=ep['termination'], elapsed=ep['elapsed'], flown=flown,
               pulses=pulses, goal_t=goal_t, ref_abs_mean=float(np.abs(a[:, 10:12]).mean()),
               heading=None if fwd is None else fwd.tolist())
    if trace:
        res['trace'] = a.tolist()
    w = a[(a[:, 0] >= T0) & (a[:, 0] < T0+window)]
    rel_t = w[:, 0]-T0
    if pos0 is None:
        pos0, fwd, left = a[0, 3:6], np.array([1., 0.]), np.array([0., 1.])
    dvec = math.cos(phi)*fwd+math.sin(phi)*left
    nvec = np.array([-dvec[1], dvec[0]])
    disp = w[:, 3:5]-pos0[:2]
    along_m, cross_m = disp@dvec, disp@nvec
    v_along = w[:, 6:8]@dvec
    tilt_tot = np.degrees(np.hypot(w[:, 1], w[:, 2]))
    dz = float(100*max(0., pos0[2]-w[:, 5].min())) if len(w) else 0.
    crash = 0. if flown else 10.+5.*float(np.clip((T0+window-ep['elapsed'])/window, 0., 1.))
    miss = window

    def first(mask, start=0.):
        i = np.flatnonzero(mask & (rel_t >= start))
        return float(rel_t[i[0]]) if len(i) else None
    res.update(height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 9].max())) if len(w) else None)
    if task == 'nod':
        bigm = w[:, 13] >= NOD_DEG
        sustained = np.array([bigm[i:i+hold].all() if i+hold <= len(bigm) else False for i in range(len(bigm))])
        t30 = first(sustained)
        trec = first(tilt_tot < 5., (t30 or 0.)+.02) if t30 is not None else None
        drift = float(np.hypot(*disp[-1])) if len(disp) else 0.
        score = (t30 if t30 is not None else miss)+((trec-t30) if (t30 is not None and trec is not None) else miss)+.03*dz+.5*drift+crash
        res.update(t30=t30, t_recover=None if (t30 is None or trec is None) else trec-t30, drift_m=drift,
                   peak_along_deg=float(w[:, 13].max()) if len(w) else None, score=float(score))
    else:
        target = B.DASH_M['value']
        treach = first(along_m >= target)
        over = float(max(0., along_m.max()-(target+B.OVER_M))) if len(along_m) else 0.
        tsettle = None
        if treach is not None:
            slow = np.abs(v_along) < .1
            for i in np.flatnonzero(rel_t >= treach):
                j = i+int(.3/DT)
                if j <= len(slow) and slow[i:j].all():
                    tsettle = float(rel_t[i]-treach)
                    break
        short = 0. if treach is not None else 2.*max(0., target-(along_m.max() if len(along_m) else 0.))
        score = (treach if treach is not None else miss+short)+(tsettle if tsettle is not None else miss)+2.*over+.03*dz \
            + .5*float(abs(cross_m[-1]) if len(cross_m) else 0.)+crash
        res.update(t_reach=treach, t_settle=tsettle, overshoot_m=over, max_along_m=float(along_m.max()) if len(along_m) else None,
                   cross_m=float(abs(cross_m[-1])) if len(cross_m) else None, score=float(score))
    return res




# ----------------------------------------------------------------------------------------- pool
def _init(source, dash_m, signs):
    B.worker_init(source, dash_m)
    SIGNS.update(signs)


def pool(workers, dash_m=.15):
    if not (SOURCE/'robot_upper7_lower8.xml').exists():
        prepare_source(SOURCE)
    return ProcessPoolExecutor(workers, initializer=_init, initargs=(str(SOURCE), dash_m, dict(SIGNS)))


def job(args):
    name, tname, c, kind, arg = args[:5]
    trace = args[5] if len(args) > 5 else False
    pol = Classical2(arg.get('law'), arg.get('schedule')) if kind == 'classical' else arg['factory'](arg)
    return dict(run_episode8(B._ENV, tname, c, pol, trace), controller=name)


def run_jobs(ex, jobs):
    return list(ex.map(job, jobs, chunksize=1))


# ----------------------------------------------------------------------------------------- stages
def stage_signs(a):
    """Constant 0.2 rad set-point on one axis for 10 beats (0.4 s) from T0, then 0; displacement in the heading frame
    at T0 + 1.2 s (mean over 8 cases)."""
    law = B.law_tuned()
    cs = cases8('dash', BASES['tune'], 8, dirs=DIRS)
    out = {}
    with pool(a.workers) as ex:
        for name, vec in (('pitch+', (0., .2)), ('roll+', (.2, 0.))):
            sched = dict(vec=vec, go_ref=1., go_beats=10, brake_beats=0, brake_ref=0., gated=False)
            rows = run_jobs(ex, [(name, 'dash', c, 'classical', dict(law=law, schedule=sched), True) for c in cs])
            fw, lf = [], []
            for r, c in zip(rows, cs):
                tr = np.array(r['trace'])
                i0, i1 = int(np.searchsorted(tr[:, 0], B.T0)), int(np.searchsorted(tr[:, 0], B.T0+1.2))
                i1 = min(i1, len(tr)-1)
                d = tr[i1, 3:5]-tr[i0, 3:5]
                h = np.array(r['heading'])
                fw.append(float(d@h))
                lf.append(float(d@np.array([-h[1], h[0]])))
            out[name] = dict(forward_m=float(np.mean(fw)), left_m=float(np.mean(lf)), forward_all=np.round(fw, 3).tolist(),
                             left_all=np.round(lf, 3).tolist(), flown=sum(r['flown'] for r in rows))
            print(name, json.dumps(out[name]), flush=True)
    print(json.dumps(out, indent=1))


def stage_pilot(a):
    """Feasibility: tune the 2-D classical schedule on 16 cases per task (2 per direction); report per direction the best
    config's score, goal rate and crashes, for the global best and for the best per direction class."""
    law = B.law_tuned()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    grids = dict(
        nod=[dict(go_ref=s*r, go_beats=n, brake_beats=b, brake_ref=-s*r2, gated=g)
             for s in (1, -1) for r in (.35, .5, .6) for n in (6, 9, 12) for b in (0, 3, 6) for r2 in ((0.,) if b == 0 else (.2, .35))
             for g in (True, False)],
        dash=[dict(go_ref=r, go_beats=n, brake_beats=b, brake_ref=-r2, gated=g)
              for r in (.2, .35, .5) for n in (18, 30, 45) for b in (3, 6, 12) for r2 in (.35, .5) for g in (True, False)])
    klass = {0: 'pitch', 180: 'pitch', 90: 'roll', 270: 'roll', 45: 'diag', 135: 'diag', 225: 'diag', 315: 'diag'}
    summary = {}
    with pool(a.workers) as ex:
        for task in ('nod', 'dash'):
            cs = cases8(task, BASES['tune'], 16, dirs=DIRS)
            t_start = time.monotonic()
            jobs = [(f'{task}_{k}', task, c, 'classical', dict(law=law, schedule=q)) for k, q in enumerate(grids[task]) for c in cs]
            rows = run_jobs(ex, jobs)
            per = {}
            for r in rows:
                per.setdefault(r['controller'], []).append(r)
            def stats(rs):
                by = {}
                for d in DIRS:
                    sel = [r for r in rs if r['dir_deg'] == d]
                    by[d] = dict(score=float(np.mean([r['score'] for r in sel])), goal=sum(r['goal_t'] is not None for r in sel),
                                 crashes=sum(not r['flown'] for r in sel), n=len(sel))
                return by
            table = {k: stats(v) for k, v in per.items()}
            mean = {k: float(np.mean([r['score'] for r in v])) for k, v in per.items()}
            best = min(mean, key=mean.get)
            best_cls = {}
            for cl in ('pitch', 'roll', 'diag'):
                ds = [d for d in DIRS if klass[d] == cl]
                m = {k: float(np.mean([table[k][d]['score'] for d in ds])) for k in table}
                kb = min(m, key=m.get)
                best_cls[cl] = dict(config=grids[task][int(kb.split('_')[1])], mean=m[kb], per_dir={d: table[kb][d] for d in ds})
            summary[task] = dict(n_configs=len(grids[task]), n_cases=len(cs), wall_s=round(time.monotonic()-t_start),
                                 best_global=dict(config=grids[task][int(best.split('_')[1])], mean=mean[best], per_dir=table[best]),
                                 best_per_class=best_cls)
            print(task, json.dumps(summary[task]['best_global']), flush=True)
            for cl, v in best_cls.items():
                print(' ', cl, json.dumps(v), flush=True)
            (out/f'pilot_{task}_rows.json').write_text(json.dumps(rows))
    (out/'pilot_summary.json').write_text(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['signs', 'pilot'])
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--output', default='runs/spike_steer/bench8_pilot')
    a = p.parse_args()
    load_signs()
    {'signs': stage_signs, 'pilot': stage_pilot}[a.stage](a)


if __name__ == '__main__':
    main()
