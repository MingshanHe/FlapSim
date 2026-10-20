"""Plan v7 (quick validation): three manoeuvres on the v6 interface.

  evade     at an unannounced time t0 ~ U[2.2, 2.8] s, move EVADE_M = 0.10 m along phi as fast as possible, no stop
            required. 4 directions (0/90/180/270). Window 1.5 s.
            J = t_reach (miss: window + 10*shortfall_m) + 0.03*height loss (cm) + 0.5*|cross-track| (m) + crash
  redirect  dash along phi1 from t0 = 2.4 s; when 0.08 m along phi1 is reached (or 1.0 s passed) the command turns to
            phi2 = phi1 +- 90 deg; reach 0.10 m along phi2 from the switch point. 8 (phi1, turn) combos. Window 4 s.
            J = t_switch + t_reach2 (miss: rest of window + 10*shortfall) + 2*overrun along phi1 after the switch
                beyond 0.05 m + 0.03*height loss + crash
  dash      exactly the plan v6 dash (bench8.score8), 8 directions, the anchor task.

Command channels (bench8.features8, unchanged 25-D input): slot 0-1 = evade (cos, sin), slot 2-3 = dash / redirect
(the direction of the current leg). The learned layer is descend8.Learned8 unchanged (gate, tail, refractory, righting).
Classical opponent: Schedule9 = bench8.Schedule2 that restarts its go phase whenever the commanded direction changes.

Every episode can share its pre-command prefix (run_cases, same mechanism and check as cache8).

    python -m experiments.algorithms.spike_steer.bench9 selftest
    python -m experiments.algorithms.spike_steer.bench9 pilot --workers 30
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ReferencePID
from ..topology_hover.environment import case as make_case
from ..neuro_gates import gates
from . import bench as B
from . import bench8 as E
from . import pilot_v3 as V

TASKS = ('evade', 'redirect', 'dash')
DIRS = dict(evade=(0, 90, 180, 270), dash=E.DIRS, redirect=(0, 90, 180, 270))
TURNS = (90, -90)
SECONDS = dict(evade=5., redirect=6.5, dash=6.5)
WINDOW = dict(evade=1.5, redirect=4., dash=B.WINDOW['dash'])
EVADE_M, SWITCH_M, SWITCH_T, REDIRECT_M, OVERRUN_M = .10, .08, 1.0, .10, .05
BASES = dict(tune=2_098_000_000, train=2_098_200_000, test=2_098_100_000)
KLASS = {0: 'pitch', 180: 'pitch', 90: 'roll', 270: 'roll', 45: 'diag', 135: 'diag', 225: 'diag', 315: 'diag'}
CLASSES = dict(evade=('pitch', 'roll'), redirect=('pitch', 'roll'), dash=('pitch', 'roll', 'diag'))


def klass(c):
    """Direction class of a case: evade / dash by phi, redirect by the direction it turns to (phi2)."""
    return KLASS[int(c['dir2_deg'] if c['task'] == 'redirect' else c['dir_deg']) % 360]


def cases9(task, base, n, offset=0):
    out = []
    for i in range(n):
        c = make_case(base+offset+i, SECONDS[task], 'stress', i % 8)
        k = (i+i//8)
        c['task'] = task
        if task == 'redirect':
            phi1 = DIRS['redirect'][k % 4]
            turn = TURNS[(k//4) % 2]
            c['dir_deg'], c['dir2_deg'] = float(phi1), float((phi1+turn) % 360)
        else:
            c['dir_deg'] = float(DIRS[task][k % len(DIRS[task])])
        if task == 'evade':
            c['t0'] = 2.2+.6*float(np.random.default_rng(np.random.SeedSequence([c['seed'], 917])).random())
        else:
            c['t0'] = B.T0
        out.append(c)
    return out


class Schedule9(E.Schedule2):
    """2-D set-point schedule that follows the commanded direction: (re)starts its go phase whenever the command
    direction changes (redirect), brakes when the command drops (gated)."""

    def reset(self, c):
        self.t_go, self.t_end, self.phi = None, None, None
        self.u = np.zeros(2)

    def ref9(self, rel, cmd):
        on = bool(np.any(cmd))
        if on:
            k = 0 if np.any(cmd[:2]) else 2
            phi = round(math.degrees(math.atan2(cmd[k+1], cmd[k])) % 360, 6)
            if phi != self.phi:
                self.phi, self.u, self.t_go, self.t_end = phi, E.att_dir(phi), None, None
        return self.ref(rel, on)


class Classical9:
    def __init__(self, law=None, schedule=None):
        self.law, self.schedule = law, Schedule9(schedule) if schedule else None

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
        ref, active = (self.schedule.ref9(rel, cmd) if self.schedule else (np.zeros(2), False))
        return (pulse and not active), ref


def _dvec(phi_deg, fwd, left):
    p = math.radians(phi_deg)
    return math.cos(p)*fwd+math.sin(p)*left


class Episode:
    """One v7 episode as setup / tick / finish (the shape of cache8.Episode, task logic of this module)."""

    def __init__(self, env, task, c, policy):
        env.configure(c)
        env.reset(seed=c['seed'])
        env.upset_axes = {}
        p, (sp, sd) = env.p, B.PID_GAINS
        env.teacher = ReferencePID(dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']]), env.air.last_rpy)
        env._teacher_step = -1
        policy.reset(c)
        self.s = dict(env=env, oracle=gates.Oracle(env, 'derocked'), policy=policy, rows=[], pos0=None, fwd=None, left=None,
                      goal_done=False, goal_t=None, pulses=0, out=None, done=False, xs=[], task=task, c=c,
                      phase=1, t_switch=None, pos_switch=None)

    def t(self):
        return self.s['env'].air.steps*DT

    def tick(self):
        s = self.s
        env, task, c = s['env'], s['task'], s['c']
        d = env.sim.data
        t0 = c['t0']
        t, now = env.air.steps*DT, float(d.time)
        rel = t-t0
        if s['pos0'] is None and t >= t0:
            pos, _, _ = env.air.env._state()
            s['pos0'] = pos.copy()
            yaw = float(env.last_reading[2])
            s['fwd'] = np.array([math.cos(yaw), math.sin(yaw)])
            s['left'] = np.array([-s['fwd'][1], s['fwd'][0]])
        phi_cmd = c['dir2_deg'] if (task == 'redirect' and s['phase'] == 2) else c['dir_deg']
        cmd = np.zeros(4)
        if rel >= 0 and not s['goal_done']:
            k = 0 if task == 'evade' else 2
            a = math.radians(phi_cmd)
            cmd[k], cmd[k+1] = math.cos(a), math.sin(a)
        x = E.features8(env.last_reading, cmd)
        if rel < 0:
            s['xs'].append((t, x))
        fire, ref = s['policy'].act(env, now, t, x, cmd, rel)
        ref = np.asarray(ref, float)
        if ref.ndim == 0:
            ref = np.array([0., float(ref)])
        ref = np.clip(ref, -B.REF_MAX, B.REF_MAX)
        env.teacher.rpy_reference = ref
        command = env.teacher_command()
        if fire and t >= gates.HANDOVER_T:
            command[2] = HIGH[2]
            s['pulses'] += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH)-CENTER)/HALF)
        s['oracle'].update()
        pos, vel, R = env.air.env._state()
        roll_true, pitch_true = (float(v) for v in s['oracle'].queue[-1][:2])
        tnow = env.air.steps*DT
        # columns: 0 t, 1 pitch, 2 roll, 3-5 pos, 6-8 vel, 9 tilt, 10-11 ref (roll, pitch), 12 fire, 13 commanded phi (nan = off)
        s['rows'].append([tnow, pitch_true, roll_true, *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2])))), *ref, float(fire),
                          phi_cmd if np.any(cmd) else float('nan')])
        if s['pos0'] is not None and not s['goal_done']:
            disp = pos[:2]-s['pos0'][:2]
            if task == 'evade':
                if float(disp@_dvec(c['dir_deg'], s['fwd'], s['left'])) >= EVADE_M:
                    s['goal_done'], s['goal_t'] = True, tnow-t0
            elif task == 'dash':
                if float(disp@_dvec(c['dir_deg'], s['fwd'], s['left'])) >= B.DASH_M['value']:
                    s['goal_done'], s['goal_t'] = True, tnow-t0
            elif s['phase'] == 1:
                if float(disp@_dvec(c['dir_deg'], s['fwd'], s['left'])) >= SWITCH_M or tnow-t0 >= SWITCH_T:
                    s['phase'], s['t_switch'], s['pos_switch'] = 2, tnow-t0, pos.copy()
            elif float((pos[:2]-s['pos_switch'][:2])@_dvec(c['dir2_deg'], s['fwd'], s['left'])) >= REDIRECT_M:
                s['goal_done'], s['goal_t'] = True, tnow-t0
        s['out'] = out
        s['done'] = bool(term or trunc)

    def finish(self, trace=False):
        return score9(self.s, trace)


def score9(s, trace=False):
    task, c = s['task'], s['c']
    ep = s['out']['episode']
    a = np.array(s['rows'])
    if task == 'dash':                                          # the v6 anchor: identical scoring
        res = E.score8('dash', c, ep, a, s['pos0'], s['fwd'], s['left'], s['pulses'], s['goal_t'], trace)
        res.update(task='dash', cls=klass(c))
        return res
    t0, W = c['t0'], WINDOW[task]
    flown = ep['termination'] == 'time_limit'
    res = dict(seed=c['seed'], task=task, dir_deg=c['dir_deg'], dir2_deg=c.get('dir2_deg'), t0=t0, cls=klass(c),
               termination=ep['termination'], elapsed=ep['elapsed'], flown=flown, pulses=s['pulses'], goal_t=s['goal_t'],
               ref_abs_mean=float(np.abs(a[:, 10:12]).mean()))
    if trace:
        res['trace'] = a.tolist()
    pos0, fwd, left = s['pos0'], s['fwd'], s['left']
    if pos0 is None:
        pos0, fwd, left = a[0, 3:6], np.array([1., 0.]), np.array([0., 1.])
    w = a[(a[:, 0] >= t0) & (a[:, 0] < t0+W)]
    rel_t = w[:, 0]-t0
    dz = float(100*max(0., pos0[2]-w[:, 5].min())) if len(w) else 0.
    crash = 0. if flown else 10.+5.*float(np.clip((t0+W-ep['elapsed'])/W, 0., 1.))
    res.update(height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 9].max())) if len(w) else None)
    if task == 'evade':
        d1 = _dvec(c['dir_deg'], fwd, left)
        n1 = np.array([-d1[1], d1[0]])
        disp = w[:, 3:5]-pos0[:2]
        along, cross = disp@d1, disp@n1
        i = np.flatnonzero(along >= EVADE_M)
        treach = float(rel_t[i[0]]) if len(i) else None
        k = i[0] if len(i) else len(w)-1
        short = 0. if treach is not None else max(0., EVADE_M-(along.max() if len(along) else 0.))
        crossm = float(abs(cross[k])) if len(w) else 0.
        score = (treach if treach is not None else W+10.*short)+.03*dz+.5*crossm+crash
        res.update(t_reach=treach, cross_m=crossm, max_along_m=float(along.max()) if len(along) else None, score=float(score))
        return res
    d1, d2 = _dvec(c['dir_deg'], fwd, left), _dvec(c['dir2_deg'], fwd, left)
    ts = s['t_switch']
    if ts is None:
        score = W+10.*REDIRECT_M+.03*dz+crash
        res.update(t_switch=None, t_reach2=None, overrun_m=None, score=float(score))
        return res
    ps = s['pos_switch']
    after = w[rel_t >= ts]
    rel_a = after[:, 0]-t0
    along2 = (after[:, 3:5]-ps[:2])@d2
    over1 = (after[:, 3:5]-ps[:2])@d1
    i = np.flatnonzero(along2 >= REDIRECT_M)
    treach2 = float(rel_a[i[0]]-ts) if len(i) else None
    short = 0. if treach2 is not None else max(0., REDIRECT_M-(along2.max() if len(along2) else 0.))
    overrun = float(max(0., (over1.max() if len(over1) else 0.)-OVERRUN_M))
    score = ts+(treach2 if treach2 is not None else (W-ts)+10.*short)+2.*overrun+.03*dz+crash
    res.update(t_switch=ts, t_reach2=treach2, overrun_m=overrun, max_along2_m=float(along2.max()) if len(along2) else None,
               score=float(score))
    return res


def run_episode9(env, task, c, policy, trace=False):
    ep = Episode(env, task, c, policy)
    while not ep.s['done']:
        ep.tick()
    return ep.finish(trace)


# ----------------------------------------------------------------------------------------- pool / jobs
def job(args):
    """(name, task, case, 'classical', {law, schedule}) or (name, task, case, 'learned', spec)[, trace]."""
    name, task, c, kind, arg = args[:5]
    trace = args[5] if len(args) > 5 else False
    if kind == 'classical':
        pol = Classical9(arg.get('law'), arg.get('schedule'))
    else:
        from .descend8 import Learned8
        pol = Learned8(arg)
    return dict(run_episode9(B._ENV, task, c, pol, trace), controller=name)


def run_jobs(ex, jobs):
    return list(ex.map(job, jobs, chunksize=1))


def run_cases(env, task, c, specs, trace=False):
    """Learned specs [(name, spec)] on one case; the pre-command prefix (t < t0) is simulated once (cache8 mechanism)."""
    from . import cache8 as C8
    from .descend8 import Learned8, Net8
    ep = Episode(env, task, c, Learned8(specs[0][1]))
    while not ep.s['done'] and ep.t() < c['t0']-1e-9:
        ep.tick()
    base = ep.s
    out = []
    for name, spec in specs:
        st = C8.snapshot(base)
        if not base['done']:
            pol = st['policy']
            pol.spec, pol.net, pol.twitch = spec, Net8(spec), 0
            for (t, x) in st['xs']:
                pol.replay(t, x)
        e = Episode.__new__(Episode)
        e.s = st
        while not e.s['done']:
            e.tick()
        out.append(dict(e.finish(trace), controller=name))
    return out


def job_group(args):
    task, c, specs = args[:3]
    return run_cases(B._ENV, task, c, specs, args[3] if len(args) > 3 else False)


def run_grouped(ex, items, chunk=8):
    """items [(name, task, case, spec)] -> rows in input order; candidates of a case share the prefix."""
    groups = {}
    for k, (name, task, c, spec) in enumerate(items):
        groups.setdefault((task, c['seed']), (task, c, []))[2].append((k, name, spec))
    jobs, index = [], []
    for task, c, lst in groups.values():
        for i in range(0, len(lst), chunk):
            part = lst[i:i+chunk]
            jobs.append((task, c, [(n, s) for _, n, s in part]))
            index.append([k for k, _, _ in part])
    out = [None]*len(items)
    for idx, rows in zip(index, ex.map(job_group, jobs, chunksize=1)):
        for k, r in zip(idx, rows):
            out[k] = r
    return out


# ----------------------------------------------------------------------------------------- stages
GRIDS = dict(
    evade=[dict(go_ref=r, go_beats=n, brake_beats=b, brake_ref=br, gated=True)
           for r in (.35, .5, .6) for n in (6, 9, 12, 18) for (b, br) in ((0, 0.), (3, -.35))],
    redirect=[dict(go_ref=r, go_beats=n, brake_beats=b, brake_ref=-br, gated=True)
              for r in (.35, .5) for n in (12, 18, 30) for b in (3, 6) for br in (.35, .5)],
)


def stage_pilot(a):
    """Classical opponents: evade / redirect tuned on 8 tune cases per task (24 configs each, best per direction class);
    dash keeps the plan-v6 opponents (artifacts/spike_steer/plan_v6/baseline.json)."""
    law = B.law_tuned()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    v6 = json.loads(Path('artifacts/spike_steer/plan_v6/baseline.json').read_text())
    opp = dict(dash=v6['opponent']['dash'])
    summary = {}
    with E.pool(a.workers) as ex:
        for task in ('evade', 'redirect'):
            cs = cases9(task, BASES['tune'], 8)
            t1 = time.monotonic()
            rows = run_jobs(ex, [(str(k), task, c, 'classical', dict(law=law, schedule=q)) for k, q in enumerate(GRIDS[task]) for c in cs])
            opp[task], summary[task] = {}, {}
            for cl in CLASSES[task]:
                m = {k: float(np.mean([r['score'] for r in rows if r['controller'] == str(k) and r['cls'] == cl])) for k in range(len(GRIDS[task]))}
                kb = min(m, key=m.get)
                sel = [r for r in rows if r['controller'] == str(kb) and r['cls'] == cl]
                opp[task][cl] = GRIDS[task][kb]
                summary[task][cl] = dict(config=GRIDS[task][kb], mean=m[kb], goal=sum(r['goal_t'] is not None for r in sel), n=len(sel),
                                         crashes=sum(not r['flown'] for r in sel))
            summary[task]['crash_rate_all_configs'] = float(np.mean([not r['flown'] for r in rows]))
            summary[task]['wall_s'] = round(time.monotonic()-t1)
            print(task, json.dumps(summary[task]), flush=True)
    (out/'pilot9.json').write_text(json.dumps(dict(opponent=opp, summary=summary, law=law), indent=1))


def selftest():
    """Case generation, class mapping, and one classical episode per task (no pool)."""
    E.load_signs()
    B.DASH_M['value'] = .15
    for task in TASKS:
        cs = cases9(task, BASES['tune'], 16)
        print(task, sorted({(c['dir_deg'], c.get('dir2_deg')) for c in cs}), 'classes', sorted({klass(c) for c in cs}),
              't0', sorted({round(c['t0'], 2) for c in cs})[:4])
    B.worker_init(str(E.SOURCE), .15)
    law = B.law_tuned()
    for task, q in (('evade', dict(go_ref=.5, go_beats=9, brake_beats=0, brake_ref=0., gated=True)),
                    ('redirect', dict(go_ref=.35, go_beats=18, brake_beats=3, brake_ref=-.35, gated=True)),
                    ('dash', dict(go_ref=.35, go_beats=18, brake_beats=3, brake_ref=-.35, gated=True))):
        c = cases9(task, BASES['tune'], 2)[1]
        r = run_episode9(B._ENV, task, c, Classical9(law, q))
        print(task, {k: r.get(k) for k in ('dir_deg', 'dir2_deg', 'score', 'flown', 'goal_t', 't_reach', 't_switch', 't_reach2', 'overrun_m', 't_settle')})
    # dash anchor = bench8 bit for bit
    c = cases9('dash', BASES['tune'], 2)[1]
    q = dict(go_ref=.35, go_beats=18, brake_beats=3, brake_ref=-.35, gated=True)
    r9 = run_episode9(B._ENV, 'dash', c, Classical9(law, q))
    r8 = E.run_episode8(B._ENV, 'dash', c, E.Classical2(law, q))
    assert r9['score'] == r8['score'] and r9['pulses'] == r8['pulses'], (r9['score'], r8['score'])
    print('dash anchor identical to bench8:', r9['score'])
    print('bench9 selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'pilot'])
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--output', default='runs/spike_steer/v7')
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    if a.stage == 'selftest':
        return selftest()
    stage_pilot(a)


if __name__ == '__main__':
    main()
