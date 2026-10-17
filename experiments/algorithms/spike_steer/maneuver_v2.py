"""Combined benchmark v2 (research plan v4, stage S0): maneuver tasks T2 (nod) and T3 (dash-and-stop) with a maneuver
module that has SETPOINT AUTHORITY over the hover PID.

Why v2: the first feasibility run (artifacts/spike_steer/maneuver/report.md) showed that a maneuver module stacked on
top of a PID that keeps targeting 0 deg cannot nod (S1 reached a sustained 30 deg in 3/8 cases) and cannot dash 0.5 m
(max 0.20 m). Here the module rewrites the PID's attitude reference (ReferencePID.rpy_reference) for the scheduled
segments, optionally adds phase-locked motor pulses and an alpha residual, then hands back (reference -> 0).

Everything stays inside the original interface (2 ms commands, LOW/HIGH, +-0.3 rad). Truth is used for scoring only.

T2 nod:  reach |pitch| >= 30 deg (held >= 0.1 s) as fast as possible, then return below 5 deg as fast as possible.
         score = t_30 + t_recover + 0.03 height_loss_cm + 0.5 |drift_m| (+ crash); misses cost WINDOW each.
T3 dash: move DASH_M = 0.25 m forward (heading frame at t0) and stop, window 4 s.
         score = t_reach + t_settle + 2 overshoot_m + 0.03 height_loss_cm + 0.5 |lateral_m| (+ crash); misses cost WINDOW.

Arms:   hover_only; S1 schedule grid (setpoint + optional pulses, 2-3 segments); S2 schedule CEM (per-segment beats,
        pitch reference, alpha residual; crash-constrained elites); S3 privileged truth-rate brake (headroom indicator).
Cases:  8 stress tuning cases, 16 held-out stress test cases (blind to tuning).

    python -m experiments.algorithms.spike_steer.maneuver_v2 selftest
    python -m experiments.algorithms.spike_steer.maneuver_v2 all --output runs/spike_steer/maneuver_v2 --workers 30 [--quick]
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
from ..robust_control.environment import ResearchEnv, ReferencePID
from ..topology_hover.environment import case as make_case
from ..neuro_gates import gates
from . import pilot_v3 as V

T0 = 2.4
WINDOW = dict(nod=2.5, dash=4.)
SECONDS = {k: T0+w+.1 for k, w in WINDOW.items()}
HOLD_S, DASH_M, OVER_M = .1, .25, .1
BEAT, PULSE_WIN = .04, math.radians(45.)
UP, DOWN = math.radians(135.), math.radians(315.)
PID_GAINS = (1.5, .85)
BASES = dict(tune=2_096_000_000, test=2_096_100_000)
N_TUNE, N_TEST = 8, 16
TASKS = ('nod', 'dash')
REF_MAX = .7      # rad, pitch reference bound for the module (the PID's own bound is the +-0.3 rad gimbal)
CEM = dict(pop=16, iters=5, elite=4)
CRASH_PENALTY = 100.   # added to a candidate's tuning score for each crashed case: elites are all-flown candidates


# segment = (beats, pulse_dir in {-1,0,1}, alpha_residual rad, pitch_reference rad)
def s1_grid(task):
    out = []
    for s in (1, -1):
        for pulse in (0, 1):
            if task == 'nod':
                for n in (3, 6, 9):
                    for th in (.35, .6):
                        for b in (0, 3, 6):
                            for th2 in ((0.,) if b == 0 else (0., .2)):
                                out.append([(n, s*pulse, 0., s*th), (b, -s*pulse, 0., -s*th2)])
            else:
                for n in (12, 18, 30):
                    for th in (.2, .35, .5):
                        for c in (0,):
                            for b in (6, 12):
                                for th2 in (.2, .35):
                                    out.append([(n, s*pulse, 0., s*th), (c, 0, 0., 0.), (b, -s*pulse, 0., -s*th2)])
    return out


S1_GRID = {t: s1_grid(t) for t in TASKS}
S2_SEGMENTS = dict(nod=2, dash=3)
S2_DIRS = dict(nod=(1, -1), dash=(1, 0, -1))
BEATS_MAX = dict(nod=12, dash=30)
S3_DIM = 7   # go beats, go alpha, go pitch ref, brake gain on truth pitch rate (pulse), brake gain alpha, threshold, hold beats


def decode_s2(x, s, task):
    x = np.asarray(x, dtype=float)
    k = S2_SEGMENTS[task]
    beats = np.clip(np.round(x[:k]*BEATS_MAX[task]), 0, BEATS_MAX[task]).astype(int)
    refs = np.clip((x[k:2*k]*2.-1.)*REF_MAX, -REF_MAX, REF_MAX)
    alphas = np.clip((x[2*k:3*k]*2.-1.)*.3, -.3, .3)
    return [(int(b), int(s*d), float(a), float(r)) for b, d, a, r in zip(beats, S2_DIRS[task], alphas, refs)]


def decode_s3(x):
    x = np.asarray(x, dtype=float)
    return [int(np.clip(round(x[0]*12.), 1, 12)), float(np.clip((x[1]*2.-1.)*.3, -.3, .3)), float(np.clip(x[2]*REF_MAX, 0., REF_MAX)),
            float(np.clip(x[3]*1.5, 0., 1.5)), float(np.clip(x[4]*.1, 0., .1)), float(np.clip(x[5], 0., 1.)), int(np.clip(round(x[6]*6.), 0, 6))]


class Maneuver:
    """Open-loop schedule of segments with setpoint authority, or the privileged feedback brake (S3)."""

    def __init__(self, kind, params, sign=1):
        self.kind, self.params, self.sign = kind, params, sign

    def duration(self):
        if self.kind == 'none':
            return 0.
        if self.kind == 'schedule':
            return sum(b for b, *_ in self.params)*BEAT
        return max(WINDOW.values())

    def command(self, rel, pitch_rate_truth):
        """(pulse_dir, alpha residual, pitch reference) for time rel since t0; outside the maneuver: (0, 0, 0)."""
        if self.kind == 'none' or rel < 0 or rel >= self.duration():
            return 0, 0., 0.
        if self.kind == 'schedule':
            acc = 0.
            for b, d, a, r in self.params:
                if acc <= rel < acc+b*BEAT:
                    return d, a, r
                acc += b*BEAT
            return 0, 0., 0.
        go_b, go_a, go_r, k_pulse, k_alpha, thr, hold_b = self.params
        if rel < go_b*BEAT:
            return self.sign, go_a, self.sign*go_r
        if rel < (go_b+hold_b)*BEAT:
            return 0, 0., 0.
        u = -k_pulse*pitch_rate_truth
        d = int(np.sign(u)) if abs(u) > thr else 0
        return d, float(np.clip(-k_alpha*pitch_rate_truth, -.3, .3)), 0.


_LAW = None


def run(env, c, task, man):
    global _LAW
    if _LAW is None:
        _LAW = json.loads(V.V2_LAW.read_text())['best']
    env.configure(dict(c, seconds=SECONDS[task]))
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, PID_GAINS
    # setpoint authority: the hover PID becomes a ReferencePID (derivative on measured attitude, not on setpoint steps)
    env.teacher = ReferencePID(dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']]), env.air.last_rpy)
    env._teacher_step = -1
    steer = V.SteerV3(None, _LAW, False)
    steer.reset()
    m, d = env.sim.model, env.sim.data
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    oracle = gates.Oracle(env, 'derocked')
    rows, heading, pos0, pulses = [], None, None, 0
    window = WINDOW[task]
    while True:
        t, now = env.air.steps*DT, float(d.time)
        rel = t-T0
        if pos0 is None and t >= T0:
            pos, _, R = env.air.env._state()
            pos0 = pos.copy()
            yaw = float(env.last_reading[2])
            heading = np.array([math.cos(yaw), math.sin(yaw)])
        dirn, alpha, ref = man.command(rel, float(oracle.rate[1]))
        env.teacher.rpy_reference = np.array([0., ref])
        cmd = env.teacher_command()
        pulse, _ = steer.step(env, now, t)
        if 0 <= rel < man.duration():
            if dirn and abs(math.remainder(float(d.qpos[crank])-(UP if dirn > 0 else DOWN), 2*math.pi)) < PULSE_WIN/2:
                cmd[2] = HIGH[2]
                pulses += 1
            cmd[0] += alpha
        elif pulse and t >= gates.HANDOVER_T:
            cmd[2] = HIGH[2]
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pos, vel, R = env.air.env._state()
        rows.append([env.air.steps*DT, oracle.queue[-1][1], *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2]))))])
        if term or trunc:
            break
    ep = out['episode']
    a = np.array(rows)
    w = a[(a[:, 0] >= T0) & (a[:, 0] < T0+window)]
    flown = ep['termination'] == 'time_limit'
    rel_t = w[:, 0]-T0
    pitch = np.degrees(w[:, 1])
    if pos0 is None:
        pos0 = a[0, 2:5]
        heading = np.array([1., 0.])
    fwd = (w[:, 2:4]-pos0[:2])@heading
    lat = (w[:, 2:4]-pos0[:2])@np.array([-heading[1], heading[0]])
    vfwd = w[:, 5:7]@heading
    dz = float(100*max(0., pos0[2]-w[:, 4].min())) if len(w) else 0.
    crash = 0. if flown else 10.+5.*float(np.clip((T0+window-ep['elapsed'])/window, 0., 1.))
    miss = window

    def first(mask, start=0.):
        i = np.flatnonzero(mask & (rel_t >= start))
        return float(rel_t[i[0]]) if len(i) else None
    res = dict(seed=c['seed'], task=task, termination=ep['termination'], flown=flown, pulses=pulses,
               height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 8].max())) if len(w) else None, duration=man.duration())
    if task == 'nod':
        hold = int(round(HOLD_S/DT))
        big = np.abs(pitch) >= 30.
        sustained = np.array([big[i:i+hold].all() if i+hold <= len(big) else False for i in range(len(big))])
        t30 = first(sustained)
        trec = first(np.abs(pitch) < 5., (t30 or 0.)+.02) if t30 is not None else None
        drift = float(abs(fwd[-1])) if len(fwd) else 0.
        score = (t30 if t30 is not None else miss)+((trec-t30) if (t30 is not None and trec is not None) else miss)+.03*dz+.5*drift+crash
        res.update(t30=t30, t_recover=None if (t30 is None or trec is None) else trec-t30, drift_m=drift,
                   peak_pitch_deg=float(np.abs(pitch).max()) if len(pitch) else None, score=float(score))
    else:
        treach = first(fwd >= DASH_M)
        over = float(max(0., fwd.max()-(DASH_M+OVER_M))) if len(fwd) else 0.
        tsettle = None
        if treach is not None:
            slow = np.abs(vfwd) < .1
            for i in np.flatnonzero((rel_t >= treach)):
                j = i+int(.3/DT)
                if j <= len(slow) and slow[i:j].all():
                    tsettle = float(rel_t[i]-treach)
                    break
        short = 0. if treach is not None else 2.*max(0., DASH_M-(fwd.max() if len(fwd) else 0.))
        score = (treach if treach is not None else miss+short)+(tsettle if tsettle is not None else miss)+2.*over+.03*dz \
            + .5*float(abs(lat[-1]) if len(lat) else 0.)+crash
        res.update(t_reach=treach, t_settle=tsettle, overshoot_m=over, max_forward_m=float(fwd.max()) if len(fwd) else None,
                   lateral_m=float(abs(lat[-1])) if len(lat) else None, score=float(score))
    return res


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')


def task(args):
    name, c, tname, kind, params, sign = args
    return dict(run(_ENV, c, tname, Maneuver(kind, params, sign)), controller=name)


def cases(base, n, task):
    return [make_case(base+i, SECONDS[task], 'stress', i % 8) for i in range(n)]


def mean_score(rows):
    return float(np.mean([r['score'] for r in rows]))


def tune_score(rows):
    """Tuning objective: mean score plus a hard crash penalty (a crashed candidate never becomes an elite)."""
    return mean_score(rows)+CRASH_PENALTY*sum(not r['flown'] for r in rows)


def summary(rows):
    keys = [k for k in ('t30', 't_recover', 'drift_m', 'peak_pitch_deg', 't_reach', 't_settle', 'overshoot_m', 'max_forward_m',
                        'lateral_m', 'height_loss_cm', 'tilt_peak_deg', 'pulses') if k in rows[0]]
    out = dict(episodes=len(rows), flown=sum(r['flown'] for r in rows), score=mean_score(rows))
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
        if k in ('t30', 't_reach', 't_settle'):
            out[k+'_hits'] = len(v)
    return out


def cem(ex, tname, kind, dim, tune, rng, decode, quick, sign):
    pop, iters = (CEM['pop'], CEM['iters']) if not quick else (6, 2)
    mu, sd = np.full(dim, .5), np.full(dim, .3)
    best, history = None, []
    for it in range(iters):
        xs = np.clip(mu+sd*rng.standard_normal((pop, dim)), 0., 1.)
        jobs = []
        for i, x in enumerate(xs):
            params = decode(x, sign, tname) if kind == 'schedule' else decode(x)
            jobs += [(str(i), c, tname, kind, params, sign) for c in tune]
        rows = list(ex.map(task, jobs, chunksize=1))
        per_i = {i: tune_score([r for r in rows if r['controller'] == str(i)]) for i in range(pop)}
        flown_i = {i: sum(r['flown'] for r in rows if r['controller'] == str(i)) for i in range(pop)}
        order = sorted(per_i, key=per_i.get)
        elite = xs[order[:CEM['elite']]]
        mu, sd = elite.mean(0), np.maximum(elite.std(0), .05)
        i0 = order[0]
        cand = dict(x=xs[i0].tolist(), sign=sign, score=per_i[i0], flown=flown_i[i0],
                    params=decode(xs[i0], sign, tname) if kind == 'schedule' else decode(xs[i0]))
        if best is None or cand['score'] < best['score']:
            best = cand
        history.append(dict(iter=it, best=per_i[i0], best_flown=flown_i[i0], mean=float(np.mean(list(per_i.values()))),
                            all_flown=sum(v == len(tune) for v in flown_i.values())))
        print(json.dumps(dict(task=tname, arm=kind, **history[-1])), flush=True)
    return best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['selftest', 'all'])
    ap.add_argument('--output', type=Path, default=Path('runs/spike_steer/maneuver_v2'))
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tasks', default='nod,dash')
    a = ap.parse_args()
    if a.stage == 'selftest':
        return selftest()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    rng = np.random.default_rng(a.seed)
    tasks = [t for t in a.tasks.split(',') if t in TASKS]
    prereg = dict(
        version='maneuver_v2 (research plan v4, stage S0)',
        change_from_v1='maneuver module has setpoint authority (ReferencePID.rpy_reference); T3 target 0.25 m / 4 s window; '
                       '8 tuning + 16 blind test cases; CEM elites must fly all tuning cases',
        tasks=dict(nod='t_30 (|pitch| >= 30 deg held >= 0.1 s) + t_recover (to < 5 deg) + .03 dz_cm + .5 |drift_m| (+ crash 10 + 5 x unflown); miss = 2.5 s each',
                   dash=f't_reach ({DASH_M} m) + t_settle + 2 overshoot_m (beyond {DASH_M+OVER_M} m) + .03 dz_cm + .5 |lateral_m| (+ crash); miss = 4 s (+2 x shortfall)'),
        arms=['hover_only', 'S1 schedule grid (setpoint + optional pulses)', 'S2 schedule CEM (crash-constrained)', 'S3 privileged truth-rate brake CEM'],
        gate_S0='classical S1 (or S2) reaches a sustained 30 deg in >= 12/16 test cases (nod) and 0.25 m in >= 12/16 (dash); '
                'otherwise the task definition is revised before any learner is trained',
        headroom_rule='S2 or S3 beats S1 by >= 20 % on the 16 test cases -> learning room; the classical opponent for the composite '
                      'benchmark is the better of S1/S2 on the TEST set only if it was selected on the TUNING set (it is)',
        cases={t: dict(tune=[c['seed'] for c in cases(BASES['tune'], 2 if a.quick else N_TUNE, t)],
                       test=[c['seed'] for c in cases(BASES['test'], 2 if a.quick else N_TEST, t)]) for t in TASKS},
        time=time.strftime('%Y-%m-%d %H:%M:%S'), quick=bool(a.quick))
    (out/'preregistration.json').write_text(json.dumps(prereg, indent=1))
    result = dict(preregistration=prereg)
    with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
        for tname in tasks:
            tune, test = cases(BASES['tune'], 2 if a.quick else N_TUNE, tname), cases(BASES['test'], 2 if a.quick else N_TEST, tname)
            t0 = time.monotonic()
            grid = S1_GRID[tname] if not a.quick else S1_GRID[tname][::len(S1_GRID[tname])//6]
            rows = list(ex.map(task, [(str(i), c, tname, 'schedule', g, 1) for i, g in enumerate(grid) for c in tune], chunksize=1))
            s1_scores = {i: tune_score([r for r in rows if r['controller'] == str(i)]) for i in range(len(grid))}
            s1_flown = {i: sum(r['flown'] for r in rows if r['controller'] == str(i)) for i in range(len(grid))}
            s1_best = min(s1_scores, key=s1_scores.get)
            print(json.dumps(dict(task=tname, arm='S1', best=grid[s1_best], score=s1_scores[s1_best], flown=s1_flown[s1_best],
                                  all_flown_candidates=sum(v == len(tune) for v in s1_flown.values()), grid=len(grid),
                                  wall=round(time.monotonic()-t0))), flush=True)
            s1_sign = 1 if grid[s1_best][0][3] >= 0 else -1
            s2, s2_hist = cem(ex, tname, 'schedule', 3*S2_SEGMENTS[tname], tune, rng, decode_s2, a.quick, s1_sign)
            s3, s3_hist = cem(ex, tname, 's3', S3_DIM, tune, rng, lambda x: decode_s3(x), a.quick, s1_sign)
            arms = [('hover_only', 'none', [], 1), ('S1', 'schedule', grid[s1_best], 1), ('S2', 'schedule', s2['params'], s2['sign']),
                    ('S3', 's3', s3['params'], s3['sign'])]
            trows = list(ex.map(task, [(n, c, tname, k, pr, s) for n, k, pr, s in arms for c in test], chunksize=1))
            table = {n: summary([r for r in trows if r['controller'] == n]) for n, *_ in arms}
            s1s = table['S1']['score']
            headroom = dict(S2_over_S1=1.-table['S2']['score']/s1s if s1s else None, S3_over_S1=1.-table['S3']['score']/s1s if s1s else None)
            headroom['room'] = bool(max(v for v in headroom.values() if v is not None) >= .2)
            hits = table['S1'].get('t30_hits' if tname == 'nod' else 't_reach_hits', 0)
            hits2 = table['S2'].get('t30_hits' if tname == 'nod' else 't_reach_hits', 0)
            gate = dict(S1_hits=hits, S2_hits=hits2, n=len(test), passed=bool(max(hits, hits2) >= (12 if not a.quick else 1)))
            result[tname] = dict(S1=dict(grid_best=grid[s1_best], tune_score=s1_scores[s1_best], tune_flown=s1_flown[s1_best]),
                                 S1_grid_scores={str(grid[i]): dict(score=s1_scores[i], flown=s1_flown[i]) for i in range(len(grid))},
                                 S2=s2, S2_history=s2_hist, S3=s3, S3_history=s3_hist, test=table, headroom=headroom, gate_S0=gate,
                                 test_rows=trows, wall=round(time.monotonic()-t0))
            (out/f'{tname}.json').write_text(json.dumps(result[tname], indent=1))
            for n, s in table.items():
                print(f"{tname} {n:10s} flown {s['flown']}/{s['episodes']} score {s['score']:.2f} " +
                      ' '.join(f"{k}={v:.2f}" for k, v in s.items() if isinstance(v, float) and k not in ('score',)), flush=True)
            print(json.dumps(dict(task=tname, headroom=headroom, gate_S0=gate, wall=round(time.monotonic()-t0))), flush=True)
    (out/'maneuver_v2.json').write_text(json.dumps({k: (v if k == 'preregistration' else {kk: vv for kk, vv in v.items() if kk != 'test_rows'})
                                                      for k, v in result.items()}, indent=1))


def selftest():
    m = Maneuver('schedule', [(3, 1, .1, .5), (6, -1, -.1, -.2)])
    assert abs(m.duration()-.36) < 1e-9 and m.command(.05, 0.) == (1, .1, .5) and m.command(.2, 0.) == (-1, -.1, -.2) and m.command(.5, 0.) == (0, 0., 0.)
    segs = decode_s2(np.array([.5, .25, 1., 0., .5, 1.]), -1, 'nod')
    assert [b for b, *_ in segs] == [6, 3] and [d for _, d, *_ in segs] == [-1, 1]
    assert abs(segs[0][3]-REF_MAX) < 1e-9 and abs(segs[1][3]+REF_MAX) < 1e-9 and abs(segs[0][2]-0.) < 1e-9 and abs(segs[1][2]-.3) < 1e-9
    segs = decode_s2(np.full(9, .5), 1, 'dash')
    assert segs[0][0] == 15 and len(segs) == 3 and [d for _, d, *_ in segs] == [1, 0, -1] and all(abs(r) < 1e-9 for *_, r in segs)
    s3 = decode_s3(np.full(7, .5))
    assert s3[0] == 6 and abs(s3[2]-REF_MAX/2) < 1e-9 and abs(s3[3]-.75) < 1e-9 and s3[6] == 3
    m3 = Maneuver('s3', s3, 1)
    assert m3.command(.1, 0.) == (1, s3[1], s3[2]) and m3.command(1.0, 2.)[0] == -1 and m3.command(1.0, .1)[0] == 0 and m3.command(1.0, 2.)[2] == 0.
    assert len(S1_GRID['nod']) == 120 and len(S1_GRID['dash']) == 144, (len(S1_GRID['nod']), len(S1_GRID['dash']))
    rows = [dict(score=1., flown=True), dict(score=1., flown=False)]
    assert abs(tune_score(rows)-(1.+CRASH_PENALTY)) < 1e-9
    assert abs(SECONDS['dash']-6.5) < 1e-9 and abs(SECONDS['nod']-5.) < 1e-9
    print('selftest ok')


if __name__ == '__main__':
    main()
