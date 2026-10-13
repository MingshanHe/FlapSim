"""Combined benchmark, maneuver tasks T2 (nod) and T3 (dash-and-stop): tasks, scores, classical baselines, headroom.

Hybrid classical controller: the hover controller (retuned PID + phase-locked pulse law) flies; at t0 a maneuver module
takes the motor-pulse channel and an alpha-gimbal residual for a scheduled time, then hands back. Everything stays
inside the original interface (2 ms commands, LOW/HIGH, +-0.3 rad). Truth is used for scoring only.

T2 nod:  reach |pitch| >= 30 deg as fast as possible and return below 5 deg as fast as possible.
         score = t_30 + t_recover + 0.03 * height_loss_cm + 0.5 * |drift_m| (+ crash penalty); misses cost 2.5 s each.
T3 dash: move 0.5 m forward (heading frame at t0) and stop.
         score = t_reach + t_settle + 2 * overshoot_m + 0.03 * height_loss_cm + 0.5 * |lateral_m| (+ crash); misses 2.5 s.

Arms:   hover_only (no maneuver module: the task is not solved by hovering), S1 simple schedule (3 knobs, grid),
        S2 rich schedule (8 knobs, CEM), S3 privileged feedback brake (truth pitch rate; CEM) = headroom indicator.
Pre-registered: scores and arms above are fixed before the runs; 4 stress tuning cases, 8 held-out stress test cases.

    python -m experiments.algorithms.spike_steer.maneuver selftest
    python -m experiments.algorithms.spike_steer.maneuver all --output runs/spike_steer/maneuver [--quick]
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
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..neuro_gates import gates
from . import pilot as P
from . import pilot_v3 as V

T0, SECONDS, WINDOW, MISS = 2.4, 5., 2.5, 2.5
BEAT, PULSE_WIN = .04, math.radians(45.)
UP, DOWN = math.radians(135.), math.radians(315.)
PID_GAINS = (1.5, .85)
BASES = dict(tune=2_095_000_000, test=2_095_100_000)
TASKS = ('nod', 'dash')
# schedule = list of segments (beats, pulse_dir in {-1,0,1}, alpha_residual rad); S1 grids:
S1_GRID = dict(
    nod=[[(n, s, g), (b, -s, g2)] for n in (3, 6, 9) for b in (0, 3, 6, 9) for s in (1, -1) for g, g2 in ((0., 0.), (.15, -.15), (.3, -.3))],
    dash=[[(n, s, g), (c, 0, 0.), (b, -s, -g)] for n in (3, 6, 9, 12) for c in (0, 6) for b in (6, 12) for s in (1, -1) for g in (0., .2)])
S2_DIM = 8   # 4 segments x (beats 0-12, alpha -0.3..0.3); pulse dirs fixed pattern (+s, +s, -s, -s) with s from sign of first alpha
S3_DIM = 6   # go beats, go alpha, brake gain on truth pitch rate (pulse), brake gain alpha, brake threshold, hold beats
CEM = dict(pop=12, iters=4, elite=4)


def decode_s2(x, s):
    x = np.asarray(x, dtype=float)
    beats = np.clip(np.round(x[:4]*12.), 0, 12).astype(int)
    alphas = np.clip(x[4:]*.3, -.3, .3)
    dirs = (s, s, -s, -s)
    return [(int(b), d, float(a)) for b, d, a in zip(beats, dirs, alphas)]


class Maneuver:
    """Open-loop schedule (segments) or privileged feedback brake (S3)."""

    def __init__(self, kind, params, sign=1):
        self.kind, self.params, self.sign = kind, params, sign

    def duration(self):
        if self.kind == 'none':
            return 0.
        if self.kind == 'schedule':
            return sum(b for b, _, _ in self.params)*BEAT
        return WINDOW  # s3: active for the window

    def command(self, rel, crank, pitch_rate_truth, pitch_truth):
        """Returns (pulse_dir in {-1,0,1} or 0, alpha residual) for time rel since t0."""
        if self.kind == 'none' or rel < 0 or rel >= self.duration():
            return 0, 0.
        if self.kind == 'schedule':
            acc = 0.
            for b, d, a in self.params:
                if acc <= rel < acc+b*BEAT:
                    return d, a
                acc += b*BEAT
            return 0, 0.
        # s3: go phase (open loop) then truth-rate feedback brake: pulse against the rate when |rate| > thr
        go_b, go_a, k_pulse, k_alpha, thr, hold_b = self.params
        if rel < go_b*BEAT:
            return self.sign, go_a
        if rel < (go_b+hold_b)*BEAT:
            return 0, 0.
        u = -k_pulse*pitch_rate_truth
        d = int(np.sign(u)) if abs(u) > thr else 0
        return d, float(np.clip(-k_alpha*pitch_rate_truth, -.3, .3))


def decode_s3(x):
    x = np.asarray(x, dtype=float)
    return [int(np.clip(round(x[0]*12.), 1, 12)), float(np.clip(x[1]*.3, -.3, .3)), float(np.clip(x[2]*2., 0., 2.)),
            float(np.clip(x[3]*.1, 0., .1)), float(np.clip(x[4]*1., 0., 1.)), int(np.clip(round(x[5]*6.), 0, 6))]


_LAW = None


def run(env, c, task, man):
    global _LAW
    if _LAW is None:
        _LAW = json.loads(V.V2_LAW.read_text())['best']
    env.configure(dict(c, seconds=SECONDS))
    env.reset(seed=c['seed'])
    p, (sp, sd) = env.p, PID_GAINS
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    steer = V.SteerV3(None, _LAW, False)
    steer.reset()
    m, d = env.sim.model, env.sim.data
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    oracle = gates.Oracle(env, 'derocked')
    rows, heading, pos0, pulses = [], None, None, 0
    while True:
        t, now = env.air.steps*DT, float(d.time)
        cmd = env.teacher_command()
        pulse, _ = steer.step(env, now, t)
        rel = t-T0
        if pos0 is None and t >= T0:
            pos, _, R = env.air.env._state()
            pos0 = pos.copy()
            yaw = float(env.last_reading[2])
            heading = np.array([math.cos(yaw), math.sin(yaw)])
        dirn, alpha = man.command(rel, float(d.qpos[crank]), float(oracle.rate[1]), float(oracle.queue[-1][1]))
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
    w = a[(a[:, 0] >= T0) & (a[:, 0] < T0+WINDOW)]
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
    crash = 0. if flown else 10.+5.*float(np.clip((T0+WINDOW-ep['elapsed'])/WINDOW, 0., 1.))

    def first(mask, start=0.):
        i = np.flatnonzero(mask & (rel_t >= start))
        return float(rel_t[i[0]]) if len(i) else None
    res = dict(seed=c['seed'], task=task, termination=ep['termination'], flown=flown, pulses=pulses,
               height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 8].max())) if len(w) else None, duration=man.duration())
    if task == 'nod':
        t30 = first(np.abs(pitch) >= 30.)
        trec = first(np.abs(pitch) < 5., (t30 or 0.)+.02) if t30 is not None else None
        drift = float(abs(fwd[-1])) if len(fwd) else 0.
        score = (t30 if t30 is not None else MISS)+((trec-t30) if (t30 is not None and trec is not None) else MISS)+.03*dz+.5*drift+crash
        res.update(t30=t30, t_recover=None if (t30 is None or trec is None) else trec-t30, drift_m=drift,
                   peak_pitch_deg=float(np.abs(pitch).max()) if len(pitch) else None, score=float(score))
    else:
        treach = first(fwd >= .5)
        over = float(max(0., fwd.max()-.6)) if len(fwd) else 0.
        tsettle = None
        if treach is not None:
            slow = np.abs(vfwd) < .1
            for i in np.flatnonzero((rel_t >= treach)):
                j = i+int(.3/DT)
                if j <= len(slow) and slow[i:j].all():
                    tsettle = float(rel_t[i]-treach)
                    break
        short = 0. if treach is not None else 2.*max(0., .5-(fwd.max() if len(fwd) else 0.))
        score = (treach if treach is not None else MISS+short)+(tsettle if tsettle is not None else MISS)+2.*over+.03*dz \
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


def cases(base, n):
    return [make_case(base+i, SECONDS, 'stress', i % 8) for i in range(n)]


def mean_score(rows):
    return float(np.mean([r['score'] for r in rows]))


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
    """CEM over the unit box; the maneuver direction sign is fixed to the S1 winner's."""
    pop, iters = (CEM['pop'], CEM['iters']) if not quick else (6, 2)
    mu, sd = np.full(dim, .5), np.full(dim, .3)
    best, history = None, []
    for it in range(iters):
        xs = np.clip(mu+sd*rng.standard_normal((pop, dim)), 0., 1.)
        jobs = []
        for i, x in enumerate(xs):
            params = decode(x, sign) if kind == 'schedule' else decode(x)
            jobs += [(str(i), c, tname, kind, params, sign) for c in tune]
        rows = list(ex.map(task, jobs, chunksize=1))
        per_i = {i: mean_score([r for r in rows if r['controller'] == str(i)]) for i in range(pop)}
        order = sorted(per_i, key=per_i.get)
        elite = xs[order[:CEM['elite']]]
        mu, sd = elite.mean(0), np.maximum(elite.std(0), .05)
        i0 = order[0]
        s0 = sign
        cand = dict(x=xs[i0].tolist(), sign=s0, score=per_i[i0], params=decode(xs[i0], s0) if kind == 'schedule' else decode(xs[i0]))
        if best is None or cand['score'] < best['score']:
            best = cand
        history.append(dict(iter=it, best=per_i[i0], mean=float(np.mean(list(per_i.values())))))
        print(json.dumps(dict(task=tname, arm=kind, **history[-1])), flush=True)
    return best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['selftest', 'all'])
    ap.add_argument('--output', type=Path, default=Path('runs/spike_steer/maneuver'))
    ap.add_argument('--workers', type=int, default=18)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    if a.stage == 'selftest':
        return selftest()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    tune, test = cases(BASES['tune'], 2 if a.quick else 4), cases(BASES['test'], 2 if a.quick else 8)
    rng = np.random.default_rng(a.seed)
    prereg = dict(tasks=dict(nod='t_30 + t_recover + .03 dz_cm + .5 |drift_m| (+ crash 10 + 5 x unflown); miss = 2.5 s each',
                             dash='t_reach + t_settle + 2 overshoot_m + .03 dz_cm + .5 |lateral_m| (+ crash); miss = 2.5 s (+2 x shortfall)'),
                  arms=['hover_only', 'S1 schedule grid (3 knobs)', 'S2 schedule CEM (8 knobs)', 'S3 privileged truth-rate brake CEM (6 knobs)'],
                  headroom_rule='maneuver has learning room if S2 or S3 beats S1 by >= 20 % on the 8 test cases; '
                                'the hybrid classical baseline for the composite is S1 (or S2 if better) on top of PID + law',
                  cases=dict(tune=[c['seed'] for c in tune], test=[c['seed'] for c in test]), time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out/'preregistration.json').write_text(json.dumps(prereg, indent=1))
    result = dict(preregistration=prereg)
    with ProcessPoolExecutor(a.workers, initializer=worker_init, initargs=(str(source),)) as ex:
        for tname in TASKS:
            t0 = time.monotonic()
            grid = S1_GRID[tname] if not a.quick else S1_GRID[tname][:4]
            rows = list(ex.map(task, [(str(i), c, tname, 'schedule', g, 1) for i, g in enumerate(grid) for c in tune], chunksize=1))
            s1_scores = {i: mean_score([r for r in rows if r['controller'] == str(i)]) for i in range(len(grid))}
            s1_best = min(s1_scores, key=s1_scores.get)
            print(json.dumps(dict(task=tname, arm='S1', best=grid[s1_best], score=s1_scores[s1_best], wall=round(time.monotonic()-t0))), flush=True)
            s1_sign = grid[s1_best][0][1]
            s2, s2_hist = cem(ex, tname, 'schedule', S2_DIM, tune, rng, decode_s2, a.quick, s1_sign)
            s3, s3_hist = cem(ex, tname, 's3', S3_DIM, tune, rng, lambda x, s=None: decode_s3(x), a.quick, s1_sign)
            arms = [('hover_only', 'none', [], 1), ('S1', 'schedule', grid[s1_best], 1), ('S2', 'schedule', s2['params'], s2['sign']),
                    ('S3', 's3', s3['params'], s3['sign'])]
            trows = list(ex.map(task, [(n, c, tname, k, pr, s) for n, k, pr, s in arms for c in test], chunksize=1))
            table = {n: summary([r for r in trows if r['controller'] == n]) for n, *_ in arms}
            s1s = table['S1']['score']
            headroom = dict(S2_over_S1=1.-table['S2']['score']/s1s if s1s else None, S3_over_S1=1.-table['S3']['score']/s1s if s1s else None)
            headroom['room'] = bool(max(v for v in headroom.values() if v is not None) >= .2)
            result[tname] = dict(S1=dict(grid_best=grid[s1_best], tune_score=s1_scores[s1_best]), S2=s2, S2_history=s2_hist,
                                 S3=s3, S3_history=s3_hist, test=table, headroom=headroom, test_rows=trows)
            (out/f'{tname}.json').write_text(json.dumps(result[tname], indent=1))
            for n, s in table.items():
                print(f"{tname} {n:10s} flown {s['flown']}/{s['episodes']} score {s['score']:.2f} " +
                      ' '.join(f"{k}={v:.2f}" for k, v in s.items() if isinstance(v, float) and k not in ('score',)), flush=True)
            print(json.dumps(dict(task=tname, headroom=headroom)), flush=True)
    (out/'maneuver.json').write_text(json.dumps({k: (v if k == 'preregistration' else {kk: vv for kk, vv in v.items() if kk != 'test_rows'})
                                                   for k, v in result.items()}, indent=1))


def selftest():
    m = Maneuver('schedule', [(3, 1, .1), (6, -1, -.1)])
    assert abs(m.duration()-.36) < 1e-9 and m.command(.05, 0., 0., 0.) == (1, .1) and m.command(.2, 0., 0., 0.) == (-1, -.1) and m.command(.5, 0., 0., 0.) == (0, 0.)
    segs = decode_s2(np.array([.5, .25, 0., 1., .5, 0., 1., .5]), -1)
    assert [b for b, _, _ in segs] == [6, 3, 0, 12] and [d for _, d, _ in segs] == [-1, -1, 1, 1] and abs(segs[1][2]-0.) < 1e-9 and abs(segs[2][2]-.3) < 1e-9
    s3 = decode_s3(np.full(6, .5))
    assert s3[0] == 6 and abs(s3[2]-1.) < 1e-9 and s3[5] == 3
    m3 = Maneuver('s3', s3, 1)
    assert m3.command(.1, 0., 0., 0.) == (1, s3[1]) and m3.command(1.0, 0., 2., 0.)[0] == -1 and m3.command(1.0, 0., .1, 0.)[0] == 0
    assert len(S1_GRID['nod']) == 72 and len(S1_GRID['dash']) == 64
    print('selftest ok')


if __name__ == '__main__':
    main()
