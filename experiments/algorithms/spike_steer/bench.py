"""Unified benchmark (research plan v4, stage S1): T1 stress hover, T2 nod, T3 dash-and-stop, T4 large-upset
recovery, one policy interface for classical and learned descending-command layers.

Architecture (fixed for every arm; the original interface is untouched: 2 ms commands, +-0.3 rad, LOW/HIGH):
  reflex layer      ReferencePID (kp x1.5, kd x0.85, ki x1.5) with an attitude setpoint;
  steering layer    motor pulses: the command sits at HIGH for the ticks a policy says so;
  descending layer  the policy. Per tick it reads the 18 rectified raw-IMU features of pilot_v6 plus two supervisor
                    command channels (nod, dash) and returns (pulse this tick, pitch reference in rad).
Supervisor: the task's goal signal, the same for every arm. nod: 1 from t0 until |pitch| >= 30 deg was held 0.1 s;
dash: 1 from t0 until the body moved DASH_M forward (heading frame at t0). hover and upset: always 0 (the upset is
unannounced). The goal is detected on truth (a navigation layer the robot does not have inside the IMU interface).

Scores per task (lower is better) are those of pilot.py (T1), maneuver_v2.py (T2, T3; T3 target revised by the
rule in revise_dash) and upset.py (T4, with upset_v2's calibration and 2.5 s window). Composite = sum of
WEIGHTS[task] x score[task] / score of the classical opponent on the same test set (classical = 1.0).

    python -m experiments.algorithms.spike_steer.bench selftest
    python -m experiments.algorithms.spike_steer.bench classical --output runs/spike_steer/bench --workers 30
    python -m experiments.algorithms.spike_steer.bench evaluate --spec <descending spec> --tag imit_s1 --output ...
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
from . import pilot as P
from . import pilot_v3 as V
from . import pilot_v6 as Y
from . import upset as U
from . import upset_v2 as U2
from . import maneuver_v2 as M

U2.patch()   # upset scenario: 2.5 s window, 5 s episodes (module globals of upset.py, read at call time)

TASKS = ('hover', 'nod', 'dash', 'upset')
WEIGHTS = dict(hover=.4, nod=.2, dash=.2, upset=.2)
PID_GAINS = (1.5, .85)
REF_MAX = .7
TWITCH = 3
T0 = 2.4
BEAT = .04
WINDOW = dict(nod=2.5, dash=4., upset=2.5)
SECONDS = dict(hover=8., nod=5., dash=6.5, upset=5.)
HOLD_S, OVER_M = .1, .1
DASH_M = dict(value=.25)          # revised by revise_dash() before the classical stage (written to dash_target.json)
N_IN = Y.INPUTS+2                 # 18 raw-IMU features + command channels (nod, dash)
IN_DELAYS_TICKS = tuple(range(0, 11))   # 0..20 ms, input -> hidden
HID_DELAYS_MS = tuple(range(0, 21, 2))  # 0..20 ms, hidden -> outputs
SPIKE_SCALE = float(P.SPEC['spike_scale'])
TAU_G = .02                       # graded descending unit (pitch reference), s
BASES = dict(hover_train=V.BASES['train'], man_train=2_096_200_000, upset_train=2_070_500_000)
NOD_GRID = [dict(go_ref=s*r, go_beats=n, brake_beats=b, brake_ref=-s*r2, gated=g)
            for s in (1, -1) for r in (.35, .6) for n in (6, 9, 12) for b in (0, 3, 6) for r2 in ((0.,) if b == 0 else (.2, .35))
            for g in (True, False)]
DASH_GRID = [dict(go_ref=s*r, go_beats=n, brake_beats=b, brake_ref=-s*r2, gated=g)
             for s in (1, -1) for r in (.2, .35, .5) for n in (18, 30, 45) for b in (3, 6, 12) for r2 in (.35, .5)
             for g in (True, False)]
QUICK = False


# ----------------------------------------------------------------------------------------- cases
def upset_levels(out):
    return json.loads((Path(out)/'upset_v2'/'calibrate.json').read_text())['levels'] if not isinstance(out, dict) else out


def cases(task, split, levels=None, n=None, offset=0):
    """Test sets are the blind sets of S0 (nod/dash: maneuver_v2; upset: upset_v2; hover: the old 16). Train sets
    are disjoint seed ranges; `offset` draws fresh batches (ES generations)."""
    if task == 'hover':
        if split == 'test':
            return P.cases('test', n or 16, 8.)
        return V.stratified(BASES['hover_train']+offset, n or 16, 6.)
    if task in ('nod', 'dash'):
        if split == 'test':
            return M.cases(M.BASES['test'], n or 16, task)
        if split == 'tune':
            return M.cases(M.BASES['tune'], n or 8, task)
        return [make_case(BASES['man_train']+offset+i, SECONDS[task], 'stress', i % 8) for i in range(n or 8)]
    assert levels is not None, 'upset cases need the calibrated levels'
    if split == 'test':
        return U2.test_cases(levels)[:n] if n else U2.test_cases(levels)
    if split == 'tune':
        return U2.train_cases(levels)
    rng = np.random.default_rng(np.random.SeedSequence([BASES['upset_train']+offset, 77]))
    out = []
    for i in range(n or 8):
        phi = float(rng.choice(U.DIRECTIONS))
        lv = int(rng.integers(0, 3))
        out.append(U.upset_case(BASES['upset_train']+offset+i, phi, levels[U.klass(phi)]['levels'][lv], i % 2 == 1, i))
    return out


# ----------------------------------------------------------------------------------------- policies
def features(reading, cmd):
    x = np.asarray(reading, dtype=float)[list(Y.IDX)]/Y.SCALE
    return np.r_[np.maximum(x, 0.), np.maximum(-x, 0.), 1., 0., cmd]


class Schedule:
    """Two-segment pitch-reference schedule from the command onset: go (ref, <= go_beats, ends early when the goal
    signal drops if gated), then brake (brake_beats at brake_ref), then hand back (0)."""

    def __init__(self, q):
        self.q = dict(q)

    def reset(self):
        self.t_go, self.t_end = None, None

    def ref(self, rel, cmd_on):
        q = self.q
        if rel < 0:
            return 0., False
        if self.t_go is None:
            self.t_go = rel
        if self.t_end is None:
            if rel-self.t_go < q['go_beats']*BEAT and (cmd_on or not q['gated']):
                return q['go_ref'], True
            self.t_end = rel
        if rel-self.t_end < q['brake_beats']*BEAT:
            return q['brake_ref'], True
        return 0., False


class Classical:
    """Hover pulse law (PLL, pilot_v3.SteerV3) on top of the reflex PID, optionally a task schedule with setpoint
    authority. law=None: plain PID (+ schedule)."""

    def __init__(self, law=None, schedule=None):
        self.law, self.schedule = law, Schedule(schedule) if schedule else None

    def reset(self):
        self.steer = V.SteerV3(None, self.law, False) if self.law else None
        if self.steer:
            self.steer.reset()
        if self.schedule:
            self.schedule.reset()

    def act(self, env, now, t, x, cmd, rel):
        pulse = False
        if self.steer:
            pulse, _ = self.steer.step(env, now, t)
        ref, active = (self.schedule.ref(rel, bool(cmd.any())) if self.schedule else (0., False))
        return (pulse and not active), float(ref)


class DescendingNet:
    """NumPy twin of descend.DescendingSNN: inputs -> H hidden LIF (a delay bank per input, 0..10 ticks) -> 2 spiking
    outputs (motor pulses; a delay bank per hidden unit, 0..20 ms) and one graded unit (pitch reference =
    REF_MAX tanh(g)). Two 1 ms substeps per 2 ms tick; threshold 1, subtractive reset; spike_scale on hidden spikes."""

    def __init__(self, spec):
        s = spec
        self.W_in, self.b_h = np.asarray(s['W_in'], float), np.asarray(s['b_h'], float)
        self.W_out, self.b_o = np.asarray(s['W_out'], float), np.asarray(s['b_o'], float)
        self.W_g, self.b_g = np.asarray(s['W_g'], float), float(s['b_g'])
        self.H, self.n_in, self.K = self.W_in.shape
        self.D = self.W_out.shape[2]
        self.S = float(s.get('spike_scale', SPIKE_SCALE))
        self.a_m, self.a_s = math.exp(-.001/s['tau_m']), math.exp(-.001/s['tau_s'])
        self.a_g = math.exp(-.001/s.get('tau_g', TAU_G))
        self.mask = np.ones(self.n_in)
        if s.get('mask') == 'ahrs':   # ablation: gyro and accelerometer channels zeroed (both rectified halves)
            fast = [2, 3, 4, 5, 6, 7]
            for i in fast:
                self.mask[i] = self.mask[i+len(Y.CHANNELS)] = 0.
        self.delays_in = np.asarray(s.get('delays_in', IN_DELAYS_TICKS), int)
        self.delays_hid = np.asarray(s.get('delays_hid', HID_DELAYS_MS), int)
        self.L = int(self.delays_hid.max())+2
        self.reset()

    def reset(self):
        self.xbuf = np.zeros((self.n_in, int(self.delays_in.max())+1))
        self.cur_h = np.zeros(self.H)
        self.v_h = np.zeros(self.H)
        self.cur_o = np.zeros(2)
        self.v_o = np.zeros(2)
        self.cur_g = self.g = 0.
        self.hist = np.zeros((self.L, self.H))
        self.p = 0
        self.spikes = 0

    def tick(self, x):
        self.xbuf = np.roll(self.xbuf, 1, axis=1)
        self.xbuf[:, 0] = np.asarray(x, float)*self.mask
        x_del = self.xbuf[:, self.delays_in]                              # n_in, K
        drive_h = np.einsum('hik,ik->h', self.W_in, x_del)+self.b_h
        fired = np.zeros(2, bool)
        for _ in range(2):
            self.cur_h = self.a_s*self.cur_h+(1-self.a_s)*drive_h
            self.v_h = self.a_m*self.v_h+(1-self.a_m)*self.cur_h
            spk = self.v_h >= 1.
            self.v_h = self.v_h-spk
            self.hist[self.p] = spk
            self.spikes += int(spk.sum())
            rec = self.hist[(self.p-1-self.delays_hid) % self.L].T          # H, D
            drive_o = self.S*np.einsum('ohd,hd->o', self.W_out, rec)+self.b_o
            self.cur_o = self.a_s*self.cur_o+(1-self.a_s)*drive_o
            self.v_o = self.a_m*self.v_o+(1-self.a_m)*self.cur_o
            spk_o = self.v_o >= 1.
            self.v_o = self.v_o-spk_o
            fired |= spk_o
            drive_g = self.S*float(np.sum(self.W_g*rec))+self.b_g
            self.cur_g = self.a_s*self.cur_g+(1-self.a_s)*drive_g
            self.g = self.a_g*self.g+(1-self.a_g)*self.cur_g
            self.p = (self.p+1) % self.L
        return fired, REF_MAX*math.tanh(self.g)


class Learned:
    """Descending-command policy from a spec: runs from takeoff (settles), acts after the handover; any output spike
    starts a TWITCH-tick motor pulse (as pilot_v3.SteerV3)."""

    def __init__(self, spec):
        self.spec = spec

    def reset(self):
        self.net = DescendingNet(self.spec)
        self.twitch = 0
        self.onsets = 0

    def act(self, env, now, t, x, cmd, rel):
        fired, ref = self.net.tick(x)
        if t < gates.HANDOVER_T:
            return False, 0.
        if self.twitch > 0:
            self.twitch -= 1
            return True, ref
        if fired.any():
            self.twitch = TWITCH-1
            self.onsets += 1
            return True, ref
        return False, ref


# ----------------------------------------------------------------------------------------- episode
def run_episode(env, task, c, policy, trace=False):
    env.configure(c)   # the case carries its own duration (train hover 6 s, test hover 8 s, nod 5 s, dash 6.5 s, upset 5 s)
    env.reset(seed=c['seed'])
    env.upset_axes = {}
    p, (sp, sd) = env.p, PID_GAINS
    env.teacher = ReferencePID(dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']]), env.air.last_rpy)
    env._teacher_step = -1
    policy.reset()
    m, d = env.sim.model, env.sim.data
    oracle = gates.Oracle(env, 'derocked')
    body, thrust = env.air.env.body, m.body('thrust_unit').id
    up = math.copysign(1., float(d.xmat[thrust][8]))
    t0 = c['upset']['t0'] if task == 'upset' else T0
    window = WINDOW.get(task)
    rows, pos0, heading, goal_done, goal_t, pulses = [], None, None, False, None, 0
    hold = int(round(HOLD_S/DT))
    big = 0
    log = []
    while True:
        t, now = env.air.steps*DT, float(d.time)
        rel = t-t0
        if task in ('nod', 'dash') and pos0 is None and t >= t0:
            pos, _, _ = env.air.env._state()
            pos0 = pos.copy()
            yaw = float(env.last_reading[2])
            heading = np.array([math.cos(yaw), math.sin(yaw)])
        cmd = np.zeros(2)
        if task in ('nod', 'dash') and rel >= 0 and not goal_done:
            cmd[0 if task == 'nod' else 1] = 1.
        x = features(env.last_reading, cmd)
        fire, ref = policy.act(env, now, t, x, cmd, rel)
        env.teacher.rpy_reference = np.array([0., float(np.clip(ref, -REF_MAX, REF_MAX))])
        command = env.teacher_command()
        if fire and t >= gates.HANDOVER_T:
            command[2] = HIGH[2]
            pulses += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH)-CENTER)/HALF)
        oracle.update()
        pos, vel, R = env.air.env._state()
        pitch_true = float(oracle.queue[-1][1])
        tnow = env.air.steps*DT
        rows.append([tnow, pitch_true, *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2])))), float(ref), float(fire), *cmd])
        if task == 'upset':
            log.append((tnow, math.acos(float(np.clip(up*d.xmat[thrust][8], -1., 1.)))))
        if task == 'nod' and not goal_done and rel >= 0:
            big = big+1 if abs(pitch_true) >= math.radians(30.) else 0
            if big >= hold:
                goal_done, goal_t = True, tnow-t0
        if task == 'dash' and not goal_done and pos0 is not None:
            if float((pos[:2]-pos0[:2])@heading) >= DASH_M['value']:
                goal_done, goal_t = True, tnow-t0
        if term or trunc:
            break
    ep = out['episode']
    a = np.array(rows)
    flown = ep['termination'] == 'time_limit'
    res = dict(seed=c['seed'], task=task, termination=ep['termination'], elapsed=ep['elapsed'], flown=flown, pulses=pulses,
               pulse_duty=pulses/max(1, env.air.steps), goal_t=goal_t, ref_abs_mean=float(np.abs(a[:, 9]).mean()))
    if trace:
        res['trace'] = a.tolist()
    if task == 'hover':
        late = a[:, 0] >= 2.
        pitch_rms = float(np.degrees(np.sqrt(np.mean(a[late, 1]**2)))) if late.any() else float('nan')
        J = pitch_rms+P.HEIGHT_WEIGHT*100*ep['height_rmse_m'] if flown else 40.-ep['elapsed']
        res.update(score=float(J), pitch_rms_deg=pitch_rms, height_rmse_cm=100*ep['height_rmse_m'], tilt_rms_deg=ep['tilt_rms_deg'])
        return res
    if task == 'upset':
        s, lg = np.asarray(env.samples), np.asarray(log)
        ts, tilt, z = s[:, 0], np.degrees(s[:, 1]), s[:, 2]
        end = t0+window
        pre, win = (ts >= 2.) & (ts < t0), (ts >= t0) & (ts < end)
        i0 = int(np.argmin(abs(ts-t0)))
        dz = float(100*np.abs(z[win]-z[i0]).max()) if win.any() else float('nan')
        J = float(tilt[pre].sum()*DT+tilt[win].sum()*DT+.2*dz) if flown else 60.+40.*float(np.clip((end-ts[-1])/window, 0., 1.))
        smooth = np.convolve(tilt, np.ones(U.SMOOTH)/U.SMOOTH, mode='full')[:len(tilt)]
        recovery = None
        if flown:
            good = (smooth < U.RECOVER_DEG) & (ts >= t0)
            run_len, hold_n = 0, round(U.RECOVER_HOLD/DT)
            for i in range(len(good)):
                run_len = run_len+1 if good[i] else 0
                if run_len >= hold_n:
                    recovery = float(ts[i-hold_n+1]-t0)
                    break
        sel = (lg[:, 0] >= t0) & (lg[:, 0] < end)
        res.update(score=J, phi=c['upset']['phi'], mag=c['upset']['mag'], double=c['upset']['double'], cls=U.klass(c['upset']['phi']),
                   tilt_peak_deg=float(tilt[win].max()) if win.any() else None,
                   thrust_tilt_peak_deg=float(np.degrees(lg[sel, 1].max())) if sel.any() else None,
                   recovery_s=recovery, recovery_censored=recovery if recovery is not None else window, dz_cm=dz)
        return res
    w = a[(a[:, 0] >= T0) & (a[:, 0] < T0+window)]
    rel_t = w[:, 0]-T0
    pitch = np.degrees(w[:, 1])
    if pos0 is None:
        pos0, heading = a[0, 2:5], np.array([1., 0.])
    fwd = (w[:, 2:4]-pos0[:2])@heading
    lat = (w[:, 2:4]-pos0[:2])@np.array([-heading[1], heading[0]])
    vfwd = w[:, 5:7]@heading
    dz = float(100*max(0., pos0[2]-w[:, 4].min())) if len(w) else 0.
    crash = 0. if flown else 10.+5.*float(np.clip((T0+window-ep['elapsed'])/window, 0., 1.))
    miss = window

    def first(mask, start=0.):
        i = np.flatnonzero(mask & (rel_t >= start))
        return float(rel_t[i[0]]) if len(i) else None
    res.update(height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 8].max())) if len(w) else None)
    if task == 'nod':
        bigm = np.abs(pitch) >= 30.
        sustained = np.array([bigm[i:i+hold].all() if i+hold <= len(bigm) else False for i in range(len(bigm))])
        t30 = first(sustained)
        trec = first(np.abs(pitch) < 5., (t30 or 0.)+.02) if t30 is not None else None
        drift = float(abs(fwd[-1])) if len(fwd) else 0.
        score = (t30 if t30 is not None else miss)+((trec-t30) if (t30 is not None and trec is not None) else miss)+.03*dz+.5*drift+crash
        res.update(t30=t30, t_recover=None if (t30 is None or trec is None) else trec-t30, drift_m=drift,
                   peak_pitch_deg=float(np.abs(pitch).max()) if len(pitch) else None, score=float(score))
    else:
        target = DASH_M['value']
        treach = first(fwd >= target)
        over = float(max(0., fwd.max()-(target+OVER_M))) if len(fwd) else 0.
        tsettle = None
        if treach is not None:
            slow = np.abs(vfwd) < .1
            for i in np.flatnonzero((rel_t >= treach)):
                j = i+int(.3/DT)
                if j <= len(slow) and slow[i:j].all():
                    tsettle = float(rel_t[i]-treach)
                    break
        short = 0. if treach is not None else 2.*max(0., target-(fwd.max() if len(fwd) else 0.))
        score = (treach if treach is not None else miss+short)+(tsettle if tsettle is not None else miss)+2.*over+.03*dz \
            + .5*float(abs(lat[-1]) if len(lat) else 0.)+crash
        res.update(t_reach=treach, t_settle=tsettle, overshoot_m=over, max_forward_m=float(fwd.max()) if len(fwd) else None,
                   lateral_m=float(abs(lat[-1])) if len(lat) else None, score=float(score))
    return res


# ----------------------------------------------------------------------------------------- pool
_ENV = None


def worker_init(source, dash_m):
    global _ENV
    DASH_M['value'] = dash_m
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')
    U.install(_ENV)


def make_policy(kind, arg):
    if kind == 'classical':
        return Classical(arg.get('law'), arg.get('schedule'))
    return Learned(arg)


def task(args):
    """(name, task, case, kind, arg[, trace]) -> result row."""
    name, tname, c, kind, arg = args[:5]
    trace = args[5] if len(args) > 5 else False
    return dict(run_episode(_ENV, tname, c, make_policy(kind, arg), trace), controller=name)


def pool(workers, dash_m=None):
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source), dash_m if dash_m is not None else DASH_M['value']))


def run_jobs(ex, jobs):
    return list(ex.map(task, jobs, chunksize=1))


# ----------------------------------------------------------------------------------------- statistics
def summary(rows):
    keys = [k for k in ('pitch_rms_deg', 'height_rmse_cm', 't30', 't_recover', 'drift_m', 'peak_pitch_deg', 't_reach', 't_settle',
                        'overshoot_m', 'max_forward_m', 'lateral_m', 'height_loss_cm', 'tilt_peak_deg', 'thrust_tilt_peak_deg',
                        'recovery_censored', 'dz_cm', 'pulse_duty', 'ref_abs_mean', 'goal_t') if rows and k in rows[0]]
    out = dict(episodes=len(rows), flown=sum(r['flown'] for r in rows), score=float(np.mean([r['score'] for r in rows])))
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None and r[k] == r[k]]
        out[k] = float(np.mean(v)) if v else None
        if k in ('t30', 't_reach', 't_settle', 'goal_t'):
            out[k+'_hits'] = len(v)
    if rows and 'recovery_s' in rows[0]:
        out['recovered'] = sum(r['recovery_s'] is not None for r in rows)
    return out


def paired(a, b):
    """Paired score difference a-b over common seeds (bootstrap CI) and win count (a lower)."""
    A = {r['seed']: r['score'] for r in a}
    B = {r['seed']: r['score'] for r in b}
    common = sorted(set(A) & set(B))
    d = np.array([A[s]-B[s] for s in common])
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) else [float('nan')]
    return dict(n=len(d), mean_delta=float(d.mean()) if len(d) else None, ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                wins=int((d < 0).sum()), losses=int((d > 0).sum()), sign_p=U.binom_p(int(min((d < 0).sum(), (d > 0).sum())), int((d != 0).sum())))


def composite(scores, normalisers, weights=None):
    """scores/normalisers: task -> mean score. Tasks missing from the weights are ignored (e.g. a dropped T3)."""
    w = weights or WEIGHTS
    tot = sum(w.values())
    return float(sum(w[t]*scores[t]/normalisers[t] for t in w)/tot)


# ----------------------------------------------------------------------------------------- stages
def revise_dash(s0_dir):
    """Pre-registered rule (maneuver_v2 preregistration: 'otherwise the task definition is revised'): if the S0
    classical S1 reached 0.25 m in < 12/16 blind cases, the T3 target becomes the 12th-largest max forward travel of
    S1 over those 16 cases, rounded down to 0.05 m. Below 0.10 m, T3 is dropped and its weight goes to T1 (0.5/0.25/0.25
    for T1/T2/T4). Returns (dash_m or None, weights, note)."""
    dash = json.loads((Path(s0_dir)/'dash.json').read_text())
    gate = dash['gate_S0']
    if gate['S1_hits'] >= 12:
        return .25, dict(WEIGHTS), 'S0 gate passed at 0.25 m'
    fwd = sorted([r['max_forward_m'] or 0. for r in dash['test_rows'] if r['controller'] == 'S1'], reverse=True)
    q = fwd[11] if len(fwd) >= 12 else 0.
    target = math.floor(q/.05+1e-9)*.05
    if target < .1-1e-9:
        return None, dict(hover=.5, nod=.25, upset=.25), f'S1 reached only {q:.3f} m on 12/16 cases: T3 dropped'
    return float(round(target, 2)), dict(WEIGHTS), f'S1 hits {gate["S1_hits"]}/16 at 0.25 m; 12th-largest travel {q:.3f} m -> target {target:.2f} m'


def law_tuned():
    b = json.loads(V.V2_LAW.read_text())['best']
    return dict(k_theta=b['k_theta'], k_q=b['k_q'], threshold=b['threshold'], width=b['width'])


def tune_schedule(ex, tname, tune_cases, grid, law):
    rows = run_jobs(ex, [(str(i), tname, c, 'classical', dict(law=law, schedule=q)) for i, q in enumerate(grid) for c in tune_cases])
    table = []
    for i, q in enumerate(grid):
        items = [r for r in rows if r['controller'] == str(i)]
        table.append(dict(q=q, score=M.tune_score(items), flown=sum(r['flown'] for r in items), mean=float(np.mean([r['score'] for r in items]))))
    best = min(range(len(grid)), key=lambda i: table[i]['score'])
    return table[best], table


def stage_classical(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    levels = upset_levels(a.output)
    upset_tune = json.loads((out/'upset_v2'/'tune.json').read_text())
    dash_m, weights, note = revise_dash(a.s0) if a.dash_m is None else (a.dash_m, dict(WEIGHTS), f'override --dash-m {a.dash_m}')
    DASH_M['value'] = dash_m if dash_m is not None else .25
    (out/'dash_target.json').write_text(json.dumps(dict(dash_m=dash_m, weights=weights, note=note), indent=1))
    print(json.dumps(dict(dash_m=dash_m, weights=weights, note=note)), flush=True)
    law = law_tuned()
    tasks = [t for t in TASKS if t in weights]
    arms = {}
    with pool(a.workers, DASH_M['value']) as ex:
        t_start = time.monotonic()
        # T1: the pulse law of pilot v2 (108-point grid) on the reflex PID; the plain PID as reference
        arms['hover'] = {'pid': dict(), 'law_tuned': dict(law=law)}
        # T2/T3: goal-gated or timed two-segment schedules, tuned on the 8 S0 tuning cases (crash-constrained)
        grids = dict(nod=NOD_GRID, dash=DASH_GRID)
        tuning = {}
        for tname in ('nod', 'dash'):
            if tname not in weights:
                continue
            grid = grids[tname] if not QUICK else grids[tname][::len(grids[tname])//4]
            best, table = tune_schedule(ex, tname, cases(tname, 'tune', n=2 if QUICK else None), grid, law)
            tuning[tname] = dict(best=best, grid=table, wall=round(time.monotonic()-t_start))
            print(json.dumps(dict(task=tname, arm='schedule_tuned', **{k: v for k, v in best.items()})), flush=True)
            arms[tname] = {'pid': dict(), 'law_tuned': dict(law=law), 'schedule': dict(law=law, schedule=best['q'])}
        # T4: pid vs the pulse law chosen by upset_v2 on its training cases
        arms['upset'] = {'pid': dict(), 'law_tuned': dict(law=law), 'pulse_tuned': dict(law=upset_tune['pulse']['best']['q'])}
        test_rows, table = {}, {}
        for tname in tasks:
            tc = cases(tname, 'test', levels, n=4 if QUICK else None)
            rows = run_jobs(ex, [(name, tname, c, 'classical', arg) for name, arg in arms[tname].items() for c in tc])
            test_rows[tname] = rows
            table[tname] = {name: summary([r for r in rows if r['controller'] == name]) for name in arms[tname]}
            for name, s in table[tname].items():
                print(f"{tname:6s} {name:13s} flown {s['flown']}/{s['episodes']} score {s['score']:.3f} " +
                      ' '.join(f"{k}={v:.2f}" for k, v in s.items() if isinstance(v, float) and k != 'score'), flush=True)
    # opponent per task, chosen on tuning/training data only (pre-registered rule), never on the test set
    upset_opp = 'pulse_tuned' if (upset_tune['pulse']['best']['flown'], -upset_tune['pulse']['best']['J']) >= \
        (upset_tune['pid']['grid'][upset_tune['pid']['best']]['flown'], -upset_tune['pid']['grid'][upset_tune['pid']['best']]['J']) else 'pid'
    opponent = dict(hover='law_tuned', nod='schedule', dash='schedule', upset=upset_opp)
    opponent = {t: opponent[t] for t in tasks}
    normalisers = {t: table[t][opponent[t]]['score'] for t in tasks}
    prereg = dict(
        version='bench (research plan v4, stage S1 preregistration)', time=time.strftime('%Y-%m-%d %H:%M:%S'),
        weights=weights, dash_m=dash_m, dash_note=note, opponent=opponent, normalisers=normalisers,
        reflex_pid=dict(kp=PID_GAINS[0], kd=PID_GAINS[1], ki=PID_GAINS[0]),
        supervisor='goal signal per task (nod: until |pitch| >= 30 deg held 0.1 s; dash: until DASH_M reached), truth-based, same for all arms',
        learned_policy='DescendingNet: 18 raw-IMU features + 2 command channels -> hidden LIF (delay banks) -> 2 spiking motor outputs + 1 graded pitch reference; '
                       'trained by imitation of the opponents (descend.py train) then fine-tuned by policy search on fresh training cases (descend.py finetune)',
        test_sets={t: [c['seed'] for c in cases(t, 'test', levels)] for t in tasks},
        hypotheses=dict(
            H1='composite <= 0.95 in >= 2 of 3 fine-tuned seeds AND T1 ratio <= 1.05 in those seeds (learned beats the classical opponents overall without losing hover)',
            H2='>= 2 of the maneuver tasks (T2, T3, T4) have ratio <= 0.90 in >= 2 of 3 seeds (the advantage comes from the maneuvers)',
            H3='mechanism: clamping gyro+accelerometer inputs raises the composite by >= 10 % or silences the pulses; zeroing all delays raises it by >= 10 %',
            H4='an ANN/rate control of the same size is expected to match (no SNN performance claim); pending',
            hybrid='if H1 fails but some task ratio <= 0.9, the best hybrid (per-task best of classical/learned) is reported as the result'),
        gate_S0=dict(nod='passed (S1 16/16)', dash=note))
    (out/'preregistration.json').write_text(json.dumps(prereg, indent=1))
    result = dict(prereg=prereg, arms={t: {n: dict(arg, law=('law' in arg)) for n, arg in arms[t].items()} for t in tasks}, arms_full=arms,
                  tuning=tuning, table=table, opponent=opponent, normalisers=normalisers, test_rows=test_rows)
    (out/'baseline.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(opponent=opponent, normalisers=normalisers)), flush=True)


def load_baseline(out):
    return json.loads((Path(out)/'baseline.json').read_text())


def evaluate_spec(ex, out, spec, tag, baseline, variants=('intact',), n=None):
    """Blind test of a descending spec (and ablation variants) on every task; paired against the opponents."""
    levels = upset_levels(out)
    tasks = list(baseline['opponent'])
    report = dict(tag=tag, variants={})
    for variant in variants:
        s = json.loads(json.dumps(spec))
        if variant == 'clamp_fast':
            s['mask'] = 'ahrs'
        elif variant == 'nodelay_all':
            W = np.asarray(s['W_in'])
            s['W_in'] = np.concatenate([W.sum(2, keepdims=True), np.zeros_like(W[:, :, 1:])], 2).tolist()
            Wo = np.asarray(s['W_out'])
            s['W_out'] = np.concatenate([Wo.sum(2, keepdims=True), np.zeros_like(Wo[:, :, 1:])], 2).tolist()
            Wg = np.asarray(s['W_g'])
            s['W_g'] = np.concatenate([Wg.sum(1, keepdims=True), np.zeros_like(Wg[:, 1:])], 1).tolist()
        elif variant == 'noref':
            s['W_g'] = np.zeros_like(np.asarray(s['W_g'])).tolist()
            s['b_g'] = 0.
        elif variant == 'nopulse':
            s['b_o'] = [-50., -50.]
        scores, table, rows_all = {}, {}, {}
        for tname in tasks:
            tc = cases(tname, 'test', levels, n=n)
            rows = run_jobs(ex, [(variant, tname, c, 'learned', s) for c in tc])
            rows_all[tname] = rows
            table[tname] = summary(rows)
            opp = [r for r in baseline['test_rows'][tname] if r['controller'] == baseline['opponent'][tname]]
            table[tname]['ratio'] = table[tname]['score']/baseline['normalisers'][tname]
            table[tname]['paired_vs_opponent'] = paired(rows, opp)
            scores[tname] = table[tname]['score']
        comp = composite(scores, baseline['normalisers'], baseline['prereg']['weights'])
        hybrid = {t: min(1., table[t]['ratio']) for t in tasks}
        report['variants'][variant] = dict(table=table, composite=comp, ratios={t: table[t]['ratio'] for t in tasks},
                                           hybrid_composite=composite(hybrid, {t: 1. for t in tasks}, baseline['prereg']['weights']), rows=rows_all)
        print(f"{tag} {variant:12s} composite {comp:.3f} " + ' '.join(f"{t}={table[t]['ratio']:.3f}({table[t]['flown']}/{table[t]['episodes']})" for t in tasks), flush=True)
    return report


def stage_evaluate(a):
    out = Path(a.output)
    baseline = load_baseline(out)
    spec = json.loads(Path(a.spec).read_text())
    variants = tuple(a.variants.split(',')) if a.variants else ('intact',)
    with pool(a.workers, baseline['prereg']['dash_m'] or .25) as ex:
        report = evaluate_spec(ex, out, spec, a.tag, baseline, variants, n=4 if QUICK else None)
    (out/f'evaluate_{a.tag}.json').write_text(json.dumps(dict(report, spec=str(a.spec)), indent=1))


# ----------------------------------------------------------------------------------------- selftest
def random_spec(hidden=4, n_in=N_IN, seed=0, scale=1.):
    rng = np.random.default_rng(seed)
    K, D = len(IN_DELAYS_TICKS), len(HID_DELAYS_MS)
    return dict(W_in=(rng.normal(0, .3*scale/math.sqrt(K), (hidden, n_in, K))).tolist(), b_h=(rng.normal(-.5, .3, hidden)).tolist(),
                W_out=(rng.normal(0, .3*scale/math.sqrt(D), (2, hidden, D))).tolist(), b_o=[-2., -2.],
                W_g=(rng.normal(0, .3*scale/math.sqrt(D), (hidden, D))).tolist(), b_g=0., tau_m=1e-3, tau_s=.5e-3, tau_g=TAU_G,
                spike_scale=SPIKE_SCALE, inputs=n_in, hidden=hidden)


def selftest():
    s = Schedule(dict(go_ref=-.6, go_beats=9, brake_beats=6, brake_ref=.2, gated=True))
    s.reset()
    assert s.ref(-.1, True) == (0., False) and s.ref(0., True) == (-.6, True) and s.ref(.2, True) == (-.6, True)
    assert s.ref(.3, False) == (.2, True) and s.ref(.3+.23, False) == (.2, True) and s.ref(.3+.25, False) == (0., False)
    s = Schedule(dict(go_ref=.5, go_beats=30, brake_beats=6, brake_ref=-.35, gated=False))
    s.reset()
    assert s.ref(0., False) == (.5, True) and s.ref(1.19, False) == (.5, True) and s.ref(1.21, False) == (-.35, True) and s.ref(1.5, False) == (0., False)
    x = features(np.r_[.05, -.05, 0., 5., -5., 0., 25., 0., 0., 0., 0.], np.array([1., 0.]))
    assert x.shape == (N_IN,) and x[0] == .5 and x[9] == .5 and x[2] == .5 and x[11] == .5 and x[16] == 1. and x[17] == 0. and x[18] == 1. and x[19] == 0.
    net = DescendingNet(random_spec(scale=6.))
    rng = np.random.default_rng(1)
    fired = 0
    for _ in range(200):
        f, ref = net.tick(rng.random(N_IN)*(rng.random(N_IN) < .4))
        fired += int(f.any())
        assert abs(ref) < REF_MAX
    assert net.spikes > 0 and fired > 0, (net.spikes, fired)
    masked = DescendingNet(dict(random_spec(scale=6.), mask='ahrs'))
    assert masked.mask[2] == 0. and masked.mask[10] == 0. and masked.mask[0] == 1. and masked.mask[16] == 1.
    assert len(NOD_GRID) == 2*2*3*(1+2+2)*2 and len(DASH_GRID) == 2*3*3*3*2*2, (len(NOD_GRID), len(DASH_GRID))
    assert abs(composite(dict(hover=1., nod=2., dash=1., upset=.5), dict(hover=1., nod=1., dash=1., upset=1.))-(.4+.4+.2+.1)) < 1e-12
    assert abs(composite(dict(hover=1., nod=1., upset=1.), dict(hover=1., nod=1., upset=1.), dict(hover=.5, nod=.25, upset=.25))-1.) < 1e-12
    fake = {c: dict(levels=[5e-3, 8e-3, 1.2e-2]) for c in U.CLASSES}
    assert len(cases('upset', 'train', fake, n=6, offset=3)) == 6 and cases('upset', 'train', fake, n=2) == cases('upset', 'train', fake, n=2)
    assert cases('upset', 'test', fake)[0]['seconds'] == 5. and len(cases('nod', 'tune')) == 8 and cases('dash', 'test')[0]['seconds'] == 6.5
    p = paired([dict(seed=i, score=1.) for i in range(8)], [dict(seed=i, score=2.) for i in range(8)])
    assert p['wins'] == 8 and p['mean_delta'] == -1.
    print('selftest ok')


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['selftest', 'classical', 'evaluate'])
    ap.add_argument('--output', default='runs/spike_steer/bench')
    ap.add_argument('--s0', default='runs/spike_steer/maneuver_v2', help='maneuver_v2 output (dash.json) for the T3 revision rule')
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--spec')
    ap.add_argument('--tag', default='spec')
    ap.add_argument('--variants', default='')
    ap.add_argument('--dash-m', type=float, default=None, help='skip the revision rule (smoke tests)')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    QUICK = a.quick
    if a.stage == 'selftest':
        return selftest()
    dict(classical=stage_classical, evaluate=stage_evaluate)[a.stage](a)


if __name__ == '__main__':
    main()
