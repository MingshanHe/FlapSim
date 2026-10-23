"""Righting reflex, Stage A (classical only): does multi-phase reflex logic help this plant recover from large upsets?

Scenario (a new test scenario; the sensors, actuators, 2 ms update, smoothing and bounds are unchanged):
  * a stress case with the periodic disturbance pulses switched off (pulse_force = pulse_torque = 0; the RNG
    order is unchanged);
  * at t0 = 2.4 s + U(0, 40 ms), a 20 ms torque about a horizontal axis on base_link. The axis is fixed in the
    heading frame: direction 0/180 deg = roll axis, 90/270 deg = pitch axis, the rest diagonal;
  * in half of the cases, a second hit of the same size 80-200 ms later in a random direction.
The torque is added inside air.advance, after ControlEnv.step has zeroed the external forces.

Controllers (IMU only: lagged AHRS, gyro, PLL wingbeat phase and wingbeat-notched gyro rates):
  pid       the base PID, retuned on upsets (kp/kd grid);
  pulse     PID + the phase-locked motor-pulse law (36-point grid);
  gs        gain-scheduled PID: above a rate threshold, the D term uses the notched gyro (CEM, 384 episodes);
  oneshot   one saturated counter twitch + attitude-integrator freeze, no brake (CEM, 384 episodes);
  deadzone  PID + a dead-zoned notched-rate gimbal term (24-point grid);
  auto      designer automaton HOVER -> COUNTER -> BRAKE -> HANDBACK per axis (10 parameters, CEM, 1536 episodes).

Score per case (lower is better): J = tilt integral over [2 s, t0) + tilt integral over [t0, t0+1.5 s] (deg s)
+ 0.2 x max|z - z(t0)| over the window (cm, truth). A crash scores 60 + 40 x the unflown fraction of the window.
GO criteria are in PREREGISTRATION and are written to disk before the test stage runs.

    python -m experiments.algorithms.spike_steer.upset selftest
    python -m experiments.algorithms.spike_steer.upset all --output runs/spike_steer/upset_s1 [--quick]
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
import mujoco
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..wingbeat_feasibility.preliminary import WingbeatPhase
from ..wingbeat_snn.plant import PHASE_OFFSET
from ..neuro_gates import gates
from ..neuro_gates.timing import PULSE_PHASE

T0, T0_JITTER, WINDOW, SECONDS, HIT_S = 2.4, .04, 1.5, 4., .02
DIRECTIONS = [45.*k for k in range(8)]
CLASSES = ('roll', 'pitch', 'diag')
CALIB_MAGS = (5e-3, 7e-3, 9e-3, 11e-3, 13e-3, 16e-3)  # N m; PID kp x1.5 survives 8e-3 about the roll axis (probe)
TARGET_SURVIVAL = (.8, .5, .2)
BASES = dict(calib=2_060_100_000, prim=2_060_200_000, train=2_060_300_000, test=2_060_400_000)
CALIB_GAINS = (1.5, 1.)  # the pilot's retuned PID, used only to calibrate the upset sizes
RECOVER_DEG, RECOVER_HOLD, SMOOTH = 8., .3, 20  # recovered: 40 ms mean tilt < 8 deg held for 0.3 s
PULSE_GRID = [dict(k_theta=a, k_q=b, threshold=c, width=math.radians(w)) for a in (3., 5., 8.) for b in (.2, .4, .8)
              for c in (.5, .8) for w in (60., 90.)]
DEADZONE_GRID = [dict(k=k, r0=r, pol=s) for k in (.02, .05, .1, .2) for r in (1., 2., 4.) for s in (1., -1.)]
BOXES = dict(
    gs=dict(r_th=(.5, 8.), kd_fast=(.2, 3.), kp_boost=(1., 3.), hold=(0., .2)),
    oneshot=dict(r_th=(.5, 8.), dwell=(.006, .08), amp=(.2, 1.), pol=(-1., 1.), freeze=(0., .4)),
    auto=dict(r_th=(.5, 8.), a_th=(.1, .8), t1=(.006, .1), a1=(.2, 1.), pol=(-1., 1.), t2=(0., .15), a2=(0., 1.),
              freeze=(0., .5), blend=(0., .3), lift=(0., 1.)))
CEM_BUDGET = dict(gs=(12, 4), oneshot=(12, 4), auto=(24, 8))  # population x iterations, 8 cases each
SINGLE = ('pulse', 'gs', 'oneshot', 'deadzone')
PREREGISTRATION = dict(
    written_before='the Stage A test stage runs',
    test_cases='96 held-out paired cases: 8 directions x 3 calibrated levels x 4 seeds, half with a second hit',
    score='J = tilt integral [2 s, t0) + tilt integral [t0, t0+1.5 s] (deg s) + 0.2 x max|z-z(t0)| (cm); '
          'crash = 60 + 40 x unflown window fraction',
    best_single='the single-mode controller (pulse, gs, oneshot, deadzone) with the lowest mean test J',
    GO=['C1: auto beats best_single: (survivals_auto - survivals_single >= 8 and McNemar exact p < 0.05) or '
        '(mean J_auto <= 0.85 x mean J_single and auto wins >= 60% of paired cases with sign-test p < 0.05)',
        'C2: auto also has lower mean thrust-unit peak tilt and lower mean recovery time (censored at 1.5 s) '
        'than best_single',
        'C3: the better of oneshot and gs captures < 80% of the auto gain over pid: '
        '(J_pid - J_best_os_gs) / (J_pid - J_auto) < 0.8, with J_auto < J_pid'],
    if_fail='Stage A NO-GO: report the classical result only; Stage B (SNN/NEAT) needs the user\'s approval anyway')
QUICK = False


def n(full, quick):
    return quick if QUICK else full


def klass(phi):
    return 'roll' if phi % 180 == 0 else 'pitch' if phi % 180 == 90 else 'diag'


def upset_case(seed, phi, mag, double, index=0):
    c = make_case(seed, SECONDS, 'stress', index % 8)
    c.update(pulse_force=0., pulse_torque=0.)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    t0 = T0+float(rng.uniform(0., T0_JITTER))
    second = (t0+float(rng.uniform(.08, .2)), float(rng.choice(DIRECTIONS)), mag)
    c['upset'] = dict(t0=t0, hits=[(t0, float(phi), float(mag))]+([second] if double else []), phi=float(phi),
                      mag=float(mag), double=bool(double))
    return c


def install(env):
    """Add the upset torque inside air.advance (after ControlEnv.step zeroes the external forces)."""
    original, body = env.air.advance, env.air.env.body

    def advance(command, p):
        u = env.case.get('upset')
        if u:
            t = env.air.steps*DT
            for k, (start, phi, mag) in enumerate(u['hits']):
                if start-1e-9 <= t < start+HIT_S-1e-9:
                    if k not in env.upset_axes:  # axis fixed in the heading frame at the hit onset
                        r = env.sim.data.xmat[body]
                        a = math.atan2(r[3], r[0])+math.radians(phi)
                        env.upset_axes[k] = np.array([math.cos(a), math.sin(a), 0.])
                    env.sim.external_body_forces[body, 3:6] += mag*env.upset_axes[k]
        original(command, p)
    env.air.advance = advance
    env.upset_axes = {}


class Notch:
    """Removes the wingbeat harmonics from a gyro channel, referenced to the pitch-gyro PLL phase."""

    def __init__(self, window=25):
        self.buffer, self.slow = deque(maxlen=window), 0.

    def update(self, theta, y):
        self.buffer.append((theta, y))
        self.slow = y
        if len(self.buffer) == self.buffer.maxlen:
            th = np.array([b[0] for b in self.buffer])
            X = np.c_[np.cos(th), np.sin(th), np.cos(2*th), np.sin(2*th), np.ones_like(th)]
            coef, *_ = np.linalg.lstsq(X, np.array([b[1] for b in self.buffer]), rcond=None)
            self.slow = y-float(X[-1, :4]@coef[:4])


def axis_cmd(k):
    """Command index of the gimbal that the PID drives from axis k (0 roll, 1 pitch): angles = trim + effort[::-1]."""
    return 1-k


class Arm:
    """IMU-only reflex layer on top of the base PID. `rate` is [roll, pitch] notched gyro (rad/s)."""

    def __init__(self, kind='pid', q=None):
        self.kind, self.q = kind, dict(q or {})

    def reset(self, env):
        self.est, self.notch = WingbeatPhase(PHASE_OFFSET), Notch()
        self.events, self.rate = [], np.zeros(2)
        self.state = [dict(mode='idle', until=0., dir=0., freeze_until=-1., refractory=0., hold=0.) for _ in range(2)]
        self.frozen = env.teacher.integral.copy()
        self.active = False

    def sense(self, env, now):
        r = env.last_reading
        self.est.update(now, float(r[4]))
        self.notch.update(self.est.buffer[-1][0], float(r[3]))
        self.rate = np.array([self.notch.slow, self.est.slow])

    def freeze(self, env, t):
        for k, s in enumerate(self.state):
            if t < s['freeze_until']:
                env.teacher.integral[k] = self.frozen[k]
        self.frozen = env.teacher.integral.copy()

    def act(self, env, cmd, t, now):
        q, kind = self.q, self.kind
        if kind == 'pulse':  # pilot.Steer law
            r = env.last_reading
            u = -(q['k_theta']*r[1]+q['k_q']*self.est.slow)
            psi = PULSE_PHASE if u > 0 else PULSE_PHASE+math.pi
            fire = abs(u) > q['threshold'] and abs(math.remainder(self.est.crank(now)-psi, 2*math.pi)) < q['width']/2
            if fire:
                cmd[2] = HIGH[2]
                if not self.active:
                    self.events.append(t)
            self.active = fire
        elif kind == 'gs':
            p = env.teacher.p
            for k in range(2):
                s = self.state[k]
                if abs(self.rate[k]) > q['r_th']:
                    if t >= s['hold']:
                        self.events.append(t)
                    s['hold'] = t+q['hold']+DT
                if t < s['hold']:
                    delta = -p['kd'][k]*(q['kd_fast']*self.rate[k]-env.teacher.rate[k]) \
                        - (q['kp_boost']-1.)*p['kp'][k]*env.last_reading[k]
                    cmd[axis_cmd(k)] += delta
        elif kind == 'deadzone':
            for k in range(2):
                z = math.copysign(max(abs(self.rate[k])-q['r0'], 0.), self.rate[k])
                cmd[axis_cmd(k)] += -q['pol']*q['k']*z
        elif kind in ('oneshot', 'auto'):
            self.reflex(env, cmd, t)
        return cmd

    def reflex(self, env, cmd, t):
        q, auto = self.q, self.kind == 'auto'
        pol = 1. if q['pol'] >= 0 else -1.
        lift = False
        for k in range(2):
            s, rate, angle = self.state[k], self.rate[k], env.last_reading[k]
            trig_rate = abs(rate) > q['r_th']
            trig = trig_rate or (auto and abs(angle) > q['a_th'])
            if trig and t >= s['refractory'] and s['mode'] in ('idle', 'handback'):
                sign = -math.copysign(1., rate if trig_rate else angle)  # the PID's restoring direction
                s.update(mode='counter', dir=pol*sign, until=t+(q['t1'] if auto else q['dwell']),
                         freeze_until=t+q['freeze'], refractory=t+(.25 if auto else .3))
                self.events.append(t)
            if s['mode'] == 'counter' and t >= s['until']:
                s.update(mode='brake' if auto and q['t2'] > 0 else 'handback' if auto else 'idle',
                         until=t+(q['t2'] if auto and q['t2'] > 0 else q['blend'] if auto else 0.))
            if s['mode'] == 'brake' and t >= s['until']:
                s.update(mode='handback', until=t+q['blend'])
            if s['mode'] == 'handback' and t >= s['until']:
                s['mode'] = 'idle'
            i = axis_cmd(k)
            if s['mode'] == 'counter':
                cmd[i] = .3*s['dir']*(q['a1'] if auto else q['amp'])
                lift = True
            elif s['mode'] == 'brake':
                cmd[i] = -.3*s['dir']*q['a2']
                lift = True
            elif s['mode'] == 'handback' and q['blend'] > 0:
                w = 1.-(s['until']-t)/q['blend']
                trim = env.teacher.p['trim'][i]
                cmd[i] = trim+w*(cmd[i]-trim)
        if auto and lift:
            cmd[2] += q['lift']*(HIGH[2]-cmd[2])
        self.freeze(env, t)


def decode(kind, x):
    """Unit-box vector -> named parameters."""
    box = BOXES[kind]
    return {name: lo+float(np.clip(v, 0., 1.))*(hi-lo) for (name, (lo, hi)), v in zip(box.items(), x)}


def run(env, c, kind, q, gains):
    env.configure(c)
    env.reset(seed=c['seed'])
    env.upset_axes = {}
    p, (sp, sd) = env.p, gains
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    arm = Arm(kind, q)
    arm.reset(env)
    m, d = env.sim.model, env.sim.data
    body, thrust = env.air.env.body, m.body('thrust_unit').id
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    up = math.copysign(1., float(d.xmat[thrust][8]))  # the thrust unit's local z points down in its own frame
    u = c['upset']
    t0, end = u['t0'], u['t0']+WINDOW
    log, phase_err = [], []
    while True:
        t, now = env.air.steps*DT, float(d.time)
        cmd = env.teacher_command()
        arm.sense(env, now)
        if t >= gates.HANDOVER_T and kind != 'pid':
            cmd = arm.act(env, cmd, t, now)
        _, _, term, trunc, out = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        t = env.air.steps*DT
        thrust_tilt = math.acos(float(np.clip(up*d.xmat[thrust][8], -1., 1.)))
        log.append((t, thrust_tilt, float(np.linalg.norm(d.cvel[body, :3]))))
        if t0 <= t < end:
            phase_err.append(abs(math.remainder(arm.est.crank(float(d.time))-float(d.qpos[crank]), 2*math.pi)))
        if term or trunc:
            break
    ep = out['episode']
    s, lg = np.asarray(env.samples), np.asarray(log)
    ts, tilt, z = s[:, 0], np.degrees(s[:, 1]), s[:, 2]
    pre, win = (ts >= 2.) & (ts < t0), (ts >= t0) & (ts < end)
    flown = ep['termination'] == 'time_limit'
    i0 = int(np.argmin(abs(ts-t0)))
    dz = float(100*np.abs(z[win]-z[i0]).max()) if win.any() else float('nan')
    if flown:
        J = float(tilt[pre].sum()*DT+tilt[win].sum()*DT+.2*dz)
    else:
        J = 60.+40.*float(np.clip((end-ts[-1])/WINDOW, 0., 1.))
    smooth = np.convolve(tilt, np.ones(SMOOTH)/SMOOTH, mode='full')[:len(tilt)]
    hold = round(RECOVER_HOLD/DT)
    recovery = None
    if flown:
        good = (smooth < RECOVER_DEG) & (ts >= t0)
        run_len = 0
        for i in range(len(good)):
            run_len = run_len+1 if good[i] else 0
            if run_len >= hold:
                recovery = float(ts[i-hold+1]-t0)
                break
    rate_ma = np.convolve(lg[:, 2], np.ones(SMOOTH)/SMOOTH, mode='full')[:len(lg)]
    early = (lg[:, 0] >= t0) & (lg[:, 0] < t0+.3)
    return dict(seed=c['seed'], phi=u['phi'], mag=u['mag'], double=u['double'], cls=klass(u['phi']),
                termination=ep['termination'], elapsed=ep['elapsed'], flown=flown, J=J,
                tilt_peak_deg=float(tilt[win].max()) if win.any() else float('nan'),
                thrust_tilt_peak_deg=float(np.degrees(lg[(lg[:, 0] >= t0) & (lg[:, 0] < end), 1].max()))
                if (lg[:, 0] >= t0).any() else float('nan'),
                recovery_s=recovery, recovery_censored=recovery if recovery is not None else WINDOW,
                altitude_loss_cm=float(max(0., 100*(z[i0]-z[win].min()))) if win.any() else float('nan'),
                dz_cm=dz, false_triggers=sum(1 for e in arm.events if 2. <= e < t0), events=len(arm.events),
                induced_rate=float(rate_ma[early].max()) if early.any() else float('nan'),
                phase_error_deg=float(np.degrees(np.mean(phase_err))) if phase_err else float('nan'))


_ENV = None


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(str(source), mode='direct', task='hover')
    install(_ENV)


def task(args):
    name, c, kind, q, gains = args
    return dict(run(_ENV, c, kind, q, gains), controller=name)


def pool(workers):
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None and r[key] == r[key]]
    return float(np.mean(v)) if v else None


def summary(rows):
    return dict(episodes=len(rows), flown=sum(r['flown'] for r in rows), J=mean(rows, 'J'),
                tilt_peak_deg=mean(rows, 'tilt_peak_deg'), thrust_tilt_peak_deg=mean(rows, 'thrust_tilt_peak_deg'),
                recovery_censored_s=mean(rows, 'recovery_censored'), recovered=sum(r['recovery_s'] is not None for r in rows),
                altitude_loss_cm=mean([r for r in rows if r['flown']], 'altitude_loss_cm'),
                false_triggers=mean(rows, 'false_triggers'), phase_error_deg=mean(rows, 'phase_error_deg'))


def binom_p(k, n_):
    """Exact two-sided binomial test against 1/2."""
    if n_ == 0:
        return 1.
    k = min(k, n_-k)
    return min(1., 2*sum(math.comb(n_, i) for i in range(k+1))/2**n_)


def compare(a, b):
    """Paired comparison of rows a vs b (same seed order). Lower J is better."""
    assert [r['seed'] for r in a] == [r['seed'] for r in b]
    only_a = sum(x['flown'] and not y['flown'] for x, y in zip(a, b))
    only_b = sum(y['flown'] and not x['flown'] for x, y in zip(a, b))
    wins = sum(x['J'] < y['J'] for x, y in zip(a, b))
    losses = sum(x['J'] > y['J'] for x, y in zip(a, b))
    Ja, Jb = mean(a, 'J'), mean(b, 'J')
    return dict(extra_survivals=only_a-only_b, mcnemar_p=binom_p(min(only_a, only_b), only_a+only_b),
                J_ratio=Ja/Jb if Jb else None, wins=wins, losses=losses, win_frac=wins/len(a),
                sign_p=binom_p(min(wins, losses), wins+losses))


def judge(table, rows_by):
    best = min(SINGLE, key=lambda k: table[k]['J'])
    cmp_ = compare(rows_by['auto'], rows_by[best])
    c1 = (cmp_['extra_survivals'] >= 8 and cmp_['mcnemar_p'] < .05) or \
         (cmp_['J_ratio'] <= .85 and cmp_['win_frac'] >= .6 and cmp_['sign_p'] < .05)
    c2 = table['auto']['thrust_tilt_peak_deg'] < table[best]['thrust_tilt_peak_deg'] and \
        table['auto']['recovery_censored_s'] < table[best]['recovery_censored_s']
    os_gs = min(('oneshot', 'gs'), key=lambda k: table[k]['J'])
    gain_auto = table['pid']['J']-table['auto']['J']
    capture = (table['pid']['J']-table[os_gs]['J'])/gain_auto if gain_auto > 0 else None
    c3 = capture is not None and capture < .8
    return dict(best_single=best, auto_vs_best=cmp_, C1=bool(c1), C2=bool(c2), best_os_gs=os_gs, capture=capture,
                C3=bool(c3), GO=bool(c1 and c2 and c3))


def logistic_levels(rows):
    """Per direction class: fit P(survive) = 1/(1+exp(b*(log m - a))) by grid MLE, invert at TARGET_SURVIVAL."""
    out = {}
    for cls in CLASSES:
        items = [(math.log(r['mag']), r['flown']) for r in rows if r['cls'] == cls]
        x, y = np.array([i[0] for i in items]), np.array([i[1] for i in items], dtype=float)
        best = None
        for a in np.linspace(math.log(CALIB_MAGS[0])-.5, math.log(CALIB_MAGS[-1])+.8, 200):
            for b in np.linspace(1., 40., 80):
                pr = np.clip(1./(1.+np.exp(b*(x-a))), 1e-6, 1-1e-6)
                ll = float(np.sum(y*np.log(pr)+(1-y)*np.log(1-pr)))
                if best is None or ll > best[0]:
                    best = (ll, a, b)
        _, a, b = best
        levels = [float(math.exp(a+math.log(1./s-1.)/b)) for s in TARGET_SURVIVAL]
        out[cls] = dict(levels=levels, a=float(a), b=float(b),
                        survival_by_mag={f'{m:g}': float(np.mean([f for xm, f in items if abs(xm-math.log(m)) < 1e-9]))
                                         for m in CALIB_MAGS})
    return out


def train_cases(levels):
    lv = (1, 2, 0, 1, 2, 0, 1, 1)
    return [upset_case(BASES['train']+k, phi, levels[klass(phi)]['levels'][lv[k]], k % 2 == 1, k)
            for k, phi in enumerate(DIRECTIONS)]


def test_cases(levels):
    out, i = [], 0
    for phi in DIRECTIONS:
        for level in range(3):
            for s in range(n(4, 1)):
                out.append(upset_case(BASES['test']+i, phi, levels[klass(phi)]['levels'][level], s % 2 == 1, i))
                i += 1
    return out[:n(96, 4)]


def run_jobs(ex, jobs):
    return list(ex.map(task, jobs, chunksize=1))


def save(out, name, obj):
    (out/name).write_text(json.dumps(obj, indent=1)+'\n')
    print(f'[{time.strftime("%H:%M:%S")}] wrote {name}', flush=True)


def load(out, name):
    path = out/name
    return json.loads(path.read_text()) if path.exists() else None


def stage_calibrate(ex, out):
    if (r := load(out, 'calibrate.json')):
        return r
    mags, seeds = (CALIB_MAGS, n(4, 1))
    jobs = [('calib', upset_case(BASES['calib']+1000*j+s, phi, m, False, s), 'pid', {}, CALIB_GAINS)
            for j, m in enumerate(mags) for phi in DIRECTIONS[:n(8, 2)] for s in range(seeds)]
    rows = run_jobs(ex, jobs)
    if QUICK:  # one class per direction subset: fill the missing classes so that later stages can run
        rows += [dict(r, cls=c) for r in rows for c in CLASSES if c != r['cls']]
    levels = logistic_levels(rows)
    result = dict(stage='calibrate', gains=CALIB_GAINS, levels=levels, rows=rows,
                  induced_rate_by_class_level={})
    save(out, 'calibrate.json', result)
    return result


def stage_primitives(ex, out, levels):
    """Paired gimbal twitches (+-0.3 rad, 20 ms) in hover and 60 ms after a mid-level upset; thrust-unit and base
    tilt 100 ms later, compared with the same case without a twitch. Characterisation only."""
    if (r := load(out, 'primitives.json')):
        return r
    jobs = []
    for s in range(n(4, 1)):
        for ctx in ('hover', 'upset'):
            phi = 90. if s % 2 == 0 else 0.
            c = upset_case(BASES['prim']+s, phi, levels[klass(phi)]['levels'][1] if ctx == 'upset' else 0., False, s)
            at = c['upset']['t0']+(.06 if ctx == 'upset' else 0.)
            for axis in (0, 1):
                for sign in (1., -1.):
                    q = dict(r_th=1e9, dwell=.02, amp=1., pol=sign, freeze=0., at=at, axis=axis)
                    jobs.append((f'{ctx}|{axis}|{sign:+g}', c, 'twitch', q, CALIB_GAINS))
            jobs.append((f'{ctx}|ref', c, 'pid', dict(at=at), CALIB_GAINS))
    rows = list(ex.map(twitch_task, jobs, chunksize=1))
    result = dict(stage='primitives', rows=rows)
    save(out, 'primitives.json', result)
    return result


def twitch_task(args):
    """Forced gimbal twitch at q['at'] on one axis. Logged at +20/50/100 ms: t, base roll, pitch, thrust roll, pitch
    (its Euler roll sits near 180 deg), base tilt, thrust tilt (rad)."""
    name, c, kind, q, gains = args
    env = _ENV
    env.configure(c)
    env.reset(seed=c['seed'])
    env.upset_axes = {}
    p, (sp, sd) = env.p, gains
    env.teacher.p = dict(p, kp=[sp*k for k in p['kp']], kd=[sd*k for k in p['kd']], ki=[sp*k for k in p['ki']])
    m, d = env.sim.model, env.sim.data
    body, thrust = env.air.env.body, m.body('thrust_unit').id
    euler = lambda r: (math.atan2(r[7], r[8]), math.asin(min(1., max(-1., -r[6]))))
    up = math.copysign(1., float(d.xmat[thrust][8]))
    tilt = lambda r, sgn=1.: math.acos(min(1., max(-1., sgn*r[8])))
    trace = []
    while env.air.steps*DT < q['at']+.15:
        t = env.air.steps*DT
        cmd = env.teacher_command()
        if kind == 'twitch' and q['at'] <= t < q['at']+q['dwell']:
            cmd[axis_cmd(q['axis'])] = .3*q['pol']
        _, _, term, trunc, _ = env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
        trace.append((env.air.steps*DT, *euler(d.xmat[body]), *euler(d.xmat[thrust]), tilt(d.xmat[body]), tilt(d.xmat[thrust], up)))
        if term or trunc:
            break
    tr = np.array(trace)
    pick = lambda dt: tr[int(np.argmin(abs(tr[:, 0]-(q['at']+dt))))].tolist() if len(tr) else None
    return dict(name=name, seed=c['seed'], at=q['at'], at_20ms=pick(.02), at_50ms=pick(.05), at_100ms=pick(.1))


def stage_base_pid(ex, out, cases):
    if (r := load(out, 'base_pid.json')):
        return r
    grid = gates.GRID if not QUICK else gates.GRID[::8]
    rows = run_jobs(ex, [(f'{g[0]:g},{g[1]:g}', c, 'pid', {}, g) for g in grid for c in cases])
    score = {}
    for g in grid:
        items = [r for r in rows if r['controller'] == f'{g[0]:g},{g[1]:g}']
        score[f'{g[0]:g},{g[1]:g}'] = (sum(r['flown'] for r in items), -mean(items, 'J'))
    best = max(score, key=score.get)
    result = dict(stage='base_pid', gains=[float(v) for v in best.split(',')], score=score, rows=rows)
    save(out, 'base_pid.json', result)
    return result


def stage_tune(ex, out, cases, gains, seed):
    """All tuned arms share the 8 training cases (common random numbers). Grids and CEM run in lockstep rounds."""
    if (r := load(out, 'tune.json')):
        return r
    rng = np.random.default_rng(seed)
    pending = [('pulse', q) for q in (PULSE_GRID if not QUICK else PULSE_GRID[:2])] + \
              [('deadzone', q) for q in (DEADZONE_GRID if not QUICK else DEADZONE_GRID[:2])]
    cem = {}
    for kind, (pop, iters) in CEM_BUDGET.items():
        dim = len(BOXES[kind])
        cem[kind] = dict(mu=np.full(dim, .5), sd=np.full(dim, .3), pop=n(pop, 2), iters=n(iters, 1), it=0)
    history, evaluated = [], {k: [] for k in ('pulse', 'deadzone', *CEM_BUDGET)}
    for rnd in range(max(v['iters'] for v in cem.values())+1):
        batch = list(pending) if rnd == 0 else []
        pending = []
        drawn = {}
        for kind, st in cem.items():
            if st['it'] >= st['iters']:
                continue
            xs = np.clip(st['mu']+st['sd']*rng.standard_normal((st['pop'], len(st['mu']))), 0., 1.)
            if st['it'] == 0:
                xs[0] = 1.  # first candidate: thresholds at their maximum, i.e. close to the plain PID
            drawn[kind] = xs
            batch += [(kind, decode(kind, x)) for x in xs]
        if not batch:
            break
        rows = run_jobs(ex, [(f'{i}', c, kind, q, gains) for i, (kind, q) in enumerate(batch) for c in cases])
        for i, (kind, q) in enumerate(batch):
            items = [r for r in rows if r['controller'] == f'{i}']
            evaluated[kind].append(dict(q=q, J=mean(items, 'J'), flown=sum(r['flown'] for r in items)))
        for kind, xs in drawn.items():
            st = cem[kind]
            Js = np.array([e['J'] for e in evaluated[kind][-len(xs):]])
            elite = xs[np.argsort(Js)[:max(2, len(xs)//4)]]
            st['mu'], st['sd'] = elite.mean(0), np.maximum(elite.std(0), .05)
            st['it'] += 1
        history.append(dict(round=rnd, best={k: min(e['J'] for e in v) for k, v in evaluated.items() if v}))
        print(json.dumps(history[-1]), flush=True)
    chosen = {k: min(v, key=lambda e: e['J']) for k, v in evaluated.items()}
    result = dict(stage='tune', gains=list(gains), chosen=chosen, history=history, evaluated=evaluated)
    save(out, 'tune.json', result)
    return result


def stage_test(ex, out, cases, gains, chosen):
    if (r := load(out, 'test.json')):
        return r
    save(out, 'preregistration.json', dict(PREREGISTRATION, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                           gains=list(gains), chosen={k: v['q'] for k, v in chosen.items()},
                                           cases=[c['seed'] for c in cases]))
    arms = [('pid', 'pid', {})]+[(k, k, chosen[k]['q']) for k in (*SINGLE, 'auto')]
    rows = run_jobs(ex, [(name, c, kind, q, gains) for name, kind, q in arms for c in cases])
    by = {name: sorted([r for r in rows if r['controller'] == name], key=lambda r: r['seed']) for name, _, _ in arms}
    table = {name: summary(v) for name, v in by.items()}
    by_class = {f'{name}|{cls}': summary([r for r in v if r['cls'] == cls]) for name, v in by.items() for cls in CLASSES}
    by_double = {f'{name}|{"double" if d else "single"}': summary([r for r in v if r['double'] == d])
                 for name, v in by.items() for d in (False, True)}
    verdict = judge(table, by)
    result = dict(stage='test', gains=list(gains), table=table, by_class=by_class, by_double=by_double,
                  pairwise={f'{a}-vs-pid': compare(by[a], by['pid']) for a in by if a != 'pid'},
                  verdict=verdict, rows=rows)
    save(out, 'test.json', result)
    for name, s in table.items():
        print(f"{name:9s} flown {s['flown']}/{s['episodes']} J {s['J']:.2f} peak {s['tilt_peak_deg']:.1f} "
              f"thrust_peak {s['thrust_tilt_peak_deg']:.1f} rec {s['recovery_censored_s']:.2f}s")
    print(json.dumps(verdict, indent=1))
    return result


def selftest():
    """No simulation: case draws, parameter decoding, statistics, the reflex state machine on a stub env."""
    c = upset_case(BASES['test'], 90., 9e-3, True)
    assert c['pulse_force'] == 0. and c['pulse_torque'] == 0. and len(c['upset']['hits']) == 2
    assert T0 <= c['upset']['t0'] < T0+T0_JITTER and .08 <= c['upset']['hits'][1][0]-c['upset']['t0'] <= .2
    assert upset_case(BASES['test'], 90., 9e-3, True) == c  # deterministic
    assert [klass(p) for p in DIRECTIONS] == ['roll', 'diag', 'pitch', 'diag']*2
    q = decode('auto', np.full(10, .5))
    assert set(q) == set(BOXES['auto']) and abs(q['r_th']-4.25) < 1e-12
    assert len(PULSE_GRID) == 36 and len(DEADZONE_GRID) == 24
    assert abs(binom_p(0, 8)-2/256) < 1e-12 and binom_p(4, 8) == 1.
    rows = [dict(seed=i, flown=True, J=1.) for i in range(10)]
    worse = [dict(seed=i, flown=i >= 9, J=2. if i >= 9 else 61.) for i in range(10)]
    cmp_ = compare(rows, worse)
    assert cmp_['extra_survivals'] == 9 and cmp_['wins'] == 10 and cmp_['mcnemar_p'] < .01
    # logistic level fit recovers a planted threshold
    rng = np.random.default_rng(0)
    fake = []
    for m in CALIB_MAGS:
        for phi in DIRECTIONS:
            for _ in range(20):
                pr = 1/(1+math.exp(12*(math.log(m)-math.log(1.1e-2))))
                fake.append(dict(mag=m, cls=klass(phi), flown=bool(rng.random() < pr)))
    lv = logistic_levels(fake)
    assert all(abs(lv[k]['levels'][1]/1.1e-2-1) < .15 for k in CLASSES), lv

    class Stub:  # minimal env for the reflex state machine
        class teacher:
            p = dict(kp=[1., 1.], kd=[1., 1.], trim=[0., 0.])
            integral = np.zeros(2)
            rate = np.zeros(2)
        last_reading = np.zeros(11)
    arm = Arm('auto', dict(r_th=1., a_th=1., t1=.01, a1=1., pol=1., t2=.01, a2=.5, freeze=.05, blend=.01, lift=0.))
    env = Stub()
    arm.state = [dict(mode='idle', until=0., dir=0., freeze_until=-1., refractory=0., hold=0.) for _ in range(2)]
    arm.frozen, arm.events, arm.active = np.zeros(2), [], False
    modes = []
    for i in range(20):
        arm.rate = np.array([0., 3. if i == 0 else 0.])  # positive pitch rate at t = 0
        cmd = arm.act(env, np.r_[0., 0., 3400.], i*DT, 0.)
        modes.append((arm.state[1]['mode'], round(cmd[0], 3)))
    assert modes[0] == ('counter', -.3), modes  # restoring direction = -sign(rate) on the alpha gimbal
    assert ('brake', .15) in modes and modes[-1][0] == 'idle' and arm.events == [0.], modes
    print('selftest ok')


def main():
    global QUICK
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all', 'calibrate', 'primitives'])
    p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.stage == 'selftest':
        return selftest()
    QUICK = a.quick
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    with pool(a.workers) as ex:
        cal = stage_calibrate(ex, out)
        levels = cal['levels']
        print(json.dumps({k: [round(x*1e3, 2) for x in v['levels']] for k, v in levels.items()}), flush=True)
        if a.stage == 'calibrate':
            return
        stage_primitives(ex, out, levels)
        if a.stage == 'primitives':
            return
        cases = train_cases(levels)
        base = stage_base_pid(ex, out, cases)
        gains = tuple(base['gains'])
        tune = stage_tune(ex, out, cases, gains, a.seed)
        stage_test(ex, out, test_cases(levels), gains, tune['chosen'])


if __name__ == '__main__':
    main()
