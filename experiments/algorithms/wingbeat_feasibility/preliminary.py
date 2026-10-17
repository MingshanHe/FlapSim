"""P0 preliminary study: does wingbeat-synchronous drive timing work in free flight?

Plant: the v3 free-flight plant (7 g + 8 g, 5 us physics, 500 Hz PID, IMU and
height sensing). The only change is inside each 2 ms control step: the motor
speed target is updated every 0.25 ms as

    target = PID_motor + A * sin(crank_angle + psi)

which, through the motor's proportional speed loop, is a phase-locked terminal
voltage modulation (split-cycle). With A = 0 the plant is the original one.

E1  open loop: authority of a 0.1 s split-cycle burst vs the v3 gimbal residual
E2  closed loop: PID (B1) vs PID + hand-designed split-cycle law (B2)
E3  timing: phase error, phase jitter and once-per-wingbeat updates

Seeds start at 1_950_000_000 (unused by v2.1/v3/v4). Simulation only.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from ..snn_pid_compare.common import DT, prepare_source
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..neat_imu.environment import rotation_to_rpy

BLOCK = 50
SEED0 = 1_950_000_000
_ENV = None


class VelocitySensor:
    """Same assumption as the v3 velocity task: 100 ms prefilter, 50 Hz, noise, 20 ms delay, 100 ms postfilter."""
    def __init__(self, seed, noise=.01, delay=.02):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 9311]))
        self.noise, self.delay = noise, delay
        self.filtered, self.value = np.zeros(3), np.zeros(2)
        self.pending, self.count = deque(), 0

    def update(self, t, velocity):
        self.filtered += (1-math.exp(-DT/.1))*(velocity-self.filtered)
        if self.count % 10 == 0:
            self.pending.append((t+self.delay, self.filtered[:2]+self.rng.normal(0., self.noise, 2)))
        while self.pending and self.pending[0][0] <= t+1e-9:
            self.value += (1-math.exp(-.02/.1))*(self.pending.popleft()[1]-self.value)
        self.count += 1


class WingbeatPhase:
    """Causal IMU-only wingbeat phase from the pitch gyro (500 Hz samples).

    Sliding least-squares phase detector (fundamental + 2nd harmonic, 25
    samples = 50 ms) inside a 2nd-order PLL of about 1.5 Hz bandwidth. The
    constant gyro-to-crank offset is a one-time calibration, like a bench
    calibration against an encoder. ``slow`` is the gyro with the fitted
    wingbeat oscillation removed (an adaptive notch).
    """
    def __init__(self, offset=0., hz=25., window=25, kp=13., ki=89.):
        self.theta, self.omega, self.offset = 0., 2*math.pi*hz, offset
        self.buffer = deque(maxlen=window)
        self.kp, self.ki, self.t, self.slow = kp, ki, None, 0.

    def update(self, t, y):
        if self.t is not None:
            self.theta += self.omega*(t-self.t)
        self.t = t
        self.buffer.append((self.theta, y))
        self.slow = y
        if len(self.buffer) == self.buffer.maxlen:
            theta = np.array([b[0] for b in self.buffer])
            X = np.c_[np.cos(theta), np.sin(theta), np.cos(2*theta), np.sin(2*theta), np.ones_like(theta)]
            coef, *_ = np.linalg.lstsq(X, np.array([b[1] for b in self.buffer]), rcond=None)
            error = math.atan2(coef[1], coef[0])  # y ~ R cos(theta - error)
            self.slow = y-float(X[-1, :4]@coef[:4])
            self.theta -= self.kp*error*DT
            self.omega -= self.ki*error*DT

    def crank(self, t):
        return self.theta+self.omega*(t-self.t)-self.offset


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def com_velocity(m, d, body):
    mujoco.mj_subtreeVel(m, d)
    return d.subtree_linvel[body].copy()


def install(env):
    """Wrap Airborne.advance: identical bookkeeping, motor target refreshed every 0.25 ms."""
    air, sim = env.air, env.air.sim
    m = sim.model
    body = air.env.body
    crank = int(m.joint('carrier_2_joint').qposadr[0])
    dof = int(m.joint('motor_joint').dofadr[0])
    c = sim.actuator.config
    blocks = round(DT/m.opt.timestep)//BLOCK
    env.original_advance = air.advance

    def advance(command, p):
        st = env.drive
        if st.get('native'):
            return env.original_advance(command, p)
        if p.get('motor_start_s', 0):
            command = command.copy()
            command[2] *= min(1., (air.steps+1)*DT/p['motor_start_s'])
        tau = p.get('action_tau', .02)
        air.action += (1. if tau == 0 else 1.-math.exp(-DT/tau))*(command-air.action)
        air.predicted_gimbal += (1.-math.exp(-DT/.02))*(air.action[:2]-air.predicted_gimbal)
        t = air.steps*DT
        estimator = st.get('estimator')
        if estimator is not None:
            estimator.update(float(sim.data.time), float(env.last_reading[4]))
            if t >= 1.5:
                st['phase_errors'].append(wrap(estimator.crank(sim.data.time)-sim.data.qpos[crank]))
        imu_phase = estimator is not None and st.get('phase_source') == 'imu'
        gimbal = air.action[:2].copy()
        window = st.get('window')
        inside = bool(window) and window[0] <= t < window[1]
        if inside:
            if st.get('hold') is None:
                st['hold'] = gimbal.copy()
            gimbal = st['hold']+np.asarray(st.get('gimbal_step', [0., 0.]))
            amplitude, phase = st.get('burst', (0., 0.))
        else:
            amplitude, phase = st['law'](env, t)
        gimbal = np.clip(gimbal, -.3, .3)
        sim.set_gimbal(pitch=float(gimbal[0]), roll=float(gimbal[1]))
        sim.external_body_forces[body] += st['disturbance'](env, t)
        e = st['electric']
        for _ in range(blocks):
            angle = estimator.crank(sim.data.time) if imu_phase else sim.data.qpos[crank]
            target = float(air.action[2]+amplitude*math.sin(angle+phase))
            v = float(sim.data.qvel[dof])
            volt = c.resistance*(c.speed_gain*(target-v)/c.torque_constant+c.no_load_current*math.tanh(v))+c.back_emf*v
            saturated = abs(volt) >= c.max_voltage
            volt = max(-c.max_voltage, min(c.max_voltage, volt))
            current = max(-c.max_current, min(c.max_current, (volt-c.back_emf*v)/c.resistance))
            key = 'window' if inside else 'all'
            e[key]['n'] += 1
            e[key]['power'] += volt*current
            e[key]['peak_current'] = max(e[key]['peak_current'], abs(current))
            e[key]['saturated'] += int(saturated)
            sim.set_speed(target)
            sim.step(BLOCK)
        air.steps += 1

    air.advance = advance


def worker_init(source):
    global _ENV
    _ENV = ResearchEnv(source, mode='pid', task='hover')
    install(_ENV)


def heading(rotation):
    yaw = rotation_to_rpy(rotation)[2]
    return np.array([math.cos(yaw), math.sin(yaw), 0.]), np.array([-math.sin(yaw), math.cos(yaw), 0.])


def make_disturbance(pulses):
    """Pulses act along the robot's heading at pulse onset: fore-aft force and pitch-axis torque."""
    frames = {}
    def disturbance(env, t):
        out = np.zeros(6)
        for i, pulse in enumerate(pulses):
            if pulse['t'] <= t < pulse['t']+pulse['duration']:
                if i not in frames:
                    frames[i] = heading(env.air.env._state()[2])
                forward, lateral = frames[i]
                out[:3] += pulse.get('force', 0.)*forward
                out[3:] += pulse.get('torque', 0.)*lateral
        return out
    return disturbance


def make_law(spec):
    law = spec.get('law')
    if not law:
        return lambda env, t: (0., 0.)
    rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], 4421]))
    state = dict(cycle=None, latched=(0., 0.))
    k_theta, k_q, k_v = law['gains']
    def control(env, t):
        _, _, rotation = env.air.env._state()
        sensor = env.drive['sensor']
        sensor.update(t, com_velocity(env.sim.model, env.sim.data, env.air.env.body))
        if t < law.get('start', 1.):
            return 0., 0.
        r = env.last_reading
        theta, q, yaw = r[1], r[4], r[2]
        if law.get('rate') == 'notch':
            q = env.drive['estimator'].slow
        forward = math.cos(yaw)*sensor.value[0]+math.sin(yaw)*sensor.value[1]
        u = -(k_theta*theta+k_q*q)-k_v*law['velocity_sign']*forward
        u = max(-1., min(1., u))
        amplitude = abs(u)*law['amplitude']
        phase = law['psi']+(0. if u >= 0 else math.pi)+math.radians(law.get('phase_error_deg', 0.))
        if law.get('jitter_deg'):
            phase += math.radians(law['jitter_deg'])*rng.standard_normal()
        if law.get('update') == 'wingbeat':
            angle = (env.drive['estimator'].crank(env.sim.data.time) if law.get('phase_source') == 'imu'
                     else env.sim.data.joint('carrier_2_joint').qpos[0])
            cycle = int(angle//(2*math.pi))
            if cycle != state['cycle']:
                state['cycle'], state['latched'] = cycle, (amplitude, phase)
            return state['latched']
        return amplitude, phase
    return control


def run(spec):
    env = _ENV
    c = make_case(spec['seed'], spec['seconds'], 'nominal', spec['phase'])
    c['task'] = 'hover'
    env.mode = 'pid'
    env.configure(c)
    electric = {k: dict(n=0, power=0., peak_current=0., saturated=0) for k in ('all', 'window')}
    env.drive = dict(native=spec.get('native', False), window=spec.get('window'), hold=None,
                     gimbal_step=spec.get('gimbal_step', [0., 0.]), burst=spec.get('burst', (0., 0.)),
                     law=make_law(spec), disturbance=make_disturbance(spec.get('pulses', [])),
                     sensor=VelocitySensor(spec['seed']), electric=electric,
                     estimator=WingbeatPhase(spec['phase_offset']) if 'phase_offset' in spec else None,
                     phase_source=(spec.get('law') or {}).get('phase_source', 'crank'), phase_errors=[])
    env.reset(seed=c['seed'])
    m, d = env.sim.model, env.sim.data
    body, unit = env.air.env.body, m.body('thrust_unit').id
    rows, info = [], {}
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        _, _, rotation = env.air.env._state()
        tu = d.xmat[unit].reshape(3, 3)
        rows.append([env.air.steps*DT, *d.subtree_com[body], *com_velocity(m, d, body),
                     *rotation_to_rpy(rotation), *(rotation.T@d.cvel[body][:3]),
                     *rotation_to_rpy(tu), *(tu.T@d.cvel[unit][:3]), float(rotation[2, 2])])
        if terminated or truncated:
            break
    rows = np.asarray(rows)
    result = dict(spec={k: v for k, v in spec.items() if k != 'pulses'}, pulses=spec.get('pulses', []),
                  termination=info['episode']['termination'], elapsed=float(rows[-1, 0]),
                  height_rmse_m=info['episode']['height_rmse_m'],
                  electric={k: dict(mean_power_W=v['power']/max(v['n'], 1), peak_current_A=v['peak_current'],
                                    saturated_fraction=v['saturated']/max(v['n'], 1))
                            for k, v in electric.items()})
    errors = np.asarray(env.drive['phase_errors'])
    if len(errors):
        mean = math.atan2(np.mean(np.sin(errors)), np.mean(np.cos(errors)))
        result['phase_error_deg'] = dict(circular_mean=math.degrees(mean),
                                         rms=math.degrees(float(np.sqrt(np.mean(errors**2)))),
                                         p95_abs=math.degrees(float(np.quantile(np.abs(errors), .95))))
    if spec.get('trace'):
        keep = (rows[:, 0] >= spec['trace'][0]-.004) & (rows[:, 0] <= spec['trace'][1]+.004)
        result['trace'] = rows[keep].tolist()
        result['qpos_end'] = d.qpos.tolist()
        return result
    result.update(metrics(rows, spec))
    return result


def metrics(rows, spec, settle=2.):
    t = rows[:, 0]
    after = t >= settle
    if not after.any():
        return {k: float('nan') for k in ('tilt_rms_deg', 'pitch_peak_deg', 'foreaft_max_m', 'foreaft_final_m',
                                          'horizontal_final_m', 'foreaft_speed_rms_m_s')}
    ref = int(np.argmax(after))
    forward, _ = heading_from_yaw(rows[ref, 9])
    displacement = (rows[:, 1:4]-rows[ref, 1:4])@forward
    tilt = np.degrees(np.arccos(np.clip(rows[:, 19], -1, 1)))
    pitch = np.degrees(rows[:, 8])
    windows = [(p['t'], p['t']+1.) for p in spec.get('pulses', [])] or [(settle, t[-1])]
    peaks = [float(np.max(np.abs(pitch[(t >= a) & (t < b)]))) for a, b in windows if np.any((t >= a) & (t < b))]
    velocity = rows[:, 4:7]@forward
    return dict(tilt_rms_deg=float(np.sqrt(np.mean(tilt[after]**2))),
                pitch_peak_deg=float(np.mean(peaks)) if peaks else float('nan'),
                foreaft_max_m=float(np.max(np.abs(displacement[after]))),
                foreaft_final_m=float(abs(displacement[-1])),
                horizontal_final_m=float(np.linalg.norm((rows[-1, 1:3]-rows[ref, 1:3]))),
                foreaft_speed_rms_m_s=float(np.sqrt(np.mean(velocity[after]**2))))


def heading_from_yaw(yaw):
    return np.array([math.cos(yaw), math.sin(yaw), 0.]), None


# ----------------------------------------------------------------------------- stages

def pool(source, workers):
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def stage_e1(executor, seeds=4, window=(2., 2.1)):
    specs = []
    for i in range(seeds):
        base = dict(seed=SEED0+i, phase=i % 8, seconds=window[1]+.01, window=list(window), trace=list(window))
        specs.append(dict(base, label='baseline'))
        specs.append(dict(base, label='gimbal_pitch_0.035', gimbal_step=[.035, 0.]))
        for amplitude, k in itertools.product((150., 300., 450.), range(8)):
            specs.append(dict(base, label=f'split_A{amplitude:g}_psi{45*k}', burst=(amplitude, k*math.pi/4)))
    # Equivalence: block-wise wrapper with A=0 must reproduce the original advance.
    check = dict(seed=SEED0+99, phase=3, seconds=.5, trace=[.4, .45])
    specs += [dict(check, label='equivalence_wrapper'), dict(check, label='equivalence_native', native=True)]
    rows = list(executor.map(run, specs))
    by = {(r['spec']['label'], r['spec']['seed']): r for r in rows}
    equivalence = float(np.max(np.abs(np.array(by[('equivalence_wrapper', SEED0+99)]['qpos_end']) -
                                      np.array(by[('equivalence_native', SEED0+99)]['qpos_end']))))
    labels = sorted({r['spec']['label'] for r in rows if not r['spec']['label'].startswith('equivalence')})
    def at(trace, time):
        trace = np.asarray(trace)
        return trace[int(np.argmin(np.abs(trace[:, 0]-time)))]
    summary = {}
    for label in labels:
        deltas = []
        for i in range(seeds):
            run_, base = by[(label, SEED0+i)], by[('baseline', SEED0+i)]
            a, b = at(run_['trace'], window[1]), at(base['trace'], window[1])
            a0 = at(run_['trace'], window[0])
            forward = np.array([math.cos(a0[9]), math.sin(a0[9]), 0.])
            deltas.append(dict(body_pitch_rate=a[11]-b[11], body_pitch_deg=math.degrees(a[8]-b[8]),
                               unit_pitch_rate=a[17]-b[17], forward_velocity=float((a[4:7]-b[4:7])@forward),
                               vertical_velocity=a[6]-b[6],
                               peak_current_A=run_['electric']['window']['peak_current_A'],
                               mean_power_W=run_['electric']['window']['mean_power_W'],
                               saturated_fraction=run_['electric']['window']['saturated_fraction']))
        summary[label] = {k: float(np.mean([x[k] for x in deltas])) for k in deltas[0]}
        summary[label]['forward_velocity_sd'] = float(np.std([x['forward_velocity'] for x in deltas]))
    return dict(window_s=list(window), seeds=seeds, equivalence_max_abs_qpos=equivalence, summary=summary,
                note='deltas are paired run-minus-baseline at window end; gimbal held at its t0 value in all runs')


def pick_phase(e1, amplitude):
    """Phase giving the largest positive body pitch-rate change, and the sign of its forward-velocity effect."""
    candidates = {k: v for k, v in e1['summary'].items() if k.startswith(f'split_A{amplitude:g}_')}
    best = max(candidates, key=lambda k: candidates[k]['body_pitch_rate'])
    psi = math.radians(float(best.split('psi')[1]))
    return psi, float(np.sign(candidates[best]['forward_velocity']) or 1.), best


def episode_specs(seed_offset, families, controllers, phases=8, seconds=8.):
    specs = []
    for family, pulses in families.items():
        for i in range(phases):
            for name, law in controllers.items():
                specs.append(dict(seed=SEED0+seed_offset+i, phase=i, seconds=seconds, family=family,
                                  controller=name, law=law, pulses=pulses))
    return specs


def gust(force_mN):
    return [dict(t=3., duration=.15, force=force_mN*1e-3), dict(t=5.5, duration=.15, force=-force_mN*1e-3)]


def e2_families():
    return {'nominal': [], **{f'gust_{g}mN': gust(g) for g in (6., 20., 40., 80.)},
            **{f'torque_{tq}uNm': [dict(t=3., duration=.05, torque=tq*1e-6)] for tq in (30., 100., 300.)}}


def summarize(rows, keys=('tilt_rms_deg', 'pitch_peak_deg', 'foreaft_max_m', 'foreaft_final_m',
                          'foreaft_speed_rms_m_s', 'height_rmse_m')):
    groups = {}
    for r in rows:
        groups.setdefault((r['spec']['family'], r['spec']['controller']), []).append(r)
    out = {}
    for (family, controller), items in sorted(groups.items()):
        flown = [r for r in items if r['termination'] == 'time_limit']
        entry = dict(episodes=len(items), completed=len(flown),
                     terminations=sorted(r['termination'] for r in items if r['termination'] != 'time_limit'))
        for k in keys:  # performance over completed episodes only; failures are counted above
            entry[k] = float(np.mean([r[k] for r in flown])) if flown else None
        entry['mean_power_W'] = float(np.mean([r['electric']['all']['mean_power_W'] for r in items]))
        entry['peak_current_A'] = float(np.max([r['electric']['all']['peak_current_A'] for r in items]))
        out.setdefault(family, {})[controller] = entry
    return out


def paired(rows, a, b, key):
    """Per-case a-b on the same family/seed, only where both flew the full episode.

    Failures are counted separately so that an early crash (short, hence small
    displacement) can never look like better control."""
    index = {(r['spec']['family'], r['spec']['seed'], r['spec']['controller']): r for r in rows}
    out = {}
    for family in sorted({r['spec']['family'] for r in rows}):
        diffs, failed = [], {a: 0, b: 0}
        for (f, s, c), r in index.items():
            if f != family or c != a or (family, s, b) not in index:
                continue
            other = index[(family, s, b)]
            failed[a] += r['termination'] != 'time_limit'
            failed[b] += other['termination'] != 'time_limit'
            if r['termination'] == other['termination'] == 'time_limit':
                diffs.append(r[key]-other[key])
        out[family] = dict(mean=float(np.mean(diffs)) if diffs else None,
                           better=int(np.sum(np.array(diffs) < 0)), n=len(diffs), failures=failed)
    return out


def stage_e2(executor, e1, amplitude):
    psi, velocity_sign, chosen = pick_phase(e1, amplitude)
    base_law = dict(amplitude=amplitude, psi=psi, velocity_sign=velocity_sign, start=1.)
    grid = list(itertools.product((0., 2., 5.), (0., .2, .5), (0., 5., 15.)))
    tune = episode_specs(500, {'tune_gust40': gust(40.)},
                         {f'g{i}': dict(base_law, gains=list(g)) for i, g in enumerate(grid)}, phases=4)
    tuned = list(executor.map(run, tune))
    reference = {r['spec']['seed']: r for r in tuned if r['spec']['controller'] == 'g0'}  # zero gains = PID
    scores = {}
    for i, g in enumerate(grid):
        items = [r for r in tuned if r['spec']['controller'] == f'g{i}']
        if any(r['termination'] != 'time_limit' for r in items):
            continue
        scores[i] = float(np.mean([r['foreaft_max_m']/reference[r['spec']['seed']]['foreaft_max_m'] +
                                   r['pitch_peak_deg']/reference[r['spec']['seed']]['pitch_peak_deg'] for r in items]))
    best = min(scores, key=scores.get)
    law = dict(base_law, gains=list(grid[best]))
    rows = list(executor.map(run, episode_specs(1000, e2_families(), {'B1_pid': None, 'B2_split': law})))
    return dict(chosen_phase=chosen, psi_rad=psi, velocity_sign=velocity_sign, amplitude=amplitude,
                tuning=dict(grid=grid, score_relative_to_pid=scores, best=list(grid[best]), cases=4,
                            family='gust 40 mN', seeds=[SEED0+500+i for i in range(4)]),
                law=law, summary=summarize(rows),
                paired_B2_minus_B1={k: paired(rows, 'B2_split', 'B1_pid', k)
                                    for k in ('pitch_peak_deg', 'foreaft_max_m', 'tilt_rms_deg')},
                episodes=[{k: v for k, v in r.items() if k != 'trace'} for r in rows])


def stage_e3(executor, e2):
    law = e2['law']
    variants = {'B1_pid': None, 'B2_tick': law, 'B2_wingbeat': dict(law, update='wingbeat'),
                'B2_jitter30': dict(law, jitter_deg=30.)}
    variants.update({f'B2_err{e}': dict(law, phase_error_deg=float(e)) for e in (30, 60, 90, 180)})
    rows = list(executor.map(run, episode_specs(2000, {'gust_40mN': gust(40.)}, variants)))
    return dict(summary=summarize(rows),
                paired_minus_tick={name: {k: paired(rows, name, 'B2_tick', k)
                                          for k in ('foreaft_max_m', 'pitch_peak_deg')}
                                   for name in variants if name != 'B2_tick'},
                episodes=[{k: v for k, v in r.items() if k != 'trace'} for r in rows])


def stage_e4(executor, e2):
    """IMU-only phase: calibrate the constant gyro-to-crank offset once, then rerun E2's exact cases."""
    calibration = list(executor.map(run, [dict(seed=SEED0+4000+i, phase=i, seconds=4., family='calibration',
                                               controller='B1_pid', law=None, pulses=[], phase_offset=0.)
                                          for i in range(3)]))
    means = np.radians([r['phase_error_deg']['circular_mean'] for r in calibration])
    offset = math.atan2(np.mean(np.sin(means)), np.mean(np.cos(means)))
    law = dict(e2['law'], phase_source='imu')
    variants = {'B2_imu': law, 'B2_imu_notch': dict(law, rate='notch')}
    specs = [dict(s, phase_offset=offset) for s in episode_specs(1000, e2_families(), variants)]
    rows = list(executor.map(run, specs))
    archived = e2['episodes']  # B1_pid and crank-phase B2_split on the same seeds, deterministic
    everything = archived+rows
    keys = ('pitch_peak_deg', 'tilt_rms_deg', 'foreaft_max_m', 'height_rmse_m')
    tracking = {}
    for name in variants:
        stats = [r['phase_error_deg'] for r in rows if r['spec']['controller'] == name and 'phase_error_deg' in r]
        tracking[name] = dict(episodes=len(stats), rms_deg_mean=float(np.mean([s['rms'] for s in stats])),
                              p95_abs_deg_max=float(np.max([s['p95_abs'] for s in stats])),
                              circular_mean_deg=float(np.mean([s['circular_mean'] for s in stats])))
    return dict(offset_deg=math.degrees(offset),
                calibration=[dict(seed=r['spec']['seed'], **r['phase_error_deg']) for r in calibration],
                closed_loop_phase_tracking=tracking, law=law,
                summary=summarize(everything),
                paired={f'{a}__minus__{b}': {k: paired(everything, a, b, k) for k in keys}
                        for a, b in [('B2_imu', 'B2_split'), ('B2_imu', 'B1_pid'),
                                     ('B2_imu_notch', 'B2_imu'), ('B2_imu_notch', 'B1_pid')]},
                episodes=[{k: v for k, v in r.items() if k != 'trace'} for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['e1', 'e2', 'e3', 'e4'])
    p.add_argument('--output', type=Path, required=True, help='directory for p0_*.json')
    p.add_argument('--scratch', type=Path, required=True, help='prepared plant source (outside the repo)')
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--amplitude', type=float, default=300.)
    p.add_argument('--e1-seeds', type=int, default=4, help='e1: number of paired free-flight cases')
    a = p.parse_args()
    source = a.scratch/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    a.output.mkdir(parents=True, exist_ok=True)
    with pool(source, a.workers) as executor:
        if a.stage == 'e1':
            result = stage_e1(executor, seeds=a.e1_seeds)
        elif a.stage == 'e2':
            result = stage_e2(executor, json.loads((a.output/'p0_e1.json').read_text()), a.amplitude)
        elif a.stage == 'e3':
            result = stage_e3(executor, json.loads((a.output/'p0_e2.json').read_text()))
        else:
            result = stage_e4(executor, json.loads((a.output/'p0_e2.json').read_text()))
    (a.output/f'p0_{a.stage}.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=1)[:12000])


if __name__ == '__main__':
    main()
