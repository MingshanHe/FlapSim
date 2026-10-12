"""Free-flight plant for wingbeat-synchronous control with IMU-only controllers.

Base: the v3 robust_control plant (7 g + 8 g, 5 us physics, 500 Hz PID whose
height loop keeps its range finder as fixed infrastructure, virtual IMU).
Added here, without editing any earlier file:

* a C++ phase-locked motor drive whose speed target is recomputed every 5 us
  substep from an estimated wingbeat phase (or from two 1 ms offsets),
* aerodynamic 1-cos wind gusts along the robot heading (blade-element wind),
* IMU-only observations for the new controller; the true crank angle is used
  only for logging phase-estimation error.
"""
import ctypes as ct
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import mujoco
import numpy as np
from flapsim.rl.native import pointer, D
from ..snn_pid_compare.common import DT
from ..robust_control.environment import ResearchEnv
from ..topology_hover.environment import case as make_case
from ..topology_hover.sensors import SensorBackend, SensorState
from ..neat_imu.environment import rotation_to_rpy
from ..wingbeat_feasibility.preliminary import WingbeatPhase, wrap

HERE = Path(__file__).resolve().parent
A_MAX = 300.  # rad/s speed-target modulation amplitude (P0: about +-16 % terminal voltage)
PHASE_OFFSET = math.radians(84.14)  # one-time gyro-to-crank calibration from P0-E4
SPLITS = dict(development=1_960_000_000, training=1_970_000_000, validation=1_980_000_000, test=1_990_000_000)


def library():
    sources = [HERE/'phase_step.cpp', HERE.parent/'topology_hover/sensor_step.cpp',
               HERE.parents[2]/'flapsim/rl/native_step.cpp']
    package = Path(mujoco.__file__).parent
    binary = next(package.glob('libmujoco.so.*'))
    key = hashlib.sha256(b''.join(p.read_bytes() for p in sources)+str(binary).encode()).hexdigest()[:20]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'flapsim'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'wingbeat-phase-{key}.so'
    if not target.exists():
        fd, temporary = tempfile.mkstemp(suffix='.so', dir=cache)
        os.close(fd)
        try:
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(sources[0]), '-I'+str(package/'include'),
                            str(binary), '-Wl,-rpath,'+str(package), '-o', temporary], check=True, capture_output=True)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ct.CDLL(str(target))
    lib.fs_advance_phase.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_double, ct.c_int,
                                     ct.c_void_p, ct.c_int, ct.c_double, ct.c_double, ct.c_double,
                                     D, D, D, D, ct.POINTER(SensorState), D]
    lib.fs_advance_phase.restype = None
    return lib


class PhaseBackend(SensorBackend):
    """A SensorBackend whose motor target follows ``self.drive`` inside each step call."""
    def advance(self, steps, command, joint_forces, body_forces, controls):
        c = self.fluid.config
        arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in (joint_forces, body_forces, controls)]
        self.phase_lib.fs_advance_phase(self.model._address, self.data._address, steps, ct.addressof(self.motor),
                                        float(command[0]), self.voltage_mode, ct.addressof(self.wings), len(self.wings),
                                        c.density, c.chord_drag, c.rotational_drag, pointer(self.wind),
                                        *(pointer(a) for a in arrays), ct.byref(self.sensor), pointer(self.drive))


def enable(backend):
    if type(backend) is not SensorBackend:
        raise TypeError('expected the topology_hover SensorBackend')
    backend.__class__ = PhaseBackend
    backend.phase_lib = library()
    backend.drive = np.zeros(8)
    return backend


def gusts(peak, onsets=(3., 5.5), duration=.3):
    """Fore-aft 1-cos gusts with alternating sign; peak in m/s."""
    return [dict(t=t, duration=duration, peak=peak*(1 if i % 2 == 0 else -1)) for i, t in enumerate(onsets)]


class Plant:
    """One worker's simulator. ``run(case, controller, gust_list)`` returns metrics."""
    def __init__(self, source):
        self.env = ResearchEnv(str(source), mode='pid', task='hover')
        self.backend = enable(self.env.sim.backend)
        air = self.env.air
        m = air.sim.model
        self.crank = int(m.joint('carrier_2_joint').qposadr[0])
        self.crank_dof = int(m.joint('carrier_2_joint').dofadr[0])
        self.unit = m.body('thrust_unit').id
        air.advance = self._advance

    # -- one 2 ms control tick, same bookkeeping as Airborne/InitializedAirborne.advance
    def _advance(self, command, p):
        air, sim, env = self.env.air, self.env.sim, self.env
        if p.get('motor_start_s', 0):
            command = command.copy()
            command[2] *= min(1., (air.steps+1)*DT/p['motor_start_s'])
        tau = p.get('action_tau', .02)
        air.action += (1. if tau == 0 else 1.-math.exp(-DT/tau))*(command-air.action)
        air.predicted_gimbal += (1.-math.exp(-DT/.02))*(air.action[:2]-air.predicted_gimbal)
        time, t = float(sim.data.time), air.steps*DT
        self.backend.wind[:] = self._wind(t)
        self._control(time, t)
        sim.set_gimbal(pitch=float(np.clip(air.action[0], -.3, .3)), roll=float(np.clip(air.action[1], -.3, .3)))
        sim.set_speed(float(air.action[2]))
        sim.step(round(DT/sim.model.opt.timestep))
        air.steps += 1

    def _wind(self, t):
        w = np.zeros(3)
        for g in self.gusts:
            if g['t'] <= t < g['t']+g['duration']:
                if 'forward' not in g:
                    yaw = rotation_to_rpy(self.env.air.env._state()[2])[2]
                    g['forward'] = np.array([math.cos(yaw), math.sin(yaw), 0.])
                w += .5*g['peak']*(1-math.cos(2*math.pi*(t-g['t'])/g['duration']))*g['forward']
        return w

    def observation(self, phase_inputs):
        r = self.env.last_reading
        x = [r[0]/.35, r[1]/.35, *(np.asarray(r[3:6])/10.), *(np.asarray(r[6:9])/9.81)]
        if phase_inputs:
            phi = self.estimator.crank(float(self.env.sim.data.time))
            x += [math.sin(phi), math.cos(phi), self.estimator.slow/10.]
        return np.asarray(x)

    def _control(self, time, t):
        drive = self.backend.drive
        drive[:] = 0.
        drive[7] = time+DT/2  # split between the two 1 ms offsets
        reading = self.env.last_reading
        if hasattr(self.estimator, 'update_reading'):  # learned estimators read the whole IMU vector
            self.estimator.update_reading(time, reading)
        else:
            self.estimator.update(time, float(reading[4]))
        true_crank = float(self.env.sim.data.qpos[self.crank])
        if t >= 1.5:
            self.phase_errors.append(wrap(self.estimator.crank(time)-true_crank))
        if self.trace is not None:
            self.trace.append([time, *reading[:9], true_crank, self.estimator.crank(time), self.estimator.omega,
                               *self.backend.wind[:2]])
        ctl = self.controller
        if ctl is None or (t < ctl.start and not ctl.prewarm):
            return
        out = ctl.act(self, time)  # networks run from t=0 so they can entrain, but drive only after start
        if t < ctl.start:
            return
        if ctl.mode == 'phasor':
            a, b = out
            norm = math.hypot(a, b)
            if norm > 1.:
                a, b = a/norm, b/norm
            # An unknown actuator lag shifts where the commanded phasor physically lands; controllers never see it.
            drive[0], drive[1], drive[2] = self.estimator.crank(time)-self.lag, self.estimator.omega, time
            drive[3], drive[4] = A_MAX*a, A_MAX*b
        else:
            drive[5], drive[6] = A_MAX*float(np.clip(out[0], -1, 1)), A_MAX*float(np.clip(out[1], -1, 1))
        self.activity.append(float(np.hypot(drive[3], drive[4])+.5*(abs(drive[5])+abs(drive[6]))))

    def run(self, c, controller=None, gust_list=(), settle=2., estimator=None, record=False):
        """``estimator`` replaces the PLL (default WingbeatPhase); ``record`` returns a per-tick IMU trace:
        time, roll, pitch, yaw, gyro xyz, accel xyz, true crank, estimated crank, estimated omega, wind xy."""
        env = self.env
        c = dict(c, task='hover')
        env.mode = 'pid'
        env.configure(c)
        self.controller, self.gusts = controller, [dict(g) for g in gust_list]
        self.lag = math.radians(c.get('phase_lag_deg', 0.))
        self.estimator = estimator if estimator is not None else WingbeatPhase(PHASE_OFFSET)
        self.phase_errors, self.activity = [], []
        self.trace = [] if record else None
        env.reset(seed=c['seed'])
        if controller is not None:
            controller.reset()
        m, d = env.sim.model, env.sim.data
        body = env.air.env.body
        rows, info = [], {}
        while True:
            _, _, terminated, truncated, info = env.step(np.zeros(3))
            _, _, rotation = env.air.env._state()
            mujoco.mj_subtreeVel(m, d)
            rows.append([env.air.steps*DT, *d.subtree_com[body], *d.subtree_linvel[body],
                         *rotation_to_rpy(rotation), float(rotation[2, 2])])
            if terminated or truncated:
                break
        rows = np.asarray(rows)
        out = dict(seed=c['seed'], seconds=float(c['seconds']), termination=info['episode']['termination'],
                   elapsed=float(rows[-1, 0]), height_rmse_m=info['episode']['height_rmse_m'],
                   return_sum=float(info['episode'].get('return_sum', env.total_reward)),
                   drive_activity=float(np.mean(self.activity)) if self.activity else 0.)
        if controller is not None and hasattr(controller, 'spikes'):
            out['spikes_per_s'] = controller.spikes/max(out['elapsed'], 1e-9)
        errors = np.asarray(self.phase_errors)
        if len(errors):
            out['phase_error_rms_deg'] = math.degrees(float(np.sqrt(np.mean(errors**2))))
        out.update(metrics(rows, self.gusts, settle))
        if record:
            out['trace'] = np.asarray(self.trace, dtype=np.float32)
        return out


def metrics(rows, gust_list, settle=2.):
    t = rows[:, 0]
    after = t >= settle
    keys = ('tilt_rms_deg', 'pitch_peak_deg', 'foreaft_max_m', 'height_peak_to_peak_m')
    if not after.any():
        return {k: float('nan') for k in keys}
    ref = int(np.argmax(after))
    forward = np.array([math.cos(rows[ref, 9]), math.sin(rows[ref, 9]), 0.])
    displacement = (rows[:, 1:4]-rows[ref, 1:4])@forward
    tilt = np.degrees(np.arccos(np.clip(rows[:, 10], -1, 1)))
    pitch = np.degrees(rows[:, 8])
    windows = [(g['t'], g['t']+1.) for g in gust_list] or [(settle, t[-1])]
    peaks = [float(np.max(np.abs(pitch[(t >= a) & (t < b)]))) for a, b in windows if np.any((t >= a) & (t < b))]
    return dict(tilt_rms_deg=float(np.sqrt(np.mean(tilt[after]**2))),
                pitch_peak_deg=float(np.mean(peaks)) if peaks else float('nan'),
                foreaft_max_m=float(np.max(np.abs(displacement[after]))),
                height_peak_to_peak_m=float(np.ptp(rows[after, 3])))


def cases(split, count, seconds=8., offset=0):
    base = SPLITS[split]+offset
    return [make_case(base+i, seconds, 'nominal', i % 8) for i in range(count)]


def with_lags(case_list, lag_range, seed):
    """Unknown actuator lag per case, uniform in +-lag_range degrees, deterministic in the case seed."""
    if not lag_range:
        return case_list
    out = []
    for c in case_list:
        rng = np.random.default_rng(np.random.SeedSequence([c['seed'], seed, 7717]))
        out.append(dict(c, phase_lag_deg=float(rng.uniform(-lag_range, lag_range))))
    return out


class Controller:
    """Interface: ``mode`` is 'phasor' (a, b around the PLL phase) or 'direct' (two 1 ms offsets).

    ``prewarm`` controllers are stepped from t=0 (to entrain) but drive only after ``start``."""
    mode, start, prewarm = 'phasor', 1., False

    def reset(self):
        pass

    def act(self, plant, time):
        raise NotImplementedError


class HandLaw(Controller):
    """B2+: PLL phase, notch-filtered pitch rate, PD on pitch; optional high-pass to drop steady output."""
    def __init__(self, k_theta=5., k_q=.2, highpass_s=None, psi=math.radians(270.), limit=1.):
        self.k_theta, self.k_q, self.highpass_s, self.psi = k_theta, k_q, highpass_s, psi
        self.limit = limit  # |u| bound in units of A_MAX; the plant still clips the phasor norm at 1

    def reset(self):
        self.mean = 0.

    def act(self, plant, time):
        r = plant.env.last_reading
        u = -(self.k_theta*r[1]+self.k_q*plant.estimator.slow)
        if self.highpass_s:
            self.mean += (1-math.exp(-DT/self.highpass_s))*(u-self.mean)
            u -= self.mean
        u = max(-self.limit, min(self.limit, u))
        return u*math.cos(self.psi), u*math.sin(self.psi)
