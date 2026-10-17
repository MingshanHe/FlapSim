"""One causal observation and command per 2 ms, unchanged physical substeps."""
import math
import time
import numpy as np
from ..topology_hover.environment import HoverEnv, case, NAMES
from ..pid_airborne.audit import clearance
from .common import DT

LOW = np.array([-.3, -.3, 3115.224079])
HIGH = np.array([.3, .3, 3807.496097])
CENTER = (LOW + HIGH)/2
HALF = (HIGH - LOW)/2


def direct_command(action):
    return CENTER + HALF*np.clip(action, -1, 1)


def residual_command(action, base):
    action = np.clip(action, -1, 1)
    base = np.clip(base, LOW, HIGH)
    return base + action*np.where(action >= 0, HIGH-base, base-LOW)


class TeacherPID:
    """Independent causal teacher; querying labels cannot advance the plant."""
    def __init__(self, parameters, initial_rpy):
        self.p = parameters
        self.last_rpy = np.asarray(initial_rpy).copy()
        self.rate = np.zeros(2)
        self.integral = np.zeros(2)
        self.z_integral = 0.
        self.raw = None

    def step(self, reading):
        p = self.p
        rpy = reading[:3]
        delta = rpy[:2]-self.last_rpy[:2]
        derivative = np.arctan2(np.sin(delta), np.cos(delta))/DT
        self.last_rpy = rpy.copy()
        self.rate += (1-np.exp(-DT/p['derivative_tau']))*(derivative-self.rate)
        error = -rpy[:2]
        candidate = np.clip(self.integral+error*DT, -.3, .3)
        effort = np.array(p['kp'])*error-np.array(p['kd'])*self.rate+np.array(p['ki'])*candidate
        angles = np.array(p['trim'])+effort[::-1]
        self.integral = np.where(np.abs(angles[::-1]) < .3, candidate, self.integral)
        z, vz = reading[9:11]
        self.z_integral = float(np.clip(self.z_integral-z*DT, -.5, .5))
        kp, kd, ki = p['height_gains']
        self.raw = np.r_[angles, p['motor']-kp*z-kd*vz+ki*self.z_integral]
        return np.clip(self.raw, LOW, HIGH)


class ControlEnv(HoverEnv):
    def __init__(self, source, mode='direct', **kwargs):
        if mode not in ('direct', 'residual', 'pid'):
            raise ValueError(mode)
        super().__init__(source=source, mode=mode, sensor_mode='height', **kwargs)

    def reset(self, **kwargs):
        training = self.training
        if training:
            rng = np.random.default_rng(np.random.SeedSequence([self.run_seed, self.rank, self.episode_index, 8163]))
            seed = int(rng.integers(1_200_000_000, 1_290_000_000))
            self.case = case(seed, self.horizon, 'nominal' if self.episode_index % 3 == 0 else 'train')
            self.episode_index += 1
            kwargs['seed'] = seed
            self.training = False
        try:
            observation, info = super().reset(**kwargs)
        finally:
            self.training = training
        self.teacher = TeacherPID(self.p, self.air.last_rpy)
        self._teacher_step = -1
        self._teacher_command = None
        self.previous_total_action = (self.command-CENTER)/HALF
        return observation, info

    def teacher_command(self):
        if self._teacher_step != self.steps:
            self._teacher_command = self.teacher.step(self.last_reading)
            self._teacher_step = self.steps
        return self._teacher_command.copy()

    def step(self, action):
        control_start = time.perf_counter()
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Expected three finite actions')
        action = np.clip(action, -1, 1)
        if self.mode == 'pid':
            command = self.teacher_command()
        elif self.mode == 'residual':
            command = residual_command(action, self.teacher_command())
        else:
            command = direct_command(action)
        self.command = np.clip(command, LOW, HIGH)
        total_action = (self.command-CENTER)/HALF
        command = self.command.copy()
        command[2] *= self.motor_gain
        controller_seconds = time.perf_counter()-control_start
        t = self.air.steps*DT
        self.sim.external_body_forces[:] = 0.
        if .6 <= t % 2. < .8:
            self.sim.external_body_forces[self.air.env.body, :3] = self.pulse_sign*np.array([1, 1, 0])*self.case.get('pulse_force', 0.)
            self.sim.external_body_forces[self.air.env.body, 3:5] = self.pulse_sign[:2]*self.case.get('pulse_torque', 0.)
        self.air.advance(command, self.p)
        pos, vel, rotation = self.air.env._state()
        t = self.air.steps*DT
        self.height.step(t, pos[2], rotation[2, 2], np.array(self.frontend.sensor.attitude))
        self.reading_queue.append(self.reading())
        while len(self.reading_queue) > self.case.get('delay_steps', 0)+1:
            self.reading_queue.popleft()
        self.last_reading = self.reading_queue[0].copy()
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        z = pos[2]-self.air.start[2]
        low = clearance(self.sim.model, self.sim.data)
        self.minimum_clearance = min(low, self.minimum_clearance)
        self.samples.append([t, tilt, z, low, float(np.linalg.norm(pos[:2]-self.air.start[:2]))])
        reward = DT*(2.+math.exp(-(tilt/.15)**2)+math.exp(-(z/.12)**2))
        reward -= DT*.02*float(np.sum((total_action-self.previous_total_action)**2))
        reason = 'time_limit'
        if not np.isfinite(self.sim.data.qpos).all() or any(w.number for w in self.sim.data.warning):
            reason = 'numerical'
        elif low <= .03:
            reason = 'clearance'
        elif pos[2] < .12:
            reason = 'low_height'
        elif abs(z) > 1.:
            reason = 'height_escape'
        elif tilt > 1.4:
            reason = 'tilt'
        terminated = reason != 'time_limit'
        if terminated:
            reward -= 1.
        self.previous_total_action = total_action.copy()
        self.steps += 1
        self.total_reward += reward
        truncated = t >= self.case['seconds']-1e-9 and not terminated
        info = dict(command=self.command.copy(), applied_command=self.air.action.copy(),
                    controller_seconds=controller_seconds)
        if self.record:
            self.trace.append([t, *pos, *vel, tilt, *self.command, *self.last_reading, *action])
        if terminated or truncated:
            info['episode'] = self.metrics(reason)
        return self.observation(), float(reward), terminated, truncated, info


def scenarios(split, count, seconds=20., condition='nominal'):
    # Split labels occupy disjoint ranges; training never reads final cases.
    offsets = dict(demo=1_100_000_000, offline=1_110_000_000, validation=1_120_000_000,
                   long_validation=1_130_000_000, stress_validation=1_140_000_000,
                   dagger=1_150_000_000, training=1_200_000_000, final=1_300_000_000)
    base = offsets[split] + (100_000 if condition == 'stress' else 0) + round(seconds)*1000
    result = [case(base+i, seconds, condition, i % 8) for i in range(count)]
    if split == 'demo':
        for i, c in enumerate(result):
            if i % 2:
                c.update(degrees=2., motor_uncertainty=.005, acceleration_bias=.003,
                         pulse_force=.0015, pulse_torque=.000002)
    return result
