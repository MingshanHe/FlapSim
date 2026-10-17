"""Original 500 Hz Python control and 5 us plant, with bounded PID residuals.

Hover retains the exact 11 observation meanings of v2.1. Velocity is a separate
16-input task and explicitly assumes a noisy, delayed horizontal velocimeter.
"""
from collections import deque
import math
import time
import numpy as np
from gymnasium import spaces
from ..snn_pid_compare.environment import ControlEnv, TeacherPID, LOW, HIGH, CENTER, HALF
from ..snn_pid_compare.common import DT
from ..topology_hover.environment import case

RESIDUAL_SCALE = np.array([.035, .035, 100.])
INPUTS = dict(hover=11, velocity=16)
VELOCITY_KP = np.array([.02, 1.])
VELOCITY_KI = np.array([.01, .3])
# Development identification gives roughly 0.14 m/s per rad pitch reference
# and -1.06 m/s per rad roll reference, not a quadrotor's acceleration map.
VELOCITY_FEEDFORWARD = np.array([7., 1.])  # body x -> pitch; body y -> -roll


def scenarios(split, count, seconds=20., condition='nominal', task='hover'):
    # New independent ranges; old validation/test scenarios are not recycled.
    offsets = dict(development=1_410_000_000, validation=1_420_000_000,
                   training=1_500_000_000, final=1_600_000_000)
    base = offsets[split] + (2_000_000 if task == 'velocity' else 0)
    base += (100_000 if condition == 'stress' else 200_000 if condition == 'ood' else 0)
    base += round(seconds)*1000
    result = []
    for i in range(count):
        c = case(base+i, seconds, condition, i % 8)
        c['task'] = task
        if task == 'velocity':
            # Four axial, two vertical and two diagonal commanded motions.
            vectors = [[.04,0,0],[-.04,0,0],[0,.04,0],[0,-.04,0],
                       [0,0,.06],[0,0,-.06],[.028,.028,.04],[-.028,-.028,-.04]]
            c['velocity_command'] = vectors[(3*i+i//8) % 8]
            c.update(velocity_noise=.01, velocity_delay=.02,
                     velocity_bias=.01 if condition != 'nominal' else 0.)
        result.append(c)
    return result


def training_case(seed, seconds, index, task):
    condition = 'nominal' if index % 4 == 0 else 'train'
    c = case(seed, seconds, condition, index % 8)
    c['task'] = task
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7301]))
    if condition == 'train':
        # Include isolated and combined changes during training; larger mass
        # shifts and doubled sensor delay remain independent stress tests.
        c['lower_mass_scale'] = float(rng.uniform(.975, 1.075))
        c['motor_uncertainty'] = .02
        c['delay_steps'] = int(rng.integers(0, 3))
    if task == 'velocity':
        c['velocity_command'] = scenarios('development', 8, seconds, task=task)[int(rng.integers(8))]['velocity_command']
        c.update(velocity_noise=.01, velocity_delay=.02, velocity_bias=.01)
    return c


def desired_velocity(c, t):
    """Slew-limited start, reverse, stop; each 20 s cycle integrates to zero."""
    local = t % 20.
    if local < 6. or local >= 18.:
        factor = 0.
    elif local < 8.:
        factor = (local-6.)/2.
    elif local < 10.:
        factor = 1.
    elif local < 14.:
        factor = 1.-(local-10.)/2.
    elif local < 16.:
        factor = -1.
    else:
        factor = -1.+(local-16.)/2.
    return factor*np.asarray(c.get('velocity_command', [0., 0., 0.]))


class ReferencePID(TeacherPID):
    """Derivative on measured attitude, not on step changes in its setpoint."""
    def __init__(self, parameters, initial_rpy):
        super().__init__(parameters, initial_rpy)
        self.rpy_reference = np.zeros(2)
        self.z_reference = 0.
        self.vz_reference = 0.

    def step(self, reading):
        p = self.p
        rpy = reading[:3]
        delta = rpy[:2]-self.last_rpy[:2]
        derivative = np.arctan2(np.sin(delta), np.cos(delta))/DT
        self.last_rpy = rpy.copy()
        self.rate += (1-np.exp(-DT/p['derivative_tau']))*(derivative-self.rate)
        error = self.rpy_reference-rpy[:2]
        candidate = np.clip(self.integral+error*DT, -.3, .3)
        effort = np.array(p['kp'])*error-np.array(p['kd'])*self.rate+np.array(p['ki'])*candidate
        angles = np.array(p['trim'])+effort[::-1]
        self.integral = np.where(np.abs(angles[::-1]) < .3, candidate, self.integral)
        z = reading[9]-self.z_reference
        vz = reading[10]-self.vz_reference
        self.z_integral = float(np.clip(self.z_integral-z*DT, -.5, .5))
        kp, kd, ki = p['height_gains']
        self.raw = np.r_[angles, p['motor']-kp*z-kd*vz+ki*self.z_integral]
        return np.clip(self.raw, LOW, HIGH)


class ResearchEnv(ControlEnv):
    def __init__(self, source, mode='residual', task='hover', **kwargs):
        if task not in INPUTS:
            raise ValueError(task)
        self.task = task
        super().__init__(source, mode=mode, **kwargs)
        self.observation_space = spaces.Box(-10., 10., (INPUTS[task],), dtype=np.float32)
        self.vxy = np.zeros(2)
        self.reference_z = 0.
        self.command_velocity = np.zeros(3)

    def reset(self, **kwargs):
        training = self.training
        if training:
            index = self.episode_index
            seed = int(np.random.default_rng(np.random.SeedSequence(
                [self.run_seed, self.rank, index, 7261])).integers(1_500_000_000, 1_590_000_000))
            self.case = training_case(seed, self.horizon, index+self.rank, self.task)
            self.episode_index += 1
            kwargs['seed'] = seed
            self.training = False
        self.vxy = np.zeros(2)
        self.reference_z = 0.
        self.command_velocity = np.zeros(3)
        self.velocity_rows = []
        self.filtered_velocity = np.zeros(3)
        self.velocity_integral = np.zeros(2)
        self.attitude_setpoint = np.zeros(2)
        try:
            _, info = super().reset(**kwargs)
        finally:
            self.training = training
        self.velocity_rng = np.random.default_rng(np.random.SeedSequence([self.case['seed'], 8211]))
        self.velocity_bias = self.velocity_rng.uniform(-1., 1., 2)*self.case.get('velocity_bias', 0.)
        self.velocity_pending = deque()
        if self.task == 'velocity':
            self.teacher = ReferencePID(self.p, self.air.last_rpy)
        return self.observation(), info

    def observation(self):
        # Do not call HoverEnv.observation() with a 16-element observation space.
        scales = np.array([.35,.35,np.pi,10.,10.,10.,9.81,9.81,9.81,.2,1.])
        x = self.last_reading.copy()
        if self.task == 'velocity':
            x[9] -= self.reference_z
            x = np.r_[x/scales, self.vxy/.3, self.command_velocity/np.array([.3,.3,.15])]
        else:
            x = x/scales
        return np.clip(x, -10, 10).astype(np.float32)

    def velocity_sensor(self, t, true_velocity):
        # Truth enters only the simulated sensor, at 50 Hz. The controller gets
        # delayed noisy readings, filtered with a 100 ms time constant.
        self.filtered_velocity += (1-np.exp(-DT/.1))*(true_velocity-self.filtered_velocity)
        if self.air.steps % 10 == 0:
            measurement = self.filtered_velocity[:2]+self.velocity_bias+self.velocity_rng.normal(
                0., self.case.get('velocity_noise', .01), 2)
            self.velocity_pending.append((t+self.case.get('velocity_delay', .02), measurement))
        while self.velocity_pending and self.velocity_pending[0][0] <= t+1e-9:
            _, measurement = self.velocity_pending.popleft()
            self.vxy += (1-np.exp(-.02/.1))*(measurement-self.vxy)

    def step(self, action):
        controller_start = time.perf_counter()
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('Expected three finite actions')
        action = np.clip(action, -1., 1.)
        mode = self.mode
        if self.task == 'velocity':
            t = self.air.steps*DT
            self.command_velocity = desired_velocity(self.case, t)
            self.reference_z += self.command_velocity[2]*DT
            yaw = self.last_reading[2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            velocity_error = self.command_velocity[:2]-self.vxy
            rotation = np.array([[cy, sy], [-sy, cy]])
            self.velocity_integral = (np.clip(self.velocity_integral+velocity_error*DT, -.3, .3)
                                      if t >= 6. else np.zeros(2))
            body_control = (rotation@self.command_velocity[:2]
                +np.asarray(self.case.get('velocity_kp', VELOCITY_KP))*(rotation@velocity_error)
                +np.asarray(self.case.get('velocity_ki', VELOCITY_KI))*(rotation@self.velocity_integral))
            mapped = body_control*VELOCITY_FEEDFORWARD
            target = np.clip(np.array([-mapped[1], mapped[0]]), -.6, .6)
            if t < 6.:
                target[:] = 0.
            self.attitude_setpoint += np.clip(target-self.attitude_setpoint, -.6*DT, .6*DT)
            self.teacher.rpy_reference = self.attitude_setpoint.copy()
            if 'attitude_reference_override' in self.case:
                self.teacher.rpy_reference = np.asarray(self.case['attitude_reference_override'])
            self.teacher.z_reference = self.reference_z
            self.teacher.vz_reference = self.command_velocity[2]
        if mode == 'pid':
            command = self.teacher_command()
        elif mode == 'residual':
            command = np.clip(self.teacher_command()+RESIDUAL_SCALE*action, LOW, HIGH)
        else:
            command = CENTER+HALF*action
        adapter_seconds = time.perf_counter()-controller_start
        # The original physical/observation/termination step remains unchanged.
        self.mode = 'direct'
        try:
            _, reward, terminated, truncated, info = super().step((command-CENTER)/HALF)
        finally:
            self.mode = mode
        info['controller_seconds'] += adapter_seconds
        if self.task == 'velocity':
            pos, velocity, _ = self.air.env._state()
            self.velocity_sensor(self.air.steps*DT, velocity)
            # Net flight velocity is a 100 ms low-pass of the body's velocity;
            # retain instantaneous values too, to expose wingbeat oscillations.
            error = self.filtered_velocity-self.command_velocity
            z_error = pos[2]-self.air.start[2]-self.reference_z
            tilt = self.samples[-1][1]
            self.velocity_rows.append([self.air.steps*DT, *self.filtered_velocity,
                                       *self.command_velocity, z_error, *velocity])
            new_reward = DT*(2.+np.exp(-(tilt/.20)**2)+np.exp(-float(error@error)/.08**2))
            new_reward -= DT*.01*float(action@action)
            if terminated:
                new_reward -= 1.
            self.total_reward += new_reward-reward
            reward = float(new_reward)
            self.command_velocity = desired_velocity(self.case, self.air.steps*DT)
            if terminated or truncated:
                result = info['episode']
                rows = np.asarray(self.velocity_rows)
                after = rows[rows[:,0] >= min(6., rows[-1,0]/2)]
                rms = np.sqrt(np.mean((after[:,1:4]-after[:,4:7])**2, axis=0))
                desired_rms = float(np.sqrt(np.mean(np.sum(after[:,4:7]**2, axis=1))))
                normalized_error = float(np.linalg.norm(rms)/max(desired_rms, .02))
                position_rmse = float(np.sqrt(np.mean(rows[:,7]**2)))
                raw_rms = np.sqrt(np.mean((after[:,8:11]-after[:,4:7])**2, axis=0))
                active = after[np.linalg.norm(after[:,4:7], axis=1) > 0.]
                response_gain = float(np.sum(active[:,1:4]*active[:,4:7]) /
                                      max(np.sum(active[:,4:7]**2), 1e-12))
                result.update(return_sum=self.total_reward, velocity_rmse_m_s=rms.tolist(),
                              velocity_vector_rmse_m_s=float(np.linalg.norm(rms)),
                              tracking_nrmse=normalized_error,
                              raw_velocity_rmse_m_s=raw_rms.tolist(),
                              commanded_response_gain=response_gain,
                              reference_height_rmse_m=position_rmse,
                              sensor_assumption='100 ms prefilter; 50 Hz noisy horizontal velocity; 20 ms delay; 100 ms postfilter',
                              metric_velocity_filter_seconds=.1)
                result['success'] = bool(result['termination'] == 'time_limit'
                    and result['elapsed'] >= self.case['seconds']-1e-6
                    and result['minimum_clearance_m'] > .03
                    and result['tilt_rms_deg'] < 12.
                    and np.linalg.norm(rms) < .10 and normalized_error < .60
                    and position_rmse < .15)
        if terminated or truncated:
            info['episode'].update(task=self.task, mode=mode, inner_steps=self.air.steps)
        return self.observation(), reward, terminated, truncated, info
