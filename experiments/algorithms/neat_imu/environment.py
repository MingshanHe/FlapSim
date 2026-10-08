"""Full free-flight dynamics with an IMU attitude-only observation contract."""
from dataclasses import replace
import math

from gymnasium import spaces
import mujoco
import numpy as np

from flapsim.kinematics import euler_matrix
from flapsim.rl.env import TrackingConfig, TrajectoryTrackingEnv
from flapsim.rl.native_fast import accelerate_fast

ANGLE_SCALE = 0.35  # radians, fixed normalization; no fitted statistics
INPUT_NAMES = {
    3: ['roll', 'pitch', 'yaw'],
    6: ['roll', 'pitch', 'yaw', 'position_error_x', 'position_error_y', 'position_error_z'],
    12: ['roll', 'pitch', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z',
         'position_error_x', 'position_error_y', 'position_error_z',
         'velocity_x', 'velocity_y', 'velocity_z'],
}


def rotation_to_rpy(rotation):
    """ZYX Euler angles (roll, pitch, yaw), in radians."""
    r = np.asarray(rotation).reshape(3, 3)
    return np.array([math.atan2(r[2, 1], r[2, 2]),
                     math.asin(float(np.clip(-r[2, 0], -1, 1))),
                     math.atan2(r[1, 0], r[0, 0])])


class ImuAttitudeEnv(TrajectoryTrackingEnv):
    """Low-dimensional policy; actions are alpha/pitch, beta/roll, motor speed.

    Real gravity and all six free-base degrees of freedom.
    No teacher, PID attitude loop or synthetic torque.
    The simulator orientation is an ideal, noiseless IMU attitude estimate proxy.
    """
    def __init__(self, seconds=3., motor_speed=3400., initial_degrees=5.,
                 render_mode=None, warmup_seconds=.1, inputs=3, simulation_config=None):
        if not math.isfinite(initial_degrees) or not 0 <= initial_degrees <= 30:
            raise ValueError('initial_degrees must be between 0 and 30')
        self.initial_degrees = initial_degrees
        if inputs not in INPUT_NAMES:
            raise ValueError('inputs must be 3, 6 or 12')
        self.inputs = inputs
        super().__init__(TrackingConfig(
            trajectories=('hover',), episode_seconds=seconds,
            initial_speed=motor_speed, randomize=False, native_physics=True,
            warmup_seconds=warmup_seconds, max_position_error=1.), simulation_config=simulation_config, render_mode=render_mode)
        accelerate_fast(self.sim)
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-10., 10., (inputs,), dtype=np.float32)

    def angles(self):
        return rotation_to_rpy(self.sim.data.xmat[self.body])

    def _observation(self):
        values = self.angles()/ANGLE_SCALE
        if self.inputs > 3:
            position, velocity, rotation = self._state()
            error = (np.asarray(self.config.center)-position) @ rotation/.2
            if self.inputs == 6:
                values = np.r_[values, error]
            else:
                gyro = self.sim.data.cvel[self.body, :3] @ rotation/10.
                values = np.r_[values, gyro, error, velocity @ rotation/.5]
        return np.clip(values, -10, 10).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        rpy = self.np_random.uniform(-1, 1, 3)*math.radians(self.initial_degrees)
        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        qpos[self.free_qpos:self.free_qpos+3] += self.np_random.normal(0, .01, 3)
        qvel[self.free_dof:self.free_dof+3] += self.np_random.normal(0, .01, 3)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, euler_matrix(rpy).ravel())
        qpos[self.free_qpos+3:self.free_qpos+7] = quaternion
        self.sim.reset(qpos=qpos, qvel=qvel)
        self.sim.set_speed(self.config.initial_speed)
        self.sim.set_gimbal(pitch=0., roll=0.)
        self._last_observation = self._observation()
        self.angle_squared = np.zeros(3)
        self.max_tilt = 0.
        return self._last_observation.copy(), dict(info, rpy=rpy.copy())

    def set_seconds(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('seconds must be positive')
        self.config = replace(self.config, episode_seconds=seconds)
        self.horizon = math.ceil(seconds/self.config.control_dt)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('action must contain three finite values: alpha, beta, speed')
        observation, reward, terminated, truncated, info = super().step(action[[2, 0, 1]])
        rpy = self.angles()
        if not np.isfinite(rpy).all():
            rpy = np.full(3, np.pi)
        tilt = math.acos(float(np.clip(self.sim.data.xmat[self.body, 8], -1, 1)))
        self.angle_squared += rpy**2
        self.max_tilt = max(self.max_tilt, tilt)
        # Use the existing hovering reward (position, velocity, tilt and control
        # smoothness). Reward can access state that is not a policy observation.
        # Fitness is divided by the requested horizon, not surviving step count.
        rmse = np.sqrt(self.angle_squared/self.steps)
        info.update(rpy=rpy, attitude_rmse_deg=np.rad2deg(rmse),
                    max_tilt_deg=math.degrees(self.max_tilt),
                    alpha=float(info['gimbal_command'][0]),
                    beta=float(info['gimbal_command'][1]),
                    attitude_success=bool(truncated and np.all(rmse < math.radians(10))))
        return observation, float(reward), terminated, truncated, info
