"""Free-flight PPO environment using the full IST_robot-V2 dynamics."""
from dataclasses import dataclass, replace
from pathlib import Path
import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from flapsim import Simulator, load_config
from flapsim.kinematics import point_velocity, euler_matrix
from .trajectories import NAMES, Trajectory


PRESET = Path(__file__).resolve().parents[1] / 'config' / 'training' / 'ist_v2_flight.json'


@dataclass(frozen=True)
class TrackingConfig:
    trajectories: tuple = NAMES
    control_dt: float = .01
    episode_seconds: float = 8.
    center: tuple = (0., 0., .8)
    radius: float = .2
    period: float = 8.
    height: float = .1
    gravity_scale: float = 1.
    initial_speed: float = 4000.
    warmup_seconds: float = .1
    action_tau: float = .02
    randomize: bool = True
    min_altitude: float = .1
    max_position_error: float = 1.
    max_tilt: float = 1.4
    native_physics: bool = False

    def __post_init__(self):
        if not self.trajectories or any(n not in NAMES for n in self.trajectories):
            raise ValueError(f'trajectories must be a nonempty subset of {NAMES}')
        Trajectory(self.trajectories[0], self.center, self.radius, self.period, self.height)
        for key in ('control_dt', 'episode_seconds', 'gravity_scale',
                    'max_position_error', 'max_tilt'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('initial_speed', 'warmup_seconds', 'action_tau', 'min_altitude'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.center[2] - self.height <= self.min_altitude:
            raise ValueError('reference altitude must clear min_altitude')
        if self.max_tilt >= math.pi:
            raise ValueError('max_tilt must be less than pi')


class TrajectoryTrackingEnv(gym.Env):
    """Actions [-1,1]^3 -> shaft speed, pitch target, roll target.

    No synthetic thrust, attitude torque, position overwrite or altered mass
    is applied. Gravity scaling is explicit and is recorded with the policy.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 100}

    def __init__(self, config=None, *, simulation_config=None, render_mode=None):
        super().__init__()
        self.config = config or TrackingConfig()
        if render_mode not in (None, *self.metadata['render_modes']):
            raise ValueError('unsupported render mode')
        self.render_mode = render_mode
        source = simulation_config or load_config(PRESET)
        if source.gimbal is None or len(source.actuator.joints) != 1:
            raise ValueError('tracking requires one shaft motor and two gimbal servos')
        if source.backend != 'cpu':
            raise ValueError('this SB3 environment uses native CPU physics; vectorize with processes')
        if source.controller.kind != 'speed':
            raise ValueError('tracking requires the speed controller')
        if self.config.initial_speed > source.actuator.max_velocity:
            raise ValueError('initial_speed exceeds the configured motor limit')
        self.source_config = source
        config = self.config
        self.substeps = round(config.control_dt / source.scene.timestep)
        if self.substeps < 1 or not math.isclose(self.substeps * source.scene.timestep,
                                               config.control_dt, abs_tol=1e-10):
            raise ValueError('control_dt must be an integer multiple of physics timestep')
        self.horizon = math.ceil(config.episode_seconds / config.control_dt)
        self.metadata = dict(self.metadata, render_fps=round(1/config.control_dt))
        scene = replace(source.scene, fixed_base=False, orientation=(1., 0., 0., 0.),
                        gravity=tuple(config.gravity_scale * g for g in source.scene.gravity))
        self.sim = Simulator(replace(source, scene=scene))
        if config.native_physics:
            from .native import accelerate
            accelerate(self.sim)
        self.body = self.sim.model.body(source.root_body).id
        self.root = int(self.sim.model.body_rootid[self.body])
        free = self.sim.model.joint('floating_base')
        self.free_qpos, self.free_dof = int(free.qposadr[0]), int(free.dofadr[0])
        self.internal = [self.sim.model.joint(i) for i in range(self.sim.model.njnt)
                         if int(self.sim.model.jnt_type[i]) != int(mujoco.mjtJoint.mjJNT_FREE)]
        if any(int(j.type[0]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
               for j in self.internal):
            raise ValueError('internal joints must be scalar hinge/slide joints')
        self.qids = np.array([j.qposadr[0] for j in self.internal], dtype=int)
        self.dids = np.array([j.dofadr[0] for j in self.internal], dtype=int)
        self.hinge = np.array([int(j.type[0]) == int(mujoco.mjtJoint.mjJNT_HINGE) for j in self.internal])
        self._initial_qpos = self.sim.model.qpos0.copy()
        self._initial_qvel = np.zeros(self.sim.model.nv)
        if config.warmup_seconds:
            fixed = Simulator(replace(source, scene=replace(scene, fixed_base=True)))
            if config.native_physics:
                accelerate(fixed)
            fixed.set_speed(config.initial_speed)
            fixed.step(math.ceil(config.warmup_seconds / scene.timestep))
            if not np.isfinite(fixed.data.qpos).all() or any(w.number for w in fixed.data.warning):
                raise RuntimeError('powered initial state is unstable')
            for joint in self.internal:
                self._initial_qpos[joint.qposadr[0]] = fixed.data.joint(joint.name).qpos[0]
                self._initial_qvel[joint.dofadr[0]] = fixed.data.joint(joint.name).qvel[0]
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.applied_action = np.zeros(3)
        self.reference = Trajectory(config.trajectories[0], config.center,
                                    config.radius, config.period, config.height)
        self.steps = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self._observation().shape,
                                            dtype=np.float32)
        self._viewer = self._renderer = None
        self._finished = True

    def _state(self):
        data = self.sim.data
        position = data.xpos[self.body].copy()
        rotation = data.xmat[self.body].reshape(3, 3)
        velocity = point_velocity(data, self.body, self.root, position)
        return position, velocity, rotation

    def _observation(self):
        position, velocity, rotation = self._state()
        time = self.steps * self.config.control_dt
        target, target_velocity, target_acceleration = self.reference.sample(time)
        data = self.sim.data
        q, v = data.qpos[self.qids], data.qvel[self.dids]
        # Sin/cos avoids discontinuities in continuously rotating gear joints.
        internal = np.r_[np.sin(q[self.hinge]), np.cos(q[self.hinge]),
                         q[~self.hinge] / .003, np.where(self.hinge, v / 2000., v)]
        future = [(self.reference.sample(time + dt)[0] - position) @ rotation
                  for dt in (.1, .3, .6)]
        return np.concatenate([(target-position) @ rotation,
                               (target_velocity-velocity) @ rotation,
                               velocity @ rotation, rotation.ravel(),
                               data.cvel[self.body, :3] @ rotation,
                               target_acceleration @ rotation, *future,
                               internal, self.applied_action]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.config
        options = options or {}
        name = options.get('trajectory', self.np_random.choice(c.trajectories))
        phase = options.get('phase', self.np_random.uniform(0, 2*np.pi) if c.randomize else 0.)
        radius = c.radius * (self.np_random.uniform(.8, 1.2) if c.randomize else 1.)
        period = c.period * (self.np_random.uniform(.8, 1.2) if c.randomize else 1.)
        self.reference = Trajectory(name, c.center, radius, period, c.height, phase)
        position, velocity, _ = self.reference.sample(0.)
        qpos, qvel = self._initial_qpos.copy(), self._initial_qvel.copy()
        qp, qv = self.free_qpos, self.free_dof
        noise = .01 if c.randomize else 0.
        qpos[qp:qp+3] = position + self.np_random.normal(0, noise, 3)
        quaternion = np.zeros(4)
        matrix = euler_matrix(self.np_random.normal(0, noise, 3))
        mujoco.mju_mat2Quat(quaternion, matrix.ravel())
        qpos[qp+3:qp+7] = quaternion
        qvel[qv:qv+3] = velocity + self.np_random.normal(0, noise, 3)
        self.sim.reset(qpos=qpos, qvel=qvel)
        self.applied_action = np.zeros(3)
        self.sim.set_speed(c.initial_speed)
        self.sim.set_gimbal(pitch=0., roll=0.)
        self.steps = 0
        self.squared_errors = 0.
        self._finished = False
        self._last_observation = self._observation()
        if self.render_mode == 'human':
            self.render()
        return self._last_observation.copy(), {'trajectory': name, 'gravity_scale': c.gravity_scale}

    def step(self, action):
        if self._finished:
            raise RuntimeError('reset must be called before stepping a finished episode')
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('action must contain three finite values')
        action = np.clip(action, -1, 1)
        previous = self.applied_action.copy()
        c = self.config
        alpha = 1. if c.action_tau == 0 else 1 - np.exp(-c.control_dt/c.action_tau)
        self.applied_action += alpha * (action-self.applied_action)
        maximum = self.source_config.actuator.max_velocity
        # Zero policy output starts at trim; both motor limits remain reachable.
        span = maximum-c.initial_speed if self.applied_action[0] >= 0 else c.initial_speed
        speed = c.initial_speed + self.applied_action[0]*span
        bounds = self.sim.model.actuator_ctrlrange
        angles = bounds[:, 0] + .5*(self.applied_action[1:]+1)*(bounds[:, 1]-bounds[:, 0])
        self.sim.set_speed(float(speed))
        self.sim.set_gimbal(pitch=float(angles[0]), roll=float(angles[1]))
        start_time = float(self.sim.data.time)
        self.sim.step(self.substeps)
        self.steps += 1
        position, velocity, rotation = self._state()
        target, target_velocity, _ = self.reference.sample(self.steps*c.control_dt)
        numerical = (not np.isfinite(self.sim.data.qpos).all()
                     or not np.isfinite(self.sim.data.qvel).all()
                     or any(w.number for w in self.sim.data.warning)
                     or not math.isclose(float(self.sim.data.time)-start_time,
                                         c.control_dt, abs_tol=1e-8))
        error = float(np.linalg.norm(position-target))
        velocity_error = float(np.linalg.norm(velocity-target_velocity))
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        reason = ('numerical' if numerical else 'ground' if position[2] < c.min_altitude
                  else 'tracking_error' if error > c.max_position_error
                  else 'tilt' if tilt > c.max_tilt else '')
        terminated = bool(reason)
        truncated = self.steps >= self.horizon and not terminated
        observation = self._observation()
        if not np.isfinite(observation).all():
            numerical, terminated, truncated, reason = True, True, False, 'numerical'
        if numerical:
            observation = self._last_observation.copy()
            error = c.max_position_error
            velocity_error = 10.
        self.squared_errors += error**2
        rmse = math.sqrt(self.squared_errors/self.steps)
        reward = (1. + 2.*np.exp(-(error/.2)**2) + .5*np.exp(-(velocity_error/.5)**2)
                  - .1*tilt**2 - .005*(speed/maximum)**2
                  - .02*float(np.sum((self.applied_action-previous)**2)))
        reward = -25. if numerical else float(reward - (25. if terminated else 0.))
        self._finished = terminated or truncated
        self._last_observation = observation.copy()
        info = {'trajectory': self.reference.name, 'position_error': error,
                'tracking_rmse': rmse, 'elapsed': self.steps*c.control_dt,
                'termination_reason': reason or ('time_limit' if truncated else ''),
                'is_success': bool(truncated and rmse < .1),
                'position': position.copy(), 'target': target,
                'motor_command': float(speed), 'gimbal_command': angles.copy(),
                'gravity_scale': c.gravity_scale}
        if self.render_mode == 'human':
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.sim.model, self.sim.data)
                self._viewer.cam.distance = 1.
            self._viewer.cam.lookat[:] = self.sim.data.xpos[self.body]
            self._viewer.sync()
        elif self.render_mode == 'rgb_array':
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.sim.data.xpos[self.body]
            camera.distance = .5
            self._renderer.update_scene(self.sim.data, camera=camera)
            return self._renderer.render()

    def close(self):
        for resource in (self._viewer, self._renderer):
            if resource is not None:
                resource.close()
        self._viewer = self._renderer = None
