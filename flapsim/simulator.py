"""Public simulation facade: configure, command, step, reset and render."""
import math
import time
import numpy as np
from .config import SimulationConfig
from .config.scene import build_model
from .actuator import create_actuator
from .controller import create_controller, SixDOFCommand, SpeedController
from .fluid import create_fluid
from .backends import CPUBackend, MJXBackend


class Simulator:
    def __init__(self, config=None, *, actuator=None, controller=None, fluid=None):
        self.config = config or SimulationConfig()
        self.actuator = actuator or create_actuator(self.config.actuator)
        self.controller = controller or create_controller(self.config.controller)
        self.fluid = fluid or create_fluid(self.config.fluid)
        self.model = build_model(self.config, self.fluid)
        self.actuator.bind(self.model)
        self.controller.bind(self.model, self.actuator, self.config.root_body)
        self.fluid.bind(self.model)
        self._command = self.controller.command()
        self._gimbal_command = np.array(self.config.gimbal.angles if self.config.gimbal else [],
                                        dtype=float)
        self.external_joint_forces = np.zeros(self.model.nv)
        self.external_body_forces = np.zeros((self.model.nbody, 6))
        args = (self.model, self.actuator, self.controller, self.fluid)
        self.backend = (CPUBackend(*args, controls=self._gimbal_command)
                        if self.config.backend == 'cpu' else
                        MJXBackend(*args, device=self.config.device, controls=self._gimbal_command))

    @property
    def data(self):
        """Native MjData on CPU, device-resident mjx.Data on MJX."""
        return self.backend.data

    def set_command(self, command):
        if isinstance(command, SixDOFCommand):
            command = command.as_array()
        command = np.atleast_1d(np.asarray(command, dtype=float))
        if command.shape != self._command.shape or not np.isfinite(command).all():
            raise ValueError(f'command must be finite with shape {self._command.shape}')
        self._command = command.copy()

    def set_speed(self, angular_velocity):
        if not isinstance(self.controller, SpeedController):
            raise ValueError('set_speed requires a SpeedController; use set_command for pose targets')
        self.set_command(angular_velocity)

    def set_gimbal(self, *, pitch, roll):
        """Set mount-relative gimbal angles [rad]; shaft speed stays independent."""
        if self.config.gimbal is None:
            raise ValueError('set_gimbal requires GimbalConfig')
        target = np.asarray([pitch, roll], dtype=float)
        limits = self.model.actuator_ctrlrange
        if (target.shape != (2,) or not np.isfinite(target).all()
                or np.any(target < limits[:, 0]) or np.any(target > limits[:, 1])):
            raise ValueError(f'gimbal angles must be finite and within {limits.tolist()} rad')
        self._gimbal_command = target.copy()

    def step(self, steps=1):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError('steps must be a non-negative integer')
        if steps:
            self.backend.advance(steps, self._command,
                                 self.external_joint_forces, self.external_body_forces,
                                 self._gimbal_command)
        return self.data

    def reset(self, *, qpos=None, qvel=None):
        for name, value, size in (('qpos', qpos, self.model.nq), ('qvel', qvel, self.model.nv)):
            if value is not None and (np.shape(value) != (size,) or not np.isfinite(value).all()):
                raise ValueError(f'{name} must contain {size} finite values')
        self._command = self.controller.command()
        self._gimbal_command = np.array(self.backend.initial_controls, copy=True)
        self.external_joint_forces[:] = 0
        self.external_body_forces[:] = 0
        self.backend.reset(qpos, qvel)
        return self.data

    def snapshot(self):
        return self.backend.snapshot()

    def run(self, *, duration=None, viewer=True):
        if duration is not None and (not math.isfinite(duration) or duration < 0):
            raise ValueError('duration must be finite and non-negative')
        if duration is None and not viewer:
            raise ValueError('headless run requires duration')
        count = None if duration is None else math.ceil(duration / self.model.opt.timestep - 1e-10)
        if not viewer:
            self.step(count)
            return count
        import mujoco
        import mujoco.viewer
        display = self.snapshot()
        steps = 0
        frame_steps = max(1, round(1 / (60 * self.model.opt.timestep)))
        with mujoco.viewer.launch_passive(self.model, display) as window:
            window.cam.lookat[:] = self.config.scene.position
            window.cam.distance = .35
            while window.is_running() and (count is None or steps < count):
                start = time.perf_counter()
                block = frame_steps if count is None else min(frame_steps, count - steps)
                self.step(block)
                mujoco.mj_copyData(display, self.model, self.snapshot())
                window.sync()
                steps += block
                time.sleep(max(0, block*self.model.opt.timestep - (time.perf_counter()-start)))
        return steps
