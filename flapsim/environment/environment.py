"""The step loop: controllers, actuators, force models and ``mj_step``."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco

from flapsim.robot.loader import RobotModel, load
from flapsim.robot.manifest import RobotManifest


class SimulationEnvironment:
    """Own a :class:`RobotModel` and step it with attached components.

    ``robot`` may be a :class:`RobotModel` or anything
    :func:`flapsim.robot.load` accepts (built-in name, manifest or model
    file).  The step order is: supervisors update commands, actuators apply
    joint commands, force models write ``xfrc_applied``, ``mj_step``, then
    kinematic actuators re-impose their targets in ``post_step``.
    """

    def __init__(
        self,
        robot: RobotModel | RobotManifest | str | Path,
        *,
        base: str | None = None,
        fixed_base: bool | None = None,
        position=None,
        timestep: float | None = None,
        scene: bool = True,
    ) -> None:
        if not isinstance(robot, RobotModel):
            if fixed_base is not None and base is None:
                base = "fixed" if fixed_base else "free"
            robot = load(
                robot, base=base, position=position, timestep=timestep, scene=scene
            )
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.fixed_base = robot.fixed_base
        self.control_pairs = []
        self.force_models = []
        self.supervisors = []
        mujoco.mj_forward(self.model, self.data)

    def attach_control(self, command_source, actuator) -> None:
        self.control_pairs.append((command_source, actuator))

    def attach_force_model(self, force_model) -> None:
        self.force_models.append(force_model)

    def attach_supervisor(self, supervisor) -> None:
        self.supervisors.append(supervisor)

    def step(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        for supervisor in self.supervisors:
            supervisor.update(self.model, self.data)
        for command_source, actuator in self.control_pairs:
            actuator.apply(self.model, self.data, command_source.targets)
        for force_model in self.force_models:
            force_model.apply_forces()
        mujoco.mj_step(self.model, self.data)
        forwarded = False
        for command_source, actuator in self.control_pairs:
            if hasattr(actuator, "post_step"):
                actuator.post_step(self.model, self.data, command_source.targets)
                forwarded = True
        if forwarded:
            mujoco.mj_forward(self.model, self.data)

    def run(self, *, viewer: bool = True, duration: float | None = None) -> int:
        """Run until duration expires or the viewer is closed."""
        steps = 0
        end_time = None if duration is None else self.data.time + duration

        if not viewer:
            if end_time is None:
                raise ValueError("headless mode requires a duration")
            while self.data.time < end_time:
                self.step()
                steps += 1
            return steps

        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as window:
            while window.is_running() and (
                end_time is None or self.data.time < end_time
            ):
                wall_start = time.perf_counter()
                frame_end = self.data.time + 1.0 / 60.0
                while self.data.time < frame_end:
                    self.step()
                    steps += 1
                window.sync()
                remaining = 1.0 / 60.0 - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        return steps
