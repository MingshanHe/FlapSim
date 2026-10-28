"""Connect the command-line robot name to the MuJoCo environment."""

from __future__ import annotations

from pathlib import Path

from flapsim.actuator.SingleThrustFlapper import (
    PositionController,
    TrajectoryController,
)
from flapsim.controller.SingleThrustFlapper import (
    HoveringController,
    PlateController,
    ThrustController,
)
from flapsim.environment import SimulationEnvironment
from flapsim.fluid import AveragedHoverAero, MuJoCoAeroAdapter

ROBOT_URDFS = {
    "SingleThrustFlapper": (
        Path(__file__).parent
        / "robot"
        / "SingleThrustFlapper"
        / "urdf"
        / "SingleThrustFlapper.urdf"
    )
}


class SimCore:
    def __init__(
        self,
        robot: str,
        *,
        pitch: float = 0.0,
        roll: float = 0.0,
        fix: bool = True,
    ) -> None:
        try:
            urdf_path = ROBOT_URDFS[robot]
        except KeyError as exc:
            available = ", ".join(ROBOT_URDFS)
            raise ValueError(
                f"Unknown robot {robot!r}. Available: {available}"
            ) from exc

        self.robot = robot
        self.environment = SimulationEnvironment(
            urdf_path,
            position=(0.0, 0.0, 0.5),
            fixed_base=fix,
        )
        self.model = self.environment.model
        self.data = self.environment.data
        joints_config = (
            Path(__file__).parent / "config" / "SingleThrustFlapper" / "joints.json"
        )
        self.plate_controller = PlateController(pitch=pitch, roll=roll)
        self.position_controller = PositionController(self.model, joints_config)
        self.thrust_controller = ThrustController(joints_config)
        self.trajectory_controller = TrajectoryController(self.model, joints_config)
        self.trajectory_controller.validate_commands(self.thrust_controller.targets)
        self.position_controller.validate_targets(
            self.model, self.plate_controller.targets
        )
        if fix:
            self.environment.attach_control(
                self.plate_controller, self.position_controller
            )
            self.environment.attach_control(
                self.thrust_controller, self.trajectory_controller
            )
            self.aero = MuJoCoAeroAdapter(self.model, self.data)
        else:
            self.position_controller.set_kinematic(True)
            self.trajectory_controller.set_dynamic_velocity(False)
            self.environment.attach_control(
                self.plate_controller, self.position_controller
            )
            self.environment.attach_control(
                self.thrust_controller, self.trajectory_controller
            )
            self.aero = AveragedHoverAero(
                self.model,
                self.data,
                self.plate_controller,
                self.thrust_controller,
            )
        self.environment.attach_force_model(self.aero)
        self.hovering_controller = None
        if not fix:
            self.enable_hovering(target_z=0.5)

    def enable_hovering(self, target_z: float) -> None:
        if self.environment.fixed_base:
            raise ValueError("hovering requires a floating base (--fix false)")
        if self.hovering_controller is None:
            self.hovering_controller = HoveringController(
                self.model,
                self.plate_controller,
                self.thrust_controller,
                target_z=target_z,
            )
            self.environment.attach_supervisor(self.hovering_controller)
        else:
            self.hovering_controller.set_target(target_z)

    def set_plate_command(
        self, *, pitch: float | None = None, roll: float | None = None
    ) -> None:
        old_targets = self.plate_controller.targets
        self.plate_controller.set_command(pitch=pitch, roll=roll)
        try:
            self.position_controller.validate_targets(
                self.model, self.plate_controller.targets
            )
        except ValueError:
            self.plate_controller.set_command(
                pitch=old_targets["pitch_joint"],
                roll=old_targets["roll_joint"],
            )
            raise

    def set_thrust_command(
        self,
        *,
        frequency: float | None = None,
        amplitude: float | None = None,
    ) -> None:
        old_commands = self.thrust_controller.targets
        self.thrust_controller.set_command(frequency=frequency, amplitude=amplitude)
        try:
            self.trajectory_controller.validate_commands(self.thrust_controller.targets)
        except ValueError:
            old = next(iter(old_commands.values()))
            self.thrust_controller.set_command(
                frequency=old["frequency"], amplitude=old["amplitude"]
            )
            raise

    def set_hover_waypoints(
        self,
        waypoints,
        *,
        interval: float = 3.0,
    ) -> None:
        if self.hovering_controller is None:
            raise ValueError("waypoints require a floating base (--fix false)")
        self.hovering_controller.set_waypoints(waypoints, interval=interval)

    def run(self, *, viewer: bool = True, duration: float | None = None) -> int:
        return self.environment.run(viewer=viewer, duration=duration)
