"""Classic single-robot simulation: a robot plus the built-in controllers.

``SimCore`` keeps the pre-platform workflow working: pick a robot (built-in
name, manifest or model file), choose a fixed or floating base, and run with
the plate, thrust and hover controllers.  Roadmap Phase 3 rebuilds this on
the plugin API; the public methods stay.
"""

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
from flapsim.robot.loader import builtin_robots, load, resolve_manifest
from flapsim.robot.manifest import ManifestError, RobotManifest


class SimCore:
    def __init__(
        self,
        robot: str | Path | RobotManifest = "SingleThrustFlapper",
        *,
        pitch: float = 0.0,
        roll: float = 0.0,
        fix: bool = True,
    ) -> None:
        try:
            manifest = resolve_manifest(robot)
        except ManifestError as exc:
            available = ", ".join(
                sorted({p.parent.name for p in builtin_robots().values()})
            )
            raise ValueError(
                f"Unknown robot {robot!r}. Available: {available}"
            ) from exc
        joints_file = manifest.extras.get("joints")
        if joints_file is None:
            raise ValueError(
                f"robot {manifest.name!r} has no extras.joints configuration; "
                "SimCore needs it for the classic controllers"
            )
        joints_config = manifest.resolve(joints_file)
        hover = manifest.extras.get("hover", {})

        self.manifest = manifest
        self.robot = manifest.name
        self.robot_model = load(manifest, base="fixed" if fix else "free")
        self.environment = SimulationEnvironment(self.robot_model)
        self.model = self.environment.model
        self.data = self.environment.data
        root_body = self.robot_model.root_body_name

        self.plate_controller = PlateController(
            pitch=pitch, roll=roll, config_path=joints_config
        )
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
            self.aero = MuJoCoAeroAdapter(self.robot_model)
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
                root_body=root_body,
            )
        self.environment.attach_force_model(self.aero)
        self.hover_frequency = float(hover.get("frequency_hz", 42.0))
        self.hovering_controller = None
        if not fix:
            self.enable_hovering(target_z=float(hover.get("target_height", 0.5)))

    def enable_hovering(self, target_z: float) -> None:
        if self.environment.fixed_base:
            raise ValueError("hovering requires a floating base (--fix false)")
        if self.hovering_controller is None:
            self.hovering_controller = HoveringController(
                self.model,
                self.plate_controller,
                self.thrust_controller,
                target_z=target_z,
                root_body=self.robot_model.root_body_name,
                hover_frequency=self.hover_frequency,
            )
            self.environment.attach_supervisor(self.hovering_controller)
        else:
            self.hovering_controller.set_target(target_z)

    def set_plate_command(
        self, *, pitch: float | None = None, roll: float | None = None
    ) -> None:
        old_pitch, old_roll = self.plate_controller.pitch, self.plate_controller.roll
        self.plate_controller.set_command(pitch=pitch, roll=roll)
        try:
            self.position_controller.validate_targets(
                self.model, self.plate_controller.targets
            )
        except ValueError:
            self.plate_controller.set_command(pitch=old_pitch, roll=old_roll)
            raise

    def set_thrust_command(
        self, *, frequency: float | None = None, amplitude: float | None = None
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

    def set_hover_waypoints(self, waypoints, *, interval: float = 3.0) -> None:
        if self.hovering_controller is None:
            raise ValueError("waypoints require a floating base (--fix false)")
        self.hovering_controller.set_waypoints(waypoints, interval=interval)

    def run(self, *, viewer: bool = True, duration: float | None = None) -> int:
        return self.environment.run(viewer=viewer, duration=duration)
