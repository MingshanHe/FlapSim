"""Classic convenience wrapper: a built-in robot with its shipped setup.

``SimCore`` builds a :class:`~flapsim.core.Simulation` from the robot's
``sim.yaml`` (fixed base) or ``hover.yaml`` (floating base) and exposes the
pre-platform method names.  New code should use :class:`Simulation` and
plugins directly.
"""

from __future__ import annotations

from pathlib import Path

from flapsim.core.simulation import Simulation
from flapsim.robot.loader import builtin_robots, resolve_manifest
from flapsim.robot.manifest import ManifestError, RobotManifest


class SimCore:
    def __init__(
        self,
        robot: str | Path | RobotManifest = "SingleThrustFlapper",
        *,
        pitch: float = 0.0,
        roll: float = 0.0,
        fix: bool = True,
        config: str | None = None,
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
        name = config or ("sim" if fix else "hover")
        config_path = (
            None if manifest.base_dir is None else manifest.base_dir / f"{name}.yaml"
        )
        if config_path is None or not config_path.is_file():
            raise ValueError(
                f"robot {manifest.name!r} ships no {name}.yaml next to its manifest; "
                "build a Simulation with plugins instead"
            )
        self.manifest = manifest
        self.robot = manifest.name
        self.simulation = Simulation.from_config(
            config_path, base="fixed" if fix else "free"
        )
        self.environment = self.simulation
        self.robot_model = self.simulation.robot
        self.model = self.simulation.model
        self.data = self.simulation.data
        self.stroke = self.simulation.find("stroke")
        self.gimbal = self.simulation.find("gimbal")
        self.aero = self.simulation.find("aero")
        self.hovering_controller = self.simulation.find("hover")
        if (pitch or roll) and self.gimbal is not None:
            self.set_plate_command(pitch=pitch, roll=roll)

    def enable_hovering(self, target_z: float) -> None:
        if self.simulation.fixed_base:
            raise ValueError("hovering requires a floating base (--fix false)")
        if self.hovering_controller is None:
            raise ValueError("this setup has no hover controller plugin")
        self.hovering_controller.set_target(z=target_z)

    def set_plate_command(
        self, *, pitch: float | None = None, roll: float | None = None
    ) -> None:
        if self.gimbal is None:
            raise ValueError("this setup has no gimbal actuator")
        if pitch is not None:
            self.gimbal.set_role_target("pitch", pitch)
        if roll is not None:
            self.gimbal.set_role_target("roll", roll)

    def set_thrust_command(
        self, *, frequency: float | None = None, amplitude: float | None = None
    ) -> None:
        if self.stroke is None:
            raise ValueError("this setup has no stroke actuator")
        self.stroke.set_command(frequency_hz=frequency, amplitude_rad=amplitude)

    def set_hover_waypoints(self, waypoints, *, interval: float = 3.0) -> None:
        if self.hovering_controller is None:
            raise ValueError("waypoints require a floating base (--fix false)")
        self.hovering_controller.set_waypoints(waypoints, interval=interval)

    def run(self, *, viewer: bool = True, duration: float | None = None) -> int:
        return self.simulation.run(duration, viewer=viewer)
