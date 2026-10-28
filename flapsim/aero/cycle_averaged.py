"""Cycle-averaged lift and control moments for floating-base hover studies.

The model treats the robot as one rigid body: lift along the body axis tilted
by the gimbal angles, proportional to the stroke frequency squared, plus
identified plate control moments and linear damping.  Coefficients come from
a fixed-base identification of the per-step model (``flapsim sweep`` in a
later phase); the defaults are the example robot's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from flapsim.plugins.base import AeroPlugin
from flapsim.plugins.registry import register


@register
class CycleAveragedAero(AeroPlugin):
    name = "cycle_averaged"
    provides = ("F_world", "torque_world", "lift", "frequency_hz")
    example_config = {
        "pitch_joint": "pitch_joint",
        "roll_joint": "roll_joint",
    }

    def __init__(
        self,
        *,
        pitch_joint: str,
        roll_joint: str,
        stroke: str = "stroke",
        gimbal: str = "gimbal",
        reference: Mapping[str, float] | None = None,
        lift_coefficient: float | None = None,
        roll_moment_per_rad: float = 0.01055 / 20.0,
        pitch_moment_per_rad: float = 0.01144 / 20.0,
        moment_reference_frequency_hz: float = 42.0,
        angular_damping: Sequence[float] = (2.0e-4, 2.0e-4, 1.0e-4),
        horizontal_damping: float = 0.06,
        root_body: str | None = None,
        id: str | None = "aero",
    ) -> None:
        reference = dict(reference or {"frequency_hz": 30.0, "lift_n": 0.16661})
        if lift_coefficient is None:
            lift_coefficient = (
                float(reference["lift_n"]) / float(reference["frequency_hz"]) ** 2
            )
        super().__init__(
            id=id,
            pitch_joint=pitch_joint,
            roll_joint=roll_joint,
            stroke=stroke,
            gimbal=gimbal,
            lift_coefficient=lift_coefficient,
            roll_moment_per_rad=roll_moment_per_rad,
            pitch_moment_per_rad=pitch_moment_per_rad,
            moment_reference_frequency_hz=moment_reference_frequency_hz,
            angular_damping=list(angular_damping),
            horizontal_damping=horizontal_damping,
            root_body=root_body,
        )
        self.pitch_joint = pitch_joint
        self.roll_joint = roll_joint
        self.stroke_id = stroke
        self.gimbal_id = gimbal
        self.lift_coefficient = float(lift_coefficient)
        self.roll_moment_per_rad = float(roll_moment_per_rad)
        self.pitch_moment_per_rad = float(pitch_moment_per_rad)
        self.moment_reference_frequency = float(moment_reference_frequency_hz)
        self.angular_damping = np.asarray(angular_damping, dtype=float)
        self.horizontal_damping = float(horizontal_damping)
        self.root_body_name = root_body
        self.base_body = -1
        self.velocity = np.zeros(6)
        self.diagnostics: dict = {}

    def setup(self) -> None:
        name = self.root_body_name or self.robot.root_body_name
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.base_body < 0:
            raise ValueError(f"root body {name!r} not found")

    def forces(self) -> None:
        signals = self.signals
        frequency = float(signals.get(f"{self.stroke_id}/frequency_hz", 0.0))
        plate_pitch = float(signals.get(f"{self.gimbal_id}/{self.pitch_joint}", 0.0))
        plate_roll = float(signals.get(f"{self.gimbal_id}/{self.roll_joint}", 0.0))
        data = self.data
        lift = self.lift_coefficient * frequency**2
        direction_body = np.array([-plate_roll, -plate_pitch, 1.0])
        direction_body /= np.linalg.norm(direction_body)
        rotation = data.xmat[self.base_body].reshape(3, 3)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_body, self.velocity, 1
        )
        linear_velocity_world = rotation @ self.velocity[3:]
        force_world = rotation @ (lift * direction_body)
        force_world[:2] -= self.horizontal_damping * linear_velocity_world[:2]
        scale = (frequency / self.moment_reference_frequency) ** 2
        torque_body = (
            np.array(
                [
                    scale * self.roll_moment_per_rad * plate_pitch,
                    -scale * self.pitch_moment_per_rad * plate_roll,
                    0.0,
                ]
            )
            - self.angular_damping * self.velocity[:3]
        )
        torque_world = rotation @ torque_body
        data.xfrc_applied[self.base_body, :3] = force_world
        data.xfrc_applied[self.base_body, 3:] = torque_world
        self.diagnostics = {
            "F_world": force_world.copy(),
            "torque_world": torque_world.copy(),
            "lift": float(lift),
            "frequency": frequency,
            "mode": "cycle_averaged_hover",
        }
        self.publish("F_world", force_world)
        self.publish("torque_world", torque_world)
        self.publish("lift", lift)
        self.publish("frequency_hz", frequency)

    def reset(self) -> None:
        self.diagnostics = {}


__all__ = ["CycleAveragedAero"]
