"""Cycle-averaged lift and control moments for floating-base hover studies.

The robot is treated as one rigid body.  Lift ``k f^2`` acts along the body
thrust axis, tilted by the gimbal angles; gimbal angles also produce control
moments, scaled by ``(f / f_ref)^2``; linear and angular damping oppose the
body velocity.  All coefficients can be identified from the per-step model
with :func:`flapsim.aero.identify.identify_cycle_averaged` (``flapsim sweep``)
and loaded from the resulting parameter file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from flapsim.plugins.base import AeroPlugin
from flapsim.plugins.registry import register

PARAMETERS_SCHEMA = "flapsim.cycle_averaged/1"


def load_parameters(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if (
        not isinstance(data, Mapping)
        or data.get("schema", PARAMETERS_SCHEMA) != PARAMETERS_SCHEMA
    ):
        raise ValueError(f"{path} is not a {PARAMETERS_SCHEMA} parameter file")
    return dict(data)


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
        pitch_joint: str | None = None,
        roll_joint: str | None = None,
        stroke: str = "stroke",
        gimbal: str = "gimbal",
        parameters: str | Path | Mapping[str, Any] | None = None,
        reference: Mapping[str, float] | None = None,
        lift_coefficient: float | None = None,
        thrust_tilt_per_rad: Mapping[str, Sequence[float]] | None = None,
        moment_per_rad: Mapping[str, Sequence[float]] | None = None,
        roll_moment_per_rad: float = 0.01055 / 20.0,
        pitch_moment_per_rad: float = 0.01144 / 20.0,
        moment_reference_frequency_hz: float = 42.0,
        linear_damping: Sequence[float] = (0.06, 0.06, 0.0),
        angular_damping: Sequence[float] = (2.0e-4, 2.0e-4, 1.0e-4),
        root_body: str | None = None,
        id: str | None = "aero",
    ) -> None:
        loaded: dict[str, Any] = {}
        if parameters is not None:
            loaded = (
                dict(parameters)
                if isinstance(parameters, Mapping)
                else load_parameters(parameters)
            )
        reference = dict(
            reference
            or loaded.get("reference")
            or {"frequency_hz": 30.0, "lift_n": 0.16661}
        )
        if lift_coefficient is None:
            lift_coefficient = loaded.get("lift_coefficient")
        if lift_coefficient is None:
            lift_coefficient = (
                float(reference["lift_n"]) / float(reference["frequency_hz"]) ** 2
            )
        if (
            moment_reference_frequency_hz is None
            or "moment_reference_frequency_hz" in loaded
        ):
            moment_reference_frequency_hz = float(
                loaded.get(
                    "moment_reference_frequency_hz", moment_reference_frequency_hz
                )
            )
        if "linear_damping" in loaded:
            linear_damping = loaded["linear_damping"]
        if "angular_damping" in loaded:
            angular_damping = loaded["angular_damping"]
        # control matrices: explicit > parameter file > legacy pitch/roll scalars
        tilt = thrust_tilt_per_rad or loaded.get("thrust_tilt_per_rad")
        moments = moment_per_rad or loaded.get("moment_per_rad")
        if tilt is None or moments is None:
            if pitch_joint is None or roll_joint is None:
                raise ValueError(
                    "CycleAveragedAero needs pitch_joint and roll_joint, or "
                    "thrust_tilt_per_rad and moment_per_rad (for example from a "
                    "parameter file)"
                )
            tilt = tilt or {roll_joint: [-1.0, 0.0, 0.0], pitch_joint: [0.0, -1.0, 0.0]}
            moments = moments or {
                pitch_joint: [roll_moment_per_rad, 0.0, 0.0],
                roll_joint: [0.0, -pitch_moment_per_rad, 0.0],
            }
        super().__init__(
            id=id,
            stroke=stroke,
            gimbal=gimbal,
            lift_coefficient=float(lift_coefficient),
            thrust_tilt_per_rad={k: list(map(float, v)) for k, v in tilt.items()},
            moment_per_rad={k: list(map(float, v)) for k, v in moments.items()},
            moment_reference_frequency_hz=float(moment_reference_frequency_hz),
            linear_damping=list(map(float, linear_damping)),
            angular_damping=list(map(float, angular_damping)),
            root_body=root_body,
            parameters=str(parameters) if isinstance(parameters, str | Path) else None,
        )
        self.stroke_id = stroke
        self.gimbal_id = gimbal
        self.lift_coefficient = float(lift_coefficient)
        self.thrust_tilt = {k: np.asarray(v, dtype=float) for k, v in tilt.items()}
        self.moments = {k: np.asarray(v, dtype=float) for k, v in moments.items()}
        self.gimbal_joints = sorted(set(self.thrust_tilt) | set(self.moments))
        self.moment_reference_frequency = float(moment_reference_frequency_hz)
        self.linear_damping = np.asarray(linear_damping, dtype=float)
        self.angular_damping = np.asarray(angular_damping, dtype=float)
        self.root_body_name = root_body
        self.base_body = -1
        self.velocity = np.zeros(6)
        self.diagnostics: dict = {}

    @property
    def hover_frequency_hz(self) -> float:
        """Frequency at which the cycle-averaged lift balances the robot weight."""
        mass = float(np.sum(self.model.body_mass[1:]))
        weight = mass * float(np.linalg.norm(self.model.opt.gravity))
        return float(np.sqrt(weight / self.lift_coefficient))

    def setup(self) -> None:
        name = self.root_body_name or self.robot.root_body_name
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.base_body < 0:
            raise ValueError(f"root body {name!r} not found")

    def forces(self) -> None:
        signals = self.signals
        frequency = float(signals.get(f"{self.stroke_id}/frequency_hz", 0.0))
        angles = {
            joint: float(signals.get(f"{self.gimbal_id}/{joint}", 0.0))
            for joint in self.gimbal_joints
        }
        data = self.data
        lift = self.lift_coefficient * frequency**2
        direction_body = np.array([0.0, 0.0, 1.0])
        for joint, tilt in self.thrust_tilt.items():
            direction_body = direction_body + angles[joint] * tilt
        direction_body /= np.linalg.norm(direction_body)
        rotation = data.xmat[self.base_body].reshape(3, 3)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_body, self.velocity, 1
        )
        v_body = self.velocity[3:]
        w_body = self.velocity[:3]
        force_body = lift * direction_body - self.linear_damping * v_body
        scale = (frequency / self.moment_reference_frequency) ** 2
        torque_body = -self.angular_damping * w_body
        for joint, moment in self.moments.items():
            torque_body = torque_body + scale * angles[joint] * moment
        force_world = rotation @ force_body
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


__all__ = ["PARAMETERS_SCHEMA", "CycleAveragedAero", "load_parameters"]
