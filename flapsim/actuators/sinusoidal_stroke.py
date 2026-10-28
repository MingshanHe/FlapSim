"""Kinematic sinusoidal stroke: ``q = direction * A sin(2 pi f t + phase)``."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco

from flapsim.plugins.base import ActuatorPlugin
from flapsim.plugins.registry import register

MODES = ("kinematic", "position_only")


@register
class SinusoidalStroke(ActuatorPlugin):
    """Impose a sinusoidal angle on one or more stroke joints.

    ``kinematic`` mode writes position and velocity (so aerodynamic models see
    the stroke rate); ``position_only`` writes the position with zero velocity,
    which the cycle-averaged floating model uses to avoid injecting energy.
    Commands ``cmd/<id>/frequency_hz`` and ``cmd/<id>/amplitude_rad`` override
    the configured values; the current values are published as
    ``<id>/frequency_hz``, ``<id>/amplitude_rad`` and ``<id>/phase``.
    """

    name = "sinusoidal_stroke"
    provides = ("frequency_hz", "amplitude_rad", "phase")
    example_config = {
        "joints": ["left_flapping_joint", "right_wing_flapping"],
        "directions": [1.0, -1.0],
        "frequency_hz": 35.0,
        "amplitude_rad": 1.5,
    }

    def __init__(
        self,
        *,
        joints: Sequence[str],
        directions: Sequence[float] | None = None,
        frequency_hz: float = 30.0,
        amplitude_rad: float = 1.0,
        phase_rad: float = 0.0,
        mode: str = "kinematic",
        id: str | None = "stroke",
    ) -> None:
        super().__init__(
            id=id,
            joints=list(joints),
            directions=None if directions is None else list(directions),
            frequency_hz=frequency_hz,
            amplitude_rad=amplitude_rad,
            phase_rad=phase_rad,
            mode=mode,
        )
        if not joints:
            raise ValueError("SinusoidalStroke needs at least one joint")
        self.joint_names = list(joints)
        self.directions = (
            [1.0] * len(joints)
            if directions is None
            else [float(d) for d in directions]
        )
        if len(self.directions) != len(self.joint_names):
            raise ValueError("directions must match joints")
        if any(d not in (-1.0, 1.0) for d in self.directions):
            raise ValueError("directions must be +1 or -1")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.phase_rad = float(phase_rad)
        self._frequency = 0.0
        self._amplitude = 0.0
        self._amplitude_limit = math.inf
        self._addresses: list[tuple[int, int]] = []
        self.set_command(
            frequency_hz=frequency_hz, amplitude_rad=amplitude_rad, validate=False
        )

    # -- setup --------------------------------------------------------------

    def setup(self) -> None:
        model = self.model
        self._addresses = []
        limit = math.inf
        for joint_name in self.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0:
                names = [model.joint(i).name for i in range(model.njnt)]
                raise ValueError(
                    f"stroke joint {joint_name!r} not found; joints are {names}"
                )
            if model.jnt_type[joint] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    f"stroke joint {joint_name!r} must be a hinge or slide"
                )
            self._addresses.append(
                (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))
            )
            if model.jnt_limited[joint]:
                lower, upper = model.jnt_range[joint]
                limit = min(limit, abs(float(lower)), abs(float(upper)))
        self._amplitude_limit = limit
        self._validate_amplitude(self._amplitude)

    def _validate_amplitude(self, amplitude: float) -> None:
        if amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if amplitude > self._amplitude_limit + 1e-12:
            raise ValueError(
                f"amplitude {amplitude} rad exceeds the joint limit "
                f"{self._amplitude_limit} rad"
            )

    # -- commands -----------------------------------------------------------

    @property
    def frequency_hz(self) -> float:
        return self._frequency

    @property
    def amplitude_rad(self) -> float:
        return self._amplitude

    def set_command(
        self,
        *,
        frequency_hz: float | None = None,
        amplitude_rad: float | None = None,
        validate: bool = True,
    ) -> None:
        if frequency_hz is not None:
            if frequency_hz < 0.0:
                raise ValueError("frequency must be non-negative")
            self._frequency = float(frequency_hz)
        if amplitude_rad is not None:
            if validate:
                self._validate_amplitude(float(amplitude_rad))
            elif amplitude_rad < 0.0:
                raise ValueError("amplitude must be non-negative")
            self._amplitude = float(amplitude_rad)

    # -- phases -------------------------------------------------------------

    def control(self) -> None:
        frequency = self.command("frequency_hz")
        amplitude = self.command("amplitude_rad")
        if frequency is not None or amplitude is not None:
            self.set_command(frequency_hz=frequency, amplitude_rad=amplitude)

    def actuate(self) -> None:
        self._impose()
        self.publish("frequency_hz", self._frequency)
        self.publish("amplitude_rad", self._amplitude)

    def post_step(self) -> bool:
        self._impose()
        return True

    def _impose(self) -> None:
        data = self.data
        omega = 2.0 * math.pi * self._frequency
        angle = omega * float(data.time) + self.phase_rad
        position = self._amplitude * math.sin(angle)
        velocity = (
            self._amplitude * omega * math.cos(angle)
            if self.mode == "kinematic"
            else 0.0
        )
        for (qpos, dof), direction in zip(
            self._addresses, self.directions, strict=True
        ):
            data.qpos[qpos] = direction * position
            data.qvel[dof] = direction * velocity
        self.signals.publish(f"{self.id}/phase", angle % (2.0 * math.pi))

    def reset(self) -> None:
        self._frequency = float(self.config["frequency_hz"])
        self._amplitude = float(self.config["amplitude_rad"])
        self._impose()


__all__ = ["SinusoidalStroke"]
