"""Position servo: PD hold with a rate-limited, first-order lagged target."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from flapsim.plugins.registry import register

from .pd_position import PDPosition


@register
class Servo(PDPosition):
    """A :class:`PDPosition` whose targets move towards the commanded values
    at most ``rate_limit_rad_s`` fast through a first-order lag of
    ``time_constant_s`` (0 for none), like a hobby servo.  Publishes
    ``<id>/<joint>`` (current target) and ``<id>/<joint>/desired``.
    """

    name = "servo"
    example_config = {
        "joints": {"pitch_joint": {"kp": 0.5, "kd": 0.05, "max_torque": 0.5}},
        "rate_limit_rad_s": 5.0,
    }

    def __init__(
        self,
        *,
        joints: Mapping[str, Mapping[str, float] | None],
        rate_limit_rad_s: float = 10.0,
        time_constant_s: float = 0.0,
        mode: str = "torque",
        id: str | None = "servo",
    ) -> None:
        super().__init__(joints=joints, mode=mode, id=id)
        self.config.update(
            rate_limit_rad_s=rate_limit_rad_s, time_constant_s=time_constant_s
        )
        if rate_limit_rad_s <= 0.0 or time_constant_s < 0.0:
            raise ValueError(
                "rate_limit_rad_s must be positive and time_constant_s non-negative"
            )
        self.rate_limit = float(rate_limit_rad_s)
        self.time_constant = float(time_constant_s)
        self.desired: dict[str, float] = {
            name: s["target"] for name, s in self.settings.items()
        }

    def setup(self) -> None:
        super().setup()
        self.desired = {name: s["target"] for name, s in self.settings.items()}

    def reset(self) -> None:
        super().reset()
        self.desired = {name: s["target"] for name, s in self.settings.items()}

    def set_desired(self, joint_name: str, value: float) -> None:
        if joint_name not in self.settings:
            raise KeyError(f"unknown joint {joint_name!r}")
        lower, upper = self._ranges.get(joint_name, (-np.inf, np.inf))
        self.desired[joint_name] = float(np.clip(value, lower, upper))

    def set_target(self, joint_name: str, value: float, *, clip: bool = False) -> float:
        """Jump the servo (and its desired value) to ``value`` immediately."""
        result = super().set_target(joint_name, value, clip=clip)
        self.desired[joint_name] = result
        return result

    def control(self) -> None:
        dt = float(self.model.opt.timestep)
        for joint_name in self.settings:
            command = self.command(joint_name)
            if command is not None:
                self.set_desired(joint_name, command)
            current = self.settings[joint_name]["target"]
            desired = self.desired[joint_name]
            if self.time_constant > 0.0:
                step = (desired - current) * (1.0 - np.exp(-dt / self.time_constant))
            else:
                step = desired - current
            step = float(np.clip(step, -self.rate_limit * dt, self.rate_limit * dt))
            self.settings[joint_name]["target"] = current + step
            self.publish(f"{joint_name}/desired", desired)


__all__ = ["Servo"]
