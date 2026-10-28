"""Runtime switches and safety limits for the aerodynamic model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AeroOptions:
    lift_scale: float = 1.0
    moment_scale: float = 1.0
    read_only: bool = False
    force_limit: float = 5.0
    damping_moment_limit: float = 0.05
    angular_velocity_limit: float = 5000.0
    stroke_plane_angle: float = 0.0
    stroke_plane_rate: float = 0.0
    deviation_angle: float = 0.0
    deviation_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "lift_scale",
            "moment_scale",
            "force_limit",
            "damping_moment_limit",
            "angular_velocity_limit",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "stroke_plane_angle",
            "stroke_plane_rate",
            "deviation_angle",
            "deviation_rate",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
