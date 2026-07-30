"""Typed torso state passed to the pure aerodynamic functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TorsoInput:
    """Torso pose and velocity normally read from ``MjData``."""

    R_world_from_body: np.ndarray
    angular_velocity_world: np.ndarray
    linear_velocity_world: np.ndarray
