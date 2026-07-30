"""Typed wing state passed to the pure aerodynamic functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WingInput:
    """One wing's joint and rigid-body state normally read from ``MjData``."""

    psi: float
    psi_dot: float
    theta_raw: float
    theta_dot_raw: float
    pitch_axis_y: float
    R_world_from_wing: np.ndarray
    body_origin_world: np.ndarray
    com_world: np.ndarray
