"""Legacy flat aerodynamic parameter object used by the tutorials."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WingParams:
    R: float = 0.0  # wing length [m]
    c: float = 0.0  # mean chord [m]
    # Non-dimensional radius of moments
    r33: float = 0.0
    r22: float = 0.0
    r11: float = 0.0
    r00: float = 0.0
    # 压力中心相关的几何积分
    z_cp2: float = 0.0
    z_cp1: float = 0.0
    z_cp0: float = 0.0
    # rotation damping coefficient
    z_rd: float = 0.0
    # constant parameters
    rho: float = 0.0  # air density [kg/m³]
    d0: float = 0.0  # half shoulder width [m]
    ds: float = 0.0  # stroke plane offset [m]
    # hinge geometry: distance from the feathering axis to the leading edge
    # [m] when the plate starts beyond the hinge, and the chordwise fraction
    # of the local chord at which the hinge sits when it lies inside the plate
    x_le: float = 0.0
    x0: float = 0.0
