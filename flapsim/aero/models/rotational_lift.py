"""Rotational circulation force (Sane & Dickinson 2002)."""

from __future__ import annotations

import math

import numpy as np

from flapsim.aero.base import AeroModel, WingLoads, register_model


@register_model
class RotationalLift(AeroModel):
    """``F_rot = C_rot rho theta_dot * integral c(r)^2 U(r) dr`` along the normal.

    ``C_rot = pi (0.75 - x0)`` by default, with ``x0`` the hinge position as a
    fraction of the chord (Sane & Dickinson 2002, eq. 4).  Positive when the
    wing pitches up relative to the oncoming flow.  The force acts at the
    ``point_fraction`` of the chord behind the leading edge (default half
    chord) at the spanwise centre of pressure.  Diagnostics: ``C_rot``,
    ``circulation_integral``.
    """

    name = "rotational_lift"

    def __init__(
        self, *, C_rot: float | None = None, point_fraction: float = 0.5
    ) -> None:
        super().__init__(C_rot=C_rot, point_fraction=point_fraction)
        self.C_rot_override = C_rot
        self.point_fraction = float(point_fraction)

    def compute(self, wing, state, kin, flow, options):
        params = wing.params
        c_rot = (
            self.C_rot_override
            if self.C_rot_override is not None
            else math.pi * (0.75 - params.x0)
        )
        R, c = params.R, params.c
        # integral c(r)^2 U_i(r) dr with U_i affine in r
        integral = c**2 * (
            kin["U_i1"] * R**2 * params.z_cp1 + kin["U_i0"] * R * params.z_cp0
        )
        force_n = c_rot * flow.rho * kin["theta_dot"] * integral
        force_n = float(np.clip(force_n, -options.force_limit, options.force_limit))
        chord_offset = params.x_le + (self.point_fraction - params.x0) * c
        point = np.array([0.0, state.side * kin["span_CoP"], -chord_offset])
        return WingLoads(
            source=self.name,
            force=np.array([force_n, 0.0, 0.0]),
            point=point,
            moment=np.zeros(3),
            diagnostics={
                "C_rot": float(c_rot),
                "circulation_integral": float(integral),
            },
        )


__all__ = ["RotationalLift"]
