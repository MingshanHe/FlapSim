"""Added-mass (non-circulatory) force of a thin plate section."""

from __future__ import annotations

import math

import numpy as np

from flapsim.aero.base import AeroModel, WingLoads, register_model


@register_model
class AddedMass(AeroModel):
    """Reaction of the accelerated fluid on a thin flat-plate wing.

    Per unit span the added mass of a flat plate moving normal to itself is
    ``m_a = rho pi c^2 / 4``.  The normal flow at the mid-chord of section
    ``r`` is ``U_n(r) = U_i1 r + U_i0 + theta_dot d_mid`` with ``d_mid`` the
    mid-chord offset from the hinge; its time derivative (finite difference
    over one step) gives the normal acceleration.  The force along ``+X`` is
    ``integral m_a(r) dU_n/dt dr``, applied at the mid-chord of the
    ``c^2``-weighted span centroid.  The rotational added inertia
    ``rho pi c^4 / 128`` per unit span contributes a moment
    ``-integral I_a theta_ddot dr`` about the span axis (Sedov; Sane &
    Dickinson 2002, eq. 5).  Diagnostics: ``dU_i1``, ``dU_i0``, ``theta_ddot``.
    """

    name = "added_mass"

    def __init__(self, *, include_rotational_inertia: bool = True) -> None:
        super().__init__(include_rotational_inertia=include_rotational_inertia)
        self.include_rotational_inertia = bool(include_rotational_inertia)
        self._previous: dict[str, tuple[float, float, float, float]] = {}

    def reset(self) -> None:
        self._previous.clear()

    def compute(self, wing, state, kin, flow, options):
        params = wing.params
        R, c = params.R, params.c
        key = wing.name
        current = (kin["U_i1"], kin["U_i0"], kin["theta_dot"], flow.time)
        previous = self._previous.get(key)
        self._previous[key] = current
        if previous is None or flow.dt <= 0.0 or current[3] <= previous[3]:
            return WingLoads.zero(self.name)
        dt = current[3] - previous[3]
        d_u1 = (current[0] - previous[0]) / dt
        d_u0 = (current[1] - previous[1]) / dt
        theta_ddot = (current[2] - previous[2]) / dt
        d_mid = params.x_le + (0.5 - params.x0) * c
        m_a = flow.rho * math.pi / 4.0  # times c(r)^2
        # integral c^2 (dU_i1 r + dU_i0 + theta_ddot d_mid) dr
        force_n = (
            m_a
            * c**2
            * (
                d_u1 * R**2 * params.z_cp1
                + (d_u0 + theta_ddot * d_mid) * R * params.z_cp0
            )
        )
        force_n = float(np.clip(force_n, -options.force_limit, options.force_limit))
        moment_y = 0.0
        if self.include_rotational_inertia:
            c4 = getattr(wing.integrals, "c4", 1.0)
            moment_y = -flow.rho * math.pi / 128.0 * theta_ddot * c**4 * R * c4
        span_centroid = (
            R * params.z_cp1 / params.z_cp0 if params.z_cp0 > 0 else kin["span_CoP"]
        )
        point = np.array([0.0, state.side * span_centroid, -d_mid])
        return WingLoads(
            source=self.name,
            force=np.array([force_n, 0.0, 0.0]),
            point=point,
            moment=np.array([0.0, moment_y, 0.0]),
            diagnostics={"dU_i1": d_u1, "dU_i0": d_u0, "theta_ddot": theta_ddot},
        )


__all__ = ["AddedMass"]
