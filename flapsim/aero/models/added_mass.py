"""Added-mass (non-circulatory) loads of a thin flat-plate wing."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from flapsim.aero.base import AeroModel, WingLoads, register_model


def feathering_added_inertia(params, rho: float, c4: float = 1.0) -> float:
    """Added rotational inertia of a flat plate about its feathering hinge.

    Per unit span the plate carries ``m_a = rho pi c^2 / 4`` at its mid-chord,
    a distance ``d_mid`` from the hinge, plus the added inertia
    ``rho pi c^4 / 128`` about the mid-chord (Sedov).  Integrated over the
    span with the planform integrals (``c^2 -> z_cp0``, ``c^4 -> c4``).
    """
    R, c = params.R, params.c
    d_mid = params.x_le + (0.5 - params.x0) * c
    translational = rho * math.pi / 4.0 * c**2 * R * params.z_cp0 * d_mid**2
    rotational = rho * math.pi / 128.0 * c**4 * R * c4
    return translational + rotational


@register_model
class AddedMass(AeroModel):
    """Reaction of the accelerated fluid on a thin flat-plate wing.

    Per unit span the added mass of a flat plate moving normal to itself is
    ``m_a = rho pi c^2 / 4``.  The model splits its effect in two:

    * **Explicit force** from the *prescribed* accelerations: the normal flow
      at section ``r`` is ``U_n(r) = U_i1 r + U_i0`` (stroke and body motion);
      its time derivative over one step gives
      ``F = integral m_a dU_n/dt dr`` along the normal, applied at the
      mid-chord of the ``c^2``-weighted span centroid.
    * **Implicit inertia** for the passive feathering: the added rotational
      inertia about the hinge (:func:`feathering_added_inertia`) is added to
      the feathering joint's armature when the model is bound.  Applying it as
      an explicit force from a finite-difference feathering acceleration is
      unstable for light wings, because the added inertia then lags the
      motion by one step; MuJoCo integrates armature implicitly.

    ``include_feathering_inertia=False`` skips the armature change.  The
    armature uses the manifest's air density at bind time.  Diagnostics:
    ``dU_i1``, ``dU_i0``, ``added_inertia``.
    """

    name = "added_mass"

    def __init__(self, *, include_feathering_inertia: bool = True) -> None:
        super().__init__(include_feathering_inertia=include_feathering_inertia)
        self.include_feathering_inertia = bool(include_feathering_inertia)
        self._previous: dict[str, tuple[float, float, float]] = {}
        self.added_inertia: dict[str, float] = {}

    def bind(self, robot) -> None:
        self.added_inertia = {}
        if not self.include_feathering_inertia:
            return
        applied = robot.__dict__.setdefault("_added_feathering_armature", {})
        rho = float(robot.manifest.air_density)
        for wing in robot.wings:
            inertia = feathering_added_inertia(wing.params, rho, wing.integrals.c4)
            self.added_inertia[wing.name] = inertia
            if wing.feathering is None or wing.name in applied:
                continue
            _, dof, _ = wing.feathering
            robot.model.dof_armature[dof] += inertia
            applied[wing.name] = inertia
        mujoco.mj_setConst(robot.model, robot.data)

    def reset(self) -> None:
        self._previous.clear()

    def compute(self, wing, state, kin, flow, options):
        params = wing.params
        R, c = params.R, params.c
        key = wing.name
        current = (kin["U_i1"], kin["U_i0"], flow.time)
        previous = self._previous.get(key)
        self._previous[key] = current
        if previous is None or current[2] <= previous[2]:
            return WingLoads.zero(self.name)
        dt = current[2] - previous[2]
        d_u1 = (current[0] - previous[0]) / dt
        d_u0 = (current[1] - previous[1]) / dt
        m_a = flow.rho * math.pi / 4.0  # times c(r)^2
        # Accelerating the plate along +X pushes the fluid, whose reaction
        # opposes the plate's acceleration.  U_i is the relative flow along
        # +X, i.e. minus the plate velocity, so the reaction is along +dU_i.
        force_n = m_a * c**2 * (d_u1 * R**2 * params.z_cp1 + d_u0 * R * params.z_cp0)
        force_n = float(np.clip(force_n, -options.force_limit, options.force_limit))
        d_mid = params.x_le + (0.5 - params.x0) * c
        span_centroid = (
            R * params.z_cp1 / params.z_cp0 if params.z_cp0 > 0 else kin["span_CoP"]
        )
        point = np.array([0.0, state.side * span_centroid, -d_mid])
        return WingLoads(
            source=self.name,
            force=np.array([force_n, 0.0, 0.0]),
            point=point,
            moment=np.zeros(3),
            diagnostics={
                "dU_i1": d_u1,
                "dU_i0": d_u0,
                "added_inertia": self.added_inertia.get(key, 0.0),
            },
        )


__all__ = ["AddedMass", "feathering_added_inertia"]
