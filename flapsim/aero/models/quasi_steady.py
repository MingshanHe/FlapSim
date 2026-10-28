"""Translational quasi-steady normal force (Whitney & Wood 2010 / FlapX)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from flapsim.aero.base import AeroModel, WingLoads, register_model
from flapsim.config import AeroOptions
from flapsim.fluid.aero import wing_loads_from_kinematics


@register_model
class QuasiSteadyModel(AeroModel):
    """Normal force ``F_N`` at the centre of pressure plus rotational damping.

    Diagnostics: ``alpha``, ``C_N``, ``d_cp``, ``M_aero``, ``M_rd``,
    ``chord_CoP``, ``span_CoP``, ``F_N_raw``.  ``options`` overrides the
    engine's :class:`~flapsim.config.AeroOptions` for this model only.
    """

    name = "quasi_steady"

    def __init__(self, *, options=None) -> None:
        super().__init__(options=None if options is None else dict(options))
        self._override = None if options is None else AeroOptions(**dict(options))

    def compute(self, wing, state, kin, flow, options):
        options = self._override or options
        params = wing.params
        if abs(params.rho - flow.rho) > 1e-12:
            params = replace(params, rho=flow.rho)
        result = wing_loads_from_kinematics(params, kin, options)
        point = np.array([0.0, state.side * result["span_CoP"], -result["chord_CoP"]])
        return WingLoads(
            source=self.name,
            force=np.array([result["F_N"], 0.0, 0.0]),
            point=point,
            moment=np.array([0.0, result["M_rd"], 0.0]),
            diagnostics={
                "alpha": result["alpha"],
                "C_N": result["C_N"],
                "d_cp": result["d_cp"],
                "M_aero": result["M_aero"],
                "M_rd": result["M_rd"],
                "chord_CoP": result["chord_CoP"],
                "span_CoP": result["span_CoP"],
                "F_N_raw": result["F_N_raw"],
            },
        )


__all__ = ["QuasiSteadyModel"]
