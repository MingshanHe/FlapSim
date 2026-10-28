"""Empirical wake-capture force after stroke reversal (Dickinson et al. 1999)."""

from __future__ import annotations

import math

import numpy as np

from flapsim.aero.base import AeroModel, WingLoads, register_model


@register_model
class WakeCapture(AeroModel):
    """Transient normal force when the wing meets the wake it shed before
    reversal.

    Empirical: when the relative flow ``U_i`` at the span centre of pressure
    changes sign, the velocity magnitude just before reversal ``U_rev`` is
    stored and a force ``C_wc 1/2 rho S U_rev^2 exp(-(t - t_rev) / tau)`` is
    applied along the new flow direction, with ``tau = tau_chords c / U_rev``.
    ``C_wc`` and ``tau_chords`` are tuning constants (defaults 1.0 and 0.5);
    there is no first-principles value.  Diagnostics: ``U_rev``, ``elapsed``.
    """

    name = "wake_capture"

    def __init__(
        self,
        *,
        C_wc: float = 1.0,
        tau_chords: float = 0.5,
        point_fraction: float = 0.5,
        min_speed: float = 0.05,
    ) -> None:
        super().__init__(
            C_wc=C_wc,
            tau_chords=tau_chords,
            point_fraction=point_fraction,
            min_speed=min_speed,
        )
        self.C_wc = float(C_wc)
        self.tau_chords = float(tau_chords)
        self.point_fraction = float(point_fraction)
        self.min_speed = float(min_speed)
        self._last_u: dict[str, float] = {}
        self._peak: dict[str, float] = {}
        self._events: dict[str, tuple[float, float, float]] = {}

    def reset(self) -> None:
        self._last_u.clear()
        self._peak.clear()
        self._events.clear()

    def compute(self, wing, state, kin, flow, options):
        params = wing.params
        key = wing.name
        u = float(kin["U_i_at_CoP"])
        last = self._last_u.get(key)
        self._last_u[key] = u
        self._peak[key] = max(self._peak.get(key, 0.0), abs(u))
        if (
            last is not None
            and u * last < 0.0
            and self._peak.get(key, 0.0) > self.min_speed
        ):
            self._events[key] = (flow.time, self._peak[key], math.copysign(1.0, u))
            self._peak[key] = 0.0
        event = self._events.get(key)
        if event is None:
            return WingLoads.zero(self.name)
        t_rev, u_rev, direction = event
        tau = self.tau_chords * params.c / max(u_rev, 1e-9)
        elapsed = flow.time - t_rev
        if elapsed > 8.0 * tau:
            del self._events[key]
            return WingLoads.zero(self.name)
        area = params.R * params.c * params.r00
        force_n = (
            direction
            * self.C_wc
            * 0.5
            * flow.rho
            * area
            * u_rev**2
            * math.exp(-elapsed / tau)
        )
        force_n = float(np.clip(force_n, -options.force_limit, options.force_limit))
        chord_offset = params.x_le + (self.point_fraction - params.x0) * params.c
        point = np.array([0.0, state.side * kin["span_CoP"], -chord_offset])
        return WingLoads(
            source=self.name,
            force=np.array([force_n, 0.0, 0.0]),
            point=point,
            moment=np.zeros(3),
            diagnostics={"U_rev": u_rev, "elapsed": elapsed},
        )


__all__ = ["WakeCapture"]
