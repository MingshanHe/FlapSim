"""Air density, steady wind, gusts and turbulence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from flapsim.plugins.base import AtmospherePlugin
from flapsim.plugins.registry import register

GUST_TYPES = ("one_minus_cos", "sine", "step")


def isa_density(altitude_m: float) -> float:
    """International Standard Atmosphere density in the troposphere [kg/m^3]."""
    if altitude_m > 11000.0:
        raise ValueError("ISA density is implemented for the troposphere (<= 11 km)")
    return 1.225 * (1.0 - 2.25577e-5 * altitude_m) ** 4.25588


@register
class Atmosphere(AtmospherePlugin):
    """Publish ``atmosphere/rho`` and ``atmosphere/wind_world`` every step.

    ``rho`` defaults to the ISA density at ``altitude_m``.  ``wind`` is a
    steady world-frame vector; ``gust`` adds one of ``one_minus_cos``
    (``amplitude, start, duration, direction``), ``sine`` (``amplitude,
    frequency_hz, direction, phase``) or ``step`` (``amplitude, start,
    direction``); ``turbulence`` adds an Ornstein-Uhlenbeck process per axis
    (``sigma`` m/s, ``tau`` s, ``seed``).  Add this plugin before the aero
    plugins that should see the wind.
    """

    name = "standard"
    provides = ("rho", "wind_world", "wind_speed")
    example_config = {"wind": [0.5, 0.0, 0.0]}

    def __init__(
        self,
        *,
        rho: float | None = None,
        altitude_m: float = 0.0,
        wind: Sequence[float] = (0.0, 0.0, 0.0),
        gust: Mapping[str, object] | None = None,
        turbulence: Mapping[str, float] | None = None,
        id: str | None = "atmosphere",
    ) -> None:
        super().__init__(
            id=id,
            rho=rho,
            altitude_m=altitude_m,
            wind=list(wind),
            gust=None if gust is None else dict(gust),
            turbulence=None if turbulence is None else dict(turbulence),
        )
        self.rho = float(rho) if rho is not None else isa_density(float(altitude_m))
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        self.wind = np.asarray(wind, dtype=float)
        if self.wind.shape != (3,):
            raise ValueError("wind must be a 3-vector")
        self.gust = None if gust is None else dict(gust)
        if self.gust is not None:
            kind = self.gust.get("type")
            if kind not in GUST_TYPES:
                raise ValueError(f"gust.type must be one of {GUST_TYPES}")
            direction = np.asarray(
                self.gust.get("direction", (1.0, 0.0, 0.0)), dtype=float
            )
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                raise ValueError("gust.direction must be non-zero")
            self._gust_direction = direction / norm
        self.turbulence = None if turbulence is None else dict(turbulence)
        if self.turbulence is not None:
            if float(self.turbulence.get("tau", 1.0)) <= 0.0:
                raise ValueError("turbulence.tau must be positive")
        self._rng = np.random.default_rng(
            None if turbulence is None else int(self.turbulence.get("seed", 0))
        )
        self._turbulent = np.zeros(3)
        self._last_time = None

    def reset(self) -> None:
        self._turbulent[:] = 0.0
        self._last_time = None
        if self.turbulence is not None:
            self._rng = np.random.default_rng(int(self.turbulence.get("seed", 0)))

    def gust_at(self, time: float) -> np.ndarray:
        if self.gust is None:
            return np.zeros(3)
        g = self.gust
        amplitude = float(g.get("amplitude", 0.0))
        kind = g["type"]
        if kind == "one_minus_cos":
            start = float(g.get("start", 0.0))
            duration = float(g.get("duration", 1.0))
            if time < start or time > start + duration or duration <= 0.0:
                return np.zeros(3)
            value = (
                0.5
                * amplitude
                * (1.0 - math.cos(2.0 * math.pi * (time - start) / duration))
            )
        elif kind == "sine":
            value = amplitude * math.sin(
                2.0 * math.pi * float(g.get("frequency_hz", 1.0)) * time
                + float(g.get("phase", 0.0))
            )
        else:  # step
            value = amplitude if time >= float(g.get("start", 0.0)) else 0.0
        return value * self._gust_direction

    def _advance_turbulence(self, time: float) -> None:
        if self.turbulence is None:
            return
        if self._last_time is None:
            self._last_time = time
            return
        dt = time - self._last_time
        self._last_time = time
        if dt <= 0.0:
            return
        sigma = float(self.turbulence.get("sigma", 0.0))
        tau = float(self.turbulence.get("tau", 1.0))
        noise = self._rng.standard_normal(3)
        self._turbulent += (
            -self._turbulent * dt / tau + sigma * math.sqrt(2.0 * dt / tau) * noise
        )

    def wind_at(self, time: float) -> np.ndarray:
        return self.wind + self.gust_at(time) + self._turbulent

    def pre_control(self) -> None:
        time = float(self.data.time)
        self._advance_turbulence(time)
        wind = self.wind_at(time)
        self.publish("rho", self.rho)
        self.publish("wind_world", wind)
        self.publish("wind_speed", float(np.linalg.norm(wind)))


__all__ = ["Atmosphere", "isa_density"]
