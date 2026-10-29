"""Downward (or any-direction) range finder such as a time-of-flight sensor."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from flapsim.plugins.base import SensorPlugin
from flapsim.plugins.registry import register

from .noise import Conditioner

METHODS = ("plane", "ray")


@register
class RangeFinder(SensorPlugin):
    """Slant range along a body-fixed direction, delayed and noisy.

    ``plane`` intersects the ray with the ground plane ``z = ground_height``
    (the slant range is ``height / cos(tilt)`` with ``cos`` clamped at
    ``min_cos``); ``ray`` casts a MuJoCo ray against all geoms.  Defaults
    reproduce the hover-policy TOF: 20 ms delay, 5 mm noise, 4 m range, and a
    range rate from differenced readings through a 50 ms low-pass.  Publishes
    ``<id>/range`` and ``<id>/range_rate``.
    """

    name = "rangefinder"
    provides = ("range", "range_rate")
    example_config = {"rate_hz": 500.0}

    def __init__(
        self,
        *,
        body: str | None = None,
        direction=(0.0, 0.0, -1.0),
        rate_hz: float = 500.0,
        delay_s: float = 0.02,
        noise: float = 0.005,
        rate_tau: float = 0.05,
        max_range: float = 4.0,
        method: str = "plane",
        ground_height: float = 0.0,
        min_cos: float = 0.2,
        seed: int = 0,
        id: str | None = "tof",
    ) -> None:
        super().__init__(
            id=id,
            body=body,
            direction=list(direction),
            rate_hz=rate_hz,
            delay_s=delay_s,
            noise=noise,
            rate_tau=rate_tau,
            max_range=max_range,
            method=method,
            ground_height=ground_height,
            min_cos=min_cos,
            seed=seed,
        )
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.body_name = body
        self.direction = np.asarray(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.rate_hz = float(rate_hz)
        self.max_range = float(max_range)
        self.method = method
        self.ground_height = float(ground_height)
        self.min_cos = float(min_cos)
        self.rate_tau = float(rate_tau)
        self.conditioner = Conditioner(
            size=1, rate_hz=rate_hz, delay_s=delay_s, noise_std=noise, seed=seed
        )
        self.body_id = -1
        self._range_rate = 0.0
        self._last_reading: float | None = None
        self._last_time: float | None = None
        self._next_time = -math.inf
        self._geomid = np.zeros(1, dtype=np.int32)

    def setup(self) -> None:
        name = self.body_name or self.robot.root_body_name
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.body_id < 0:
            raise ValueError(f"range finder body {name!r} not found")
        self.reset()

    def true_range(self) -> float:
        data = self.data
        origin = data.xpos[self.body_id]
        direction_world = data.xmat[self.body_id].reshape(3, 3) @ self.direction
        if self.method == "ray":
            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                direction_world,
                None,
                1,
                self.body_id,
                self._geomid,
            )
            if distance < 0.0:
                return self.max_range
            return float(min(distance, self.max_range))
        cos_tilt = -float(direction_world[2])
        height = float(origin[2]) - self.ground_height
        if cos_tilt <= 0.0:
            return self.max_range
        return float(min(height / max(cos_tilt, self.min_cos), self.max_range))

    def reset(self) -> None:
        self._range_rate = 0.0
        self._last_reading = None
        self._last_time = None
        self._next_time = -math.inf
        self.conditioner.reset(np.array([self.true_range()]), reseed=True)

    def pre_control(self) -> None:
        time = float(self.data.time)
        if time + 1e-12 < self._next_time:
            return
        self._next_time = time + 1.0 / self.rate_hz
        reading = float(self.conditioner.update(np.array([self.true_range()]), time)[0])
        if (
            self._last_reading is not None
            and self._last_time is not None
            and time > self._last_time
        ):
            raw_rate = (reading - self._last_reading) / (time - self._last_time)
            blend = (
                1.0 - math.exp(-(time - self._last_time) / self.rate_tau)
                if self.rate_tau > 0
                else 1.0
            )
            self._range_rate += blend * (raw_rate - self._range_rate)
        self._last_reading = reading
        self._last_time = time
        self.publish("range", reading)
        self.publish("range_rate", self._range_rate)


__all__ = ["RangeFinder"]
