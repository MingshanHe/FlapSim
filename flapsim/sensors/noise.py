"""Measurement conditioning shared by the sensor plugins."""

from __future__ import annotations

import collections
import math

import numpy as np


class Conditioner:
    """Sample-and-hold, first-order low-pass, pure delay, noise, bias, quantization.

    ``update(true_value, time)`` returns the current reading.  A new sample is
    taken whenever ``time`` reaches the next sampling instant (``rate_hz``;
    every call when ``None``).  The sample passes a low-pass with time
    constant ``tau`` (seconds, 0 disables), enters a delay line of
    ``round(delay_s * rate)`` samples, then receives Gaussian noise of
    standard deviation ``noise_std``, a bias drawn uniformly from
    ``[-bias_range, bias_range]`` at every :meth:`reset`, and is quantised to
    multiples of ``resolution`` (0 disables).  All parameters may be scalars
    or arrays matching ``size``.
    """

    def __init__(
        self,
        *,
        size: int = 1,
        rate_hz: float | None = None,
        tau: float = 0.0,
        delay_s: float = 0.0,
        noise_std=0.0,
        bias_range=0.0,
        resolution=0.0,
        seed: int | None = 0,
        wrap_angles: bool = False,
    ) -> None:
        self.size = int(size)
        self.rate_hz = None if rate_hz is None else float(rate_hz)
        self.tau = float(tau)
        self.delay_s = float(delay_s)
        self.noise_std = np.broadcast_to(
            np.asarray(noise_std, dtype=float), (self.size,)
        ).copy()
        self.bias_range = np.broadcast_to(
            np.asarray(bias_range, dtype=float), (self.size,)
        ).copy()
        self.resolution = np.broadcast_to(
            np.asarray(resolution, dtype=float), (self.size,)
        ).copy()
        self.wrap_angles = bool(wrap_angles)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros(self.size)
        self.filtered = np.zeros(self.size)
        self.reading = np.zeros(self.size)
        self._queue: collections.deque = collections.deque()
        self._next_time = -math.inf
        self._last_time: float | None = None
        self._initialised = False

    # -- lifecycle ----------------------------------------------------------

    def reset(self, initial=None, *, reseed: bool = False) -> None:
        if reseed:
            self.rng = np.random.default_rng(self.seed)
        self.bias = self.rng.uniform(-1.0, 1.0, self.size) * self.bias_range
        self._queue.clear()
        self._next_time = -math.inf
        self._last_time = None
        self._initialised = False
        if initial is not None:
            self._prime(np.asarray(initial, dtype=float))

    def _prime(self, value: np.ndarray) -> None:
        self.filtered = value.copy()
        self.reading = value.copy()
        for _ in range(self.delay_steps):
            self._queue.append(value.copy())
        self._initialised = True

    @property
    def delay_steps(self) -> int:
        if self.delay_s <= 0.0:
            return 0
        rate = self.rate_hz if self.rate_hz is not None else self._fallback_rate
        return int(round(self.delay_s * rate))

    _fallback_rate = 20000.0  # used only when rate_hz is None and no dt is known yet

    # -- update -------------------------------------------------------------

    def due(self, time: float) -> bool:
        return time + 1e-12 >= self._next_time

    def update(self, true_value, time: float) -> np.ndarray:
        value = np.asarray(true_value, dtype=float).reshape(self.size)
        if not self.due(time):
            return self.reading
        if self._last_time is None:
            dt = 1.0 / self.rate_hz if self.rate_hz else 0.0
        else:
            dt = time - self._last_time
            if self.rate_hz is None and dt > 0.0:
                self._fallback_rate = 1.0 / dt
        self._last_time = time
        self._next_time = time + (1.0 / self.rate_hz if self.rate_hz else 0.0)
        if not self._initialised:
            self._prime(value)
        if self.tau > 0.0 and dt > 0.0:
            blend = 1.0 - math.exp(-dt / self.tau)
            delta = value - self.filtered
            if self.wrap_angles:
                delta = np.arctan2(np.sin(delta), np.cos(delta))
            self.filtered = self.filtered + blend * delta
        else:
            self.filtered = value.copy()
        self._queue.append(self.filtered.copy())
        while len(self._queue) > self.delay_steps + 1:
            self._queue.popleft()
        delayed = self._queue[0]
        noisy = (
            delayed + self.bias + self.rng.normal(0.0, 1.0, self.size) * self.noise_std
        )
        if np.any(self.resolution > 0.0):
            quantised = np.where(
                self.resolution > 0.0,
                np.round(noisy / np.where(self.resolution > 0.0, self.resolution, 1.0))
                * self.resolution,
                noisy,
            )
            noisy = quantised
        if self.wrap_angles:
            noisy = np.arctan2(np.sin(noisy), np.cos(noisy))
        self.reading = noisy
        return self.reading


__all__ = ["Conditioner"]
