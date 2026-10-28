"""Run a user Python function as a controller."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from flapsim.plugins.base import ControllerPlugin
from flapsim.plugins.registry import register


@register
class CallbackController(ControllerPlugin):
    """Call ``callback(sim, controller) -> {command: value}`` at ``rate_hz``.

    Returned keys are published as commands: ``"stroke/frequency_hz"`` becomes
    ``cmd/stroke/frequency_hz`` (keys already starting with ``cmd/`` are used
    as is).  ``None`` or an empty mapping publishes nothing.  The controller
    keeps a ``state`` dictionary for the callback's own use.
    """

    name = "callback"
    example_config = None  # needs a Python callable

    def __init__(
        self,
        *,
        callback: Callable[..., Mapping[str, Any] | None],
        rate_hz: float | None = None,
        id: str | None = "callback",
    ) -> None:
        super().__init__(
            id=id, rate_hz=rate_hz, callback=getattr(callback, "__name__", "callable")
        )
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback
        self.rate_hz = None if rate_hz is None else float(rate_hz)
        self.state: dict[str, Any] = {}
        self._next_time = -math.inf

    def reset(self) -> None:
        self.state.clear()
        self._next_time = -math.inf

    def control(self) -> None:
        time = float(self.data.time)
        if time + 1e-12 < self._next_time:
            return
        self._next_time = time + (1.0 / self.rate_hz if self.rate_hz else 0.0)
        commands = self.callback(self.sim, self)
        if not commands:
            return
        for key, value in commands.items():
            name = key if key.startswith("cmd/") else f"cmd/{key}"
            self.signals.publish(name, value)


__all__ = ["CallbackController"]
