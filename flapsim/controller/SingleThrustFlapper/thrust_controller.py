"""Frequency/amplitude command source for the flapping joints."""

from __future__ import annotations

import json
from pathlib import Path


class ThrustController:
    def __init__(self, config_path: str | Path) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self._commands: dict[str, dict[str, float]] = {}
        for name, settings in config.items():
            if settings.get("controller") != "trajectory_controller":
                continue
            self._commands[name] = {
                "frequency": float(settings["frequency"]),
                "amplitude": float(settings["amplitude"]),
                "phase": float(settings.get("phase", 0.0)),
                "direction": float(settings.get("direction", 1.0)),
            }
        self._validate()

    def _validate(self) -> None:
        for name, command in self._commands.items():
            if command["frequency"] < 0.0:
                raise ValueError(f"{name} frequency must be non-negative")
            if command["amplitude"] < 0.0:
                raise ValueError(f"{name} amplitude must be non-negative")
            if command["direction"] not in (-1.0, 1.0):
                raise ValueError(f"{name} direction must be -1 or 1")

    @property
    def targets(self) -> dict[str, dict[str, float]]:
        return {name: command.copy() for name, command in self._commands.items()}

    def set_command(
        self,
        *,
        frequency: float | None = None,
        amplitude: float | None = None,
    ) -> None:
        for command in self._commands.values():
            if frequency is not None:
                command["frequency"] = float(frequency)
            if amplitude is not None:
                command["amplitude"] = float(amplitude)
        self._validate()
