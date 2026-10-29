"""Time-scheduled set points for any plugin's commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from flapsim.plugins.base import ControllerPlugin
from flapsim.plugins.registry import register


@register
class SetpointSchedule(ControllerPlugin):
    """Publish ``cmd/<target>/<name>`` from ``[[time, value], ...]`` tables.

    With ``interpolate`` the value is linearly interpolated between the table
    times, otherwise it holds the latest entry (step changes).  Before the
    first time nothing is published, so the plugin's own default applies.
    Useful for frequency ramps, gimbal step inputs and scripted manoeuvres.
    """

    name = "schedule"
    example_config = {
        "target": "stroke",
        "schedule": {"frequency_hz": [[0.0, 30.0], [0.5, 40.0]]},
    }

    def __init__(
        self,
        *,
        target: str,
        schedule: Mapping[str, Sequence[Sequence[float]]],
        interpolate: bool = False,
        id: str | None = "schedule",
    ) -> None:
        super().__init__(
            id=id,
            target=target,
            schedule={k: [list(map(float, p)) for p in v] for k, v in schedule.items()},
            interpolate=interpolate,
        )
        if not schedule:
            raise ValueError("schedule must contain at least one command")
        self.target = target
        self.interpolate = bool(interpolate)
        self.tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, points in schedule.items():
            array = np.asarray(points, dtype=float)
            if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
                raise ValueError(
                    f"schedule[{name!r}] must be a list of [time, value] pairs"
                )
            if np.any(np.diff(array[:, 0]) < 0.0):
                raise ValueError(f"schedule[{name!r}] times must be non-decreasing")
            self.tables[name] = (array[:, 0], array[:, 1])

    def value_at(self, name: str, time: float) -> float | None:
        times, values = self.tables[name]
        if time < times[0]:
            return None
        if self.interpolate:
            return float(np.interp(time, times, values))
        index = int(np.searchsorted(times, time, side="right")) - 1
        return float(values[index])

    def control(self) -> None:
        time = float(self.data.time)
        for name in self.tables:
            value = self.value_at(name, time)
            if value is not None:
                self.send_command(self.target, name, value)


__all__ = ["SetpointSchedule"]
