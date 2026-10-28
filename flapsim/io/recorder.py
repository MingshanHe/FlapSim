"""Sample signals into memory and export them."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from flapsim.plugins.base import RecorderPlugin
from flapsim.plugins.registry import register


@register
class MemoryRecorder(RecorderPlugin):
    """Record matching signals every ``every`` seconds (default: every step).

    ``signals`` is a list of shell-style patterns (``"aero/F_world"``,
    ``"wings/*/alpha"``).  Use :meth:`get` for one signal as an array of shape
    ``(samples, ...)``, :meth:`to_dict`, :meth:`to_npz` or :meth:`to_csv`.
    """

    name = "memory"
    example_config = {"signals": ["time", "aero/*"], "every": 1.0e-3}

    def __init__(
        self,
        *,
        signals: Sequence[str] = ("*",),
        every: float | None = None,
        max_samples: int | None = None,
        id: str | None = "recorder",
    ) -> None:
        super().__init__(
            id=id, signals=list(signals), every=every, max_samples=max_samples
        )
        self.patterns = list(signals)
        self.every = None if every is None else float(every)
        self.max_samples = max_samples
        self._next_time = -np.inf
        self._samples: dict[str, list] = {}
        self._times: list[float] = []

    def reset(self) -> None:
        self._next_time = -np.inf
        self._samples = {}
        self._times = []

    def record(self) -> None:
        now = float(self.data.time)
        if self.every is not None and now + 1e-12 < self._next_time:
            return
        if self.max_samples is not None and len(self._times) >= self.max_samples:
            return
        self._next_time = now + (self.every or 0.0)
        names = self.signals.matching(*self.patterns)
        snapshot = self.signals.snapshot(names)
        self._times.append(now)
        for name, value in snapshot.items():
            if isinstance(value, str):
                continue
            self._samples.setdefault(name, []).append(np.asarray(value, dtype=float))
        # keep rows aligned when a signal appears late
        for values in self._samples.values():
            while len(values) < len(self._times):
                values.insert(0, np.full_like(values[0], np.nan))

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._times)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._times, dtype=float)

    def names(self) -> list[str]:
        return sorted(self._samples)

    def get(self, name: str) -> np.ndarray:
        try:
            return np.stack(self._samples[name])
        except KeyError:
            raise KeyError(
                f"signal {name!r} was not recorded; recorded: {self.names()}"
            ) from None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get(name)

    def to_dict(self) -> dict[str, np.ndarray]:
        out = {"time": self.times}
        for name in self.names():
            out[name] = self.get(name)
        return out

    def to_npz(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path, **{k.replace("/", "__"): v for k, v in self.to_dict().items()}
        )
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Write one row per sample; vector signals expand to ``name[i]`` columns."""
        path = Path(path)
        data = self.to_dict()
        columns: list[tuple[str, str, int | None]] = [("time", "time", None)]
        for name, array in data.items():
            if name == "time":
                continue
            if array.ndim == 1:
                columns.append((name, name, None))
            else:
                flat = array.reshape(len(array), -1)
                for index in range(flat.shape[1]):
                    columns.append((f"{name}[{index}]", name, index))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([c[0] for c in columns])
            for row in range(len(self._times)):
                values = []
                for _, name, index in columns:
                    array = data[name]
                    if index is None:
                        values.append(float(array[row]))
                    else:
                        values.append(float(array.reshape(len(array), -1)[row, index]))
                writer.writerow(values)
        return path


__all__ = ["MemoryRecorder"]
