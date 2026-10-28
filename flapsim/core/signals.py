"""Named signals exchanged between plugins.

A signal is a float or a NumPy array published under a path such as
``wings/left/alpha`` or ``cmd/stroke/frequency_hz``.  Plugins publish what
they compute and read what they need, so they compose without importing each
other.  Commands follow the convention ``cmd/<plugin id>/<name>``.
"""

from __future__ import annotations

from collections.abc import Iterator
from fnmatch import fnmatchcase
from typing import Any

import numpy as np


class SignalBus:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def publish(self, name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            self._values[name] = value.copy()
        elif isinstance(value, bool | int | float | np.floating | np.integer):
            self._values[name] = float(value)
        elif isinstance(value, str):
            self._values[name] = value
        else:
            self._values[name] = np.asarray(value, dtype=float)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def __getitem__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(
                f"no signal {name!r}; published signals: {sorted(self._values)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._values))

    def __len__(self) -> int:
        return len(self._values)

    def names(self) -> list[str]:
        return sorted(self._values)

    def matching(self, *patterns: str) -> list[str]:
        """Signal names matching any shell-style pattern (``wings/*/F_N``)."""
        return [
            name
            for name in sorted(self._values)
            if any(fnmatchcase(name, pattern) for pattern in patterns)
        ]

    def snapshot(self, names=None) -> dict[str, Any]:
        selected = (
            self._values if names is None else {n: self._values[n] for n in names}
        )
        return {
            name: (value.copy() if isinstance(value, np.ndarray) else value)
            for name, value in selected.items()
        }

    def clear(self) -> None:
        self._values.clear()


__all__ = ["SignalBus"]
