"""Parameter studies: evaluate a function over a grid of cases, in parallel."""

from __future__ import annotations

import csv
import itertools
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def grid_cases(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of parameter values as a list of case dictionaries."""
    keys = list(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _run_case(args):
    evaluate, case = args
    try:
        return evaluate(**case), None
    except Exception:  # noqa: BLE001 - reported per case
        return None, traceback.format_exc()


@dataclass
class StudyResult:
    cases: list[dict[str, Any]]
    metrics: list[dict[str, Any] | None]
    errors: list[str | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def succeeded(self) -> list[int]:
        return [i for i, m in enumerate(self.metrics) if m is not None]

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**case, **(metrics or {}), "error": error}
            for case, metrics, error in zip(
                self.cases, self.metrics, self.errors, strict=True
            )
        ]

    def best(
        self, metric: str, *, maximize: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidates = [
            (self.metrics[i][metric], i)
            for i in self.succeeded
            if metric in self.metrics[i]
        ]
        if not candidates:
            raise ValueError(f"no successful case reports metric {metric!r}")
        value, index = max(candidates) if maximize else min(candidates)
        return self.cases[index], self.metrics[index]

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        rows = self.rows()
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path


def parameter_study(
    evaluate: Callable[..., Mapping[str, Any]],
    grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> StudyResult:
    """Call ``evaluate(**case)`` for every case of ``grid`` (or ``cases``).

    ``evaluate`` must be importable (module level) when ``workers > 1`` since
    cases run in separate processes.  Exceptions are captured per case.
    """
    if (grid is None) == (cases is None):
        raise ValueError("give either grid or cases")
    case_list = [dict(c) for c in cases] if cases is not None else grid_cases(grid)
    metrics: list[dict[str, Any] | None] = []
    errors: list[str | None] = []
    jobs = [(evaluate, case) for case in case_list]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, (result, error) in enumerate(pool.map(_run_case, jobs)):
                metrics.append(None if result is None else dict(result))
                errors.append(error)
                if progress:
                    progress(index + 1, len(jobs))
    else:
        for index, job in enumerate(jobs):
            result, error = _run_case(job)
            metrics.append(None if result is None else dict(result))
            errors.append(error)
            if progress:
                progress(index + 1, len(jobs))
    return StudyResult(case_list, metrics, errors)


__all__ = ["StudyResult", "grid_cases", "parameter_study"]
