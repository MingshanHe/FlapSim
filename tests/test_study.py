"""Parameter studies run serially and in parallel with identical results."""

from __future__ import annotations

import pytest

from flapsim.study import grid_cases, parameter_study


def evaluate(span: float, frequency_hz: float) -> dict[str, float]:
    return {"lift": span * frequency_hz**2, "ratio": frequency_hz / span}


def failing(span: float) -> dict[str, float]:
    if span > 0.05:
        raise ValueError("too long")
    return {"lift": span}


def test_grid_cases_and_serial_study(tmp_path):
    cases = grid_cases({"span": [0.05, 0.07], "frequency_hz": [20.0, 30.0]})
    assert len(cases) == 4 and cases[0] == {"span": 0.05, "frequency_hz": 20.0}
    result = parameter_study(
        evaluate, {"span": [0.05, 0.07], "frequency_hz": [20.0, 30.0]}
    )
    assert len(result) == 4 and result.errors == [None] * 4
    case, metrics = result.best("lift")
    assert case == {"span": 0.07, "frequency_hz": 30.0} and metrics[
        "lift"
    ] == pytest.approx(63.0)
    case, _ = result.best("ratio", maximize=False)
    assert case == {"span": 0.07, "frequency_hz": 20.0}
    path = result.to_csv(tmp_path / "study.csv")
    header = path.read_text().splitlines()[0]
    assert header == "span,frequency_hz,lift,ratio,error"


def test_parallel_matches_serial_and_captures_errors():
    grid = {"span": [0.04, 0.05, 0.06, 0.07]}
    serial = parameter_study(failing, grid)
    parallel = parameter_study(failing, grid, workers=2)
    assert serial.metrics == parallel.metrics
    assert serial.metrics[:2] == [{"lift": 0.04}, {"lift": 0.05}] and serial.metrics[
        2:
    ] == [None, None]
    assert "too long" in parallel.errors[2]
    with pytest.raises(ValueError, match="either grid or cases"):
        parameter_study(failing)
    explicit = parameter_study(evaluate, cases=[{"span": 0.1, "frequency_hz": 10.0}])
    assert explicit.metrics[0]["lift"] == pytest.approx(10.0)
