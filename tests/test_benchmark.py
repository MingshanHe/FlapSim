"""Performance guard: steps per second of the shipped fixed-base setup.

Runs only when ``FLAPSIM_BENCH=1``; set ``FLAPSIM_BENCH_FLOOR`` (steps per
second) to turn the report into an assertion on a known machine.
"""

from __future__ import annotations

import os
import time

import pytest

from flapsim.core import Simulation
from flapsim.robot import builtin_robots

pytestmark = pytest.mark.skipif(
    os.environ.get("FLAPSIM_BENCH") != "1",
    reason="set FLAPSIM_BENCH=1 to run the benchmark",
)


def test_steps_per_second():
    sim = Simulation.from_config(
        builtin_robots()["SingleThrustFlapper"].parent / "sim.yaml"
    )
    sim.advance(0.01)
    start = time.perf_counter()
    steps = sim.advance(0.2)
    rate = steps / (time.perf_counter() - start)
    print(f"\n{rate:.0f} steps/s ({steps} steps)")
    floor = os.environ.get("FLAPSIM_BENCH_FLOOR")
    if floor:
        assert rate >= float(floor), f"{rate:.0f} steps/s is below the floor {floor}"
