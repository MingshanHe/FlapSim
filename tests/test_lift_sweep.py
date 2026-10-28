"""Regression test: fixed-base mean lift versus flapping frequency.

The values were measured on the ``open-platform`` branch on 2026-10-28 with the
protocol below (fixed base, amplitude 1.5 rad, 0.2 s settle, 10 cycles
averaged), after the pitching-moment correction and with the example robot's
spar-plate wing mass model (see docs/derivations/quasi_steady_model.md).
They follow a frequency-squared law to within 0.3 percent.  Any change to
force signs, wing frames, mass models or wing parameters must either
reproduce them or update them deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from flapsim import SimCore

EXPECTED_MEAN_LIFT_N = {20.0: 0.07402, 30.0: 0.16661, 40.0: 0.29653, 50.0: 0.46376}
AMPLITUDE_RAD = 1.5
SETTLE_S = 0.2
CYCLES = 10


def mean_lift(frequency_hz: float) -> float:
    sim = SimCore("SingleThrustFlapper", fix=True)
    sim.set_thrust_command(frequency=frequency_hz, amplitude=AMPLITUDE_RAD)
    env = sim.environment
    while env.data.time < SETTLE_S:
        env.step()
    samples = []
    end = env.data.time + CYCLES / frequency_hz
    while env.data.time < end:
        env.step()
        samples.append(sim.aero.diagnostics["F_world"][2])
    return float(np.mean(samples))


@pytest.mark.parametrize("frequency_hz", sorted(EXPECTED_MEAN_LIFT_N))
def test_mean_lift_matches_reference(frequency_hz):
    assert mean_lift(frequency_hz) == pytest.approx(
        EXPECTED_MEAN_LIFT_N[frequency_hz], rel=0.01
    )


def test_lift_grows_with_frequency():
    values = [EXPECTED_MEAN_LIFT_N[f] for f in sorted(EXPECTED_MEAN_LIFT_N)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


def test_reference_follows_a_frequency_squared_law():
    base = EXPECTED_MEAN_LIFT_N[30.0] / 30.0**2
    for frequency, lift in EXPECTED_MEAN_LIFT_N.items():
        assert lift == pytest.approx(base * frequency**2, rel=0.01)
