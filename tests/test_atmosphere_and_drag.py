"""Atmosphere (density, wind, gusts, turbulence) and body drag."""

from __future__ import annotations

import numpy as np
import pytest

from flapsim.aero import Atmosphere, BodyDrag, isa_density
from flapsim.core import Simulation

STROKE = {
    "type": "actuator/sinusoidal_stroke",
    "id": "stroke",
    "joints": ["left_flapping_joint", "right_wing_flapping"],
    "directions": [1, -1],
    "frequency_hz": 30.0,
    "amplitude_rad": 1.5,
}


def test_isa_density():
    assert isa_density(0.0) == pytest.approx(1.225)
    assert isa_density(1000.0) == pytest.approx(1.112, rel=1e-3)
    with pytest.raises(ValueError):
        isa_density(20000.0)


def test_atmosphere_publishes_and_aero_uses_density():
    sim = Simulation("SingleThrustFlapper")
    sim.add(STROKE)
    sim.add(Atmosphere(rho=0.6))
    aero = sim.add("aero/quasi_steady")
    sim.advance(0.01)
    assert sim.signals["atmosphere/rho"] == 0.6
    assert aero.diagnostics["rho"] == 0.6
    np.testing.assert_allclose(sim.signals["atmosphere/wind_world"], 0.0)


def test_steady_wind_changes_the_relative_flow():
    def mean_force(wind):
        sim = Simulation("SingleThrustFlapper")
        sim.add(STROKE)
        sim.add(Atmosphere(wind=wind))
        aero = sim.add("aero/quasi_steady")
        sim.advance(0.2)
        total = np.zeros(3)
        n = 0
        end = sim.time + 10 / 30.0
        while sim.time < end:
            sim.step()
            total += aero.diagnostics["F_world"]
            n += 1
        return total / n

    still = mean_force([0.0, 0.0, 0.0])
    headwind = mean_force([0.0, 0.0, 1.0])  # air moving up past the fixed robot
    assert not np.allclose(still, headwind, atol=1e-4)
    # upward flow through the stroke plane reduces the mean lift of a hovering wing
    assert headwind[2] != pytest.approx(still[2], rel=0.01)


def test_gust_and_turbulence_models():
    gust = Atmosphere(
        wind=[1.0, 0.0, 0.0],
        gust={
            "type": "one_minus_cos",
            "amplitude": 2.0,
            "start": 1.0,
            "duration": 2.0,
            "direction": [0, 1, 0],
        },
    )
    np.testing.assert_allclose(gust.wind_at(0.5), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(gust.wind_at(2.0), [1.0, 2.0, 0.0])  # peak at mid gust
    np.testing.assert_allclose(gust.wind_at(3.5), [1.0, 0.0, 0.0])
    step = Atmosphere(
        gust={"type": "step", "amplitude": 0.5, "start": 1.0, "direction": [1, 0, 0]}
    )
    assert step.wind_at(0.9)[0] == 0.0 and step.wind_at(1.1)[0] == 0.5
    sine = Atmosphere(
        gust={
            "type": "sine",
            "amplitude": 1.0,
            "frequency_hz": 1.0,
            "direction": [0, 0, 1],
        }
    )
    assert sine.wind_at(0.25)[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gust.type"):
        Atmosphere(gust={"type": "tornado"})

    sim = Simulation("SingleThrustFlapper", check=False)
    turb = sim.add(Atmosphere(turbulence={"sigma": 0.5, "tau": 0.1, "seed": 3}))
    sim.advance(0.05)
    speeds = []
    for _ in range(200):
        sim.step()
        speeds.append(sim.signals["atmosphere/wind_speed"])
    assert np.std(speeds) > 0.0
    turb.reset()
    np.testing.assert_allclose(turb.wind_at(0.0), 0.0)


def test_body_drag_matches_the_quadratic_law():
    sim = Simulation("SingleThrustFlapper", check=False)
    sim.add(Atmosphere(rho=1.0, wind=[2.0, 0.0, 0.0]))
    drag = sim.add(BodyDrag(bodies={"link2": {"cd": 1.2, "area": 0.001}}))
    sim.step()
    force = sim.signals["drag/link2/F_world"]
    # fixed body in a 2 m/s wind along +x: F = +1/2 rho cd A v^2 along the wind
    assert force[0] == pytest.approx(0.5 * 1.0 * 1.2 * 0.001 * 4.0, rel=1e-3)
    assert force[1] == pytest.approx(0.0, abs=2e-5) and force[2] == pytest.approx(
        0.0, abs=2e-5
    )
    assert drag.describe()["type"] == "force/body_drag"
    with pytest.raises(ValueError, match="not found"):
        Simulation("SingleThrustFlapper", check=False).add(
            BodyDrag(bodies={"fuselage": {}})
        )
    with pytest.raises(ValueError, match="non-negative"):
        BodyDrag(bodies={"link2": {"cd": -1.0}})
