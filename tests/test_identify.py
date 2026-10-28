"""Cycle-averaged parameter identification and the parameter file round trip."""

from __future__ import annotations

import numpy as np
import pytest

from flapsim.aero import CycleAveragedAero, load_parameters
from flapsim.aero.identify import identify_cycle_averaged, lift_sweep, write_parameters
from flapsim.core import Simulation
from flapsim.robot import builtin_robots

ROBOT_DIR = builtin_robots()["SingleThrustFlapper"].parent


def make_sim():
    return Simulation.from_config(ROBOT_DIR / "sim.yaml")


def test_lift_sweep_recovers_the_frequency_squared_law():
    sweep = lift_sweep(make_sim, [30.0, 40.0], settle=0.2, cycles=5)
    assert sweep.lift_coefficient == pytest.approx(0.16661 / 900.0, rel=0.02)
    assert sweep.fit_rms_relative < 0.01
    assert sweep.amplitude_rad == 1.5


def test_identification_writes_a_loadable_parameter_file(tmp_path):
    result = identify_cycle_averaged(
        make_sim,
        frequencies=[30.0, 40.0],
        reference_frequency_hz=40.0,
        gimbal_angle_rad=0.1,
        include_damping=False,
        settle=0.15,
        cycles=4,
    )
    p = result.parameters
    assert p["schema"] == "flapsim.cycle_averaged/1"
    assert 38.0 < p["hover_frequency_hz"] < 42.0
    assert set(p["moment_per_rad"]) == {"pitch_joint", "roll_joint"}
    # deflecting a gimbal joint tilts the thrust and produces a control moment
    for joint in ("pitch_joint", "roll_joint"):
        assert np.linalg.norm(p["thrust_tilt_per_rad"][joint][:2]) > 0.1
        assert np.linalg.norm(p["moment_per_rad"][joint]) > 1e-5
    path = write_parameters(p, tmp_path / "params.yaml")
    loaded = load_parameters(path)
    assert loaded["lift_coefficient"] == pytest.approx(p["lift_coefficient"])

    sim = Simulation("SingleThrustFlapper", base="free")
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint", "right_wing_flapping"],
            "directions": [1, -1],
            "frequency_hz": p["hover_frequency_hz"],
            "amplitude_rad": 1.5,
            "mode": "position_only",
        }
    )
    sim.add(
        {
            "type": "actuator/pd_position",
            "id": "gimbal",
            "mode": "kinematic",
            "joints": {
                "pitch_joint": {"role": "pitch"},
                "roll_joint": {"role": "roll"},
            },
        }
    )
    aero = sim.add(CycleAveragedAero(parameters=path))
    assert aero.hover_frequency_hz == pytest.approx(p["hover_frequency_hz"], rel=0.01)
    sim.advance(0.05)
    # at the identified hover frequency the lift balances the weight
    weight = float(np.sum(sim.model.body_mass[1:])) * 9.81
    assert aero.diagnostics["lift"] == pytest.approx(weight, rel=0.02)


def test_cycle_averaged_requires_joints_or_matrices():
    with pytest.raises(ValueError, match="needs pitch_joint and roll_joint"):
        CycleAveragedAero()
    legacy = CycleAveragedAero(pitch_joint="p", roll_joint="r")
    assert set(legacy.gimbal_joints) == {"p", "r"}
