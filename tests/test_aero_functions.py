"""Unit tests for the pure quasi-steady aerodynamic functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flapsim.config import AeroOptions, TorsoInput, WingInput, WingParams
from flapsim.fluid import (
    aero_step,
    center_of_pressure_fraction,
    load_wing_parameters,
    normal_force_coefficient,
    save_wing_parameters,
    wing_aero,
    world_wrench,
)


@pytest.fixture(scope="module")
def params() -> WingParams:
    return load_wing_parameters()


def test_normal_force_coefficient_is_odd_and_zero_at_rest():
    assert normal_force_coefficient(0.0) == pytest.approx(0.0)
    for alpha in np.linspace(0.05, math.pi / 2, 12):
        assert normal_force_coefficient(-alpha) == pytest.approx(
            -normal_force_coefficient(alpha)
        )


def test_normal_force_coefficient_at_45_degrees():
    expected = (1.8 + 1.95) * math.sqrt(0.5)
    assert normal_force_coefficient(math.pi / 4) == pytest.approx(expected)


def test_center_of_pressure_moves_aft_with_angle_of_attack():
    alphas = np.linspace(0.0, math.pi / 2, 50)
    values = [center_of_pressure_fraction(a) for a in alphas]
    assert values[0] == pytest.approx(0.46 - 0.332 - 0.037 - 0.013)
    assert values[-1] == pytest.approx(0.46)
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))


def _still(params, side=1, psi_dot=0.0, theta=0.3, options=None):
    return wing_aero(
        params,
        side,
        psi=0.2,
        psi_dot=psi_dot,
        theta=theta,
        theta_dot=0.0,
        linear_velocity_body=np.zeros(3),
        angular_velocity_body=np.zeros(3),
        options=options,
    )


def test_wing_aero_is_zero_without_motion(params):
    result = _still(params, psi_dot=0.0)
    assert result["F_N"] == pytest.approx(0.0)
    assert result["M_aero"] == pytest.approx(0.0)
    assert result["M_rd"] == pytest.approx(0.0)


def test_wing_aero_force_scales_with_velocity_squared(params):
    options = AeroOptions(force_limit=1e9)
    slow = _still(params, psi_dot=10.0, options=options)["F_N"]
    fast = _still(params, psi_dot=20.0, options=options)["F_N"]
    assert slow != 0.0
    assert fast == pytest.approx(4.0 * slow, rel=1e-9)


def test_wing_aero_lift_scale_and_force_limit(params):
    base = _still(params, psi_dot=50.0, options=AeroOptions(force_limit=1e9))
    scaled = _still(
        params, psi_dot=50.0, options=AeroOptions(force_limit=1e9, lift_scale=0.5)
    )
    assert scaled["F_N"] == pytest.approx(0.5 * base["F_N"])
    limited = _still(params, psi_dot=50.0, options=AeroOptions(force_limit=1e-4))
    assert abs(limited["F_N"]) == pytest.approx(1e-4)
    assert limited["F_N_raw"] == pytest.approx(base["F_N"])


def test_pitching_moment_keeps_centre_of_pressure_on_the_plate(params):
    """M_aero = -d_cp c F_N: the CoP stays behind the hinge for either flow sign."""
    options = AeroOptions(force_limit=1e9)
    forward = _still(params, psi_dot=40.0, theta=0.0, options=options)
    backward = _still(params, psi_dot=-40.0, theta=0.0, options=options)
    assert forward["F_N"] > 0.0 > backward["F_N"]
    assert forward["chord_CoP"] > 0.0 and backward["chord_CoP"] > 0.0
    assert forward["M_aero"] < 0.0 < backward["M_aero"]
    assert forward["M_aero"] == pytest.approx(-forward["chord_CoP"] * forward["F_N"])
    assert backward["M_aero"] == pytest.approx(-backward["chord_CoP"] * backward["F_N"])
    # at theta = 0 both directions see |alpha| = pi/2, so the CoP agrees
    assert forward["chord_CoP"] == pytest.approx(backward["chord_CoP"], rel=1e-9)
    pitched = _still(params, psi_dot=40.0, theta=0.3, options=options)
    assert pitched["M_aero"] == pytest.approx(-pitched["chord_CoP"] * pitched["F_N"])


def test_hinge_geometry_shifts_the_centre_of_pressure(params):
    """x_le adds a rigid offset; x0 moves the hinge into the chord."""
    from dataclasses import replace

    options = AeroOptions(force_limit=1e9)
    base = _still(params, psi_dot=40.0, theta=0.2, options=options)
    offset = _still(
        replace(params, x_le=0.0125), psi_dot=40.0, theta=0.2, options=options
    )
    assert offset["F_N"] == pytest.approx(base["F_N"])
    assert offset["chord_CoP"] == pytest.approx(base["chord_CoP"] + 0.0125)
    inside = _still(replace(params, x0=0.25), psi_dot=40.0, theta=0.2, options=options)
    assert inside["chord_CoP"] < base["chord_CoP"]
    # with the hinge far aft the pressure centre lies ahead of it: unstable
    far_aft = _still(replace(params, x0=0.9), psi_dot=40.0, theta=0.2, options=options)
    assert far_aft["chord_CoP"] < 0.0


def test_wing_aero_rejects_bad_side(params):
    with pytest.raises(ValueError):
        _still(params, side=0)


def test_world_wrench_identity_rotation():
    wing = WingInput(
        psi=0.0,
        psi_dot=0.0,
        theta_raw=0.0,
        theta_dot_raw=0.0,
        pitch_axis_y=1.0,
        R_world_from_wing=np.eye(3),
        body_origin_world=np.array([0.0, 0.0, 1.0]),
        com_world=np.array([0.0, 0.0, 1.0]),
    )
    aero = {"F_N": 2.0, "M_rd": 0.1, "span_CoP": 0.05, "chord_CoP": 0.01}
    result = world_wrench(wing, 1, aero)
    np.testing.assert_allclose(result["F_world"], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(result["cop_world"], [0.0, 0.05, 0.99])
    expected_torque = np.cross([0.0, 0.05, -0.01], [2.0, 0.0, 0.0]) + [0.0, 0.1, 0.0]
    np.testing.assert_allclose(result["torque_world"], expected_torque)
    assert np.all(world_wrench(wing, 1, aero, read_only=True)["applied_wrench"] == 0)


def test_aero_step_mirrors_wings_for_a_still_torso(params):
    def wing(sign):
        return WingInput(
            psi=0.4,
            psi_dot=30.0,
            theta_raw=0.2,
            theta_dot_raw=1.0,
            pitch_axis_y=1.0,
            R_world_from_wing=np.eye(3),
            body_origin_world=np.array([0.0, sign * 0.02, 0.5]),
            com_world=np.array([0.0, sign * 0.03, 0.5]),
        )

    torso = TorsoInput(np.eye(3), np.zeros(3), np.zeros(3))
    result = aero_step(params, torso, wing(1), wing(-1))
    diag = result["diagnostics"]
    assert diag["F_N_L"] == pytest.approx(diag["F_N_R"])
    assert diag["M_aero_L"] == pytest.approx(diag["M_aero_R"])
    assert diag["lift_total"] == pytest.approx(2.0 * abs(diag["F_N_L"]))


def test_wing_parameters_round_trip(tmp_path, params):
    path = tmp_path / "wing.json"
    save_wing_parameters(params, path)
    assert load_wing_parameters(path) == params


def test_aero_options_validation():
    with pytest.raises(ValueError):
        AeroOptions(lift_scale=-1.0)
    with pytest.raises(ValueError):
        AeroOptions(stroke_plane_angle=float("nan"))
