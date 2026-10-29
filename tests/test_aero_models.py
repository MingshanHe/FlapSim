"""Additive aero models, modifiers and the engine that composes them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flapsim.aero import (
    AddedMass,
    AeroEngine,
    AeroModel,
    GroundEffect,
    RotationalLift,
    WakeCapture,
    WingLoads,
    available_models,
    available_modifiers,
    build_model,
    register_model,
)
from flapsim.config import AeroOptions
from flapsim.core import Simulation
from flapsim.robot import load


@pytest.fixture
def robot():
    return load("SingleThrustFlapper")


def test_registries_and_builders():
    assert {"quasi_steady", "rotational_lift", "added_mass", "wake_capture"} <= set(
        available_models()
    )
    assert "ground_effect" in available_modifiers()
    model = build_model({"type": "rotational_lift", "C_rot": 2.0})
    assert isinstance(model, RotationalLift) and model.C_rot_override == 2.0
    with pytest.raises(ValueError, match="unknown aero model"):
        build_model("vortex_lattice")
    with pytest.raises(ValueError, match="needs a 'type'"):
        build_model({"C_rot": 1.0})


def test_single_model_stack_equals_quasi_steady_adapter(robot):
    """The engine with one model reproduces the legacy adapter numbers."""
    from flapsim.fluid import MuJoCoAeroAdapter

    sim = Simulation("SingleThrustFlapper")
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint", "right_wing_flapping"],
            "directions": [1, -1],
            "frequency_hz": 30.0,
            "amplitude_rad": 1.5,
        }
    )
    stack = sim.add({"type": "aero/stack", "models": ["quasi_steady"]})
    sim.advance(0.05)
    # the signals mirror the diagnostics of the last forces phase
    assert sim.signals["wings/left/quasi_steady/C_N"] == pytest.approx(
        stack.diagnostics["left/quasi_steady/C_N"]
    )
    assert "wings/left/F_N" in sim.signals and "aero/rho" in sim.signals
    # evaluate engine and adapter at the same instant (forces ran before mj_step)
    stack.engine.apply()
    adapter = MuJoCoAeroAdapter(sim.robot)
    adapter.apply_forces()
    np.testing.assert_allclose(
        adapter.diagnostics["F_world"], stack.diagnostics["F_world"]
    )
    assert adapter.diagnostics["left/quasi_steady/C_N"] == pytest.approx(
        stack.diagnostics["left/quasi_steady/C_N"]
    )


def test_stack_sums_model_contributions(robot):
    sim = Simulation("SingleThrustFlapper")
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint", "right_wing_flapping"],
            "directions": [1, -1],
            "frequency_hz": 30.0,
            "amplitude_rad": 1.5,
        }
    )
    stack = sim.add(
        {
            "type": "aero/stack",
            "models": ["quasi_steady", "rotational_lift", "added_mass"],
        }
    )
    sim.advance(0.05)
    d = stack.diagnostics
    total = d["left/F_N"]
    parts = (
        d["left/quasi_steady/F_N"]
        + d["left/rotational_lift/F_N"]
        + d["left/added_mass/F_N"]
    )
    assert total == pytest.approx(parts)
    assert d["left/modifier_factor"] == 1.0
    assert set(stack.describe()["engine"]["models"][1].keys()) == {"model", "config"}


class _StillWing:
    """Minimal stand-ins to call models without MuJoCo."""


def _make_context(robot, *, theta_dot=0.0, side=1):
    from flapsim.fluid.aero import wing_kinematics

    wing = robot.wings[0] if side == 1 else robot.wings[1]
    state = wing.state(robot.data)
    state.theta_dot = theta_dot
    kin = wing_kinematics(
        wing.params,
        int(state.side),
        0.0,
        40.0,
        0.0,
        theta_dot,
        np.zeros(3),
        np.zeros(3),
    )
    from flapsim.aero import FlowState

    flow = FlowState(1.225, np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 5e-5)
    return wing, state, kin, flow


def test_rotational_lift_sign_and_scaling(robot):
    options = AeroOptions(force_limit=1e9)
    wing, state, kin, flow = _make_context(robot, theta_dot=0.0)
    assert RotationalLift().compute(wing, state, kin, flow, options).force[0] == 0.0
    wing, state, kin, flow = _make_context(robot, theta_dot=50.0)
    up = RotationalLift().compute(wing, state, kin, flow, options)
    wing, state, kin, flow = _make_context(robot, theta_dot=-50.0)
    down = RotationalLift().compute(wing, state, kin, flow, options)
    # pitching up in a +X flow augments the normal force along +X
    assert up.force[0] > 0.0 > down.force[0]
    assert up.force[0] == pytest.approx(-down.force[0])
    assert up.diagnostics["C_rot"] == pytest.approx(math.pi * 0.75)
    wing, state, kin, flow = _make_context(robot, theta_dot=100.0)
    double = RotationalLift().compute(wing, state, kin, flow, options)
    assert double.force[0] == pytest.approx(2.0 * up.force[0])
    assert double.point[2] == pytest.approx(-0.5 * wing.mean_chord)


def test_rotational_lift_is_mirror_symmetric_between_strokes(robot):
    """Reversing the stroke and the pitch rate mirrors the force (F -> -F)."""
    from flapsim.fluid.aero import wing_kinematics

    options = AeroOptions(force_limit=1e9)
    wing, state, _, flow = _make_context(robot)
    zero = np.zeros(3)
    forward = wing_kinematics(wing.params, 1, 0.0, 40.0, 0.2, 50.0, zero, zero)
    backward = wing_kinematics(wing.params, 1, 0.0, -40.0, -0.2, -50.0, zero, zero)
    f_fwd = RotationalLift().compute(wing, state, forward, flow, options).force[0]
    f_bwd = RotationalLift().compute(wing, state, backward, flow, options).force[0]
    assert f_fwd > 0.0
    assert f_bwd == pytest.approx(-f_fwd)


def test_added_mass_is_stable_on_the_passive_wing():
    """Explicit feathering added mass diverged; it is now implicit armature."""
    sim = Simulation("SingleThrustFlapper")
    sim.add(
        {
            "type": "actuator/sinusoidal_stroke",
            "id": "stroke",
            "joints": ["left_flapping_joint", "right_wing_flapping"],
            "directions": [1, -1],
            "frequency_hz": 40.0,
            "amplitude_rad": 1.5,
        }
    )
    dof = sim.model.joint("left_wing_feathering").dofadr[0]
    before = float(sim.model.dof_armature[dof])
    stack = sim.add({"type": "aero/stack", "models": ["quasi_steady", "added_mass"]})
    added = stack.models[1].added_inertia["left"]
    assert sim.model.dof_armature[dof] == pytest.approx(before + added)
    assert 1e-10 < added < 1e-7
    sim.advance(0.15)
    peak = 0.0
    while sim.time < 0.2:
        sim.step()
        peak = max(peak, abs(stack.diagnostics["left/added_mass/F_N"]))
    assert 0.01 < peak < 0.5  # finite, well below the 5 N force limit
    # a second engine on the same robot must not add the inertia twice
    sim.add(
        {
            "type": "aero/stack",
            "id": "aero2",
            "models": ["added_mass"],
            "options": {"read_only": True},
        }
    )
    assert sim.model.dof_armature[dof] == pytest.approx(before + added)


def test_added_mass_needs_acceleration(robot):
    options = AeroOptions(force_limit=1e9)
    model = AddedMass()
    wing, state, kin, flow = _make_context(robot)
    first = model.compute(wing, state, kin, flow, options)
    assert np.all(first.force == 0.0)  # no history yet
    flow.time = 1e-4
    steady = model.compute(wing, state, kin, flow, options)
    assert steady.force[0] == pytest.approx(
        0.0, abs=1e-12
    )  # same kinematics: no acceleration
    from flapsim.fluid.aero import wing_kinematics

    faster = wing_kinematics(
        wing.params, int(state.side), 0.0, 50.0, 0.0, 0.0, np.zeros(3), np.zeros(3)
    )
    flow.time = 2e-4
    accelerating = model.compute(wing, state, faster, flow, options)
    # flow accelerating along +X pushes the plate along +X
    assert accelerating.force[0] > 0.0
    model.reset()
    assert np.all(model.compute(wing, state, faster, flow, options).force == 0.0)


def test_wake_capture_fires_after_reversal_and_decays(robot):
    options = AeroOptions(force_limit=1e9)
    model = WakeCapture(C_wc=1.0, tau_chords=0.5)
    wing, state, kin_forward, flow = _make_context(robot)
    from flapsim.fluid.aero import wing_kinematics

    kin_backward = wing_kinematics(
        wing.params, 1, 0.0, -40.0, 0.0, 0.0, np.zeros(3), np.zeros(3)
    )
    flow.time = 0.0
    assert np.all(model.compute(wing, state, kin_forward, flow, options).force == 0.0)
    flow.time = 1e-3
    first = model.compute(wing, state, kin_backward, flow, options)
    assert first.force[0] < 0.0  # along the new (negative) flow direction
    assert first.diagnostics["U_rev"] == pytest.approx(abs(kin_forward["U_i_at_CoP"]))
    flow.time = 2e-3
    later = model.compute(wing, state, kin_backward, flow, options)
    assert abs(later.force[0]) < abs(first.force[0])
    flow.time = 10.0
    assert np.all(model.compute(wing, state, kin_backward, flow, options).force == 0.0)


def test_ground_effect_factor(robot):
    wing, state, kin, flow = _make_context(robot)
    high = GroundEffect(gain=0.3, length_scale=0.5)
    state.origin_world = np.array([0.0, 0.0, 10.0])
    assert high.factor(wing, state, kin, flow) == pytest.approx(1.0, abs=1e-6)
    state.origin_world = np.array([0.0, 0.0, 0.0])
    assert high.factor(wing, state, kin, flow) == pytest.approx(1.3, rel=1e-4)
    cb = GroundEffect(model="cheeseman_bennett", max_factor=2.0)
    state.origin_world = np.array([0.0, 0.0, wing.length])  # h = R: 1/(1-1/16)
    assert cb.factor(wing, state, kin, flow) == pytest.approx(1.0 / (1.0 - 1.0 / 16.0))
    state.origin_world = np.array([0.0, 0.0, 0.01])
    assert cb.factor(wing, state, kin, flow) == 2.0
    with pytest.raises(ValueError, match="model must be"):
        GroundEffect(model="magic")


def test_custom_model_registration_and_engine_use(robot):
    @register_model
    class ConstantPush(AeroModel):
        name = "constant_push_test"

        def compute(self, wing, state, kin, flow, options):
            return WingLoads(
                self.name,
                np.array([0.01, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(3),
                {"k": 1.0},
            )

    engine = AeroEngine(
        robot, ["constant_push_test"], options=AeroOptions(read_only=True)
    )
    diagnostics = engine.apply()
    assert diagnostics["left/F_N"] == pytest.approx(0.01)
    assert diagnostics["left/constant_push_test/k"] == 1.0
    assert np.all(robot.data.xfrc_applied == 0.0)  # read only
    with pytest.raises(ValueError, match="unknown wings"):
        AeroEngine(robot, ["quasi_steady"], wings=["tail"])
