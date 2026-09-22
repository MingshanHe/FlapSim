"""Checks for orientation fitting and the independent rectangular geometry."""
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/fixbase_curved_real'))
from rect_compare import fit_yaw, evaluate, waveform, yaw_rotate, NPHASE, HARMONICS, momentum_to_fixture_coefficients
from rect_simulation import config_for, ProfileBladeElement, GEOMETRY, mean_between
from flapsim import Simulator
from flapsim.fluid.blade_element import BladeElement


def test_yaw_and_time_phase_are_recovered_without_force_scaling():
    rng = np.random.default_rng(81)
    sim = waveform(rng.normal(size=(1+2*HARMONICS,3)))
    real = yaw_rotate(np.roll(sim, 37, axis=0), np.radians(32.5))
    pair = dict(sim_wave=sim, real_wave=real)
    theta, _, _ = fit_yaw([pair])
    pred, metrics = evaluate(pair, theta)
    assert abs(np.degrees(theta)-32.5) < 1e-8
    assert metrics['phase_shift_deg'] == 37*360/NPHASE
    np.testing.assert_allclose(pred, real, atol=1e-12)
    np.testing.assert_allclose(yaw_rotate(sim, theta)[:,2], sim[:,2])
    # Rotation cannot repair a wrong horizontal magnitude.
    scaled = real.copy(); scaled[:,:2] *= 2
    _, metrics = evaluate(dict(sim_wave=sim, real_wave=scaled), theta)
    assert np.isclose(metrics['xy_rms_ratio'], .5)


def test_rectangular_profile_equals_analytical_blade_elements():
    config = config_for('rect')
    sim = Simulator(config, fluid=ProfileBladeElement(config.fluid, GEOMETRY/'rect_profile.npz'))
    analytical = BladeElement(config.fluid)
    analytical.bind(sim.model)
    sim.data.qvel[:] = np.linspace(-2,2,sim.model.nv)
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(sim.fluid.forces(sim.data), analytical.forces(sim.data), atol=1e-14)
    wing = sim.model.body('left_wing')
    assert wing.mass[0] > 0
    assert sim.model.body('right_flapping').id > 0
    assert mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_BODY,'right_wing') == -1
    np.testing.assert_allclose(np.degrees(sim.model.joint('left_feathering_joint').range),[-26.1,26.1],atol=1e-8)
    assert config.fluid.wings[0].span == .080 and config.fluid.wings[0].chord == .026


def test_complete_cycle_mean_does_not_include_partial_cycles():
    time = np.linspace(0,2,4001)
    values = .03+.2*np.sin(2*np.pi*10*time)
    assert abs(mean_between(time, values, .65,1.95)-.03) < 1e-12


def test_momentum_derivative_sign_and_harmonic_scaling():
    aero=np.zeros((1+2*HARMONICS,3));momentum=aero.copy()
    aero[0,2]=.04
    momentum[1,0]=.003  # cos(omega*t): negative derivative gives positive sin.
    momentum[4,1]=.002  # sin(2*omega*t): negative derivative gives negative cos.
    force=momentum_to_fixture_coefficients(aero,momentum,10.)
    assert force[0,2]==.04
    assert np.isclose(force[2,0],2*np.pi*10*.003)
    assert np.isclose(force[3,1],-2*np.pi*20*.002)
