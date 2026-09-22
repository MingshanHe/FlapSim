"""Validate the native Purdue bridge and the IST frame/geometry adapter."""
from pathlib import Path
import sys
import numpy as np
import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/fixbase_curved_real')]
from flappy_audit_models import NativeWing, StripAuditFluid, PurdueBenchFluid, build_reference, cn, dcp
from rect_material_experiment import config_for, OUTPUT
from rect_simulation import ProfileBladeElement
from flapsim import Simulator


@pytest.fixture(scope='module', autouse=True)
def native_library():
    build_reference()


def test_native_rectangular_wing_matches_analytic_integrals():
    # Constant chord starting at the flapping axis: integral r^k/R^k = R/(k+1).
    R, c, rho = .09, .026, 1.225
    native = NativeWing([R, c, .25, 1/3, .5, 1., 1/3, .5, 1., 1.], rho)
    try:
        for rate in (-120., 0., 120.):
            for pitch in (-.7, -.2, .2, .7):
                for pitch_rate in (-200., 0., 200.):
                    fn, span, chord, ma, rd, alpha = native.evaluate(.1, rate, pitch, pitch_rate)
                    expected_alpha = pitch + np.sign(rate)*np.pi/2
                    expected_fn = .5*rho*c*cn(expected_alpha)*rate**2*R**3/3
                    np.testing.assert_allclose([alpha, fn, span, ma, rd],
                        [expected_alpha, expected_fn, .75*R,
                         -expected_fn*c*dcp(expected_alpha),
                         -.125*rho*5*R*c**4*abs(pitch_rate)*pitch_rate], atol=1e-14)
                    np.testing.assert_allclose(chord, c*dcp(alpha) if fn else 0., atol=1e-14)
                    assert rd*pitch_rate <= 0
    finally:
        native.close()


def test_native_ist_force_matches_high_resolution_blade_integral():
    # Native moments and blade quadrature use exactly the same 2400 CAD bins.
    config = config_for(strips=2400)
    profile = OUTPUT/'rect_profile.npz'
    native = PurdueBenchFluid(config.fluid, profile)
    sim = Simulator(config, fluid=native)
    strips = StripAuditFluid(config.fluid, profile, pressure=True, exclude_pitch=True, chord_drag=False)
    strips.bind(sim.model)
    # The saved membrane starts 2.75 mm from the pitch origin. Use its bins
    # directly, rather than interpolating onto a zero-to-tip uniform grid.
    p = np.load(profile)
    r, c, z = p['radius'], p['width'], p['center']
    dr = r[1]-r[0]
    wing, body, root, _, selector, _, _ = strips.profiles[0]
    local = np.column_stack([np.zeros_like(r),r,z])
    strips.profiles[0] = (wing,body,root,local,selector,c*dr,float(np.sum(c**4)*dr))
    strips.widths[0] = c
    body = sim.model.body('left_wing').id
    try:
        for flap, pitch, rate, pitch_rate in [(.2,.3,100.,50.),(-.7,-.4,-120.,-80.),(.5,-.2,60.,-90.)]:
            sim.data.joint('left_flapping_joint').qpos[0] = flap
            sim.data.joint('left_feathering_joint').qpos[0] = pitch
            sim.data.joint('left_flapping_joint').qvel[0] = rate
            sim.data.joint('left_feathering_joint').qvel[0] = pitch_rate
            mujoco.mj_forward(sim.model,sim.data)
            np.testing.assert_allclose(native.forces(sim.data)[body],
                                       strips.forces(sim.data)[body], rtol=2e-9, atol=1e-12)
    finally:
        native.native.close()


def test_disabled_ablation_is_identical_to_original_fluid():
    config = config_for()
    profile = OUTPUT/'rect_profile.npz'
    baseline = ProfileBladeElement(config.fluid, profile)
    sim = Simulator(config, fluid=baseline)
    audit = StripAuditFluid(config.fluid, profile)
    audit.bind(sim.model)
    states = np.load(ROOT/'experiments/fixbase_curved_real/flappy_audit/baseline_states.npz')
    for index in (501, 1003, 1729, 2197, 3501):
        sim.data.qpos[:] = states['qpos'][index]
        sim.data.qvel[:] = states['qvel'][index]
        mujoco.mj_forward(sim.model,sim.data)
        np.testing.assert_allclose(audit.forces(sim.data), baseline.forces(sim.data), rtol=1e-13, atol=1e-15)
