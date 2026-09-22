import mujoco
import numpy as np
import pytest
from flapsim import Simulator, SceneConfig
from flapsim.measurement import FixedBaseWrenchProbe


def test_mount_orientation_and_passive_pitch():
    sim = Simulator()
    root = sim.model.body('base_link').id
    wing = sim.model.body('left_wing').id
    assert sim.data.xpos[wing, 2] > sim.data.xpos[root, 2]
    for name in ('left_feathering_joint', 'right_feathering_joint'):
        joint = sim.model.joint(name)
        assert joint.stiffness[0] == 0
        np.testing.assert_allclose(np.degrees(joint.range), [-26.1, 26.1], atol=1e-8)
        assert joint.dofadr[0] not in sim.actuator.dof_ids
    with pytest.raises(ValueError):
        SceneConfig(orientation=(0, 0, 0, 0))


def test_fixture_sign_and_moment_reference():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base_link" pos="0 0 1"><inertial pos="0.1 0 0" mass="2"
      diaginertia="0.01 0.01 0.01"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.xfrc_applied[1] = [0, 0, 3, 0, .2, 0]
    mujoco.mj_forward(model, data)
    probe = FixedBaseWrenchProbe(model)
    result = probe.read(data, data.xfrc_applied)
    expected_external = np.array([0, 0, 3, 0, -.1, 0])
    np.testing.assert_allclose(result['aero'], expected_external, atol=1e-12)
    np.testing.assert_allclose(result['gravity'], [0, 0, -19.62, 0, 1.962, 0])
    np.testing.assert_allclose(result['fixture_dynamic'], expected_external, atol=1e-12)
    np.testing.assert_allclose(result['support'], -result['fixture'], atol=1e-12)


def test_blade_force_reverses_with_local_motion():
    sim = Simulator()
    for name, velocity in (('left_flapping_joint', 20), ('right_flapping_joint', -20),
                           ('left_feathering_joint', 3), ('right_feathering_joint', -3)):
        sim.data.joint(name).qvel[0] = velocity
    mujoco.mj_forward(sim.model, sim.data)
    forward = sim.fluid.forces(sim.data)
    assert np.linalg.norm(forward) > 0
    sim.data.qvel[:] *= -1
    mujoco.mj_forward(sim.model, sim.data)
    reverse = sim.fluid.forces(sim.data)
    np.testing.assert_allclose(reverse, -forward, atol=1e-12)
