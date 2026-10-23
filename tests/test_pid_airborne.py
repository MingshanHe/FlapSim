"""Flight audit contracts; no hard-coded successful trajectory claims."""
from pathlib import Path
import mujoco
import numpy as np
import pytest
from experiments.algorithms.pid_airborne.core import Airborne
from experiments.algorithms.pid_airborne.initialization import InitializedAirborne
from experiments.algorithms.pid_airborne.audit import clearance

ROOT=Path('artifacts/pid_airborne_20261008')


@pytest.fixture
def plant():
    if not ROOT.exists():pytest.skip('Recorded PID model assets are unavailable')
    e=InitializedAirborne(ROOT)
    yield e
    e.env.close()


def test_imu_controller_does_not_read_true_height(plant):
    p=dict(release='rest',motor_start_s=.1,height_gains=[900,500,80])
    plant.reset(310700001,p)
    def forbidden():raise AssertionError('Truth state entered IMU-only controller')
    plant.env._state=forbidden
    action=plant.control(p)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action[:2]))<=.3
    assert 3115<=action[2]<=3808


def test_startup_changes_commands_not_flight_pose(plant):
    p=dict(release='rest',motor_start_s=.1,motor=3190.,action_tau=0.)
    plant.reset(310700002,p,0.)
    assert np.max(np.abs(plant.sim.data.qvel))==0
    qpos=plant.sim.data.qpos.copy()
    plant.advance(np.array([.01366,.00088,3190.]),p)
    assert plant.action[2]==pytest.approx(63.8)
    assert plant.sim.data.time==pytest.approx(.002)
    assert not np.array_equal(qpos,plant.sim.data.qpos)
    assert plant.sim.model.body_mass.sum()==pytest.approx(.015)
    np.testing.assert_array_equal(plant.sim.model.opt.gravity,[0,0,-9.81])


def test_geometric_clearance_is_conservative(plant):
    plant.reset(310700003,dict(release='rest'),15.)
    m,d=plant.sim.model,plant.sim.data
    vertices=[]
    for geom in range(m.ngeom):
        if m.geom_bodyid[geom]==0 or m.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=m.geom_dataid[geom];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
        local=m.mesh_vert[start:start+count]
        vertices.append(float(np.min(local@d.geom_xmat[geom].reshape(3,3)[2,:]+d.geom_xpos[geom,2])))
    assert vertices
    assert clearance(m,d)<=min(vertices)+1e-8
    assert clearance(m,d)>.5


def test_phase_override_does_not_poison_later_resets(plant):
    p=dict(kp=[0,0],kd=[0,0])
    plant.reset(310700004,p,phase=1)
    plant.reset(310700005,p)
    fresh=Airborne(ROOT)
    try:
        fresh.reset(310700005,p)
        np.testing.assert_array_equal(plant.sim.data.qpos,fresh.sim.data.qpos)
        np.testing.assert_array_equal(plant.sim.data.qvel,fresh.sim.data.qvel)
    finally:fresh.env.close()
