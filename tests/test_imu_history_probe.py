import mujoco
import numpy as np

from experiments.algorithms.neat_imu.imu_history_probe import integrate
from flapsim.kinematics import euler_matrix


def test_specific_force_sign_and_gravity_in_freefall():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
      <site name="imu"/></body></worldbody><sensor><accelerometer name="acc" site="imu"/>
      </sensor></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.sensor('acc').data,0.,atol=1e-12)
    data.xfrc_applied[1,2] = 9.81
    mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.sensor('acc').data,[0,0,9.81],atol=1e-12)


def test_constant_acceleration_and_known_initial_velocity():
    time = np.linspace(0,2,2001)
    rotation = np.repeat(np.eye(3)[None],len(time),axis=0)
    force = np.tile([2.,0.,9.81],(len(time),1))
    p,v = integrate(time,rotation,force,[.3,0,0])
    np.testing.assert_allclose(p[:,0],time**2+.3*time,atol=1e-11)
    np.testing.assert_allclose(v[:,0],2*time+.3,atol=1e-11)


def test_rotated_stationary_imu_and_causal_prefix():
    time = np.linspace(0,1,1001)
    r = euler_matrix([.2,.3,.4])
    rotation = np.repeat(r[None],len(time),axis=0)
    force = np.tile(r.T@np.array([0,0,9.81]),(len(time),1))
    p,v = integrate(time,rotation,force)
    np.testing.assert_allclose(p,0,atol=1e-12)
    short,_ = integrate(time[:500],rotation[:500],force[:500])
    np.testing.assert_array_equal(short,p[:500])
