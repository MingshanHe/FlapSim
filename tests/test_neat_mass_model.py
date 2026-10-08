import xml.etree.ElementTree as ET
import numpy as np
import pytest
from flapsim import load_config
from flapsim.rl.env import PRESET
from experiments.algorithms.neat_imu.mass_model import make_variant,describe,MassVariantEnv
from experiments.algorithms.neat_imu.activation import cases


def test_mass_variant_preserves_moving_wing_and_gear_inertials(tmp_path):
    source=load_config(PRESET).path
    original=source.read_bytes()
    path=tmp_path/'upper7_lower9.xml'
    make_variant(path,7,9)
    old,new=ET.fromstring(original),ET.parse(path).getroot()
    for body in old.findall('.//body'):
        inertia=body.find('inertial')
        if inertia is None: continue
        replacement=new.find(f".//body[@name='{body.get('name')}']/inertial")
        if body.get('name') not in ('base_link','thrust_unit'):
            assert inertia.attrib==replacement.attrib
        else:
            assert inertia.get('pos')==replacement.get('pos')
            ratio=float(replacement.get('mass'))/float(inertia.get('mass'))
            np.testing.assert_allclose(np.fromstring(replacement.get('fullinertia'),sep=' '),
                                       ratio*np.fromstring(inertia.get('fullinertia'),sep=' '))
    props=describe(path)
    assert props['upper']['mass_g']==pytest.approx(7.)
    assert props['lower']['mass_g']==pytest.approx(9.)
    assert props['total']['mass_g']==pytest.approx(16.)
    assert props['total']['com_base_mm'][2]==pytest.approx(-.7316515273,abs=1e-8)
    assert source.read_bytes()==original


def test_mass_variant_reaches_both_warmup_and_free_flight(tmp_path):
    path=tmp_path/'variant.xml'; make_variant(path,7,9)
    trim={'command':[3500.,.005,-.001],'hover_attitude_rad':[0.,0.,0.]}
    env=MassVariantEnv(trim,model_path=path,initialization='trim',sensor_inputs=6)
    env.configure_case(cases([20],.03)[0])
    try:
        obs,_=env.reset(seed=20)
        assert env.sim.model.body_mass.sum()==pytest.approx(.016)
        assert env.source_config.path==path.resolve()
        assert np.linalg.norm(env._initial_qvel)>0.
        assert obs.shape==(6,)
        env.step([0.,0.,0.])
        assert not env.sim.external_body_forces.any()
        assert not env.sim.external_joint_forces.any()
    finally: env.close()


def test_training_worker_uses_variant_mass_and_six_imu_inputs(tmp_path):
    from experiments.algorithms.neat_imu import architecture,robust
    path=tmp_path/'worker_variant.xml'; make_variant(path,7,10)
    trim={'command':[4150.,.01,0.],'hover_attitude_rad':[0.,0.,0.]}
    architecture.init_worker(dict(trim=trim,model_path=str(path),sensor_inputs=6,
                                  initialization='trim',warmup_seconds=.01))
    try:
        assert robust.ENV.sim.model.body_mass.sum()==pytest.approx(.017)
        robust.ENV.configure_case(cases([17],.03)[0])
        obs,_=robust.ENV.reset(seed=17)
        assert obs.shape==(6,)
        assert robust.ENV.source_config.path==path.resolve()
    finally:
        robust.ENV.close()
        robust.ENV=None
