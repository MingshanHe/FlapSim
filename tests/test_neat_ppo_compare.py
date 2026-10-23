import json
import numpy as np
import neat
import pytest
from experiments.algorithms.neat_imu.signed_activations import bipolar_sigmoid,softsign,register
from experiments.algorithms.neat_imu.mass_model import make_variant,MassVariantEnv
from experiments.algorithms.neat_imu.compare_env import CompareEnv,training_case
from experiments.algorithms.neat_imu.activation import cases


def test_sigmoid_centering_reaches_both_signs_and_matches_library_tanh():
    acts=register(neat.activations.ActivationFunctionSet())
    for x in np.linspace(-10,10,101):
        assert bipolar_sigmoid(x)==pytest.approx(acts.get('tanh')(x),abs=1e-15)
        assert -1<=softsign(x)<=1
    assert bipolar_sigmoid(-2)<-.99 and bipolar_sigmoid(2)>.99
    assert bipolar_sigmoid(0)==softsign(0)==0.


@pytest.fixture
def comparison_root(tmp_path):
    make_variant(tmp_path/'robot_upper7_lower8.xml',7,8)
    trim={'command':[3460.,.01,.001],'hover_attitude_rad':[0.,0.,0.]}
    (tmp_path/'calibration').mkdir()
    (tmp_path/'calibration/trim.json').write_text(json.dumps(trim))
    env=MassVariantEnv(trim,model_path=tmp_path/'robot_upper7_lower8.xml',sensor_inputs=6,warmup_seconds=.01)
    try: np.savez(tmp_path/'phase_bank.npz',qpos=np.array([env._initial_qpos]),qvel=np.array([env._initial_qvel]))
    finally: env.close()
    return tmp_path


def test_algorithms_receive_identical_six_imu_observations_and_physics(comparison_root):
    first,second=CompareEnv(comparison_root),CompareEnv(comparison_root)
    try:
        for env in (first,second): env.configure_case(cases([631],.03,True)[0])
        a,info_a=first.reset(seed=631); b,info_b=second.reset(seed=631)
        assert a.shape==b.shape==(6,)
        assert first.sim.model.body_mass.sum()==pytest.approx(.015)
        np.testing.assert_array_equal(a,b)
        for action in [[-1,1,-1],[1,-1,1],[0,0,0]]:
            a,ra,ta,ua,ia=first.step(action); b,rb,tb,ub,ib=second.step(action)
            np.testing.assert_array_equal(a,b)
            assert (ra,ta,ua)==(rb,tb,ub)
            assert ia['motor_command']==ib['motor_command']
            assert abs(ia['alpha'])<=.3 and abs(ia['beta'])<=.3
            assert 3460*.9<=ia['motor_command']<=3460*1.1
        assert not first.sim.external_body_forces.any()
        assert not first.sim.external_joint_forces.any()
    finally: first.close(); second.close()


def test_same_curriculum_generator_for_both_methods():
    assert training_case(2,.1,True)['seconds']==2.
    assert training_case(2,.1,True)['degrees']==2.
    assert training_case(2,.4,True)['seconds']==4.
    assert training_case(2,.7,True)['seconds']==8.
