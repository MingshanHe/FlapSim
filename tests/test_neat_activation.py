import json
import math
import random
import neat
import numpy as np
import pytest
from experiments.algorithms.neat_imu.activation import (
    SignedHoverEnv, MappedPolicy, signed_map, gyro_network, cases, initialize, export)
from experiments.algorithms.neat_imu.residual import ResidualPolicy, gain_network, action_map
from experiments.algorithms.neat_imu.robust import MEMORY
from experiments.algorithms.neat_imu.robust_train import make_config

TRIM = {'command':[3377.7,.00589,-.00061], 'hover_attitude_rad':[.00066,-.00606,0.]}


def test_six_inputs_are_body_gyro_and_angles_only():
    env = SignedHoverEnv(TRIM,sensor_inputs=6,warmup_seconds=0.)
    env.configure_case(cases([32],.04)[0])
    try:
        obs,_=env.reset(seed=32)
        assert env.observation_space.shape == obs.shape == (6,)
        for _ in range(3):
            obs,_,_,_,info=env.step([.2,-.3,.1])
            rot=env.sim.data.xmat[env.body].reshape(3,3)
            np.testing.assert_allclose(obs[:3],env.angles()/.35,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(obs[3:],env.sim.data.cvel[env.body,:3]@rot/10.,rtol=1e-6,atol=1e-7)
        assert not env.sim.external_body_forces.any()
        assert not env.sim.external_joint_forces.any()
    finally: env.close()


@pytest.mark.parametrize('direction',[-1.,1.])
def test_full_signed_physical_command_mapping(direction):
    mapping=signed_map(TRIM)
    assert mapping['center'][:2] == [0.,0.]
    np.testing.assert_allclose(mapping['span'][:2],[.3,.3])
    env=SignedHoverEnv(TRIM,sensor_inputs=6,warmup_seconds=0.)
    env.configure_case(cases([4],.1)[0])
    try:
        env.reset(seed=4)
        for step in range(1,5):
            _,_,_,_,info=env.step([direction,direction,direction])
            expected=direction*(1-math.exp(-step*.01/.02))
            assert info['alpha'] == pytest.approx(.3*expected)
            assert info['beta'] == pytest.approx(.3*expected)
            assert info['motor_command'] == pytest.approx(TRIM['command'][0]*(1+.1*expected))
    finally: env.close()


def test_old_three_input_policy_retains_physical_commands_in_six_input_env():
    payload=gain_network(.2,.02,.02,action_map(TRIM))
    old=ResidualPolicy(payload); mapped=MappedPolicy(payload,signed_map(TRIM))
    obs=np.array([.1,-.2,.05,4.,-3.,7.])
    old.reset(obs[:3]); mapped.reset(obs)
    for i in range(8):
        obs[:3]*=-.9
        expected=old.commands(obs[:3])
        actual=np.array(signed_map(TRIM)['center'])+np.array(signed_map(TRIM)['span'])*mapped.act(obs)
        np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)


def test_evolved_activations_memory_and_json_match_neat(tmp_path):
    random.seed(901)
    config=make_config(tmp_path/'six.ini',12,'exploratory','gyro6')
    population=neat.Population(config)
    genome=next(iter(population.population.values()))
    mapping=signed_map(TRIM)
    initialize(genome,config,mapping,np.random.default_rng(3),gyro_network(mapping,.2,.02))
    taus=[]
    for _ in range(50):
        genome.mutate(config.genome_config)
        assert all(genome.nodes[k].activation in ('tanh','clamped') for k in (0,1,2))
        for key,(source,_) in MEMORY.items():
            tau=genome.nodes[key].time_constant
            assert .01 <= tau <= .15
            assert genome.nodes[key].activation == 'identity'
            assert genome.connections[(key,key)].weight == pytest.approx(math.exp(-.01/tau))
            assert genome.connections[(key,key)].weight+genome.connections[(source,key)].weight == pytest.approx(1.)
        taus.append(genome.nodes[3].time_constant)
    assert len(set(taus)) > 1
    payload=export(genome,config,mapping)
    assert len(payload['inputs']) == len(payload['input_names']) == 6
    policy=ResidualPolicy(json.loads(json.dumps(payload)))
    network=neat.nn.RecurrentNetwork.create(genome,config)
    initial=np.array([.1,-.1,.05,.2,-.2,.01])
    policy.reset(initial)
    for _ in range(payload['priming_steps']): network.activate(initial)
    rng=np.random.default_rng(7)
    for _ in range(40):
        obs=rng.normal(0.,.3,6)
        np.testing.assert_array_equal(policy.act(obs),np.clip(network.activate(obs),-1,1))
