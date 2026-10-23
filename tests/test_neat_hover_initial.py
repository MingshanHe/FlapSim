import itertools
import neat
import numpy as np
from experiments.algorithms.neat_imu.activation import cases,signed_map,initialize,export,gyro_network
from experiments.algorithms.neat_imu.architecture import add_hidden,trim_network
from experiments.algorithms.neat_imu.hover_initial import HoverInitialEnv
from experiments.algorithms.neat_imu.robust_train import make_config
from experiments.algorithms.neat_imu.residual import ResidualPolicy

TRIM={'command':[3377.7,.00589,-.00061],'hover_attitude_rad':[.00066,-.00606,0.]}


def test_gyro_filter_time_constant_survives_parent_transfer(tmp_path):
    config=make_config(tmp_path/'transfer.ini',8,'exploratory','gyro6')
    population=neat.Population(config)
    genome=next(iter(population.population.values()))
    parent=gyro_network(signed_map(TRIM),.2,.04)
    rho=np.exp(-.01/.087)
    for node in parent['nodes']:
        node['links']=[[10 if source==-5 else source,weight] for source,weight in node['links']]
    parent['nodes'].append(dict(key=10,activation='identity',aggregation='sum',bias=0.,response=1.,links=[[-5,1-rho],[10,rho]]))
    parent['memory_design']='stable memory'; parent['gyro_memory']='stable gyro memory'
    initialize(genome,config,signed_map(TRIM),np.random.default_rng(1),parent)
    old=ResidualPolicy(parent); new=ResidualPolicy(export(genome,config,signed_map(TRIM)))
    obs=np.zeros(6); old.reset(obs); new.reset(obs)
    rng=np.random.default_rng(1)
    for _ in range(30):
        obs=rng.normal(0.,.3,6)
        np.testing.assert_allclose(old.act(obs),new.act(obs),rtol=1e-13,atol=1e-13)


def test_hidden_seed_preserves_base_policy_until_mutation(tmp_path):
    config=make_config(tmp_path/'hidden.ini',8,'exploratory','gyro6')
    population=neat.Population(config)
    genome=next(iter(population.population.values()))
    initialize(genome,config,signed_map(TRIM),np.random.default_rng(1),trim_network(TRIM))
    old=ResidualPolicy(export(genome,config,signed_map(TRIM)))
    config.genome_config.node_indexer=itertools.count(200)
    add_hidden(genome,config,8,np.random.default_rng(3))
    new=ResidualPolicy(export(genome,config,signed_map(TRIM)))
    obs=np.zeros(6); old.reset(obs); new.reset(obs)
    rng=np.random.default_rng(7)
    for _ in range(30):
        obs=rng.normal(0.,.2,6)
        np.testing.assert_array_equal(old.act(obs),new.act(obs))
    assert len(config.genome_config.initial_hidden_keys)==8
    assert all(key in genome.nodes for key in config.genome_config.initial_hidden_keys)


def test_momentum_initialization_is_only_a_reset_condition():
    env=HoverInitialEnv(TRIM,sensor_inputs=6,initialization='momentum',warmup_seconds=.1)
    env.configure_case(cases([19],.04)[0])
    try:
        obs,info=env.reset(seed=19)
        assert obs.shape==(6,)
        assert np.linalg.norm(info['initial_momentum_after'][3:])<1e-12
        assert np.linalg.norm(info['root_velocity_correction'])>0.
        assert not env.sim.external_body_forces.any()
        assert not env.sim.external_joint_forces.any()
        _,_,_,_,step_info=env.step([0.,0.,0.])
        assert step_info['elapsed']==.01
        assert env.sim.data.time>0
    finally: env.close()
