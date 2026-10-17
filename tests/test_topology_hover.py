"""Meaningful controller, sensor, optimization and graph audit contracts."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import neat
import random
from experiments.algorithms.topology_hover.networks import NumpyPolicy,torch_actor,random_graph
from experiments.algorithms.topology_hover.ppo import gae
from experiments.algorithms.topology_hover.environment import HoverEnv,case
from experiments.algorithms.topology_hover.sensors import HeightSensor
from experiments.algorithms.pid_airborne.initialization import InitializedAirborne
from experiments.algorithms.topology_hover.evaluate import remove_height_inputs,interventions
from experiments.algorithms.topology_hover.evolve import config_file,initialize
from experiments.algorithms.topology_hover.networks import export_genome


def test_export_matches_neat_python_recurrent_network_after_mutations(tmp_path):
    random.seed(523);config=config_file(tmp_path,4,11)
    population=neat.Population(config);initialize(population,config,523)
    rng=np.random.default_rng(917)
    for genome in population.population.values():
        for _ in range(12):genome.mutate(config.genome_config)
        original=neat.nn.RecurrentNetwork.create(genome,config)
        exported=NumpyPolicy(export_genome(genome,11))
        for observation in rng.normal(size=(100,11)).astype(np.float32):
            np.testing.assert_allclose(exported.act(observation),original.activate(observation),rtol=0,atol=2e-6)


@pytest.mark.parametrize('kind',['graph','mlp'])
def test_exported_recurrent_sequence_matches_torch(kind):
    a=torch_actor(kind,11,13);p=NumpyPolicy(a.export());h=torch.zeros(1,a.state_size)
    rng=np.random.default_rng(64)
    for x in rng.normal(size=(200,11)).astype(np.float32):
        with torch.no_grad():mean,h=a(torch.tensor(x[None]),h)
        np.testing.assert_allclose(p.act(x),torch.tanh(mean)[0].numpy(),atol=2e-6)


def test_recurrent_gradient_crosses_time():
    a=torch_actor('graph',11,3);h=torch.zeros(1,a.state_size)
    x=torch.ones(1,11,requires_grad=True)
    _,h=a(x,h)
    for _ in range(3):y,h=a(torch.zeros(1,11),h)
    y.sum().backward();assert x.grad.abs().sum()>0


def test_gae_failure_breaks_episode_and_timeout_bootstraps():
    # A failure at t=0 must not receive t=1 episode's value or return.
    r=np.array([[1.],[2.]],dtype=np.float32);v=np.array([[.2],[.4]],dtype=np.float32)
    a,ret=gae(r,v,np.array([[True],[False]]),np.array([.6]),gamma=.9,lam=1.)
    np.testing.assert_allclose(ret[:,0],[1.,2.54],atol=1e-6)
    # Time-limit value is explicitly added to reward before GAE.
    a,ret=gae(np.array([[1.+.9*4]]),np.array([[.2]]),np.array([[True]]),np.array([999.]),gamma=.9)
    assert ret[0,0]==pytest.approx(4.6)


def test_random_graph_matches_indegree_and_parameters():
    source=random_graph(11,8,1);other=random_graph(11,seed=12,reference=source)
    for node in source['nodes']:
        for recurrent in [True,False]:
            def count(s):return sum(e['enabled'] and e['target']==node['id'] and (e['source']>=0)==recurrent for e in s['edges'])
            assert count(source)==count(other)
    assert len(other['edges'])==len(source['edges'])


def test_height_observer_waits_for_sensor_delay():
    h=HeightSensor(.8,np.random.default_rng(1),noise=0,delay=.02)
    for i in range(1,20):h.step(i*.002,1.,1.,[0,0,0])
    assert h.h==pytest.approx(.8)
    h.step(.04,1.,1.,[0,0,0]);assert h.h>.8


def test_controller_inputs_do_not_read_truth_position():
    e=HoverEnv();e.configure(case(329800001,.1));e.reset(seed=329800001)
    def forbidden():raise AssertionError('Truth state entered controller')
    e.air.env._state=forbidden
    assert e.observation().shape==(11,)
    assert np.isfinite(e.pid()).all();assert np.isfinite(e.reading()).all()


def test_sensor_instrumentation_preserves_physics():
    e=HoverEnv();e.configure(case(329800002,.1,'nominal',phase=2));e.reset(seed=329800002)
    original=InitializedAirborne(Path('artifacts/pid_airborne_20261008'))
    original.reset(329800002,e.p,2.,phase=2)
    np.testing.assert_allclose(e.sim.data.qpos,original.sim.data.qpos,atol=1e-12)
    np.testing.assert_allclose(e.sim.data.qvel,original.sim.data.qvel,atol=1e-12)
    for _ in range(20):
        command=np.array([.01,-.005,3190.])
        e.air.advance(command,e.p);original.advance(command,e.p)
    np.testing.assert_allclose(e.sim.data.qpos,original.sim.data.qpos,atol=1e-12)
    np.testing.assert_allclose(e.sim.data.qvel,original.sim.data.qvel,atol=1e-12)


def test_mass_ood_changes_both_lower_bodies_and_restores_reset():
    e=HoverEnv();e.configure(case(329800003,.1,'ood'));e.reset(seed=329800003)
    assert e.sim.model.body_mass.sum()==pytest.approx(.016)
    e.configure(case(329800004,.1,'nominal'));e.reset(seed=329800004)
    assert e.sim.model.body_mass.sum()==pytest.approx(.015)


def test_sensor_removal_does_not_reindex_remaining_channels():
    graph=random_graph(11,3,11);short=remove_height_inputs(graph)
    full=NumpyPolicy(graph);nine=NumpyPolicy(short);rng=np.random.default_rng(1)
    for x in rng.normal(size=(20,9)):
        np.testing.assert_allclose(full.act(np.r_[x,0,0]),nine.act(x),atol=1e-6)


def test_intervention_controls_remove_matching_edge_counts():
    spec=random_graph(11,3,42);variants=dict(interventions(spec,13))
    for name in ['remove_past_state_paths','remove_direct_cross_axis']:
        count=sum(e['enabled'] for e in variants[name]['edges'])
        for i in range(3):assert sum(e['enabled'] for e in variants[f'{name}_random_control_{i}']['edges'])==count
    assert all(e['source']<0 for e in variants['remove_past_state_paths']['edges'] if e['enabled'])
