"""Checks for scientific boundaries introduced by the v3 study."""
import numpy as np
import pytest
import torch
from experiments.algorithms.snn_pid_compare.common import prepare_source
from experiments.algorithms.snn_pid_compare.environment import ControlEnv, TeacherPID
from experiments.algorithms.robust_control.environment import (
    ResearchEnv, ReferencePID, scenarios, desired_velocity, RESIDUAL_SCALE, LOW, HIGH)
from experiments.algorithms.robust_control.networks import Actor, policy, initial_spec


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    return prepare_source(tmp_path_factory.mktemp('v3_pid_source'))


@pytest.mark.parametrize('kind', ['snn','ann'])
@pytest.mark.parametrize('inputs', [11,16])
def test_continuous_actor_export(kind, inputs):
    torch.set_num_threads(1)
    spec = initial_spec(kind, 7181, inputs=inputs)
    actor, deployed = Actor(spec), policy(spec)
    state = actor.initial_state(1)
    with torch.no_grad():
        for obs in np.random.default_rng(180).normal(0, .3, (1100, inputs)):
            output, state = actor(torch.tensor(obs[None]), state)
            np.testing.assert_allclose(deployed.act(obs), output.tanh()[0].numpy(), atol=1e-10)


def test_pid_and_zero_residual_retain_original_plant(source):
    baseline = ControlEnv(source, mode='pid')
    residual = ResearchEnv(source, mode='residual')
    c = scenarios('development', 4, .1, 'stress')[3]
    for env in (baseline, residual):
        env.configure(c)
        env.reset()
    for _ in range(50):
        a, b = baseline.step(np.zeros(3)), residual.step(np.zeros(3))
        np.testing.assert_allclose(baseline.sim.data.qpos, residual.sim.data.qpos, atol=1e-10)
        np.testing.assert_allclose(a[0], b[0], atol=1e-7)
    assert residual.observation_space.shape == (11,)


def test_reference_pid_zero_setpoint_has_original_equations(source):
    env = ResearchEnv(source)
    env.reset()
    a, b = TeacherPID(env.p, np.zeros(3)), ReferencePID(env.p, np.zeros(3))
    for reading in np.random.default_rng(13).normal(0, .05, (100, 11)):
        np.testing.assert_array_equal(a.step(reading), b.step(reading))


def test_command_schedule_reverses_and_integrates_to_zero():
    c = dict(velocity_command=[.12, -.12, .06])
    np.testing.assert_array_equal(desired_velocity(c, 0), np.zeros(3))
    np.testing.assert_allclose(desired_velocity(c, 7), [.06, -.06, .03])
    np.testing.assert_allclose(desired_velocity(c, 9), [.12, -.12, .06])
    np.testing.assert_allclose(desired_velocity(c, 15), [-.12, .12, -.06])
    np.testing.assert_allclose(sum(desired_velocity(c, i*.002) for i in range(10000)), 0, atol=1e-10)


def test_velocity_sensor_is_causal_and_command_observed(source):
    env = ResearchEnv(source, task='velocity', mode='pid')
    c = scenarios('development', 1, 20., task='velocity')[0]
    env.configure(c)
    obs, _ = env.reset()
    assert obs.shape == (16,)
    np.testing.assert_array_equal(obs[-5:], np.zeros(5))
    env.air.steps = 10
    env.velocity_sensor(.02, np.array([1., 2., 0.]))
    np.testing.assert_array_equal(env.vxy, np.zeros(2))
    env.air.steps = 11
    env.velocity_sensor(.04, np.zeros(3))
    assert np.linalg.norm(env.vxy) > 0
    env.command_velocity = np.array([.12, 0, .06])
    np.testing.assert_allclose(env.observation()[-3:], [.4, 0, .4])


@pytest.mark.parametrize('kind', ['snn','ann'])
def test_neat_import_velocity_inputs_and_fixed_topology(tmp_path, kind):
    from experiments.algorithms.robust_control.evolution import create_population, export
    spec = initial_spec(kind, 188, inputs=16, zero_readout=True)
    cfg, pop = create_population(tmp_path, spec, 188, 8, fixed_topology=True)
    g = pop.population[min(pop.population)]
    imported = export(g, kind, 16)
    a, b = policy(spec), policy(imported)
    for x in np.random.default_rng(189).normal(size=(100,16)):
        np.testing.assert_allclose(a.act(x), b.act(x), atol=1e-10)
    structure = (set(g.nodes), set(g.connections))
    for _ in range(10):
        g.mutate(cfg.genome_config)
    assert structure == (set(g.nodes), set(g.connections))


def test_splits_and_allocations_are_distinct():
    from experiments.algorithms.robust_control.train_neat import allocated_budget
    sets = [set(c['seed'] for c in scenarios(s, 32, task=t))
            for s in ('development','validation','final') for t in ('hover','velocity')]
    assert len(set.union(*sets)) == sum(map(len,sets))
    assert allocated_budget(16,12) == 3_584_000
    assert allocated_budget(16,12) % 4000 == 0
    assert allocated_budget(16,12,'velocity') == 5_120_000
