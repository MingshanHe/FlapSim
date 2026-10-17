import json
import neat
import numpy as np
import pytest
from experiments.algorithms.neat_imu.residual import (
    ResidualHoverEnv, ResidualPolicy, action_map, gain_network, encode, seed_genome)

TRIM = {'command': [3376.8672644667213, 0., 0.]}


@pytest.mark.parametrize('direction', [-1, 1])
def test_symmetric_physical_residual_smoothing_and_limits(direction):
    env = ResidualHoverEnv(TRIM, seconds=.02, warmup_seconds=0.)
    try:
        obs, _ = env.reset(seed=2)
        assert obs.shape == (3,)
        _, reward, _, _, info = env.step([direction, -direction, direction])
        smooth = 1-np.exp(-.5)
        assert info['motor_command'] == pytest.approx(TRIM['command'][0]*(1+.1*direction*smooth))
        assert info['alpha'] == pytest.approx(.2*direction*smooth)
        assert info['beta'] == pytest.approx(-.2*direction*smooth)
        assert np.isfinite(reward)
        assert env.sim.model.body_mass.sum() == pytest.approx(.015)
        assert not env.sim.external_body_forces.any()
    finally:
        env.close()


def test_recurrent_seed_export_sequence_and_episode_reset():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'experiments/algorithms/neat_imu/config.ini')
    config.genome_config.feed_forward = False
    pop = neat.Population(config)
    g = next(iter(pop.population.values()))
    payload = gain_network(.3, .03, .05, action_map(TRIM))
    seed_genome(g, config, payload)
    original = neat.nn.RecurrentNetwork.create(g, config)
    exported = ResidualPolicy(json.loads(json.dumps(encode(g, config, action_map(TRIM)))))
    for obs in np.random.default_rng(4).normal(0, .1, (40, 3)):
        np.testing.assert_allclose(exported.act(obs), original.activate(obs), atol=1e-15)
    obs = [.1, -.2, .03]
    exported.reset(obs)
    a = exported.act(obs)
    exported.act([.4, .7, -.2])
    exported.reset(obs)
    np.testing.assert_array_equal(a, exported.act(obs))
    steady = ResidualPolicy(payload)
    steady.reset([0, 0, 0])
    np.testing.assert_allclose(steady.commands([0, 0, 0]), [0, 0, TRIM['command'][0]])


def test_initial_genomes_share_innovations_and_inputs_only_three():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'experiments/algorithms/neat_imu/config.ini')
    config.genome_config.feed_forward = False
    pop = neat.Population(config)
    a, b = list(pop.population.values())[:2]
    payload = gain_network(.3, .03, .05, action_map(TRIM))
    for genome in (a, b):
        seed_genome(genome, config, payload)
    assert {k: c.innovation for k, c in a.connections.items()} == {k: c.innovation for k, c in b.connections.items()}
    assert config.genome_config.input_keys == [-1, -2, -3]


def test_nonzero_trim_attitude_and_gimbal_survive_reset():
    trim = dict(command=[3377.7, .00589, -.00061], hover_attitude_rad=[.00066, -.00606, 0.])
    env = ResidualHoverEnv(trim, seconds=.02, warmup_seconds=.01, initial_degrees=0.)
    try:
        obs, _ = env.reset(seed=9)
        np.testing.assert_allclose(obs, np.array(trim['hover_attitude_rad'])/.35, atol=1e-8)
        policy = ResidualPolicy(gain_network(.3, .1, .05, action_map(trim)))
        policy.reset(obs)
        np.testing.assert_allclose(policy.act(obs), 0., atol=1e-8)
        _, _, _, _, info = env.step([0, 0, 0])
        np.testing.assert_allclose([info['alpha'], info['beta'], info['motor_command']], env.center)
        assert not env.sim.external_body_forces.any()
    finally:
        env.close()
