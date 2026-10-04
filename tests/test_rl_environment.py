import numpy as np
import pytest
from flapsim.rl.env import TrackingConfig, TrajectoryTrackingEnv
from flapsim.rl.features import CentroidalTrackingEnv


@pytest.mark.parametrize('environment', [TrajectoryTrackingEnv, CentroidalTrackingEnv])
def test_full_physics_episode_and_seed_contract(environment):
    env = environment(TrackingConfig(trajectories=('hover',), warmup_seconds=0,
                                               episode_seconds=.02, native_physics=True))
    try:
        a, _ = env.reset(seed=902)
        before = env.sim.data.qpos.copy()
        env.step(np.zeros(3))
        b, _ = env.reset(seed=902)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(before, env.sim.data.qpos)
        assert env.sim.model.body_mass.sum() == pytest.approx(.015)
        np.testing.assert_array_equal(env.sim.model.opt.gravity, [0, 0, -9.81])
        for bad in ([np.nan, 0, 0], [0, 0], [0, np.inf, 0]):
            with pytest.raises(ValueError):
                env.step(bad)
        env.step(np.zeros(3))
        _, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert np.isfinite(reward)
        assert not terminated and truncated
        assert info['elapsed'] == pytest.approx(.02)
        assert env.sim.data.time == pytest.approx(.02)
        assert not env.sim.external_body_forces.any()
        assert not env.sim.external_joint_forces.any()
        with pytest.raises(RuntimeError):
            env.step(np.zeros(3))
    finally:
        env.close()
