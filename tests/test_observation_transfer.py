import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from flapsim.rl.transfer import expand_normalizer, expand_policy_weights, initialize_reference_acceleration


class ToyEnv(gym.Env):
    def __init__(self, size):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32)


def test_new_features_preserve_policy_output_and_old_normalization(tmp_path):
    old_env = VecNormalize(DummyVecEnv([lambda: ToyEnv(20)]))
    old_env.obs_rms.mean[:] = np.linspace(-1, 1, 20)
    old_env.obs_rms.var[:] = np.linspace(.2, 2., 20)
    old_env.save(str(tmp_path/'normalizer.pkl'))
    new_env = expand_normalizer(tmp_path/'normalizer.pkl', DummyVecEnv([lambda: ToyEnv(26)]))
    source = PPO('MlpPolicy', old_env, n_steps=2, batch_size=2, seed=4, device='cpu')
    target = PPO('MlpPolicy', new_env, n_steps=2, batch_size=2, seed=5, device='cpu')
    expand_policy_weights(source, target)
    raw = np.random.default_rng(3).normal(size=(16, 26)).astype(np.float32)
    original = old_env.normalize_obs(raw[:, :20])
    expanded = new_env.normalize_obs(raw)
    np.testing.assert_array_equal(original, expanded[:, :20])
    np.testing.assert_allclose(source.predict(original, deterministic=True)[0],
                               target.predict(expanded, deterministic=True)[0], atol=1e-7)
    old_env.close()
    new_env.close()


def test_hover_curriculum_does_not_jump_on_new_acceleration():
    env = VecNormalize(DummyVecEnv([lambda: ToyEnv(40)]))
    env.obs_rms.var[21:24] = 1e-12
    model = PPO('MlpPolicy', env, n_steps=2, batch_size=2, seed=7, device='cpu')
    raw = np.random.default_rng(8).normal(size=(8, 40)).astype(np.float32)
    raw[:, 21:24] = 0.
    before = model.predict(env.normalize_obs(raw), deterministic=True)[0]
    initialize_reference_acceleration(env, model)
    raw[:, 21:24] = .03  # Gentle moving reference, formerly clipped to 10.
    after = model.predict(env.normalize_obs(raw), deterministic=True)[0]
    np.testing.assert_allclose(before, after, atol=1e-7)
    assert np.max(np.abs(env.normalize_obs(raw)[:, 21:24])) < .031
    env.close()
