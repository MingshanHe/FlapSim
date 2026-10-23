"""Explicit six-feature observation upgrade without changing the old policy mean."""
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO


def expand_normalizer(path, env):
    with open(path, 'rb') as stream:
        normalizer = pickle.load(stream)
    old_size = normalizer.observation_space.shape[0]
    if env.observation_space.shape != (old_size+6,):
        raise ValueError('only the six centroidal features can be appended')
    normalizer.obs_rms.mean = np.r_[normalizer.obs_rms.mean, np.zeros(6)]
    normalizer.obs_rms.var = np.r_[normalizer.obs_rms.var, np.ones(6)]
    normalizer.observation_space = env.observation_space
    normalizer.set_venv(env)
    return normalizer


def expand_policy_weights(source, target):
    old = source.policy.state_dict()
    new = target.policy.state_dict()
    expandable = {'mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'}
    for name, value in new.items():
        if value.shape == old[name].shape:
            new[name] = old[name].clone()
        elif name in expandable and value.shape[0] == old[name].shape[0] and value.shape[1] == old[name].shape[1]+6:
            new[name] = torch.zeros_like(value)
            new[name][:, :old[name].shape[1]] = old[name]
        else:
            raise ValueError(f'unsupported policy transfer at {name}')
    target.policy.load_state_dict(new, strict=True)


def expand_policy(path, env, *, n_steps, batch_size, seed, device):
    source = PPO.load(str(path), device=device)
    target = PPO('MlpPolicy', env, learning_rate=source.learning_rate, n_steps=n_steps,
                 batch_size=batch_size, n_epochs=source.n_epochs, gamma=source.gamma,
                 gae_lambda=source.gae_lambda, clip_range=source.clip_range,
                 ent_coef=source.ent_coef, vf_coef=source.vf_coef,
                 max_grad_norm=source.max_grad_norm, target_kl=source.target_kl,
                 policy_kwargs=source.policy_kwargs, seed=seed, device=device, verbose=1)
    expand_policy_weights(source, target)
    target.num_timesteps = source.num_timesteps
    target._n_updates = source._n_updates
    return target


def initialize_reference_acceleration(normalizer, model):
    """Unfreeze hover-only reference inputs without an initial action jump.

    In the full observation contract columns 21:24 are reference acceleration.
    Their variance collapses to zero during hover. New trajectories would clip
    even tiny accelerations to +/-10 and activate previously untrained weights.
    """
    if normalizer.observation_space.shape[0] < 33:
        raise ValueError('expected the full IST V2 observation layout')
    normalizer.obs_rms.mean[21:24] = 0.
    normalizer.obs_rms.var[21:24] = 1.
    with torch.no_grad():
        for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
            network[0].weight[:, 21:24].zero_()
            state = model.policy.optimizer.state.get(network[0].weight, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == network[0].weight.shape:
                    value[:, 21:24].zero_()
