"""Train or resume a PPO policy, saving matching normalization and config."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from flapsim import load_config, save_config
from .env import PRESET, TrackingConfig, TrajectoryTrackingEnv
from .trajectories import NAMES
from .preflight import scan


def make_env(tracking, simulation):
    return Monitor(TrajectoryTrackingEnv(tracking, simulation_config=simulation),
                   info_keywords=('tracking_rmse', 'is_success', 'termination_reason'))


def implementation_sha256():
    """Prevent resuming across changes to the physics or observation/action contract."""
    root = Path(__file__).resolve().parents[1]
    files = list(root.glob('*.py'))
    for folder in ('actuator', 'config', 'controller', 'fluid'):
        files.extend((root/folder).glob('*.py'))
    files.extend([root/'rl/env.py', root/'rl/trajectories.py'])
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).replace('\\', '/').encode())
        digest.update(path.read_text(encoding='utf-8').replace('\r\n', '\n').encode())
    return digest.hexdigest()


class EpisodeMetrics(BaseCallback):
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.logger.record_mean('tracking/rmse_m', info['tracking_rmse'])
                self.logger.record_mean('tracking/success', float(info['is_success']))
                self.logger.record_mean('tracking/seconds', info['elapsed'])
        return True


def load_run(path):
    path = Path(path)
    metadata = json.loads((path/'run.json').read_text())
    simulation = load_config(path/'simulation.json')
    digest = hashlib.sha256(simulation.path.read_bytes()).hexdigest()
    if digest != metadata['model_sha256']:
        raise ValueError('robot XML changed since this run; use the matching model or start a new run')
    if metadata.get('implementation_sha256') != implementation_sha256():
        raise ValueError('physics/environment code changed or was not recorded; start a new run')
    return TrackingConfig(**metadata['tracking']), simulation, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PRESET)
    parser.add_argument('--output', type=Path, default=Path('runs/ist_v2_ppo'))
    parser.add_argument('--resume', type=Path, help='existing run; keep its task/physics and normalizer')
    parser.add_argument('--trajectories', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--control-dt', type=float, default=.01)
    parser.add_argument('--episode-seconds', type=float, default=8.)
    parser.add_argument('--period', type=float, default=8.)
    parser.add_argument('--radius', type=float, default=.2)
    parser.add_argument('--gravity-scale', type=float, default=1.)
    parser.add_argument('--trim-speed', type=float, default=4000.)
    parser.add_argument('--checkpoint-steps', type=int, default=10000)
    parser.add_argument('--skip-preflight', action='store_true',
                        help='only after a saved thrust scan for the same configuration')
    args = parser.parse_args()
    if min(args.total_timesteps, args.n_envs, args.n_steps, args.batch_size, args.checkpoint_steps) < 1:
        parser.error('step counts and batch/environment sizes must be positive')
    if args.batch_size < 2 or args.n_steps*args.n_envs % args.batch_size:
        parser.error('batch-size must be >=2 and divide n-steps*n-envs')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose --device cpu')
    torch.set_num_threads(1)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty; use a new output directory even when resuming')
    if args.resume:
        tracking, simulation, _ = load_run(args.resume)
    else:
        simulation = load_config(args.config)
        tracking = TrackingConfig(trajectories=tuple(args.trajectories), control_dt=args.control_dt,
                                  episode_seconds=args.episode_seconds, radius=args.radius,
                                  period=args.period, gravity_scale=args.gravity_scale,
                                  initial_speed=args.trim_speed)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the resolved source model path; run.json detects later regeneration.
    save_config(replace(simulation, model_path=simulation.path), output/'simulation.json')
    metadata = {'tracking': asdict(tracking), 'seed': args.seed,
                'model_sha256': hashlib.sha256(simulation.path.read_bytes()).hexdigest(),
                'implementation_sha256': implementation_sha256(),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'python': platform.python_version(), 'status': 'preparing'}
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if not args.skip_preflight:
        checked = replace(simulation, scene=replace(simulation.scene,
                           gravity=tuple(g*tracking.gravity_scale for g in simulation.scene.gravity)))
        rows = scan(checked, [simulation.actuator.max_velocity])
        (output/'preflight.json').write_text(json.dumps(rows, indent=2)+'\n')
        if rows[0]['lift_to_weight'] <= 1.:
            raise SystemExit('measured lift is below weight; inspect preflight.json before training')
    constructors = [partial(make_env, tracking, simulation) for _ in range(args.n_envs)]
    env = DummyVecEnv(constructors) if args.n_envs == 1 else SubprocVecEnv(constructors, start_method='spawn')
    if args.resume:
        env = VecNormalize.load(str(args.resume/'vecnormalize.pkl'), env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(str(args.resume/'policy.zip'), env=env, device=args.device)
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        # Rebuild the rollout buffer when the process count or rollout length changes.
        model.rollout_buffer = model.rollout_buffer_class(model.n_steps, env.observation_space,
                               env.action_space, device=model.device, gamma=model.gamma,
                               gae_lambda=model.gae_lambda, n_envs=env.num_envs)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=.99)
        model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=10, gamma=.99, gae_lambda=.95,
                    clip_range=.2, ent_coef=.005, vf_coef=.5, max_grad_norm=.5,
                    target_kl=.03, policy_kwargs={'net_arch': {'pi': [128, 128], 'vf': [128, 128]}},
                    seed=args.seed, device=args.device, verbose=1)
    env.seed(args.seed)
    model.set_logger(configure(str(output), ['stdout', 'csv']))
    checkpoint = CheckpointCallback(save_freq=max(1, args.checkpoint_steps//args.n_envs),
                                    save_path=str(output/'checkpoints'), name_prefix='ppo',
                                    save_vecnormalize=True)
    started = time.monotonic()
    metadata['status'] = 'running'
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    try:
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=not bool(args.resume),
                    callback=[EpisodeMetrics(), checkpoint])
        metadata['status'] = 'finished'
    except KeyboardInterrupt:
        metadata['status'] = 'interrupted'
    except BaseException:
        metadata['status'] = 'failed'
        raise
    finally:
        model.save(str(output/'policy'))
        env.save(str(output/'vecnormalize.pkl'))
        metadata.update(num_timesteps=int(model.num_timesteps), wall_seconds=time.monotonic()-started)
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
        env.close()
    print(f'Saved policy, normalizer and config in {output}. Evaluate before claiming tracking success.')


if __name__ == '__main__':
    main()
