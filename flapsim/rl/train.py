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
import mujoco
import numpy as np
import gymnasium
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from flapsim import load_config, save_config
from .env import PRESET, TrackingConfig, TrajectoryTrackingEnv
from .trajectories import NAMES
from .preflight import scan


def make_env(tracking, simulation, split=False, features='full'):
    env_type = TrajectoryTrackingEnv
    if features == 'centroidal':
        from .features import CentroidalTrackingEnv
        env_type = CentroidalTrackingEnv
    env = env_type(tracking, simulation_config=simulation)
    if split:
        from .native_fast import accelerate_fast
        accelerate_fast(env.sim)
    return Monitor(env,
                   info_keywords=('tracking_rmse', 'is_success', 'termination_reason'))


def implementation_sha256():
    """Prevent resuming across changes to the physics or observation/action contract."""
    root = Path(__file__).resolve().parents[1]
    files = list(root.glob('*.py'))
    for folder in ('actuator', 'config', 'controller', 'fluid'):
        files.extend((root/folder).glob('*.py'))
    files.extend([root/'rl'/name for name in ('env.py', 'trajectories.py', 'native.py', 'native_step.cpp')])
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
                for reason in ('tilt', 'tracking_error', 'ground', 'numerical', 'time_limit'):
                    self.logger.record_mean('termination/'+reason, float(info['termination_reason'] == reason))
        return True


class SaveBestNormalizer(BaseCallback):
    """An evaluation policy must use the statistics from that same evaluation."""
    def _on_step(self):
        self.model.get_vec_normalize_env().save(str(Path(self.parent.best_model_save_path)/'vecnormalize.pkl'))
        return True


class FixedSeedEvalCallback(EvalCallback):
    """Compare checkpoints on the same held-out reset sequence."""
    def __init__(self, *args, evaluation_seed=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_seed = evaluation_seed

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.evaluation_seed)
        return super()._on_step()


def load_run(path):
    path = Path(path)
    metadata = json.loads((path/'run.json').read_text())
    for file, key in (('policy.zip', 'exported_policy_sha256'),
                      ('vecnormalize.pkl', 'exported_normalizer_sha256')):
        if key in metadata and hashlib.sha256((path/file).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f'exported checkpoint changed: {file}')
    simulation = load_config(path/'simulation.json')
    digest = hashlib.sha256(simulation.path.read_bytes()).hexdigest()
    if digest != metadata['model_sha256']:
        raise ValueError('robot XML changed since this run; use the matching model or start a new run')
    current_digest = implementation_sha256()
    saved_digest = metadata.get('implementation_sha256')
    # The sole compatible extension adds ellipse without changing the six older
    # references, observations, actions, or physics. Both complete source hashes
    # are pinned; tests/trajectory_reference_v1.json records the older outputs.
    ellipse_extension = (
        '483e7652e4859f42d4e105d5d92813c083bef63bc510bbe88a5facb71f4ce012',
        'c20a0ed845126ef8a4e96c9908b40ff2d747c1002b48839ad2e17754cdd8240e')
    if saved_digest != current_digest:
        if (saved_digest, current_digest) != ellipse_extension:
            raise ValueError('physics/environment code changed or was not recorded; start a new run')
        metadata['loaded_compatibility'] = 'ellipse reference added; existing trajectories unchanged'
    if metadata.get('split_sha256'):
        from .native_fast import implementation_sha256 as split_sha256
        if metadata['split_sha256'] != split_sha256():
            raise ValueError('split-step implementation changed since this run')
    if metadata.get('features_sha256'):
        from .features import implementation_sha256 as features_sha256
        if metadata['features_sha256'] != features_sha256():
            raise ValueError('centroidal feature implementation changed since this run')
    runtime = metadata.get('versions', metadata.get('export_runtime_versions', {}))
    if runtime.get('mujoco', mujoco.__version__) != mujoco.__version__:
        raise ValueError('MuJoCo version differs from the recorded training runtime')
    return TrackingConfig(**metadata['tracking']), simulation, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PRESET)
    parser.add_argument('--output', type=Path, default=Path('runs/ist_v2_ppo'))
    transfer = parser.add_mutually_exclusive_group()
    transfer.add_argument('--resume', type=Path, help='existing run; keep its task/physics and normalizer')
    transfer.add_argument('--warm-start', type=Path, help='compatible policy/normalizer, with a new trajectory curriculum')
    parser.add_argument('--from-best', action='store_true', help='resume/warm-start the best policy and matching statistics')
    parser.add_argument('--trajectories', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--native', action='store_true', help='compiled equivalent physics loop (Linux/WSL, g++)')
    parser.add_argument('--split-step', action='store_true', help='validated split-step optimization of native physics')
    parser.add_argument('--features', choices=['full', 'centroidal'], help='resume preserves the existing feature set')
    parser.add_argument('--learning-rate', type=float, help='fresh runs default to 3e-4; resume preserves it unless specified')
    parser.add_argument('--log-std-init', type=float, default=-2.)
    parser.add_argument('--ent-coef', type=float, default=0.)
    parser.add_argument('--gamma', type=float, help='discount; fresh runs default to .995, resume preserves it')
    parser.add_argument('--control-dt', type=float, default=.01)
    parser.add_argument('--episode-seconds', type=float, default=8.)
    parser.add_argument('--period', type=float, default=8.)
    parser.add_argument('--radius', type=float, default=.2)
    parser.add_argument('--gravity-scale', type=float, default=1.)
    parser.add_argument('--trim-speed', type=float, default=4000.)
    parser.add_argument('--checkpoint-steps', type=int, default=10000)
    parser.add_argument('--eval-steps', type=int, default=16384)
    parser.add_argument('--skip-preflight', action='store_true',
                        help='only after a saved thrust scan for the same configuration')
    args = parser.parse_args()
    if min(args.total_timesteps, args.n_envs, args.n_steps, args.batch_size, args.checkpoint_steps, args.eval_steps) < 1:
        parser.error('step counts and batch/environment sizes must be positive')
    if args.batch_size < 2 or args.n_steps*args.n_envs % args.batch_size:
        parser.error('batch-size must be >=2 and divide n-steps*n-envs')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose --device cpu')
    if args.gamma is not None and not 0 < args.gamma < 1:
        parser.error('gamma must lie strictly between zero and one')
    if args.learning_rate is not None and not 0 < args.learning_rate < 1:
        parser.error('learning-rate must lie strictly between zero and one')
    if args.from_best and not (args.resume or args.warm_start):
        parser.error('from-best requires resume or warm-start')
    torch.set_num_threads(1)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty; use a new output directory even when resuming')
    if args.resume or args.warm_start:
        tracking, simulation, source_metadata = load_run(args.resume or args.warm_start)
        args.split_step = args.split_step or source_metadata['arguments'].get('split_step', False)
        source_features = source_metadata['arguments'].get('features') or 'full'
        args.features = args.features or source_features
        if source_features == 'centroidal' and args.features != source_features:
            parser.error('removing centroidal features is not a supported transfer')
        if args.warm_start:
            tracking = replace(tracking, trajectories=tuple(args.trajectories),
                               episode_seconds=args.episode_seconds, radius=args.radius, period=args.period)
    else:
        source_features = None
        args.features = args.features or 'full'
        simulation = load_config(args.config)
        tracking = TrackingConfig(trajectories=tuple(args.trajectories), control_dt=args.control_dt,
                                  episode_seconds=args.episode_seconds, radius=args.radius,
                                  period=args.period, gravity_scale=args.gravity_scale,
                                  initial_speed=args.trim_speed, native_physics=args.native)
    if args.split_step and not tracking.native_physics:
        parser.error('split-step requires --native (or a run already using native physics)')
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the resolved source model path; run.json detects later regeneration.
    save_config(replace(simulation, model_path=simulation.path), output/'simulation.json')
    metadata = {'tracking': asdict(tracking), 'seed': args.seed,
                'model_sha256': hashlib.sha256(simulation.path.read_bytes()).hexdigest(),
                'implementation_sha256': implementation_sha256(),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'python': platform.python_version(), 'status': 'preparing',
                'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__,
                             'torch': torch.__version__, 'gymnasium': gymnasium.__version__,
                             'stable_baselines3': stable_baselines3.__version__}}
    if args.split_step:
        from .native_fast import implementation_sha256 as split_sha256
        metadata['split_sha256'] = split_sha256()
    if args.features == 'centroidal':
        from .features import implementation_sha256 as features_sha256
        metadata['features_sha256'] = features_sha256()
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if not args.skip_preflight:
        checked = replace(simulation, scene=replace(simulation.scene,
                           gravity=tuple(g*tracking.gravity_scale for g in simulation.scene.gravity)))
        rows = scan(checked, [simulation.actuator.max_velocity], native=tracking.native_physics)
        (output/'preflight.json').write_text(json.dumps(rows, indent=2)+'\n')
        if rows[0]['lift_to_weight'] <= 1.:
            raise SystemExit('measured lift is below weight; inspect preflight.json before training')
    constructors = [partial(make_env, tracking, simulation, args.split_step, args.features) for _ in range(args.n_envs)]
    env = DummyVecEnv(constructors) if args.n_envs == 1 else SubprocVecEnv(constructors, start_method='spawn')
    transfer_source = args.resume or args.warm_start
    if transfer_source:
        metadata['transfer_source'] = str(transfer_source.resolve())
        source_dir = transfer_source/'best' if args.from_best else transfer_source
        policy_file = source_dir/('best_model.zip' if args.from_best else 'policy.zip')
        metadata['transfer_policy'] = str(policy_file.resolve())
        expand = source_features == 'full' and args.features == 'centroidal'
        if expand:
            from .transfer import expand_normalizer, expand_policy
            env = expand_normalizer(source_dir/'vecnormalize.pkl', env)
            metadata['observation_upgrade'] = 'append 6 centroidal features; zero new input weights; reset Adam state'
        else:
            env = VecNormalize.load(str(source_dir/'vecnormalize.pkl'), env)
        env.training = True
        env.norm_reward = True
        model = (expand_policy(policy_file, env, n_steps=args.n_steps, batch_size=args.batch_size,
                               seed=args.seed, device=args.device) if expand else
                 PPO.load(str(policy_file), env=env, device=args.device))
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model._setup_lr_schedule()
        if args.gamma is not None:
            model.gamma = args.gamma
            env.gamma = args.gamma
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        # Rebuild the rollout buffer when the process count or rollout length changes.
        model.rollout_buffer = model.rollout_buffer_class(model.n_steps, env.observation_space,
                               env.action_space, device=model.device, gamma=model.gamma,
                               gae_lambda=model.gae_lambda, n_envs=env.num_envs)
        if (args.warm_start and set(source_metadata['tracking']['trajectories']) == {'hover'}
                and set(tracking.trajectories) != {'hover'}):
            from .transfer import initialize_reference_acceleration
            initialize_reference_acceleration(env, model)
            metadata['reference_input_upgrade'] = 'zero acceleration input weights at 21:24; initialize variance to 1'
    else:
        gamma = args.gamma if args.gamma is not None else .995
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=gamma)
        model = PPO('MlpPolicy', env, learning_rate=args.learning_rate or 3e-4, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=10, gamma=gamma, gae_lambda=.95,
                    clip_range=.2, ent_coef=args.ent_coef, vf_coef=.5, max_grad_norm=.5,
                    target_kl=.03, policy_kwargs={'net_arch': {'pi': [128, 128], 'vf': [128, 128]},
                                                'log_std_init': args.log_std_init},
                    seed=args.seed, device=args.device, verbose=1)
    env.seed(args.seed)
    model.set_logger(configure(str(output), ['stdout', 'csv']))
    checkpoint = CheckpointCallback(save_freq=max(1, args.checkpoint_steps//args.n_envs),
                                    save_path=str(output/'checkpoints'), name_prefix='ppo',
                                    save_vecnormalize=True)
    evaluation_tasks = list(dict.fromkeys(tracking.trajectories))
    while len(evaluation_tasks) < 3:
        evaluation_tasks.append(evaluation_tasks[0])
    evaluation = SubprocVecEnv([
        partial(make_env, replace(tracking, trajectories=(task,)), simulation, args.split_step, args.features)
        for task in evaluation_tasks], start_method='spawn')
    evaluation = VecNormalize(evaluation, training=False, norm_reward=False)
    evaluation.seed(args.seed+10000)
    best = output/'best'
    best.mkdir()
    evaluator = FixedSeedEvalCallback(evaluation, eval_freq=max(1, args.eval_steps//args.n_envs),
                             n_eval_episodes=len(evaluation_tasks), deterministic=True, best_model_save_path=str(best),
                             log_path=str(output/'evaluation_history'),
                             evaluation_seed=args.seed+10000,
                             callback_on_new_best=SaveBestNormalizer())
    metadata['evaluation'] = {'trajectories': evaluation_tasks,
                              'seeds': [args.seed+10000+i for i in range(len(evaluation_tasks))]}
    started = time.monotonic()
    metadata['status'] = 'running'
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    try:
        model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=not bool(args.resume),
                    callback=[EpisodeMetrics(), checkpoint, evaluator])
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
        evaluation.close()
    print(f'Saved policy, normalizer and config in {output}. Evaluate before claiming tracking success.')


if __name__ == '__main__':
    main()
