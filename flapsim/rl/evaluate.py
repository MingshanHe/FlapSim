"""Evaluate saved PPO policies on named trajectories with frozen normalization."""
import argparse
import csv
import hashlib
from io import BytesIO
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .env import TrajectoryTrackingEnv
from .train import load_run
from .trajectories import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--trajectories', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--randomize', action='store_true', help='held-out reset perturbations and trajectory dimensions')
    parser.add_argument('--best', action='store_true', help='evaluate the best policy and its matching normalizer')
    parser.add_argument('--zero-policy', action='store_true', help='evaluate zero actions as the fixed-trim baseline')
    parser.add_argument('--video', action='store_true', help='save MP4 at 25 fps; requires imageio and imageio-ffmpeg')
    parser.add_argument('--reference-physics', action='store_true', help='validate using the unsplit original stepping order')
    parser.add_argument('--seconds', type=float, help='evaluate for a different duration, e.g. two trajectory periods')
    parser.add_argument('--period', type=float, help='override the nominal reference period (reported with the results)')
    parser.add_argument('--plot', action='store_true', help='save XY and altitude plots (requires matplotlib)')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.video and args.render:
        parser.error('choose video or the interactive viewer')
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('seconds must be finite and positive')
    if args.period is not None and (not np.isfinite(args.period) or args.period <= 0):
        parser.error('period must be finite and positive')
    tracking, simulation, metadata = load_run(args.run)
    if args.seconds is not None:
        tracking = replace(tracking, episode_seconds=args.seconds)
    if args.period is not None:
        tracking = replace(tracking, period=args.period)
    output = args.output or args.run/'evaluation'
    if output.exists() and any(output.iterdir()):
        parser.error('evaluation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    env_type = TrajectoryTrackingEnv
    if metadata['arguments'].get('features') == 'centroidal':
        from .features import CentroidalTrackingEnv
        env_type = CentroidalTrackingEnv
    env = env_type(replace(tracking, randomize=args.randomize), simulation_config=simulation,
                                render_mode='human' if args.render else 'rgb_array' if args.video else None)
    if metadata['arguments'].get('split_step') and not args.reference_physics:
        from .native_fast import accelerate_fast
        accelerate_fast(env.sim)
    # Use the saved transform on raw Gym observations. Never update eval statistics.
    policy_dir = args.run/'best' if args.best else args.run
    normalizer = VecNormalize.load(str(policy_dir/'vecnormalize.pkl'), DummyVecEnv([lambda: env]))
    normalizer.training = False
    normalizer.norm_reward = False
    policy_bytes = (policy_dir/('best_model.zip' if args.best else 'policy.zip')).read_bytes()
    policy_hash = hashlib.sha256(policy_bytes).hexdigest()
    normalizer_hash = hashlib.sha256((policy_dir/'vecnormalize.pkl').read_bytes()).hexdigest()
    policy = PPO.load(BytesIO(policy_bytes), device='cpu')
    summary = []
    video_writer = None
    try:
        for name in args.trajectories:
            rows, episodes = [], []
            for episode in range(args.episodes):
                options = {'trajectory': name}
                if not args.randomize:
                    options['phase'] = 2*np.pi*episode/args.episodes
                obs, _ = env.reset(seed=args.seed+episode, options=options)
                total_reward = 0.
                errors = []
                video_writer = None
                if args.video:
                    import imageio.v2 as imageio
                    video_writer = imageio.get_writer(str(output/f'{name}_{episode}.mp4'), fps=25)
                frame_stride = max(1, round(1/(25*tracking.control_dt)))
                while True:
                    started = time.monotonic()
                    action, _ = policy.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
                    if args.zero_policy:
                        action[:] = 0
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    errors.append(info['position_error'])
                    total_reward += reward
                    if video_writer is not None and (len(errors)-1) % frame_stride == 0:
                        video_writer.append_data(env.render())
                    rotation = env.sim.data.xmat[env.body].reshape(3, 3)
                    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
                    pitch = np.arcsin(np.clip(-rotation[2, 0], -1, 1))
                    tilt = np.arccos(np.clip(rotation[2, 2], -1, 1))
                    rows.append([episode, info['elapsed'], *info['position'], *info['target'],
                                 info['position_error'], info['motor_command'], *info['gimbal_command'],
                                 tilt, roll, pitch, *(env.sim.data.cvel[env.body, :3] @ rotation),
                                 float(env.sim.data.joint('motor_joint').qvel[0]),
                                 float(env.sim.data.joint('pitch_joint').qpos[0]),
                                 float(env.sim.data.joint('roll_joint').qpos[0])])
                    if args.render:
                        time.sleep(max(0., tracking.control_dt-(time.monotonic()-started)))
                    if terminated or truncated:
                        break
                if video_writer is not None:
                    video_writer.close()
                    video_writer = None
                episodes.append({'episode': episode, 'rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
                                 'max_error_m': float(max(errors)), 'seconds': info['elapsed'],
                                 'completion_fraction': min(1., info['elapsed']/tracking.episode_seconds),
                                 'success': info['is_success'], 'termination_reason': info['termination_reason'],
                                 'reference_radius_m': env.reference.radius,
                                 'reference_period_s': env.reference.period,
                                 'reference_phase_rad': env.reference.phase,
                                 'reward': total_reward})
            with (output/f'{name}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['episode', 'time', 'x', 'y', 'z', 'target_x', 'target_y', 'target_z',
                                 'position_error', 'motor_rad_s', 'pitch_rad', 'roll_rad',
                                 'body_tilt', 'body_roll', 'body_pitch', 'omega_x', 'omega_y', 'omega_z',
                                 'shaft_rad_s', 'actual_pitch', 'actual_roll'])
                writer.writerows(rows)
            result = {'trajectory': name, 'mean_rmse_m': float(np.mean([e['rmse_m'] for e in episodes])),
                      'success_rate': float(np.mean([e['success'] for e in episodes])),
                      'mean_completion_fraction': float(np.mean([e['completion_fraction'] for e in episodes])),
                      'randomized_resets': args.randomize, 'seed': args.seed,
                      'controller': 'zero_action' if args.zero_policy else 'ppo',
                      'episode_seconds': tracking.episode_seconds,
                      'nominal_period_s': tracking.period,
                      'physics_engine': 'split' if metadata['arguments'].get('split_step') and not args.reference_physics else 'original',
                      'policy_timesteps': int(policy.num_timesteps),
                      'policy_sha256': policy_hash,
                      'normalizer_sha256': normalizer_hash,
                      'episodes': episodes}
            summary.append(result)
            print(json.dumps(result), flush=True)
            if args.plot:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                values = np.array(rows)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for episode in range(args.episodes):
                    run = values[values[:, 0] == episode]
                    axes[0].plot(run[:, 5], run[:, 6], '--', label='target' if episode == 0 else None)
                    if name == 'hover':
                        axes[0].scatter(run[0, 5], run[0, 6], marker='+', color='tab:blue', s=70)
                    axes[0].plot(run[:, 2], run[:, 3], label='PPO' if episode == 0 else None)
                    axes[1].plot(run[:, 1], run[:, 7], '--')
                    axes[1].plot(run[:, 1], run[:, 4])
                axes[0].set(xlabel='x (m)', ylabel='y (m)', aspect='equal')
                axes[1].set(xlabel='time (s)', ylabel='altitude (m)')
                axes[0].legend()
                fig.suptitle(f'{name}: RMSE {result["mean_rmse_m"]:.3f} m; '
                             f'completion {result["mean_completion_fraction"]:.0%}')
                fig.tight_layout()
                fig.savefig(output/f'{name}.png', dpi=160)
                plt.close(fig)
    finally:
        if video_writer is not None:
            video_writer.close()
        normalizer.close()
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
