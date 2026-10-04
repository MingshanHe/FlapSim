"""Evaluate saved PPO policies on named trajectories with frozen normalization."""
import argparse
import csv
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
    parser.add_argument('--plot', action='store_true', help='save XY and altitude plots (requires matplotlib)')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    tracking, simulation, _ = load_run(args.run)
    output = args.output or args.run/'evaluation'
    if output.exists() and any(output.iterdir()):
        parser.error('evaluation output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    env = TrajectoryTrackingEnv(replace(tracking, randomize=False), simulation_config=simulation,
                                render_mode='human' if args.render else None)
    # Use the saved transform on raw Gym observations. Never update eval statistics.
    normalizer = VecNormalize.load(str(args.run/'vecnormalize.pkl'), DummyVecEnv([lambda: env]))
    normalizer.training = False
    normalizer.norm_reward = False
    policy = PPO.load(str(args.run/'policy.zip'), device='cpu')
    summary = []
    try:
        for name in args.trajectories:
            rows, episodes = [], []
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed+episode,
                                   options={'trajectory': name, 'phase': 2*np.pi*episode/args.episodes})
                total_reward = 0.
                errors = []
                while True:
                    started = time.monotonic()
                    action, _ = policy.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    errors.append(info['position_error'])
                    total_reward += reward
                    rows.append([episode, info['elapsed'], *info['position'], *info['target'],
                                 info['position_error'], info['motor_command'], *info['gimbal_command']])
                    if args.render:
                        time.sleep(max(0., tracking.control_dt-(time.monotonic()-started)))
                    if terminated or truncated:
                        break
                episodes.append({'episode': episode, 'rmse_m': float(np.sqrt(np.mean(np.square(errors)))),
                                 'max_error_m': float(max(errors)), 'seconds': info['elapsed'],
                                 'completion_fraction': min(1., info['elapsed']/tracking.episode_seconds),
                                 'success': info['is_success'], 'termination_reason': info['termination_reason'],
                                 'reward': total_reward})
            with (output/f'{name}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['episode', 'time', 'x', 'y', 'z', 'target_x', 'target_y', 'target_z',
                                 'position_error', 'motor_rad_s', 'pitch_rad', 'roll_rad'])
                writer.writerows(rows)
            result = {'trajectory': name, 'mean_rmse_m': float(np.mean([e['rmse_m'] for e in episodes])),
                      'success_rate': float(np.mean([e['success'] for e in episodes])),
                      'mean_completion_fraction': float(np.mean([e['completion_fraction'] for e in episodes])),
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
        normalizer.close()
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
