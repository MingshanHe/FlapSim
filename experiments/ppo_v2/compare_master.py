"""Run from the archived master checkout (PYTHONPATH points there).

Compare saved PPO with the zero-residual stabilizer on identical seeds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from flapsim.rl.hover_env import HoverEnv
from flapsim.rl.trajectory_env import TrajectoryTrackingEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    results = []
    for task in ('hover', 'circle', 'figure8', 'ellipse'):
        env = HoverEnv(randomize_target=True) if task == 'hover' else TrajectoryTrackingEnv(trajectory=task)
        model = PPO.load('flapsim/rl/models/'+('hover_ppo.zip' if task == 'hover' else 'trajectory_ppo.zip'),
                         device='cpu', custom_objects={'lr_schedule': lambda _: 3e-4, 'clip_range': lambda _: .2})
        for mode in ('zero_residual', 'saved_ppo'):
            for seed in (1000, 1001, 1002):
                obs, _ = env.reset(seed=seed)
                errors = []
                while True:
                    action = np.zeros(3) if mode == 'zero_residual' else model.predict(obs, deterministic=True)[0]
                    obs, _, terminated, truncated, info = env.step(action)
                    errors.append(float(np.linalg.norm(info['position_error'])))
                    if terminated or truncated:
                        break
                results.append(dict(task=task, mode=mode, seed=seed,
                                    rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                                    completed=not terminated, seconds=len(errors)*env.control_dt,
                                    terminal_success=info['is_success']))
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    for task in ('hover', 'circle', 'figure8', 'ellipse'):
        for mode in ('zero_residual', 'saved_ppo'):
            rows = [r for r in results if r['task'] == task and r['mode'] == mode]
            print(task, mode, 'RMSE', np.mean([r['rmse_m'] for r in rows]),
                  'completion', np.mean([r['completed'] for r in rows]), flush=True)


if __name__ == '__main__':
    main()
