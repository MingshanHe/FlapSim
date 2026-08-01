"""Train residual PPO for position and roll/pitch trajectory tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .trajectory_env import TrajectoryTrackingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--trajectory",
        choices=("mixed", *TrajectoryTrackingEnv.SUPPORTED_TRAJECTORIES),
        default="mixed",
        help="train one path type or sample a type every episode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flapsim/rl/models/trajectory_ppo"),
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("flapsim/rl/runs"),
    )
    parser.add_argument("--run-name", default="trajectory_ppo")
    parser.add_argument("--checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if not args.run_name.strip():
        raise ValueError("--run-name must not be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    check_env(TrajectoryTrackingEnv(randomize_reset=False), warn=True)
    mixed = args.trajectory == "mixed"
    trajectory = "circle" if mixed else args.trajectory
    env = DummyVecEnv(
        [
            lambda: Monitor(
                TrajectoryTrackingEnv(
                    trajectory=trajectory,
                    randomize_trajectory=mixed,
                    randomize_reset=True,
                )
            )
        ]
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64], "log_std_init": -2.0},
        verbose=1,
        seed=args.seed,
        device="auto",
        tensorboard_log=str(args.tensorboard_log),
    )
    callback = None
    if args.checkpoints:
        callback = CheckpointCallback(
            save_freq=10_000,
            save_path=str(args.output.parent / "checkpoints"),
            name_prefix=args.output.name,
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=args.run_name,
    )
    model.save(args.output)
    env.close()
    print(f"saved trajectory PPO policy to {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
