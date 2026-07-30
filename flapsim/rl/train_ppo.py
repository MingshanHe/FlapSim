"""Train a small Stable-Baselines3 PPO policy for the hover task."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .hover_env import HoverEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flapsim/rl/models/hover_ppo"),
    )
    parser.add_argument("--checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    check_env(HoverEnv(randomize_reset=False), warn=True)
    env = DummyVecEnv([lambda: Monitor(HoverEnv())])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
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
    )
    callback = None
    if args.checkpoints:
        callback = CheckpointCallback(
            save_freq=10_000,
            save_path=str(args.output.parent / "checkpoints"),
            name_prefix=args.output.name,
        )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.output)
    env.close()
    print(f"saved PPO policy to {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
