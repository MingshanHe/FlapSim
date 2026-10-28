"""Train PPO on the IMU + TOF hover task (needs stable-baselines3).

python examples/rl/train_ppo.py --timesteps 500000 --n-envs 4 \
        --output models/imu_hover_ppo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imu_hover_env import make_hover_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("models/imu_hover_ppo"))
    parser.add_argument("--tensorboard-log", type=Path, default=None)
    parser.add_argument("--no-tof", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def factory(rank: int):
        return lambda: make_hover_env(seed=args.seed + rank, use_tof=not args.no_tof)

    env = (
        SubprocVecEnv([factory(i) for i in range(args.n_envs)])
        if args.n_envs > 1
        else DummyVecEnv([factory(0)])
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048 // max(args.n_envs, 1),
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=None
        if args.tensorboard_log is None
        else str(args.tensorboard_log),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    env.close()
    print(f"saved policy to {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
