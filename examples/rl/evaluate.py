"""Evaluate a trained hover policy (or the zero-action baseline).

python examples/rl/evaluate.py --model models/imu_hover_ppo.zip --episodes 10
python examples/rl/evaluate.py --baseline --episodes 3
"""

from __future__ import annotations

import argparse

import numpy as np
from imu_hover_env import make_hover_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--baseline", action="store_true", help="zero actions instead of a policy"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--no-tof", action="store_true")
    args = parser.parse_args()
    if args.model is None and not args.baseline:
        parser.error("give --model or --baseline")

    policy = None
    if args.model is not None:
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device="cpu")

    env = make_hover_env(seed=args.seed, use_tof=not args.no_tof)
    returns, durations, heights = [], [], []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total = 0.0
        while True:
            action = (
                np.zeros(env.action_space.shape)
                if policy is None
                else policy.predict(obs, deterministic=True)[0]
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        durations.append(info["time"])
        heights.append(float(env.sim.data.xpos[env.sim.robot.root_body][2]))
        print(
            f"episode {episode}: return {total:8.2f}  duration {info['time']:.2f} s  "
            f"final height {heights[-1]:.3f} m"
        )
    print(
        f"mean return {np.mean(returns):.2f}, mean duration {np.mean(durations):.2f} s"
    )


if __name__ == "__main__":
    main()
