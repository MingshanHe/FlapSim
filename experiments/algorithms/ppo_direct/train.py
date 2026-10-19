"""在完整扑翼模型上训练 PPO 直接控制策略（nod / dash / hover 混合）。

    python -m experiments.algorithms.ppo_direct.train --timesteps 400000 --n-envs 18 --run-name quick
    python -m experiments.algorithms.ppo_direct.train --timesteps 2500000 --n-envs 18 --run-name full --init-model runs/ppo_direct/models/quick.zip

每个并行环境各自持有一份 ResearchEnv（约 200 拍/秒，每 10 ms 决策一次 → 约 40 决策/秒/进程）。
模型存到 runs/ppo_direct/models/<run-name>.zip，TensorBoard 在 runs/ppo_direct/tb。在 CPU 上训练。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import DirectManeuverEnv, prepare_source

OUT = Path("runs/ppo_direct")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=2_500_000)
    p.add_argument("--n-envs", type=int, default=18)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--task", default="mixed", choices=["mixed", "nod", "dash", "hover"])
    p.add_argument("--task-weights", type=float, nargs=3, default=(0.4, 0.4, 0.2), metavar=("NOD", "DASH", "HOVER"))
    p.add_argument("--run-name", default="ppo_direct")
    p.add_argument("--init-model", type=Path, default=None, help="热启动模型")
    p.add_argument("--n-steps", type=int, default=256, help="每个环境每次更新采样的步数")
    p.add_argument("--checkpoint-every", type=int, default=100_000, help="按总步数计")
    return p.parse_args()


def make_env(args, rank: int):
    def _make():
        env = DirectManeuverEnv(task=args.task, task_weights=tuple(args.task_weights), seed_offset=rank)
        env.reset(seed=args.seed * 1000 + rank)
        return Monitor(env, info_keywords=("is_success",))
    return _make


def main() -> None:
    args = parse_args()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "tb").mkdir(parents=True, exist_ok=True)
    prepare_source()
    env = SubprocVecEnv([make_env(args, i) for i in range(args.n_envs)], start_method="fork")
    if args.init_model is not None:
        model = PPO.load(args.init_model, env=env, device="cpu", tensorboard_log=str(OUT / "tb"), seed=args.seed)
        print(f"warm start from {args.init_model}", flush=True)
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=args.n_steps, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.5}, verbose=1, seed=args.seed,
                    device="cpu", tensorboard_log=str(OUT / "tb"))
    callback = CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(OUT / "models" / "checkpoints"), name_prefix=args.run_name)
    model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=args.run_name, reset_num_timesteps=args.init_model is None)
    model.save(OUT / "models" / args.run_name)
    env.close()
    print(f"saved {OUT / 'models' / args.run_name}.zip", flush=True)


if __name__ == "__main__":
    main()
