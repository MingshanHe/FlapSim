"""Evaluate a trained hover policy without reset randomization."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from stable_baselines3 import PPO

from .hover_env import HoverEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("flapsim/rl/models/hover_ppo.zip"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="disable initial position and attitude disturbances",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="display the MuJoCo viewer while evaluating",
    )
    parser.add_argument(
        "--random-target",
        action="store_true",
        help="sample a new target point for every episode",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="visual playback speed; 1.0 is real time",
    )
    return parser.parse_args()


def _set_target_marker(viewer, target: np.ndarray) -> None:
    """Draw a green sphere at the current target in the passive viewer."""
    with viewer.lock():
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0.0, 0.0]),
            np.asarray(target, dtype=float),
            np.eye(3).ravel(),
            np.array([0.1, 1.0, 0.1, 0.85]),
        )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.realtime_factor <= 0.0:
        raise ValueError("--realtime-factor must be positive")
    env = HoverEnv(
        randomize_reset=not args.no_randomize,
        randomize_target=args.random_target,
        episode_seconds=12.0 if args.random_target else 6.0,
    )
    model = PPO.load(args.model)
    summaries = []

    viewer_context = nullcontext(None)
    if args.render:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(env.model, env.data)

    with viewer_context as viewer:
        if viewer is not None:
            camera_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera"
            )
            if camera_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id

        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            errors = []
            tilts = []
            rewards = []
            success = False
            if viewer is not None:
                _set_target_marker(viewer, env.target)
                viewer.sync()
            while True:
                wall_start = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                errors.append(float(np.linalg.norm(info["position_error"])))
                tilts.append(math.degrees(info["tilt"]))
                rewards.append(reward)
                success = bool(info["is_success"])
                if viewer is not None:
                    if not viewer.is_running():
                        env.close()
                        return
                    viewer.sync()
                    frame_time = env.model.opt.timestep * env.frame_skip
                    remaining = (
                        frame_time / args.realtime_factor
                        - (time.perf_counter() - wall_start)
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
                if terminated or truncated:
                    break
            summaries.append(
                (
                    float(np.mean(errors)),
                    float(errors[-1]),
                    float(max(tilts)),
                    float(sum(rewards)),
                    success and not terminated,
                )
            )
            print(
                f"episode {episode + 1}: target={np.round(env.target, 3)}, "
                f"final_error={errors[-1]:.4f} m, "
                f"success={success and not terminated}"
            )

    values = np.asarray([row[:4] for row in summaries])
    success_rate = np.mean([row[4] for row in summaries])
    print(f"episodes: {args.episodes}")
    print(f"success rate: {success_rate:.1%}")
    print(f"mean position error: {values[:, 0].mean():.4f} m")
    print(f"mean final error: {values[:, 1].mean():.4f} m")
    print(f"worst tilt: {values[:, 2].max():.2f} deg")
    print(f"mean return: {values[:, 3].mean():.2f}")
    env.close()


if __name__ == "__main__":
    main()
