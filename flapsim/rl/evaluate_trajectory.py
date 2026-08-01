"""Evaluate and visualize position/attitude trajectory tracking."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from stable_baselines3 import PPO

from .trajectory_env import TrajectoryTrackingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("flapsim/rl/models/trajectory_ppo.zip"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument(
        "--trajectory",
        choices=TrajectoryTrackingEnv.SUPPORTED_TRAJECTORIES,
        default="figure8",
    )
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    return parser.parse_args()


def _desired_body_z(roll_tilt: float, pitch_tilt: float) -> np.ndarray:
    roll = float(roll_tilt)
    pitch = float(pitch_tilt)
    return np.array(
        [
            math.sin(pitch) * math.cos(roll),
            -math.sin(roll),
            math.cos(pitch) * math.cos(roll),
        ]
    )


def _set_reference_visuals(viewer, env: TrajectoryTrackingEnv) -> None:
    """Draw path dots, moving target and desired body-z direction."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        for sample_time in np.linspace(0.0, env.period, 64, endpoint=False):
            position = env.reference_at(float(sample_time))["position"]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.005, 0.0, 0.0]),
                position,
                np.eye(3).ravel(),
                np.array([0.2, 0.55, 1.0, 0.55]),
            )
            scene.ngeom += 1

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.022, 0.0, 0.0]),
            env.target,
            np.eye(3).ravel(),
            np.array([0.1, 1.0, 0.1, 0.9]),
        )
        scene.ngeom += 1
        desired_z = _desired_body_z(
            env.desired_roll_tilt, env.desired_pitch_tilt
        )
        mujoco.mjv_connector(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.006,
            env.target,
            env.target + 0.12 * desired_z,
        )
        scene.geoms[scene.ngeom].rgba = np.array([1.0, 0.8, 0.1, 0.9])
        scene.ngeom += 1


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.realtime_factor <= 0.0:
        raise ValueError("--realtime-factor must be positive")
    env = TrajectoryTrackingEnv(
        trajectory=args.trajectory,
        randomize_reset=not args.no_randomize,
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
            position_errors = []
            velocity_errors = []
            attitude_errors = []
            rewards = []
            while True:
                wall_start = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                position_errors.append(
                    float(np.linalg.norm(info["position_error"]))
                )
                velocity_errors.append(
                    float(np.linalg.norm(info["velocity_error"]))
                )
                attitude_errors.append(
                    float(np.linalg.norm(info["attitude_error"]))
                )
                rewards.append(reward)
                if viewer is not None:
                    if not viewer.is_running():
                        env.close()
                        return
                    _set_reference_visuals(viewer, env)
                    viewer.sync()
                    remaining = (
                        env.control_dt / args.realtime_factor
                        - (time.perf_counter() - wall_start)
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
                if terminated or truncated:
                    break

            position_rms = float(
                np.sqrt(np.mean(np.square(position_errors)))
            )
            velocity_rms = float(
                np.sqrt(np.mean(np.square(velocity_errors)))
            )
            attitude_rms = math.degrees(
                float(np.sqrt(np.mean(np.square(attitude_errors))))
            )
            success = bool(info["is_success"] and not terminated)
            summaries.append(
                (position_rms, velocity_rms, attitude_rms, sum(rewards), success)
            )
            print(
                f"episode {episode + 1}: trajectory={env.trajectory}, "
                f"position_rms={position_rms:.4f} m, "
                f"attitude_rms={attitude_rms:.3f} deg, success={success}"
            )

    values = np.asarray([row[:4] for row in summaries])
    print(f"episodes: {args.episodes}")
    print(f"success rate: {np.mean([row[4] for row in summaries]):.1%}")
    print(f"mean position RMS: {values[:, 0].mean():.4f} m")
    print(f"mean velocity RMS: {values[:, 1].mean():.4f} m/s")
    print(f"mean attitude RMS: {values[:, 2].mean():.3f} deg")
    print(f"mean return: {values[:, 3].mean():.2f}")
    env.close()


if __name__ == "__main__":
    main()
