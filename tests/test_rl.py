"""Gymnasium wrapper, hover task and a PPO smoke run (needs the rl extra)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from flapsim.core import Simulation  # noqa: E402
from flapsim.envs import (  # noqa: E402  # noqa: E402
    Action,
    Observation,
    SimulationEnv,
    hover_reward,
    hover_terminated,
)
from flapsim.robot import builtin_robots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
from imu_hover_env import make_hover_env  # noqa: E402

HOVER = builtin_robots()["SingleThrustFlapper"].parent / "hover.yaml"


def _make_sim():
    sim = Simulation.from_config(HOVER)
    sim.remove("hover")
    sim.add({"type": "sensor/imu", "rate_hz": 500.0})
    return sim


def test_action_and_observation_specs():
    action = Action.parse({"command": "stroke/frequency_hz", "low": 30.0, "high": 50.0})
    assert action.signal == "cmd/stroke/frequency_hz"
    assert (
        action.to_value(0.0) == 40.0
        and action.to_value(1.0) == 50.0
        and action.to_value(-3.0) == 30.0
    )
    assert action.to_normalized(45.0) == pytest.approx(0.5)
    obs = Observation.parse({"signal": "imu/accel", "scale": 0.1, "clip": 1.0})
    assert obs.signal == "imu/accel" and obs.clip == 1.0


def test_env_spaces_commands_and_episode_flow():
    env = SimulationEnv(
        _make_sim,
        observations=["imu/euler", "imu/gyro", {"signal": "imu/accel", "scale": 0.1}],
        actions=[
            {"command": "gimbal/pitch_joint", "low": -0.3, "high": 0.3},
            {
                "command": "stroke/frequency_hz",
                "low": 36.0,
                "high": 44.0,
                "initial": 40.0,
                "tau": 0.005,
            },
        ],
        decision_dt=0.01,
        episode_seconds=0.1,
        reward=hover_reward(0.5),
        terminated=hover_terminated(),
        seed=1,
    )
    assert env.observation_space.shape == (9 + 2,)
    assert env.action_space.shape == (2,)
    obs, info = env.reset(seed=1)
    assert obs.dtype == np.float32 and obs.shape == (11,)
    assert env.sim.signals["stroke/frequency_hz"] == pytest.approx(40.0)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
    assert env.sim["gimbal"].targets["pitch_joint"] == pytest.approx(0.3)
    # the smoothed frequency command moves towards 44 Hz but has not arrived
    assert 40.0 < env.sim.signals["stroke/frequency_hz"] < 44.0
    assert (
        "reward_terms" in info
        and 0.0 < reward < 2.0
        and not terminated
        and not truncated
    )
    for _ in range(9):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        if terminated or truncated:
            break
    assert truncated or terminated
    env.close()


def test_env_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    env = make_hover_env(episode_seconds=0.2, seed=3)
    assert env.observation_space.shape == (14,)
    check_env(env, skip_render_check=True)
    env.close()


def test_hover_termination_and_crash_penalty():
    env = make_hover_env(episode_seconds=3.0, seed=5)
    env.reset(seed=5)
    terminated = hover_terminated(min_height=0.08)
    assert not terminated(env)
    env.sim.data.qpos[2] = 0.01
    import mujoco

    mujoco.mj_forward(env.sim.model, env.sim.data)
    assert terminated(env)
    value, terms = hover_reward(0.5)(env, None, np.zeros(3))
    assert value < -5.0 and terms["crashed"] == 1.0


def test_zero_action_episode_runs_in_the_hover_task():
    env = make_hover_env(episode_seconds=1.0, seed=7)
    obs, _ = env.reset(seed=7)
    total = 0.0
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        total += reward
        steps += 1
        if terminated or truncated:
            break
    assert steps >= 5
    assert np.all(np.isfinite(obs))


def test_ppo_smoke_training():
    sb3 = pytest.importorskip("stable_baselines3")
    env = make_hover_env(episode_seconds=0.5, seed=11)
    model = sb3.PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=128)
    obs, _ = env.reset(seed=12)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    env.close()
