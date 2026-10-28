"""Reward and termination helpers for hover tasks (true state, training only)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import mujoco
import numpy as np


def root_state(sim) -> dict[str, np.ndarray | float]:
    """Position, world velocity, tilt angle and body rates of the root body."""
    data = sim.data
    body = sim.robot.root_body
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        sim.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
    )
    body_z = data.xmat[body].reshape(3, 3)[:, 2]
    tilt = math.acos(float(np.clip(body_z[2], -1.0, 1.0)))
    rotation = data.xmat[body].reshape(3, 3)
    return {
        "position": data.xpos[body].copy(),
        "velocity": velocity[3:].copy(),
        "rates": rotation.T @ velocity[:3],
        "tilt": tilt,
    }


def hover_reward(
    target_height: float = 0.5,
    *,
    weights: Mapping[str, float] | None = None,
    stable_bonus: float = 0.5,
    stable_tolerance: Mapping[str, float] | None = None,
    crash_penalty: float = -10.0,
    terminated=None,
):
    """``exp(-(w_z |dz| + w_tilt tilt + w_v |v| + w_w |w| + w_a |da|^2))`` plus a
    bonus when hovering steadily and a penalty when the termination test fires.

    Defaults reproduce the IMU-hover training reward.  Returns a callable
    ``reward(env, obs, action) -> (value, terms)``.
    """
    w = {"z": 4.0, "tilt": 1.5, "velocity": 0.2, "rates": 0.03, "action_change": 0.01}
    if weights:
        w.update(weights)
    tol = {"z": 0.05, "tilt": 0.1, "velocity": 0.1}
    if stable_tolerance:
        tol.update(stable_tolerance)
    terminated_fn = terminated or hover_terminated()

    def reward(env, obs, action):
        state = root_state(env.sim)
        dz = abs(float(state["position"][2]) - target_height)
        speed = float(np.linalg.norm(state["velocity"]))
        rates = float(np.linalg.norm(state["rates"]))
        action_change = float(np.sum((np.asarray(action) - env.previous_action) ** 2))
        cost = (
            w["z"] * dz
            + w["tilt"] * state["tilt"]
            + w["velocity"] * speed
            + w["rates"] * rates
            + w["action_change"] * action_change
        )
        value = math.exp(-cost)
        stable = (
            dz < tol["z"] and state["tilt"] < tol["tilt"] and speed < tol["velocity"]
        )
        if stable:
            value += stable_bonus
        crashed = bool(terminated_fn(env))
        if crashed:
            value += crash_penalty
        terms = {
            "dz": dz,
            "tilt": state["tilt"],
            "speed": speed,
            "rates": rates,
            "stable": float(stable),
            "crashed": float(crashed),
        }
        return value, terms

    return reward


def hover_terminated(
    *,
    min_height: float = 0.08,
    max_height: float = 1.5,
    max_horizontal: float = 1.5,
    max_tilt_deg: float = 65.0,
):
    """Episode failure: too low or high, drifted away, or tipped over."""
    max_tilt = math.radians(max_tilt_deg)

    def terminated(env) -> bool:
        state = root_state(env.sim)
        position = state["position"]
        if not np.all(np.isfinite(position)):
            return True
        return bool(
            position[2] < min_height
            or position[2] > max_height
            or abs(position[0]) > max_horizontal
            or abs(position[1]) > max_horizontal
            or state["tilt"] > max_tilt
        )

    return terminated


__all__ = ["hover_reward", "hover_terminated", "root_state"]
