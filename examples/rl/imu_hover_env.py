"""IMU + TOF hover task built from public FlapSim plugins.

The policy sees only sensor readings (IMU attitude, gyro, accelerometer and a
downward range finder) plus its previous action, and commands the two gimbal
angles and the stroke frequency.  Physics: the floating-base cycle-averaged
model of the example robot.  Training uses the true state for the reward.
"""

from __future__ import annotations

import math

import numpy as np

from flapsim import Simulation
from flapsim.envs import SimulationEnv, hover_reward, hover_terminated
from flapsim.robot import builtin_robots

GIMBAL_LIMIT = 0.3  # rad, the URDF joint range
FREQUENCY_BAND = 0.10  # +/- 10 % of the hover frequency
COMMAND_SMOOTHING_S = 0.0082  # first-order lag between command and actuator


def make_sim():
    """Floating base, cycle-averaged aero, sensors; no classic hover PID."""
    setup = builtin_robots()["SingleThrustFlapper"].parent / "hover.yaml"
    sim = Simulation.from_config(setup)
    sim.remove("hover")
    sim.add({"type": "sensor/imu", "id": "imu", "rate_hz": 500.0})
    sim.add({"type": "sensor/rangefinder", "id": "tof", "rate_hz": 500.0})
    return sim


def randomize(env: SimulationEnv) -> None:
    """Small initial pose perturbation and a per-episode motor gain error."""
    sim = env.sim
    rng = env.rng
    qpos = sim.data.qpos
    qpos[0:2] += rng.uniform(-0.02, 0.02, 2)
    qpos[2] += rng.uniform(-0.03, 0.03)
    tilt = rng.uniform(-0.05, 0.05, 2)
    qpos[3:7] = np.array([1.0, tilt[0], tilt[1], 0.0]) / np.linalg.norm(
        [1.0, tilt[0], tilt[1], 0.0]
    )
    aero = sim["aero"]
    nominal = env.nominal_lift_coefficient
    aero.lift_coefficient = nominal * (1.0 + rng.uniform(-0.005, 0.005))


def make_hover_env(
    *,
    decision_dt: float = 0.01,
    episode_seconds: float = 6.0,
    target_height: float = 0.5,
    use_tof: bool = True,
    seed: int | None = None,
    render_mode: str | None = None,
) -> SimulationEnv:
    probe = make_sim()
    hover_hz = probe["aero"].hover_frequency_hz
    observations = [
        {"signal": "imu/euler"},
        {"signal": "imu/gyro"},
        {"signal": "imu/accel", "scale": 0.1},
    ]
    if use_tof:
        observations += [{"signal": "tof/range"}, {"signal": "tof/range_rate"}]
    actions = [
        {
            "command": "gimbal/pitch_joint",
            "low": -GIMBAL_LIMIT,
            "high": GIMBAL_LIMIT,
            "tau": COMMAND_SMOOTHING_S,
        },
        {
            "command": "gimbal/roll_joint",
            "low": -GIMBAL_LIMIT,
            "high": GIMBAL_LIMIT,
            "tau": COMMAND_SMOOTHING_S,
        },
        {
            "command": "stroke/frequency_hz",
            "low": hover_hz * (1.0 - FREQUENCY_BAND),
            "high": hover_hz * (1.0 + FREQUENCY_BAND),
            "tau": COMMAND_SMOOTHING_S,
            "initial": hover_hz,
        },
    ]
    env = SimulationEnv(
        make_sim,
        observations=observations,
        actions=actions,
        decision_dt=decision_dt,
        episode_seconds=episode_seconds,
        reward=hover_reward(target_height),
        terminated=hover_terminated(),
        on_reset=randomize,
        include_previous_action=True,
        render_mode=render_mode,
        seed=seed,
    )
    env.nominal_lift_coefficient = env.sim["aero"].lift_coefficient
    env.hover_frequency_hz = hover_hz
    return env


if __name__ == "__main__":
    env = make_hover_env(seed=0)
    obs, _ = env.reset()
    total = 0.0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(np.zeros(3))
        total += reward
        if terminated or truncated:
            break
    print(
        f"zero-action episode: {info['time']:.2f} s, return {total:.2f}, "
        f"obs dim {obs.shape[0]}"
    )
    tilt_deg = math.degrees(info["reward_terms"]["tilt"])
    print(f"hover frequency {env.hover_frequency_hz:.2f} Hz, tilt {tilt_deg:.1f} deg")
