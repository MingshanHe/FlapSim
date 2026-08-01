"""Trajectory tracking task for position and controllable attitude."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from .hover_env import HoverEnv


class TrajectoryTrackingEnv(HoverEnv):
    """Track a smooth 3-D loop and its compatible roll/pitch attitude.

    Yaw is intentionally not included: the current robot has no independent
    yaw control channel. Desired roll and pitch include the acceleration and
    drag compensation required by the reference trajectory.
    """

    SUPPORTED_TRAJECTORIES = ("circle", "figure8", "ellipse")

    def __init__(
        self,
        *,
        trajectory: str = "circle",
        randomize_trajectory: bool = False,
        radius: float = 0.20,
        height_amplitude: float = 0.04,
        period: float = 8.0,
        episode_seconds: float = 16.0,
        randomize_reset: bool = True,
    ) -> None:
        if trajectory not in self.SUPPORTED_TRAJECTORIES:
            raise ValueError(
                f"unknown trajectory {trajectory!r}; available: "
                f"{self.SUPPORTED_TRAJECTORIES}"
            )
        if radius <= 0.0 or height_amplitude < 0.0 or period <= 0.0:
            raise ValueError("radius and period must be positive")
        self.trajectory = trajectory
        self.randomize_trajectory = bool(randomize_trajectory)
        self.radius = float(radius)
        self.height_amplitude = float(height_amplitude)
        self.period = float(period)
        self.trajectory_time = 0.0
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.target_acceleration = np.zeros(3, dtype=np.float64)
        self.desired_roll_tilt = 0.0
        self.desired_pitch_tilt = 0.0
        initial = self.reference_at(0.0)
        super().__init__(
            target=tuple(initial["position"]),
            spawn_position=tuple(initial["position"]),
            randomize_reset=randomize_reset,
            randomize_target=False,
            episode_seconds=episode_seconds,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),
            dtype=np.float32,
        )
        self._set_reference(0.0)

    def reference_at(self, time_value: float) -> dict[str, Any]:
        """Return desired position, derivatives and compatible attitude."""
        omega = 2.0 * math.pi / self.period
        phase = omega * float(time_value)
        sin_phase, cos_phase = math.sin(phase), math.cos(phase)
        if self.trajectory == "circle":
            position = np.array(
                [
                    self.radius * (cos_phase - 1.0),
                    self.radius * sin_phase,
                    0.55 + self.height_amplitude * sin_phase,
                ]
            )
            velocity = np.array(
                [
                    -self.radius * omega * sin_phase,
                    self.radius * omega * cos_phase,
                    self.height_amplitude * omega * cos_phase,
                ]
            )
            acceleration = np.array(
                [
                    -self.radius * omega**2 * cos_phase,
                    -self.radius * omega**2 * sin_phase,
                    -self.height_amplitude * omega**2 * sin_phase,
                ]
            )
        elif self.trajectory == "figure8":
            sin_double = math.sin(2.0 * phase)
            cos_double = math.cos(2.0 * phase)
            position = np.array(
                [
                    self.radius * sin_phase,
                    0.5 * self.radius * sin_double,
                    0.55 + self.height_amplitude * sin_phase,
                ]
            )
            velocity = np.array(
                [
                    self.radius * omega * cos_phase,
                    self.radius * omega * cos_double,
                    self.height_amplitude * omega * cos_phase,
                ]
            )
            acceleration = np.array(
                [
                    -self.radius * omega**2 * sin_phase,
                    -2.0 * self.radius * omega**2 * sin_double,
                    -self.height_amplitude * omega**2 * sin_phase,
                ]
            )
        else:  # ellipse
            position = np.array(
                [
                    1.2 * self.radius * (cos_phase - 1.0),
                    0.7 * self.radius * sin_phase,
                    0.55 + self.height_amplitude * sin_phase,
                ]
            )
            velocity = np.array(
                [
                    -1.2 * self.radius * omega * sin_phase,
                    0.7 * self.radius * omega * cos_phase,
                    self.height_amplitude * omega * cos_phase,
                ]
            )
            acceleration = np.array(
                [
                    -1.2 * self.radius * omega**2 * cos_phase,
                    -0.7 * self.radius * omega**2 * sin_phase,
                    -self.height_amplitude * omega**2 * sin_phase,
                ]
            )
        # AveragedHoverAero uses 0.06 N/(m/s) horizontal damping for a
        # 0.03 kg body, equivalent to 2 /s in acceleration units.
        compensated_acceleration = acceleration[:2] + 2.0 * velocity[:2]
        desired_pitch_tilt = float(
            np.clip(compensated_acceleration[0] / 9.81, -0.10, 0.10)
        )
        desired_roll_tilt = float(
            np.clip(-compensated_acceleration[1] / 9.81, -0.10, 0.10)
        )
        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "desired_roll_tilt": desired_roll_tilt,
            "desired_pitch_tilt": desired_pitch_tilt,
            "phase": phase,
        }

    def _set_reference(self, time_value: float) -> None:
        reference = self.reference_at(time_value)
        self.trajectory_time = float(time_value)
        self.target[:] = reference["position"]
        self.target_velocity[:] = reference["velocity"]
        self.target_acceleration[:] = reference["acceleration"]
        self.desired_roll_tilt = reference["desired_roll_tilt"]
        self.desired_pitch_tilt = reference["desired_pitch_tilt"]

    def _state(self) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = super()._state()
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        body_z = rotation[:, 2]
        roll_tilt = -float(body_z[1])
        pitch_tilt = float(body_z[0])
        velocity_error = info["linear_velocity"] - self.target_velocity
        attitude_error = np.array(
            [
                roll_tilt - self.desired_roll_tilt,
                pitch_tilt - self.desired_pitch_tilt,
            ],
            dtype=np.float64,
        )
        phase = 2.0 * math.pi * self.trajectory_time / self.period
        observation = np.concatenate(
            (
                info["position_error"],
                velocity_error,
                attitude_error,
                info["angular_velocity"],
                self.target_velocity,
                self.target_acceleration,
                (math.sin(phase), math.cos(phase)),
            )
        ).astype(np.float32)
        info.update(
            {
                "target_velocity": self.target_velocity.copy(),
                "target_acceleration": self.target_acceleration.copy(),
                "velocity_error": velocity_error,
                "roll_tilt": roll_tilt,
                "pitch_tilt": pitch_tilt,
                "desired_roll_tilt": self.desired_roll_tilt,
                "desired_pitch_tilt": self.desired_pitch_tilt,
                "attitude_error": attitude_error,
                "trajectory_time": self.trajectory_time,
                "trajectory_phase": phase,
                "trajectory": self.trajectory,
            }
        )
        return observation, info

    def _reference_action(self, info: dict[str, Any]) -> np.ndarray:
        error = info["position_error"]
        velocity_error = info["velocity_error"]
        angular_velocity = info["angular_velocity"]
        vertical_ratio = max(
            0.2, (9.81 + float(self.target_acceleration[2])) / 9.81
        )
        frequency = (
            self.HOVER_FREQUENCY * math.sqrt(vertical_ratio)
            - 8.0 * float(error[2])
            - 4.0 * float(velocity_error[2])
        )
        if frequency >= self.HOVER_FREQUENCY:
            frequency_action = (frequency - self.HOVER_FREQUENCY) / (
                55.0 - self.HOVER_FREQUENCY
            )
        else:
            frequency_action = (frequency - self.HOVER_FREQUENCY) / (
                self.HOVER_FREQUENCY - 20.0
            )

        corrected_pitch = float(
            np.clip(
                self.desired_pitch_tilt
                - 0.12 * error[0]
                - 0.55 * velocity_error[0],
                -0.10,
                0.10,
            )
        )
        corrected_roll = float(
            np.clip(
                self.desired_roll_tilt
                + 0.12 * error[1]
                + 0.55 * velocity_error[1],
                -0.10,
                0.10,
            )
        )
        plate_pitch = (
            -0.80 * (info["roll_tilt"] - corrected_roll)
            - 0.06 * float(angular_velocity[0])
        )
        plate_roll = (
            0.80 * (info["pitch_tilt"] - corrected_pitch)
            + 0.06 * float(angular_velocity[1])
        )
        return np.clip(
            np.array(
                [frequency_action, plate_pitch / 0.1, plate_roll / 0.1],
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

    def _reward(self, info: dict[str, Any]) -> float:
        position_cost = 3.0 * float(np.linalg.norm(info["position_error"]))
        velocity_cost = 0.6 * float(np.linalg.norm(info["velocity_error"]))
        attitude_cost = 2.0 * float(np.linalg.norm(info["attitude_error"]))
        angular_cost = 0.02 * float(np.linalg.norm(info["angular_velocity"]))
        effort_cost = 0.01 * float(self.last_action @ self.last_action)
        return float(
            math.exp(
                -position_cost
                - velocity_cost
                - attitude_cost
                - angular_cost
                - effort_cost
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        previous_initial = self.reference_at(0.0)["position"]
        self._set_reference(0.0)
        observation, _ = super().reset(seed=seed, options=options)
        if self.randomize_trajectory:
            self.trajectory = str(
                self.np_random.choice(self.SUPPORTED_TRAJECTORIES)
            )
            new_initial = self.reference_at(0.0)["position"]
            self.data.qpos[self.free_qpos : self.free_qpos + 3] += (
                new_initial - previous_initial
            )
        self._set_reference(0.0)
        # Start on the trajectory velocity to avoid an artificial impulse at
        # the beginning of every episode.
        self.data.qvel[self.free_dof : self.free_dof + 3] = self.target_velocity
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._set_reference(self.trajectory_time + self.control_dt)
        observation, reward, terminated, truncated, info = super().step(action)
        info["is_success"] = bool(
            not terminated
            and np.linalg.norm(info["position_error"]) < 0.12
            and np.linalg.norm(info["velocity_error"]) < 0.20
            and np.linalg.norm(info["attitude_error"]) < math.radians(8.0)
        )
        return observation, reward, terminated, truncated, info
