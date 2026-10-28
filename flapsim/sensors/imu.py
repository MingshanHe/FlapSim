"""Inertial measurement unit: attitude estimate, gyro and accelerometer."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from flapsim.plugins.base import SensorPlugin
from flapsim.plugins.registry import register

from .noise import Conditioner


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix to roll, pitch, yaw (Z-Y-X, radians)."""
    pitch = -math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw])


@register
class IMU(SensorPlugin):
    """Attitude (AHRS-like low-pass of the true angles), body rates and
    specific force at ``rate_hz``.

    Defaults reproduce the IMU model used for the IMU-only hover policies:
    attitude low-pass 20 ms with 0.02 degree noise, gyro noise 0.002 rad/s,
    accelerometer noise 0.02 m/s^2 with a per-episode bias of up to
    0.01 m/s^2.  The accelerometer measures ``R^T (a - g)``, so a hovering or
    resting body reads ``[0, 0, +9.81]``.  Publishes ``<id>/euler``,
    ``<id>/gyro``, ``<id>/accel`` and ``<id>/quat`` (true attitude, for
    reference).
    """

    name = "imu"
    provides = ("euler", "gyro", "accel", "quat")
    example_config = {"rate_hz": 500.0}

    def __init__(
        self,
        *,
        body: str | None = None,
        rate_hz: float = 500.0,
        attitude_tau: float = 0.02,
        attitude_noise_deg: float = 0.02,
        gyro_noise: float = 0.002,
        gyro_bias_range: float = 0.0,
        accel_noise: float = 0.02,
        accel_bias_range: float = 0.01,
        seed: int = 0,
        id: str | None = "imu",
    ) -> None:
        super().__init__(
            id=id,
            body=body,
            rate_hz=rate_hz,
            attitude_tau=attitude_tau,
            attitude_noise_deg=attitude_noise_deg,
            gyro_noise=gyro_noise,
            gyro_bias_range=gyro_bias_range,
            accel_noise=accel_noise,
            accel_bias_range=accel_bias_range,
            seed=seed,
        )
        self.body_name = body
        self.rate_hz = float(rate_hz)
        self.attitude = Conditioner(
            size=3,
            rate_hz=rate_hz,
            tau=attitude_tau,
            noise_std=math.radians(attitude_noise_deg),
            seed=seed,
            wrap_angles=True,
        )
        self.gyro = Conditioner(
            size=3,
            rate_hz=rate_hz,
            noise_std=gyro_noise,
            bias_range=gyro_bias_range,
            seed=seed + 1,
        )
        self.accel = Conditioner(
            size=3,
            rate_hz=rate_hz,
            noise_std=accel_noise,
            bias_range=accel_bias_range,
            seed=seed + 2,
        )
        self.body_id = -1
        self._velocity = np.zeros(6)
        self._previous_velocity: np.ndarray | None = None
        self._previous_time: float | None = None
        self._next_time = -math.inf

    def setup(self) -> None:
        name = self.body_name or self.robot.root_body_name
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.body_id < 0:
            raise ValueError(f"IMU body {name!r} not found")
        self.reset()

    def reset(self) -> None:
        self._previous_velocity = None
        self._previous_time = None
        self._next_time = -math.inf
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        self.attitude.reset(rotation_to_rpy(rotation), reseed=True)
        self.gyro.reset(np.zeros(3), reseed=True)
        gravity = np.asarray(self.model.opt.gravity, dtype=float)
        self.accel.reset(rotation.T @ (-gravity), reseed=True)

    def pre_control(self) -> None:
        data = self.data
        time = float(data.time)
        if time + 1e-12 < self._next_time:
            return
        self._next_time = time + 1.0 / self.rate_hz
        rotation = data.xmat[self.body_id].reshape(3, 3)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, self._velocity, 1
        )
        angular_body = self._velocity[:3].copy()
        velocity_world = rotation @ self._velocity[3:]
        if (
            self._previous_velocity is None
            or self._previous_time is None
            or time <= self._previous_time
        ):
            acceleration_world = np.zeros(3)
        else:
            acceleration_world = (velocity_world - self._previous_velocity) / (
                time - self._previous_time
            )
        self._previous_velocity = velocity_world.copy()
        self._previous_time = time
        gravity = np.asarray(self.model.opt.gravity, dtype=float)
        specific_force = rotation.T @ (acceleration_world - gravity)
        self.publish("euler", self.attitude.update(rotation_to_rpy(rotation), time))
        self.publish("gyro", self.gyro.update(angular_body, time))
        self.publish("accel", self.accel.update(specific_force, time))
        self.publish("quat", data.xquat[self.body_id].copy())


__all__ = ["IMU", "rotation_to_rpy"]
