"""Cascaded altitude and attitude controller for hovering flight."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np


def _quaternion_to_euler(quaternion) -> tuple[float, float, float]:
    w, x, y, z = map(float, quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class HoveringController:
    def __init__(
        self,
        model: mujoco.MjModel,
        plate_controller,
        thrust_controller,
        *,
        target_z: float,
        hover_frequency: float = 42.0,
        waypoints: Sequence[Sequence[float]] | None = None,
        waypoint_interval: float = 3.0,
    ) -> None:
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body < 0:
            raise ValueError("base_link body not found")
        self.plate_controller = plate_controller
        self.thrust_controller = thrust_controller
        self.target_z = float(target_z)
        self.target_x = 0.0
        self.target_y = 0.0
        self.waypoints = [
            tuple(map(float, point))
            for point in (
                waypoints
                if waypoints is not None
                else (
                    (0.0, 0.0, target_z),
                    (2.0, 0.0, target_z),
                    (-2.0, 0.0, target_z),
                )
            )
        ]
        if not self.waypoints or any(len(point) != 3 for point in self.waypoints):
            raise ValueError("waypoints must contain at least one (x, y, z) point")
        if waypoint_interval <= 0.0:
            raise ValueError("waypoint_interval must be positive")
        self.waypoint_interval = float(waypoint_interval)
        self.waypoint_index = 0
        self.next_waypoint_time = self.waypoint_interval
        self._select_waypoint(0)
        self.hover_frequency = float(hover_frequency)
        self.last_update = -math.inf
        self.height_integral = 0.0
        self.velocity = np.zeros(6)
        self.diagnostics = {}
        self.sample_count = 0
        self.state_sum = np.zeros(10)

        self.height_kp = 8.0
        self.height_kd = 4.0
        self.height_ki = 2.0
        self.integral_limit = 0.5
        self.frequency_limits = (20.0, 55.0)
        # Use physically visible plate deflections.  The averaged plant's
        # plate-to-moment gains are scaled down by the same factor (20), so
        # this changes the commanded joint angle without multiplying torque.
        self.attitude_kp = 0.80
        self.attitude_kd = 0.06
        self.plate_limit = 0.1
        self.horizontal_kp = 0.10
        self.horizontal_kd = 0.50
        self.tilt_limit = 0.10

    def set_target(self, target_z: float) -> None:
        self.target_z = float(target_z)
        self.height_integral = 0.0

    def set_waypoints(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        interval: float = 3.0,
    ) -> None:
        points = [tuple(map(float, point)) for point in waypoints]
        if not points or any(len(point) != 3 for point in points):
            raise ValueError("waypoints must contain at least one (x, y, z) point")
        if interval <= 0.0:
            raise ValueError("waypoint interval must be positive")
        self.waypoints = points
        self.waypoint_interval = float(interval)
        self.waypoint_index = 0
        self.next_waypoint_time = self.waypoint_interval
        self._select_waypoint(0)
        self.height_integral = 0.0

    def _select_waypoint(self, index: int) -> None:
        self.waypoint_index = index % len(self.waypoints)
        self.target_x, self.target_y, self.target_z = self.waypoints[
            self.waypoint_index
        ]

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        time = float(data.time)
        while time >= self.next_waypoint_time:
            self._select_waypoint(self.waypoint_index + 1)
            self.next_waypoint_time += self.waypoint_interval
        position = data.xpos[self.base_body]
        quaternion = data.xquat[self.base_body]
        roll, pitch, yaw = _quaternion_to_euler(quaternion)
        body_z = data.xmat[self.base_body].reshape(3, 3)[:, 2]
        roll_tilt = -float(body_z[1])
        pitch_tilt = float(body_z[0])
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body,
            self.velocity,
            0,
        )
        angular_velocity = self.velocity[:3]
        vertical_velocity = float(self.velocity[5])
        self.state_sum += np.array(
            [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(self.velocity[3]),
                float(self.velocity[4]),
                vertical_velocity,
                roll_tilt,
                pitch_tilt,
                float(angular_velocity[0]),
                float(angular_velocity[1]),
            ]
        )
        self.sample_count += 1

        current_frequency = next(iter(self.thrust_controller.targets.values()))[
            "frequency"
        ]
        control_period = 1.0 / max(float(current_frequency), 1.0)
        if time - self.last_update < control_period:
            return
        dt = (
            control_period
            if not math.isfinite(self.last_update)
            else time - self.last_update
        )
        self.last_update = time
        averaged = self.state_sum / self.sample_count
        self.state_sum[:] = 0.0
        self.sample_count = 0
        (
            x,
            y,
            z,
            velocity_x,
            velocity_y,
            vertical_velocity,
            roll_tilt,
            pitch_tilt,
            roll_rate,
            pitch_rate,
        ) = averaged

        height_error = self.target_z - z
        self.height_integral = float(
            np.clip(
                self.height_integral + height_error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        frequency = (
            self.hover_frequency
            + self.height_kp * height_error
            - self.height_kd * vertical_velocity
            + self.height_ki * self.height_integral
        )
        frequency = float(np.clip(frequency, *self.frequency_limits))

        # URDF naming and physical axes are crossed:
        # pitch_joint rotates about +X and controls body roll;
        # roll_joint rotates about -Y and controls body pitch.
        desired_pitch_tilt = float(
            np.clip(
                self.horizontal_kp * (self.target_x - x)
                - self.horizontal_kd * velocity_x,
                -self.tilt_limit,
                self.tilt_limit,
            )
        )
        desired_roll_tilt = float(
            np.clip(
                -self.horizontal_kp * (self.target_y - y)
                + self.horizontal_kd * velocity_y,
                -self.tilt_limit,
                self.tilt_limit,
            )
        )
        plate_pitch = float(
            np.clip(
                -self.attitude_kp * (roll_tilt - desired_roll_tilt)
                - self.attitude_kd * roll_rate,
                -self.plate_limit,
                self.plate_limit,
            )
        )
        plate_roll = float(
            np.clip(
                self.attitude_kp * (pitch_tilt - desired_pitch_tilt)
                + self.attitude_kd * pitch_rate,
                -self.plate_limit,
                self.plate_limit,
            )
        )
        self.thrust_controller.set_command(frequency=frequency)
        self.plate_controller.set_command(pitch=plate_pitch, roll=plate_roll)
        self.diagnostics = {
            "z": float(z),
            "x": float(x),
            "y": float(y),
            "target_z": self.target_z,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "waypoint_index": self.waypoint_index,
            "next_waypoint_time": self.next_waypoint_time,
            "vertical_velocity": vertical_velocity,
            "frequency": frequency,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "roll_tilt": float(roll_tilt),
            "pitch_tilt": float(pitch_tilt),
            "desired_roll_tilt": desired_roll_tilt,
            "desired_pitch_tilt": desired_pitch_tilt,
            "plate_roll": plate_roll,
            "plate_pitch": plate_pitch,
            "control_period": control_period,
        }
