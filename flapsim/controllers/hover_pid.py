"""Cascaded altitude and attitude PID for hovering with a floating base."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from flapsim.plugins.base import ControllerPlugin
from flapsim.plugins.registry import register

DEFAULT_GAINS = {
    "height_kp": 8.0,
    "height_kd": 4.0,
    "height_ki": 2.0,
    "integral_limit": 0.5,
    "attitude_kp": 0.80,
    "attitude_kd": 0.06,
    "plate_limit": 0.1,
    "horizontal_kp": 0.10,
    "horizontal_kd": 0.50,
    "tilt_limit": 0.10,
}


def _quaternion_to_euler(quaternion) -> tuple[float, float, float]:
    w, x, y, z = map(float, quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


@register
class HoverPID(ControllerPlugin):
    """Hold a position (or cycle through waypoints) with a flapping-frequency
    altitude loop and a gimbal attitude loop.

    State is averaged over one stroke period and the loops update once per
    period.  Commands are published for a stroke actuator
    (``cmd/<stroke>/frequency_hz``) and a gimbal actuator
    (``cmd/<gimbal>/<pitch_joint>``, ``cmd/<gimbal>/<roll_joint>``).
    """

    name = "hover_pid"
    provides = (
        "z",
        "x",
        "y",
        "frequency_hz",
        "plate_pitch",
        "plate_roll",
        "waypoint_index",
    )
    example_config = {
        "stroke": "stroke",
        "gimbal": "gimbal",
        "pitch_joint": "pitch_joint",
        "roll_joint": "roll_joint",
        "hover_frequency_hz": 40.0,
    }

    def __init__(
        self,
        *,
        pitch_joint: str,
        roll_joint: str,
        stroke: str = "stroke",
        gimbal: str = "gimbal",
        target: Sequence[float] = (0.0, 0.0, 0.5),
        waypoints: Sequence[Sequence[float]] | None = None,
        waypoint_interval: float = 3.0,
        hover_frequency_hz: float = 40.0,
        frequency_limits: Sequence[float] = (20.0, 55.0),
        gains: Mapping[str, float] | None = None,
        root_body: str | None = None,
        id: str | None = "hover",
    ) -> None:
        super().__init__(
            id=id,
            pitch_joint=pitch_joint,
            roll_joint=roll_joint,
            stroke=stroke,
            gimbal=gimbal,
            target=list(target),
            waypoints=None if waypoints is None else [list(p) for p in waypoints],
            waypoint_interval=waypoint_interval,
            hover_frequency_hz=hover_frequency_hz,
            frequency_limits=list(frequency_limits),
            gains=dict(gains or {}),
            root_body=root_body,
        )
        self.pitch_joint = pitch_joint
        self.roll_joint = roll_joint
        self.stroke_id = stroke
        self.gimbal_id = gimbal
        self.root_body_name = root_body
        self.hover_frequency = float(hover_frequency_hz)
        self.frequency_limits = (float(frequency_limits[0]), float(frequency_limits[1]))
        unknown = sorted(set(gains or {}) - set(DEFAULT_GAINS))
        if unknown:
            raise ValueError(
                f"unknown gain(s) {unknown}; gains are {sorted(DEFAULT_GAINS)}"
            )
        self.gains = {
            **DEFAULT_GAINS,
            **{k: float(v) for k, v in (gains or {}).items()},
        }
        target = tuple(float(v) for v in target)
        if len(target) != 3:
            raise ValueError("target must be (x, y, z)")
        self.set_waypoints(
            waypoints if waypoints is not None else [target], interval=waypoint_interval
        )
        self.base_body = -1
        self.velocity = np.zeros(6)
        self.diagnostics: dict[str, float] = {}
        self._reset_state()

    def _reset_state(self) -> None:
        self.last_update = -math.inf
        self.height_integral = 0.0
        self.sample_count = 0
        self.state_sum = np.zeros(10)
        self.waypoint_index = 0
        self.next_waypoint_time = self.waypoint_interval
        self._select_waypoint(0)

    # -- setup --------------------------------------------------------------

    def setup(self) -> None:
        name = self.root_body_name or self.robot.root_body_name
        self.base_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.base_body < 0:
            raise ValueError(f"root body {name!r} not found")
        if self.robot.fixed_base:
            raise ValueError("HoverPID needs a floating base (base: free)")

    def reset(self) -> None:
        self._reset_state()

    # -- targets ------------------------------------------------------------

    def set_target(
        self, x: float | None = None, y: float | None = None, z: float | None = None
    ) -> None:
        point = list(self.waypoints[self.waypoint_index])
        for index, value in enumerate((x, y, z)):
            if value is not None:
                point[index] = float(value)
        self.waypoints[self.waypoint_index] = tuple(point)
        self._select_waypoint(self.waypoint_index)
        self.height_integral = 0.0

    def set_waypoints(
        self, waypoints: Sequence[Sequence[float]], *, interval: float = 3.0
    ) -> None:
        points = [tuple(float(v) for v in point) for point in waypoints]
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

    # -- control ------------------------------------------------------------

    def control(self) -> None:
        model, data, g = self.model, self.data, self.gains
        now = float(data.time)
        while now >= self.next_waypoint_time:
            self._select_waypoint(self.waypoint_index + 1)
            self.next_waypoint_time += self.waypoint_interval
        position = data.xpos[self.base_body]
        body_z = data.xmat[self.base_body].reshape(3, 3)[:, 2]
        roll_tilt = -float(body_z[1])
        pitch_tilt = float(body_z[0])
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_body, self.velocity, 0
        )
        angular = self.velocity[:3]
        self.state_sum += np.array(
            [
                position[0],
                position[1],
                position[2],
                self.velocity[3],
                self.velocity[4],
                self.velocity[5],
                roll_tilt,
                pitch_tilt,
                angular[0],
                angular[1],
            ],
            dtype=float,
        )
        self.sample_count += 1

        current_frequency = float(
            self.signals.get(f"{self.stroke_id}/frequency_hz", self.hover_frequency)
        )
        period = 1.0 / max(current_frequency, 1.0)
        if now - self.last_update < period:
            return
        dt = period if not math.isfinite(self.last_update) else now - self.last_update
        self.last_update = now
        averaged = self.state_sum / self.sample_count
        self.state_sum[:] = 0.0
        self.sample_count = 0
        x, y, z, vx, vy, vz, roll_tilt, pitch_tilt, roll_rate, pitch_rate = averaged

        height_error = self.target_z - z
        self.height_integral = float(
            np.clip(
                self.height_integral + height_error * dt,
                -g["integral_limit"],
                g["integral_limit"],
            )
        )
        frequency = float(
            np.clip(
                self.hover_frequency
                + g["height_kp"] * height_error
                - g["height_kd"] * vz
                + g["height_ki"] * self.height_integral,
                *self.frequency_limits,
            )
        )
        # The example robot's URDF names are crossed: the "pitch" joint tilts
        # the thrust about body x and controls roll, and vice versa.
        desired_pitch_tilt = float(
            np.clip(
                g["horizontal_kp"] * (self.target_x - x) - g["horizontal_kd"] * vx,
                -g["tilt_limit"],
                g["tilt_limit"],
            )
        )
        desired_roll_tilt = float(
            np.clip(
                -g["horizontal_kp"] * (self.target_y - y) + g["horizontal_kd"] * vy,
                -g["tilt_limit"],
                g["tilt_limit"],
            )
        )
        plate_pitch = float(
            np.clip(
                -g["attitude_kp"] * (roll_tilt - desired_roll_tilt)
                - g["attitude_kd"] * roll_rate,
                -g["plate_limit"],
                g["plate_limit"],
            )
        )
        plate_roll = float(
            np.clip(
                g["attitude_kp"] * (pitch_tilt - desired_pitch_tilt)
                + g["attitude_kd"] * pitch_rate,
                -g["plate_limit"],
                g["plate_limit"],
            )
        )
        self.send_command(self.stroke_id, "frequency_hz", frequency)
        self.send_command(self.gimbal_id, self.pitch_joint, plate_pitch)
        self.send_command(self.gimbal_id, self.roll_joint, plate_roll)
        roll, pitch, yaw = _quaternion_to_euler(data.xquat[self.base_body])
        self.diagnostics = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_z": self.target_z,
            "vertical_velocity": float(vz),
            "frequency_hz": frequency,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "plate_pitch": plate_pitch,
            "plate_roll": plate_roll,
            "waypoint_index": float(self.waypoint_index),
        }
        for key in (
            "x",
            "y",
            "z",
            "frequency_hz",
            "plate_pitch",
            "plate_roll",
            "waypoint_index",
        ):
            self.publish(key, self.diagnostics[key])


__all__ = ["HoverPID"]
