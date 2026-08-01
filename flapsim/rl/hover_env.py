"""Gymnasium environment for learning averaged-model hovering with PPO."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from flapsim.sim_core import SimCore


def _euler_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a MuJoCo ``[w, x, y, z]`` quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class HoverEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control hover task using FlapSim's averaged aero model.

    Residual action components are normalized to ``[-1, 1]``:

    ``[flapping_frequency, pitch_joint, roll_joint]``.

    A local stabilizing prior supplies the nominal command and PPO learns a
    bounded residual. The original :class:`HoveringController` is not attached.
    """

    metadata = {"render_modes": []}
    HOVER_FREQUENCY = 42.22125202238374

    def __init__(
        self,
        *,
        target: tuple[float, float, float] = (0.0, 0.0, 0.5),
        control_dt: float = 0.02,
        physics_dt: float = 0.001,
        episode_seconds: float = 6.0,
        randomize_reset: bool = True,
        randomize_target: bool = False,
        target_low: tuple[float, float, float] = (-0.25, -0.25, 0.40),
        target_high: tuple[float, float, float] = (0.25, 0.25, 0.65),
        spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.5),
    ) -> None:
        super().__init__()
        if control_dt <= 0.0 or physics_dt <= 0.0:
            raise ValueError("control_dt and physics_dt must be positive")
        ratio = control_dt / physics_dt
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        if episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")

        self.sim = SimCore("SingleThrustFlapper", fix=False)
        # Residual PPO replaces the separately attached classical supervisor.
        self.sim.environment.supervisors.clear()
        self.model = self.sim.model
        self.data = self.sim.data
        self.model.opt.timestep = physics_dt
        self.control_dt = float(control_dt)
        self.frame_skip = int(round(ratio))
        self.max_episode_steps = int(math.ceil(episode_seconds / control_dt))
        self.randomize_reset = bool(randomize_reset)
        self.randomize_target = bool(randomize_target)
        self.target = np.asarray(target, dtype=np.float64)
        if self.target.shape != (3,):
            raise ValueError("target must contain exactly (x, y, z)")
        self.target_low = np.asarray(target_low, dtype=np.float64)
        self.target_high = np.asarray(target_high, dtype=np.float64)
        self.spawn_position = np.asarray(spawn_position, dtype=np.float64)
        if (
            self.target_low.shape != (3,)
            or self.target_high.shape != (3,)
            or self.spawn_position.shape != (3,)
        ):
            raise ValueError("target bounds and spawn_position must have shape (3,)")
        if np.any(self.target_low >= self.target_high):
            raise ValueError("every target_low value must be below target_high")

        self.base_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        free_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        self.free_qpos = int(self.model.jnt_qposadr[free_joint])
        self.free_dof = int(self.model.jnt_dofadr[free_joint])
        self.velocity = np.zeros(6, dtype=np.float64)
        self.episode_step = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.command_action = np.zeros(3, dtype=np.float32)
        self.residual_scale = 0.20

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(object_id)

    @staticmethod
    def _frequency(action: float) -> float:
        """Map zero to hover and retain the full safe frequency interval."""
        hover = HoverEnv.HOVER_FREQUENCY
        if action >= 0.0:
            return hover + (55.0 - hover) * action
        return hover + (hover - 20.0) * action

    def _apply_action(
        self, action: np.ndarray, reference: np.ndarray
    ) -> None:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_action = clipped
        self.command_action = np.clip(
            reference + self.residual_scale * clipped, -1.0, 1.0
        )
        self.sim.set_thrust_command(
            frequency=self._frequency(float(self.command_action[0]))
        )
        self.sim.set_plate_command(
            pitch=0.1 * float(self.command_action[1]),
            roll=0.1 * float(self.command_action[2]),
        )

    def _state(self) -> tuple[np.ndarray, dict[str, Any]]:
        position = self.data.xpos[self.base_body].copy()
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body,
            self.velocity,
            0,
        )
        angular_velocity = self.velocity[:3].copy()
        linear_velocity = self.velocity[3:].copy()
        body_z = rotation[:, 2]
        position_error = position - self.target
        observation = np.concatenate(
            (
                position_error,
                linear_velocity,
                body_z[:2],
                angular_velocity,
            )
        ).astype(np.float32)
        info = {
            "position": position,
            "target": self.target.copy(),
            "position_error": position_error,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "tilt": float(math.acos(np.clip(body_z[2], -1.0, 1.0))),
            "frequency": self._frequency(float(self.command_action[0])),
            "plate_pitch": 0.1 * float(self.command_action[1]),
            "plate_roll": 0.1 * float(self.command_action[2]),
            "residual_action": self.last_action.copy(),
            "command_action": self.command_action.copy(),
        }
        return observation, info

    def _reward(self, info: dict[str, Any]) -> float:
        error = info["position_error"]
        velocity = info["linear_velocity"]
        angular_velocity = info["angular_velocity"]
        tilt = info["tilt"]
        position_cost = (
            1.5 * float(np.linalg.norm(error[:2]))
            + 4.0 * abs(float(error[2]))
        )
        velocity_cost = 0.20 * float(np.linalg.norm(velocity))
        attitude_cost = 1.5 * tilt
        angular_cost = 0.03 * float(np.linalg.norm(angular_velocity))
        effort_cost = 0.01 * float(self.last_action @ self.last_action)
        total_cost = (
            position_cost
            + velocity_cost
            + attitude_cost
            + angular_cost
            + effort_cost
        )
        reward = math.exp(-total_cost)
        if (
            np.linalg.norm(error) < 0.05
            and np.linalg.norm(velocity) < 0.10
            and tilt < math.radians(5.0)
        ):
            reward += 0.5
        return float(reward)

    def _reference_action(self, info: dict[str, Any]) -> np.ndarray:
        """Return the nominal local hover command for residual PPO."""
        error = info["position_error"]
        velocity = info["linear_velocity"]
        angular_velocity = info["angular_velocity"]
        body_z = self.data.xmat[self.base_body].reshape(3, 3)[:, 2]
        roll_tilt = -float(body_z[1])
        pitch_tilt = float(body_z[0])
        desired_pitch_tilt = float(
            np.clip(-0.10 * error[0] - 0.50 * velocity[0], -0.10, 0.10)
        )
        desired_roll_tilt = float(
            np.clip(0.10 * error[1] + 0.50 * velocity[1], -0.10, 0.10)
        )
        frequency = (
            self.HOVER_FREQUENCY
            - 8.0 * float(error[2])
            - 4.0 * float(velocity[2])
        )
        if frequency >= self.HOVER_FREQUENCY:
            frequency_action = (frequency - self.HOVER_FREQUENCY) / (
                55.0 - self.HOVER_FREQUENCY
            )
        else:
            frequency_action = (frequency - self.HOVER_FREQUENCY) / (
                self.HOVER_FREQUENCY - 20.0
            )
        plate_pitch = (
            -0.80 * (roll_tilt - desired_roll_tilt)
            - 0.06 * float(angular_velocity[0])
        )
        plate_roll = (
            0.80 * (pitch_tilt - desired_pitch_tilt)
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

    def _failed(self, info: dict[str, Any]) -> bool:
        position = info["position"]
        values_finite = np.isfinite(self.data.qpos).all() and np.isfinite(
            self.data.qvel
        ).all()
        return bool(
            not values_finite
            or position[2] < 0.08
            or position[2] > 1.5
            or abs(position[0] - self.target[0]) > 1.5
            or abs(position[1] - self.target[1]) > 1.5
            or info["tilt"] > math.radians(65.0)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        options = options or {}
        randomize = bool(options.get("randomize", self.randomize_reset))
        randomize_target = bool(
            options.get("randomize_target", self.randomize_target)
        )

        if randomize_target:
            self.target[:] = self.np_random.uniform(
                self.target_low, self.target_high
            )
            position = self.spawn_position.copy()
        else:
            position = self.target.copy()
        roll = pitch = yaw = 0.0
        if randomize:
            position += self.np_random.uniform(
                low=(-0.10, -0.10, -0.08),
                high=(0.10, 0.10, 0.08),
            )
            roll, pitch = self.np_random.uniform(
                -math.radians(8.0), math.radians(8.0), size=2
            )
            yaw = float(
                self.np_random.uniform(-math.radians(10.0), math.radians(10.0))
            )

        self.data.qpos[self.free_qpos : self.free_qpos + 3] = position
        self.data.qpos[self.free_qpos + 3 : self.free_qpos + 7] = (
            _euler_quaternion(float(roll), float(pitch), yaw)
        )
        self.data.qvel[self.free_dof : self.free_dof + 6] = 0.0
        self.episode_step = 0
        self.last_action[:] = 0.0
        self.command_action[:] = 0.0
        self.sim.set_thrust_command(frequency=self.HOVER_FREQUENCY)
        self.sim.set_plate_command(pitch=0.0, roll=0.0)
        # Synchronize the kinematic display joints before the first forward.
        for command_source, actuator in self.sim.environment.control_pairs:
            actuator.apply(self.model, self.data, command_source.targets)
        self.sim.aero.reset()
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, current_info = self._state()
        self._apply_action(action, self._reference_action(current_info))
        for _ in range(self.frame_skip):
            self.sim.environment.step()
        self.episode_step += 1
        observation, info = self._state()
        terminated = self._failed(info)
        truncated = self.episode_step >= self.max_episode_steps
        reward = self._reward(info)
        if terminated:
            reward -= 10.0
        info["is_success"] = bool(
            not terminated
            and np.linalg.norm(info["position_error"]) < 0.10
            and np.linalg.norm(info["linear_velocity"]) < 0.20
            and info["tilt"] < math.radians(10.0)
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.aero.reset()
