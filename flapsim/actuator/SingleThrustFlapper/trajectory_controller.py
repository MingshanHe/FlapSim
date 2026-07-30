"""Sinusoidal trajectory actuator for the flapping joints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco


class TrajectoryController:
    def __init__(self, model: mujoco.MjModel, config_path: str | Path) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.dynamic_velocity = True
        self.joints: dict[str, dict[str, float | int]] = {}
        for command_name, settings in config.items():
            if settings.get("controller") != "trajectory_controller":
                continue
            joint_name = settings.get("joint", command_name)
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Joint not found: {joint_name}")
            amplitude = float(settings["amplitude"])
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                if amplitude > min(abs(lower), abs(upper)):
                    raise ValueError(
                        f"{command_name} amplitude {amplitude} rad exceeds "
                        f"joint range [{lower}, {upper}] rad"
                    )
            self.joints[command_name] = {
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
                "amplitude_limit": float(
                    min(abs(model.jnt_range[joint_id]))
                    if model.jnt_limited[joint_id]
                    else math.inf
                ),
            }

    def validate_commands(self, commands) -> None:
        for command_name, command in commands.items():
            limit = float(self.joints[command_name]["amplitude_limit"])
            if command["amplitude"] > limit:
                raise ValueError(
                    f"{command_name} amplitude {command['amplitude']} rad "
                    f"exceeds joint limit {limit} rad"
                )

    def set_dynamic_velocity(self, enabled: bool) -> None:
        self.dynamic_velocity = bool(enabled)

    def apply(self, model, data, commands) -> None:
        self.validate_commands(commands)
        self._set_trajectory_state(data, commands)

    def post_step(self, model, data, commands) -> None:
        self._set_trajectory_state(data, commands)

    def _set_trajectory_state(self, data, commands) -> None:
        time = float(data.time)
        for command_name, command in commands.items():
            joint = self.joints[command_name]
            frequency = command["frequency"]
            amplitude = command["amplitude"]
            phase = command["phase"]
            direction = command["direction"]
            omega = 2.0 * math.pi * frequency
            angle = omega * time + phase
            target = direction * amplitude * math.sin(angle)
            target_velocity = direction * amplitude * omega * math.cos(angle)
            qpos = int(joint["qpos_address"])
            dof = int(joint["dof_address"])
            data.qpos[qpos] = target
            data.qvel[dof] = target_velocity if self.dynamic_velocity else 0.0
