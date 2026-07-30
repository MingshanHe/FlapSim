"""PD position actuator configured by config/SingleThrustFlapper/joints.json."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


class PositionController:
    def __init__(self, model: mujoco.MjModel, config_path: str | Path) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.kinematic = False
        self.joints: dict[str, dict[str, float | int]] = {}

        for joint_name, settings in config.items():
            if settings.get("controller") != "position_controller":
                continue
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise ValueError(f"Joint not found: {joint_name}")
            self.joints[joint_name] = {
                "joint_id": int(joint_id),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
                "kp": float(settings["kp"]),
                "kd": float(settings["kd"]),
                "max_torque": float(settings["max_torque"]),
            }

    def validate_targets(
        self, model: mujoco.MjModel, targets: dict[str, float]
    ) -> None:
        for joint_name, target in targets.items():
            joint = self.joints[joint_name]
            joint_id = int(joint["joint_id"])
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                if not lower <= target <= upper:
                    raise ValueError(
                        f"{joint_name} target {target} rad is outside "
                        f"[{lower}, {upper}] rad"
                    )

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        targets: dict[str, float],
    ) -> None:
        self.validate_targets(model, targets)
        if self.kinematic:
            self._set_joint_state(data, targets)
            return
        for joint_name, target in targets.items():
            joint = self.joints[joint_name]
            qpos_address = int(joint["qpos_address"])
            dof_address = int(joint["dof_address"])
            torque = (
                data.qfrc_bias[dof_address]
                + float(joint["kp"]) * (target - data.qpos[qpos_address])
                - float(joint["kd"]) * data.qvel[dof_address]
            )
            limit = float(joint["max_torque"])
            data.qfrc_applied[dof_address] += np.clip(torque, -limit, limit)

    def set_kinematic(self, enabled: bool) -> None:
        """Follow commands visually without applying joint torque."""
        self.kinematic = bool(enabled)

    def post_step(self, model, data, targets: dict[str, float]) -> None:
        if self.kinematic:
            self._set_joint_state(data, targets)

    def _set_joint_state(self, data, targets: dict[str, float]) -> None:
        for joint_name, target in targets.items():
            joint = self.joints[joint_name]
            data.qpos[int(joint["qpos_address"])] = float(target)
            data.qvel[int(joint["dof_address"])] = 0.0
