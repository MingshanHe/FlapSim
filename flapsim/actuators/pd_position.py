"""PD position actuator with torque limits, or a kinematic position hold."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from flapsim.plugins.base import ActuatorPlugin
from flapsim.plugins.registry import register

MODES = ("torque", "kinematic")
JOINT_KEYS = ("kp", "kd", "max_torque", "target", "role")


@register
class PDPosition(ActuatorPlugin):
    """Hold joints at targets with a gravity-compensated PD law.

    ``joints`` maps joint names to ``{kp, kd, max_torque, target, role}``.
    ``role`` is a free label (for example ``pitch`` or ``roll``) that lets
    controllers address joints by function.  Targets come from
    ``cmd/<id>/<joint>`` signals when present and are clipped to the joint
    range; the active targets are published as ``<id>/<joint>``.  In
    ``kinematic`` mode the joint is set to the target after each step.
    """

    name = "pd_position"
    example_config = {
        "joints": {
            "pitch_joint": {"role": "pitch", "kp": 0.5, "kd": 0.05, "max_torque": 0.5},
            "roll_joint": {"role": "roll", "kp": 0.5, "kd": 0.05, "max_torque": 0.5},
        }
    }

    def __init__(
        self,
        *,
        joints: Mapping[str, Mapping[str, float] | None],
        mode: str = "torque",
        id: str | None = "gimbal",
    ) -> None:
        super().__init__(
            id=id, joints={k: dict(v or {}) for k, v in joints.items()}, mode=mode
        )
        if not joints:
            raise ValueError("PDPosition needs at least one joint")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.settings: dict[str, dict[str, float]] = {}
        self.roles: dict[str, str] = {}
        for joint_name, settings in joints.items():
            settings = dict(settings or {})
            unknown = sorted(set(settings) - set(JOINT_KEYS))
            if unknown:
                raise ValueError(f"joint {joint_name!r}: unknown setting(s) {unknown}")
            role = settings.pop("role", None)
            if role is not None:
                self.roles[str(role)] = joint_name
            self.settings[joint_name] = {
                "kp": float(settings.get("kp", 0.0)),
                "kd": float(settings.get("kd", 0.0)),
                "max_torque": float(settings.get("max_torque", np.inf)),
                "target": float(settings.get("target", 0.0)),
            }
        self._addresses: dict[str, tuple[int, int, int]] = {}
        self._ranges: dict[str, tuple[float, float]] = {}

    # -- setup --------------------------------------------------------------

    def setup(self) -> None:
        model = self.model
        for joint_name in self.settings:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0:
                names = [model.joint(i).name for i in range(model.njnt)]
                raise ValueError(f"joint {joint_name!r} not found; joints are {names}")
            self._addresses[joint_name] = (
                int(joint),
                int(model.jnt_qposadr[joint]),
                int(model.jnt_dofadr[joint]),
            )
            if model.jnt_limited[joint]:
                lower, upper = model.jnt_range[joint]
                self._ranges[joint_name] = (float(lower), float(upper))
            else:
                self._ranges[joint_name] = (-np.inf, np.inf)
            self.set_target(joint_name, self.settings[joint_name]["target"])

    # -- targets ------------------------------------------------------------

    @property
    def targets(self) -> dict[str, float]:
        return {name: s["target"] for name, s in self.settings.items()}

    def joint_for_role(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(
                f"no joint with role {role!r}; roles are {sorted(self.roles)}"
            ) from None

    def set_target(self, joint_name: str, value: float, *, clip: bool = False) -> float:
        if joint_name not in self.settings:
            raise KeyError(
                f"unknown joint {joint_name!r}; joints are {list(self.settings)}"
            )
        lower, upper = self._ranges.get(joint_name, (-np.inf, np.inf))
        value = float(value)
        if value < lower or value > upper:
            if not clip:
                raise ValueError(
                    f"{joint_name} target {value} is outside [{lower}, {upper}]"
                )
            value = float(np.clip(value, lower, upper))
        self.settings[joint_name]["target"] = value
        return value

    def set_role_target(self, role: str, value: float, *, clip: bool = False) -> float:
        return self.set_target(self.joint_for_role(role), value, clip=clip)

    # -- phases -------------------------------------------------------------

    def control(self) -> None:
        for joint_name in self.settings:
            command = self.command(joint_name)
            if command is not None:
                self.set_target(joint_name, command, clip=True)

    def actuate(self) -> None:
        data = self.data
        for joint_name, settings in self.settings.items():
            _, qpos, dof = self._addresses[joint_name]
            self.publish(joint_name, settings["target"])
            if self.mode == "torque":
                torque = (
                    data.qfrc_bias[dof]
                    + settings["kp"] * (settings["target"] - data.qpos[qpos])
                    - settings["kd"] * data.qvel[dof]
                )
                limit = settings["max_torque"]
                data.qfrc_applied[dof] += float(np.clip(torque, -limit, limit))

    def post_step(self) -> bool:
        if self.mode != "kinematic":
            return False
        data = self.data
        for joint_name, settings in self.settings.items():
            _, qpos, dof = self._addresses[joint_name]
            data.qpos[qpos] = settings["target"]
            data.qvel[dof] = 0.0
        return True

    def reset(self) -> None:
        if self.mode == "kinematic":
            self.post_step()


__all__ = ["PDPosition"]
