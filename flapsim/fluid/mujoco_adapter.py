"""Connect the standalone aerodynamic model to this MuJoCo robot."""

from __future__ import annotations

import mujoco
import numpy as np

from flapsim.config import AeroOptions, TorsoInput, WingInput
from .aero import aero_step, load_wing_parameters


class MuJoCoAeroAdapter:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.params = load_wing_parameters()
        self.options = AeroOptions()
        self.diagnostics = {}
        self.torso_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "link2")
        self.left_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "left_wing")
        self.right_body = self._id(mujoco.mjtObj.mjOBJ_BODY, "right_wing")
        self.left_stroke = self._joint_addresses("left_flapping_joint")
        self.right_stroke = self._joint_addresses("right_wing_flapping")
        self.left_pitch = self._joint_addresses("left_wing_feathering")
        self.right_pitch = self._joint_addresses("right_wing_feathering")
        self.torso_velocity = np.zeros(6)
        # Aero frame expressed in the URDF wing-body frame.  The aerodynamic
        # model uses +X as wing normal and +Y as the feathering/span axis;
        # this URDF's feathering joint is aligned with body -Z.
        self.aero_from_body_left = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        ) @ np.diag([-1.0, 1.0, -1.0])
        left_body_rotation = self.data.xmat[self.left_body].reshape(3, 3)
        right_body_rotation = self.data.xmat[self.right_body].reshape(3, 3)
        neutral_aero_rotation = left_body_rotation @ self.aero_from_body_left
        self.aero_from_body_right = (
            right_body_rotation.T @ neutral_aero_rotation
        )

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(object_id)

    def _joint_addresses(self, name: str) -> tuple[int, int]:
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _wing_input(
        self,
        body: int,
        stroke,
        pitch,
        *,
        stroke_sign: float = 1.0,
        pitch_sign: float = 1.0,
        aero_from_body=None,
    ) -> WingInput:
        q_stroke, v_stroke = stroke
        q_pitch, v_pitch = pitch
        return WingInput(
            psi=stroke_sign * float(self.data.qpos[q_stroke]),
            psi_dot=stroke_sign * float(self.data.qvel[v_stroke]),
            theta_raw=pitch_sign * float(self.data.qpos[q_pitch]),
            theta_dot_raw=pitch_sign * float(self.data.qvel[v_pitch]),
            pitch_axis_y=1.0,
            R_world_from_wing=(
                self.data.xmat[body].reshape(3, 3) @ aero_from_body
            ),
            body_origin_world=self.data.xpos[body].copy(),
            com_world=self.data.xipos[body].copy(),
        )

    def apply_forces(self) -> None:
        self.data.xfrc_applied[self.left_body] = 0.0
        self.data.xfrc_applied[self.right_body] = 0.0
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body,
            self.torso_velocity,
            0,
        )
        torso = TorsoInput(
            self.data.xmat[self.torso_body].reshape(3, 3).copy(),
            self.torso_velocity[:3].copy(),
            self.torso_velocity[3:].copy(),
        )
        result = aero_step(
            self.params,
            torso,
            self._wing_input(
                self.left_body,
                self.left_stroke,
                self.left_pitch,
                aero_from_body=self.aero_from_body_left,
            ),
            self._wing_input(
                self.right_body,
                self.right_stroke,
                self.right_pitch,
                stroke_sign=-1.0,
                aero_from_body=self.aero_from_body_right,
            ),
            self.options,
        )
        # The imported STL wing normals are opposite to the oriented normal
        # convention used by FlapX. Reverse the translational normal force,
        # while retaining the aerodynamic moment that correctly drives the
        # passive feathering phase.
        left_wrench = result["xfrc_applied_L"].copy()
        right_wrench = result["xfrc_applied_R"].copy()
        left_wrench[:3] *= -1.0
        right_wrench[:3] *= -1.0
        self.data.xfrc_applied[self.left_body] = left_wrench
        self.data.xfrc_applied[self.right_body] = right_wrench
        self.diagnostics = dict(result["diagnostics"])
        self.diagnostics["F_world"] = -np.asarray(
            result["diagnostics"]["F_world"]
        )

    def reset(self) -> None:
        self.data.xfrc_applied[self.left_body] = 0.0
        self.data.xfrc_applied[self.right_body] = 0.0
        self.diagnostics = {}
