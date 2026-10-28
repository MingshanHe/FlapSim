"""Apply the quasi-steady model to every wing of a :class:`RobotModel`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from flapsim.config import AeroOptions, WingInput

from .aero import wing_aero, world_wrench

if TYPE_CHECKING:
    from flapsim.robot.loader import RobotModel


class MuJoCoAeroAdapter:
    """Per-step blade-element forces for all wings declared in the manifest.

    Each wing's state comes from its :class:`~flapsim.wings.Wing`; the torso
    velocity from the manifest's torso body.  The normal force is applied at
    the wing's centre of pressure (as a force at the centre of mass plus the
    matching torque), together with the rotational damping moment.
    """

    def __init__(self, robot: RobotModel, options: AeroOptions | None = None) -> None:
        if not robot.wings:
            raise ValueError(
                f"robot {robot.manifest.name!r} declares no wings; add a wings section "
                "to its manifest"
            )
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.wings = list(robot.wings)
        self.options = options or AeroOptions()
        self.torso_body = robot.torso_body
        self.torso_velocity = np.zeros(6)
        self.diagnostics: dict = {}

    def apply_forces(self) -> None:
        data = self.data
        for wing in self.wings:
            wing.clear(data)
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body,
            self.torso_velocity,
            0,
        )
        rotation = data.xmat[self.torso_body].reshape(3, 3)
        v_body = rotation.T @ self.torso_velocity[3:]
        w_body = rotation.T @ self.torso_velocity[:3]
        magnitude = float(np.linalg.norm(w_body))
        if magnitude > self.options.angular_velocity_limit:
            w_body = w_body * (self.options.angular_velocity_limit / magnitude)

        total_force = np.zeros(3)
        lift_total = 0.0
        diagnostics: dict = {}
        for wing in self.wings:
            state = wing.state(data)
            aero = wing_aero(
                wing.params,
                int(state.side),
                state.psi,
                state.psi_dot,
                state.theta,
                state.theta_dot,
                v_body,
                w_body,
                self.options,
            )
            wrench = world_wrench(
                WingInput(
                    psi=state.psi,
                    psi_dot=state.psi_dot,
                    theta_raw=state.theta,
                    theta_dot_raw=state.theta_dot,
                    pitch_axis_y=1.0,
                    R_world_from_wing=state.R_world_from_aero,
                    body_origin_world=state.origin_world,
                    com_world=state.com_world,
                ),
                int(state.side),
                aero,
                self.options.read_only,
            )
            force = wrench["F_world"]
            torque = wrench["torque_world"]
            if not self.options.read_only:
                wing.apply_wrench(data, force, torque)
            total_force += force
            lift_total += abs(aero["F_N"])
            prefix = wing.name
            diagnostics[f"{prefix}/F_N"] = aero["F_N"]
            diagnostics[f"{prefix}/M_aero"] = aero["M_aero"]
            diagnostics[f"{prefix}/M_rd"] = aero["M_rd"]
            diagnostics[f"{prefix}/alpha"] = aero["alpha"]
            diagnostics[f"{prefix}/C_N"] = aero["C_N"]
            diagnostics[f"{prefix}/chord_CoP"] = aero["chord_CoP"]
            diagnostics[f"{prefix}/span_CoP"] = aero["span_CoP"]
            diagnostics[f"{prefix}/cop_world"] = wrench["cop_world"]
            diagnostics[f"{prefix}/F_world"] = force
            diagnostics[f"{prefix}/torque_world"] = torque
        diagnostics["F_world"] = total_force
        diagnostics["lift_total"] = lift_total
        diagnostics["read_only"] = self.options.read_only
        names = {wing.name for wing in self.wings}
        if names == {"left", "right"}:
            for key, suffix in (
                ("F_N", "F_N"),
                ("M_aero", "M_aero"),
                ("M_rd", "M_rd"),
                ("alpha", "alpha"),
                ("chord_CoP", "chord_CoP"),
            ):
                diagnostics[f"{key}_L"] = diagnostics[f"left/{suffix}"]
                diagnostics[f"{key}_R"] = diagnostics[f"right/{suffix}"]
        self.diagnostics = diagnostics

    def reset(self) -> None:
        for wing in self.wings:
            wing.clear(self.data)
        self.diagnostics = {}
