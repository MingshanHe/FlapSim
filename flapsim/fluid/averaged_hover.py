"""Cycle-averaged aerodynamic model used by the floating hover controller."""

from __future__ import annotations

import mujoco
import numpy as np


class AveragedHoverAero:
    """Lift proportional to frequency squared plus identified plate moments.

    The coefficients were identified on the example robot's fixed-base
    sweep.  Roadmap Phase 4 replaces them with a per-robot parameter file
    produced by ``flapsim sweep``.
    """

    def __init__(
        self, model, data, plate_controller, thrust_controller, *, root_body: str
    ) -> None:
        self.model = model
        self.data = data
        self.plate_controller = plate_controller
        self.thrust_controller = thrust_controller
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        if self.base_body < 0:
            raise ValueError(f"root body {root_body!r} not found")
        # Fit from the fixed-base sweep of the example robot (2026-10-28,
        # corrected quasi-steady model, spar-plate wings): 30 Hz -> 0.1666 N.
        self.lift_coefficient = 0.16661 / (30.0**2)
        # Fixed-base plate identification at 42 Hz [N.m / rad].
        # Scaled effective gains for the visible plate-angle convention used
        # by HoveringController (20x joint angle, unchanged control moment).
        self.roll_moment_per_rad = 0.01055 / 20.0
        self.pitch_moment_per_rad = 0.01144 / 20.0
        self.angular_damping = np.array([2.0e-4, 2.0e-4, 1.0e-4])
        self.horizontal_damping = 0.06
        self.velocity = np.zeros(6)
        self.diagnostics = {}

    def apply_forces(self) -> None:
        frequency = next(iter(self.thrust_controller.targets.values()))["frequency"]
        plate = self.plate_controller.targets
        plate_pitch = plate[self.plate_controller.pitch_joint]
        plate_roll = plate[self.plate_controller.roll_joint]
        lift = self.lift_coefficient * frequency**2
        direction_body = np.array([-plate_roll, -plate_pitch, 1.0])
        direction_body /= np.linalg.norm(direction_body)
        rotation = self.data.xmat[self.base_body].reshape(3, 3)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_body,
            self.velocity,
            1,
        )
        linear_velocity_world = rotation @ self.velocity[3:]
        force_world = rotation @ (lift * direction_body)
        force_world[:2] -= self.horizontal_damping * linear_velocity_world[:2]
        scale = (frequency / 42.0) ** 2
        torque_body = (
            np.array(
                [
                    scale * self.roll_moment_per_rad * plate_pitch,
                    -scale * self.pitch_moment_per_rad * plate_roll,
                    0.0,
                ]
            )
            - self.angular_damping * self.velocity[:3]
        )
        torque_world = rotation @ torque_body
        self.data.xfrc_applied[self.base_body, :3] = force_world
        self.data.xfrc_applied[self.base_body, 3:] = torque_world
        self.diagnostics = {
            "F_world": force_world.copy(),
            "torque_world": torque_world.copy(),
            "lift": float(lift),
            "frequency": float(frequency),
            "mode": "cycle_averaged_hover",
        }

    def reset(self) -> None:
        self.data.xfrc_applied[self.base_body] = 0.0
        self.diagnostics = {}
