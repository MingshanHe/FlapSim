"""Brushed DC motor driving the stroke through a gearbox and crank."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np

from flapsim.plugins.base import ActuatorPlugin
from flapsim.plugins.registry import register

LOADS = ("aero", "none")


@register
class DCMotorCrank(ActuatorPlugin):
    """Voltage-driven DC motor, gearbox and crank imposing ``q = d A sin(theta_c)``.

    Electrical: ``i = (V - k_e w) / R``, ``tau_m = k_t i``.  Mechanical, on the
    motor shaft: ``J dw/dt = tau_m - b w - c sign(w) + tau_aero / N`` where
    ``tau_aero`` is the aerodynamic torque about the stroke joints projected
    onto the crank (``load: aero``, from the aero plugin's wing signals) and
    ``N`` the gear ratio.  The crank angle ``theta_c`` advances at ``w / N``
    and every stroke joint follows ``direction * A sin(theta_c + phase)``.
    Commands: ``cmd/<id>/voltage_v``.  Publishes ``speed_rad_s``,
    ``frequency_hz``, ``current_a``, ``voltage_v``, ``torque_motor``,
    ``torque_load``, ``power_w``, ``crank_angle``.
    """

    name = "dc_motor_crank"
    provides = (
        "speed_rad_s",
        "frequency_hz",
        "current_a",
        "voltage_v",
        "torque_motor",
        "torque_load",
        "power_w",
        "crank_angle",
    )
    example_config = {
        "joints": ["left_flapping_joint", "right_wing_flapping"],
        "directions": [1.0, -1.0],
        "amplitude_rad": 1.5,
        "voltage_v": 3.0,
        "initial_frequency_hz": 30.0,
    }

    def __init__(
        self,
        *,
        joints: Sequence[str],
        directions: Sequence[float] | None = None,
        amplitude_rad: float = 1.5,
        phase_rad: float = 0.0,
        gear_ratio: float = 10.0,
        resistance_ohm: float = 1.0,
        kt: float = 1.0e-3,
        ke: float | None = None,
        inertia: float = 2.0e-8,
        viscous: float = 1.0e-7,
        coulomb: float = 0.0,
        voltage_v: float = 3.0,
        max_voltage: float | None = None,
        initial_frequency_hz: float = 0.0,
        load: str = "aero",
        id: str | None = "motor",
    ) -> None:
        super().__init__(
            id=id,
            joints=list(joints),
            directions=None if directions is None else list(directions),
            amplitude_rad=amplitude_rad,
            phase_rad=phase_rad,
            gear_ratio=gear_ratio,
            resistance_ohm=resistance_ohm,
            kt=kt,
            ke=ke,
            inertia=inertia,
            viscous=viscous,
            coulomb=coulomb,
            voltage_v=voltage_v,
            max_voltage=max_voltage,
            initial_frequency_hz=initial_frequency_hz,
            load=load,
        )
        if not joints:
            raise ValueError("DCMotorCrank needs at least one stroke joint")
        if load not in LOADS:
            raise ValueError(f"load must be one of {LOADS}")
        if gear_ratio <= 0 or resistance_ohm <= 0 or kt <= 0 or inertia <= 0:
            raise ValueError(
                "gear_ratio, resistance_ohm, kt and inertia must be positive"
            )
        self.joint_names = list(joints)
        self.directions = (
            [1.0] * len(joints)
            if directions is None
            else [float(d) for d in directions]
        )
        if len(self.directions) != len(self.joint_names):
            raise ValueError("directions must match joints")
        self.amplitude = float(amplitude_rad)
        self.phase = float(phase_rad)
        self.gear_ratio = float(gear_ratio)
        self.resistance = float(resistance_ohm)
        self.kt = float(kt)
        self.ke = float(ke) if ke is not None else float(kt)
        self.inertia = float(inertia)
        self.viscous = float(viscous)
        self.coulomb = float(coulomb)
        self.max_voltage = None if max_voltage is None else float(max_voltage)
        self.voltage = float(voltage_v)
        self.load = load
        self.initial_speed = (
            2.0 * math.pi * float(initial_frequency_hz) * self.gear_ratio
        )
        self.omega = self.initial_speed
        self.crank_angle = 0.0
        self.current = 0.0
        self.torque_motor = 0.0
        self.torque_load = 0.0
        self._addresses: list[tuple[int, int, int]] = []
        self._wing_of_joint: dict[str, object] = {}

    # -- setup --------------------------------------------------------------

    def setup(self) -> None:
        model = self.model
        self._addresses = []
        for name in self.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"stroke joint {name!r} not found")
            self._addresses.append(
                (
                    int(joint),
                    int(model.jnt_qposadr[joint]),
                    int(model.jnt_dofadr[joint]),
                )
            )
            if model.jnt_limited[joint]:
                lower, upper = model.jnt_range[joint]
                if self.amplitude > min(abs(lower), abs(upper)) + 1e-12:
                    raise ValueError(
                        f"amplitude {self.amplitude} exceeds the range of {name!r}"
                    )
        self._wing_of_joint = {}
        for wing in self.robot.wings:
            if wing.spec.stroke_joint is not None:
                self._wing_of_joint[wing.spec.stroke_joint.name] = wing
        self.reset()

    def reset(self) -> None:
        self.omega = self.initial_speed
        self.crank_angle = 0.0
        self.current = 0.0
        self.torque_motor = 0.0
        self.torque_load = 0.0
        self._impose()

    # -- commands -----------------------------------------------------------

    def set_voltage(self, voltage_v: float) -> None:
        voltage = float(voltage_v)
        if self.max_voltage is not None:
            voltage = float(np.clip(voltage, -self.max_voltage, self.max_voltage))
        self.voltage = voltage

    @property
    def frequency_hz(self) -> float:
        return self.omega / (2.0 * math.pi * self.gear_ratio)

    def no_load_speed(self, voltage_v: float | None = None) -> float:
        """Motor speed where the back-EMF balances the voltage (no friction)."""
        return float(voltage_v if voltage_v is not None else self.voltage) / self.ke

    # -- phases -------------------------------------------------------------

    def control(self) -> None:
        command = self.command("voltage_v")
        if command is not None:
            self.set_voltage(command)

    def _aero_load_torque(self) -> float:
        """Aerodynamic torque on the crank, from the wing signals (N.m)."""
        if self.load != "aero":
            return 0.0
        data = self.data
        signals = self.signals
        total = 0.0
        for (joint, _, _), name, direction in zip(
            self._addresses, self.joint_names, self.directions, strict=True
        ):
            wing = self._wing_of_joint.get(name)
            if wing is None:
                continue
            force = signals.get(f"wings/{wing.name}/F_world")
            torque = signals.get(f"wings/{wing.name}/torque_world")
            if force is None or torque is None:
                continue
            axis = data.xaxis[joint]
            anchor = data.xanchor[joint]
            arm = data.xipos[wing.body_id] - anchor
            torque_about_joint = float(axis @ (torque + np.cross(arm, force)))
            dq_dtheta = (
                direction * self.amplitude * math.cos(self.crank_angle + self.phase)
            )
            total += torque_about_joint * dq_dtheta
        return total

    def actuate(self) -> None:
        dt = float(self.model.opt.timestep)
        self.torque_load = self._aero_load_torque()
        self.current = (self.voltage - self.ke * self.omega) / self.resistance
        self.torque_motor = self.kt * self.current
        friction = self.viscous * self.omega + self.coulomb * math.copysign(
            1.0, self.omega
        )
        # semi-implicit Euler on the shaft, viscous term implicit for stability
        numerator = self.omega + dt / self.inertia * (
            self.torque_motor
            - self.coulomb * math.copysign(1.0, self.omega)
            + self.torque_load / self.gear_ratio
        )
        denominator = 1.0 + dt / self.inertia * (
            self.viscous + self.kt * self.ke / self.resistance
        )
        # the back-EMF term was included explicitly in torque_motor; remove its
        # explicit part and keep the implicit one for consistency
        numerator += (
            dt / self.inertia * (self.kt * self.ke / self.resistance) * self.omega
        )
        self.omega = numerator / denominator
        self.crank_angle += self.omega / self.gear_ratio * dt
        self._impose()
        power = self.voltage * self.current
        self.publish("speed_rad_s", self.omega)
        self.publish("frequency_hz", self.frequency_hz)
        self.publish("current_a", self.current)
        self.publish("voltage_v", self.voltage)
        self.publish("torque_motor", self.torque_motor)
        self.publish("torque_load", self.torque_load)
        self.publish("power_w", power)
        self.publish("crank_angle", self.crank_angle % (2.0 * math.pi))
        _ = friction

    def post_step(self) -> bool:
        self._impose()
        return True

    def _impose(self) -> None:
        data = self.data
        angle = self.crank_angle + self.phase
        omega_c = self.omega / self.gear_ratio
        for (_, qpos, dof), direction in zip(
            self._addresses, self.directions, strict=True
        ):
            data.qpos[qpos] = direction * self.amplitude * math.sin(angle)
            data.qvel[dof] = direction * self.amplitude * omega_c * math.cos(angle)


__all__ = ["DCMotorCrank"]
