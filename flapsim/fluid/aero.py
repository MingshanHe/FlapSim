"""Standalone FlapX aerodynamic calculation used by the tutorials.

This module reproduces one call to ``FlappingAeroPlugin.apply_forces`` without
creating a MuJoCo model or advancing simulation time.  Inputs and outputs use
SI units.  ``wrench`` follows MuJoCo's world-frame order
``[Fx, Fy, Fz, Tx, Ty, Tz]`` about the wing centre of mass.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from flapsim.config import AeroOptions, TorsoInput, WingInput, WingParams


DEFAULT_PARAMETERS_PATH = Path(__file__).parents[1] / "config" / "Wing.json"

def load_wing_parameters(
    path: str | Path = DEFAULT_PARAMETERS_PATH,
) -> WingParams:
    """Load the flat JSON written by ``06_wing_parameter_calculator``."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    # ``Wing.json`` is also the runtime configuration file and therefore
    # carries schema/options metadata.  Preserve compatibility with the old
    # flat tutorial JSON by selecting only WingParams fields.
    parameter_names = {field.name for field in fields(WingParams)}
    parameter_values = {
        name: values[name] for name in parameter_names if name in values
    }
    missing = sorted(parameter_names - parameter_values.keys())
    if missing:
        raise ValueError(f"missing wing parameters in {path}: {missing}")
    return WingParams(**parameter_values)


def save_wing_parameters(
    params: WingParams,
    path: str | Path = DEFAULT_PARAMETERS_PATH,
) -> Path:
    """Save a ``WingParams`` instance in the flat tutorial JSON format."""

    output = Path(path)
    payload = asdict(params)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and "schema_version" in existing:
            existing.update(payload)
            payload = existing
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def normal_force_coefficient(alpha: float) -> float:
    """FlapX ``C_N(alpha)`` empirical fit."""

    return (
        1.8 * math.sin(2.0 * alpha) * math.cos(alpha)
        + 1.95 * math.sin(alpha)
        - 1.5 * math.cos(2.0 * alpha) * math.sin(alpha)
    )


def center_of_pressure_fraction(alpha: float) -> float:
    """FlapX chordwise ``d_cp(alpha)`` empirical fit."""

    return (
        0.46
        - 0.332 * math.cos(alpha)
        - 0.037 * math.cos(3.0 * alpha)
        - 0.013 * math.cos(5.0 * alpha)
    )


def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def _rotation(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    # MuJoCo's xmat/site_xmat values are rotations by construction.  An
    # orthogonality check here used to run three times per physics step and
    # dominated the 5--20 kHz viewer loop.
    return matrix


def wing_aero(
    params: WingParams,
    side: int,
    psi: float,
    psi_dot: float,
    theta: float,
    theta_dot: float,
    linear_velocity_body: np.ndarray,
    angular_velocity_body: np.ndarray,
    options: AeroOptions | None = None,
) -> dict:
    """Calculate one wing's local aerodynamic force, moments and CoP."""

    options = options or AeroOptions()
    if side not in (-1, 1):
        raise ValueError("side must be +1 (left) or -1 (right)")

    u, v, w = map(float, linear_velocity_body)
    p, q, yaw_rate = map(float, angular_velocity_body)
    phi = options.deviation_angle
    phi_dot = options.deviation_rate
    Phi = options.stroke_plane_angle
    Phi_dot = options.stroke_plane_rate
    s = float(side)

    S_psi, C_psi = math.sin(psi), math.cos(psi)
    S_phi, C_phi = math.sin(phi), math.cos(phi)
    S_Phi, C_Phi = math.sin(Phi), math.cos(Phi)
    d0, ds = params.d0, params.ds

    U_o1 = (
        s * (p * C_psi * C_Phi - Phi_dot * S_psi)
        + q * S_psi
        + yaw_rate * C_psi * S_Phi
        + phi_dot
    )
    U_o0 = (
        s
        * (
            -(u + q * ds) * C_phi * S_Phi
            - yaw_rate * d0 * S_phi * S_psi * C_Phi
            - (v - p * ds) * S_phi * C_psi
            + w * S_phi * S_psi * S_Phi
            + p * d0 * C_phi * C_Phi
        )
        + (u + q * ds) * S_phi * S_psi * C_Phi
        + yaw_rate * d0 * C_phi * S_Phi
        + w * C_phi * C_Phi
        + p * d0 * S_phi * S_psi * S_Phi
    )
    U_i1 = (
        s
        * (
            p * S_phi * S_psi * C_Phi
            + yaw_rate * C_phi * C_Phi
            + Phi_dot * S_phi * C_psi
        )
        - p * C_phi * S_Phi
        - q * S_phi * C_psi
        + yaw_rate * S_phi * S_psi * S_Phi
        + psi_dot * C_phi
    )
    U_i0 = (
        s
        * (
            yaw_rate * d0 * C_psi * C_Phi
            - (v - p * ds) * S_psi
            - w * C_psi * S_Phi
        )
        - (u + q * ds) * C_psi * C_Phi
        - p * d0 * C_psi * S_Phi
    )

    span_cop = params.R * params.r33 / params.r22
    U_i = U_i1 * span_cop + U_i0
    U_o = U_o1 * span_cop + U_o0
    flow_angle = math.atan(U_o / U_i) if abs(U_i) > 1e-12 else 0.0
    alpha = theta + _sign(U_i) * math.pi / 2.0 - flow_angle
    c_n = normal_force_coefficient(alpha)
    d_cp = center_of_pressure_fraction(alpha)

    a_U2 = U_i1**2 + U_o1**2
    a_U1 = 2.0 * (U_i1 * U_i0 + U_o1 * U_o0)
    a_U0 = U_i0**2 + U_o0**2
    R, c, rho = params.R, params.c, params.rho

    force_raw = (
        0.5
        * rho
        * c
        * c_n
        * (
            a_U2 * R**3 * params.r22
            + a_U1 * R**2 * params.r11
            + a_U0 * R * params.r00
        )
        * options.lift_scale
    )
    aero_moment = (
        -0.5
        * _sign(alpha)
        * rho
        * c**2
        * c_n
        * d_cp
        * (
            a_U2 * R**3 * params.z_cp2
            + a_U1 * R**2 * params.z_cp1
            + a_U0 * R * params.z_cp0
        )
        * options.moment_scale
    )
    chord_cop = -aero_moment / force_raw if abs(force_raw) > 1e-12 else 0.0
    damping_raw = (
        -0.125
        * rho
        * abs(theta_dot)
        * theta_dot
        * 5.0
        * R
        * c**4
        * params.z_rd
        * options.moment_scale
    )

    return {
        "F_N": float(np.clip(force_raw, -options.force_limit, options.force_limit)),
        "M_aero": aero_moment,
        "M_rd": float(
            np.clip(
                damping_raw,
                -options.damping_moment_limit,
                options.damping_moment_limit,
            )
        ),
        "span_CoP": span_cop,
        "chord_CoP": chord_cop,
        "alpha": alpha,
        "U_i1": U_i1,
        "U_i0": U_i0,
        "U_o1": U_o1,
        "U_o0": U_o0,
        "U_i_at_CoP": U_i,
        "U_o_at_CoP": U_o,
        "a_U2": a_U2,
        "a_U1": a_U1,
        "a_U0": a_U0,
        "C_N": c_n,
        "d_cp": d_cp,
        "F_N_raw": force_raw,
        "M_rd_raw": damping_raw,
    }


def world_wrench(
    wing: WingInput,
    side: int,
    aero: dict,
    read_only: bool = False,
) -> dict:
    """Convert one wing's local aero result into a world-frame wrench."""

    R = _rotation(wing.R_world_from_wing, "wing rotation")
    cop_local = np.array([0.0, side * aero["span_CoP"], -aero["chord_CoP"]])
    cop_world = np.asarray(wing.body_origin_world, dtype=float) + R @ cop_local
    force_local = np.array([aero["F_N"], 0.0, 0.0])
    force_world = R @ force_local
    arm_world = cop_world - np.asarray(wing.com_world, dtype=float)
    torque_force = np.cross(arm_world, force_world)
    torque_damping = R @ np.array([0.0, aero["M_rd"], 0.0])
    torque_world = torque_force + torque_damping
    wrench = np.r_[force_world, torque_world]
    return {
        "cop_local": cop_local,
        "cop_world": cop_world,
        "arm_world": arm_world,
        "F_local": force_local,
        "F_world": force_world,
        "torque_from_force_world": torque_force,
        "torque_rd_world": torque_damping,
        "torque_world": torque_world,
        "wrench": wrench,
        "applied_wrench": np.zeros(6) if read_only else wrench.copy(),
    }


def aero_step(
    params: WingParams,
    torso: TorsoInput,
    left: WingInput,
    right: WingInput,
    options: AeroOptions | None = None,
) -> dict:
    """Calculate both wings, equivalent to one ``apply_forces`` call."""

    options = options or AeroOptions()
    R_torso = _rotation(torso.R_world_from_body, "torso rotation")
    v_body = R_torso.T @ np.asarray(torso.linear_velocity_world, dtype=float)
    w_body = R_torso.T @ np.asarray(torso.angular_velocity_world, dtype=float)
    magnitude = float(np.linalg.norm(w_body))
    if magnitude > options.angular_velocity_limit:
        w_body *= options.angular_velocity_limit / magnitude

    theta_left_sign = 1.0 if left.pitch_axis_y >= 0.0 else -1.0
    theta_right_sign = 1.0 if right.pitch_axis_y >= 0.0 else -1.0
    left_aero = wing_aero(
        params,
        1,
        left.psi,
        left.psi_dot,
        theta_left_sign * left.theta_raw,
        theta_left_sign * left.theta_dot_raw,
        v_body,
        w_body,
        options,
    )
    right_aero = wing_aero(
        params,
        -1,
        right.psi,
        right.psi_dot,
        theta_right_sign * right.theta_raw,
        theta_right_sign * right.theta_dot_raw,
        v_body,
        w_body,
        options,
    )
    left_world = world_wrench(left, 1, left_aero, options.read_only)
    right_world = world_wrench(right, -1, right_aero, options.read_only)
    total_force = left_world["F_world"] + right_world["F_world"]

    diagnostics = {
        "F_N_L": left_aero["F_N"],
        "F_N_R": right_aero["F_N"],
        "M_aero_L": left_aero["M_aero"],
        "M_aero_R": right_aero["M_aero"],
        "M_rd_L": left_aero["M_rd"],
        "M_rd_R": right_aero["M_rd"],
        "alpha_L": left_aero["alpha"],
        "alpha_R": right_aero["alpha"],
        "chord_CoP_L": left_aero["chord_CoP"],
        "chord_CoP_R": right_aero["chord_CoP"],
        "F_world": total_force,
        "lift_total": abs(left_aero["F_N"]) + abs(right_aero["F_N"]),
        "read_only": options.read_only,
    }
    return {
        "diagnostics": diagnostics,
        "left_aero": left_aero,
        "right_aero": right_aero,
        "left_world": left_world,
        "right_world": right_world,
        "xfrc_applied_L": left_world["applied_wrench"],
        "xfrc_applied_R": right_world["applied_wrench"],
        "v_body": v_body,
        "w_body": w_body,
    }


__all__ = [
    "AeroOptions",
    "DEFAULT_PARAMETERS_PATH",
    "TorsoInput",
    "WingInput",
    "aero_step",
    "center_of_pressure_fraction",
    "load_wing_parameters",
    "normal_force_coefficient",
    "save_wing_parameters",
    "wing_aero",
    "world_wrench",
]
