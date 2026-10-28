"""Identify cycle-averaged parameters from the per-step model.

The fixed-base robot is a virtual wind tunnel: sweep the stroke frequency to
fit the lift coefficient, deflect each gimbal joint to measure how the mean
force direction and the mean moment change per radian, and impose body
velocities and rates on the aero stack to measure damping.  The result is a
``flapsim.cycle_averaged/1`` parameter file for
:class:`~flapsim.aero.cycle_averaged.CycleAveragedAero`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flapsim.aero.cycle_averaged import PARAMETERS_SCHEMA


@dataclass
class MeanLoads:
    force_world: np.ndarray
    torque_root: np.ndarray
    samples: int


@dataclass
class LiftSweep:
    frequencies: list[float]
    mean_forces: list[list[float]]
    lift_coefficient: float
    fit_rms_relative: float
    amplitude_rad: float


@dataclass
class IdentificationResult:
    parameters: dict[str, Any]
    lift_sweep: LiftSweep
    gimbal: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    damping: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "lift_sweep": asdict(self.lift_sweep),
            "gimbal": self.gimbal,
            "damping": self.damping,
        }


def _root_torque(sim, aero) -> np.ndarray:
    """Torque of all wing wrenches about the root body's centre of mass."""
    data = sim.data
    root_com = data.xipos[sim.robot.root_body]
    torque = np.zeros(3)
    for wing in aero.wings:
        wrench = data.xfrc_applied[wing.body_id]
        torque += wrench[3:] + np.cross(data.xipos[wing.body_id] - root_com, wrench[:3])
    return torque


def mean_loads(
    sim, *, settle: float, cycles: float, frequency_hz: float, aero_id: str = "aero"
) -> MeanLoads:
    aero = sim[aero_id]
    sim.advance(settle)
    end = sim.time + cycles / max(frequency_hz, 1e-9)
    force = np.zeros(3)
    torque = np.zeros(3)
    count = 0
    while sim.time < end:
        sim.step()
        force += aero.diagnostics["F_world"]
        torque += _root_torque(sim, aero)
        count += 1
    return MeanLoads(force / count, torque / count, count)


def lift_sweep(
    make_sim: Callable[[], Any],
    frequencies: Sequence[float],
    *,
    amplitude_rad: float | None = None,
    settle: float = 0.2,
    cycles: float = 10.0,
    stroke_id: str = "stroke",
    aero_id: str = "aero",
) -> LiftSweep:
    """Mean world force per frequency and the least-squares ``lift = k f^2`` fit."""
    forces = []
    for frequency in frequencies:
        sim = make_sim()
        sim[stroke_id].set_command(frequency_hz=frequency, amplitude_rad=amplitude_rad)
        loads = mean_loads(
            sim, settle=settle, cycles=cycles, frequency_hz=frequency, aero_id=aero_id
        )
        forces.append(loads.force_world)
    f2 = np.asarray(frequencies, dtype=float) ** 2
    lift = np.asarray([f[2] for f in forces])
    k = float(f2 @ lift / (f2 @ f2))
    residual = lift - k * f2
    rms_rel = float(
        np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(lift**2)), 1e-12)
    )
    amplitude = sim[stroke_id].amplitude_rad
    return LiftSweep(
        [float(f) for f in frequencies],
        [list(map(float, f)) for f in forces],
        k,
        rms_rel,
        float(amplitude),
    )


def gimbal_sensitivity(
    make_sim: Callable[[], Any],
    *,
    frequency_hz: float,
    joints: Sequence[str] | None = None,
    angle_rad: float = 0.1,
    settle: float = 0.2,
    cycles: float = 10.0,
    stroke_id: str = "stroke",
    gimbal_id: str = "gimbal",
    aero_id: str = "aero",
) -> dict[str, dict[str, list[float]]]:
    """Per gimbal joint: d(mean force)/d(angle) and d(mean torque)/d(angle).

    Central differences at ``+/- angle_rad``; forces and torques are expressed
    in the root body frame.
    """
    result: dict[str, dict[str, list[float]]] = {}
    probe = make_sim()
    joint_names = (
        list(joints) if joints is not None else list(probe[gimbal_id].settings)
    )
    for joint in joint_names:
        samples = {}
        for sign in (-1.0, 1.0):
            sim = make_sim()
            sim[stroke_id].set_command(frequency_hz=frequency_hz)
            sim[gimbal_id].set_target(joint, sign * angle_rad)
            loads = mean_loads(
                sim,
                settle=settle,
                cycles=cycles,
                frequency_hz=frequency_hz,
                aero_id=aero_id,
            )
            rotation = sim.data.xmat[sim.robot.root_body].reshape(3, 3)
            samples[sign] = (
                rotation.T @ loads.force_world,
                rotation.T @ loads.torque_root,
            )
        d_force = (samples[1.0][0] - samples[-1.0][0]) / (2.0 * angle_rad)
        d_torque = (samples[1.0][1] - samples[-1.0][1]) / (2.0 * angle_rad)
        result[joint] = {
            "force_per_rad": list(map(float, d_force)),
            "torque_per_rad": list(map(float, d_torque)),
        }
    return result


def damping_sensitivity(
    make_sim: Callable[[], Any],
    *,
    frequency_hz: float,
    speed: float = 0.5,
    rate: float = 1.0,
    settle: float = 0.2,
    cycles: float = 10.0,
    stroke_id: str = "stroke",
    aero_id: str = "aero",
) -> dict[str, list[float]]:
    """Linear and angular damping from imposed body velocities and rates.

    Returns ``linear`` (``-dF_i/dv_i``) and ``angular`` (``-dT_i/dw_i``) in the
    root body frame, using central differences on the aero stack's imposed
    body velocity and rates.
    """
    out: dict[str, list[float]] = {"linear": [], "angular": []}
    for which in ("linear", "angular"):
        for axis in range(3):
            responses = {}
            for sign in (-1.0, 1.0):
                sim = make_sim()
                sim[stroke_id].set_command(frequency_hz=frequency_hz)
                aero = sim[aero_id]
                vector = np.zeros(3)
                vector[axis] = sign * (speed if which == "linear" else rate)
                if which == "linear":
                    aero.imposed_body_velocity = vector
                else:
                    aero.imposed_body_rates = vector
                loads = mean_loads(
                    sim,
                    settle=settle,
                    cycles=cycles,
                    frequency_hz=frequency_hz,
                    aero_id=aero_id,
                )
                rotation = sim.data.xmat[sim.robot.root_body].reshape(3, 3)
                responses[sign] = (
                    rotation.T @ loads.force_world,
                    rotation.T @ loads.torque_root,
                )
            step = 2.0 * (speed if which == "linear" else rate)
            if which == "linear":
                slope = (responses[1.0][0][axis] - responses[-1.0][0][axis]) / step
            else:
                slope = (responses[1.0][1][axis] - responses[-1.0][1][axis]) / step
            out[which].append(float(max(-slope, 0.0)))
    return out


def identify_cycle_averaged(
    make_sim: Callable[[], Any],
    *,
    frequencies: Sequence[float] = (20.0, 30.0, 40.0, 50.0),
    reference_frequency_hz: float | None = None,
    amplitude_rad: float | None = None,
    gimbal_angle_rad: float = 0.1,
    include_damping: bool = True,
    settle: float = 0.2,
    cycles: float = 10.0,
    stroke_id: str = "stroke",
    gimbal_id: str | None = "gimbal",
    aero_id: str = "aero",
) -> IdentificationResult:
    """Run all sweeps and assemble a ``flapsim.cycle_averaged/1`` parameter set."""
    sweep = lift_sweep(
        make_sim,
        frequencies,
        amplitude_rad=amplitude_rad,
        settle=settle,
        cycles=cycles,
        stroke_id=stroke_id,
        aero_id=aero_id,
    )
    f_ref = float(reference_frequency_hz or frequencies[len(frequencies) // 2])
    probe = make_sim()
    mass = float(np.sum(probe.model.body_mass[1:]))
    gravity = float(np.linalg.norm(probe.model.opt.gravity))
    lift_ref = sweep.lift_coefficient * f_ref**2
    parameters: dict[str, Any] = {
        "schema": PARAMETERS_SCHEMA,
        "robot": probe.robot.manifest.name,
        "stroke_amplitude_rad": sweep.amplitude_rad,
        "reference": {"frequency_hz": f_ref, "lift_n": float(lift_ref)},
        "lift_coefficient": sweep.lift_coefficient,
        "lift_fit_rms_relative": sweep.fit_rms_relative,
        "mass_kg": mass,
        "hover_frequency_hz": float(np.sqrt(mass * gravity / sweep.lift_coefficient))
        if sweep.lift_coefficient > 0
        else None,
        "moment_reference_frequency_hz": f_ref,
    }
    gimbal: dict[str, dict[str, list[float]]] = {}
    if gimbal_id is not None and gimbal_id in probe:
        gimbal = gimbal_sensitivity(
            make_sim,
            frequency_hz=f_ref,
            angle_rad=gimbal_angle_rad,
            settle=settle,
            cycles=cycles,
            stroke_id=stroke_id,
            gimbal_id=gimbal_id,
            aero_id=aero_id,
        )
        parameters["thrust_tilt_per_rad"] = {
            joint: [v / lift_ref for v in data["force_per_rad"]]
            for joint, data in gimbal.items()
        }
        parameters["moment_per_rad"] = {
            joint: data["torque_per_rad"] for joint, data in gimbal.items()
        }
    damping: dict[str, list[float]] = {}
    if include_damping:
        damping = damping_sensitivity(
            make_sim,
            frequency_hz=f_ref,
            settle=settle,
            cycles=cycles,
            stroke_id=stroke_id,
            aero_id=aero_id,
        )
        parameters["linear_damping"] = damping["linear"]
        parameters["angular_damping"] = damping["angular"]
    return IdentificationResult(parameters, sweep, gimbal, damping)


def write_parameters(parameters: dict[str, Any], path: str | Path) -> Path:
    import yaml

    path = Path(path)
    path.write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
    return path


__all__ = [
    "IdentificationResult",
    "LiftSweep",
    "MeanLoads",
    "damping_sensitivity",
    "gimbal_sensitivity",
    "identify_cycle_averaged",
    "lift_sweep",
    "mean_loads",
    "write_parameters",
]
