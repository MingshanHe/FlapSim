"""Feasibility probes for wingbeat-synchronous control (fixed mount, IST_robot-V2).

rotation  Does the wing-root load carry a measurable body-rotation signal
          (Coriolis, haltere-like) once the periodic flapping load is removed
          by wingbeat-phase binning?
split     Does phase-locked voltage modulation inside each wingbeat
          (split-cycle) change the cycle-mean aerodynamic wrench, compared
          with the v3 gimbal residual authority (+-0.035 rad)?

Quasi-steady blade elements, ideal sensors, fixed mount: feasibility only.

    python -m experiments.algorithms.wingbeat_feasibility.probe split --output artifacts/wingbeat_feasibility_20261012/split_cycle.json
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from flapsim import Simulator, load_config
from flapsim.measurement import FixedBaseWrenchProbe
from flapsim.rl.env import PRESET
from flapsim.rl.native import accelerate
from flapsim.rl.preflight import VoltageDrive

BASE = load_config(PRESET)
BASE = replace(BASE, scene=replace(BASE.scene, fixed_base=True))
BLOCK = 50  # 0.25 ms of 5 us steps; the drive is updated once per block
BINS = 72


def make(voltage):
    sim = Simulator(BASE, actuator=VoltageDrive(BASE.actuator))
    accelerate(sim, voltage_mode=True)
    sim.set_command([voltage])
    return sim, FixedBaseWrenchProbe(sim.model)


def crank(sim):
    """Unwrapped crank angle; one revolution is one wingbeat."""
    return float(sim.data.joint('carrier_2_joint').qpos[0])


def rotation_run(roll_amp, roll_hz, seconds, voltage=4.0, settle=.3):
    sim, probe = make(voltage)
    m = sim.model
    dt = m.opt.timestep
    wing, unit = m.body('left_wing').id, m.body('thrust_unit').id
    sim.step(round(settle/dt))
    rows, t0 = [], sim.data.time
    for _ in range(round(seconds/(BLOCK*dt))):
        sim.set_gimbal(pitch=0., roll=roll_amp*math.sin(2*math.pi*roll_hz*(sim.data.time-t0)))
        sim.step(BLOCK)
        d = sim.data
        mujoco.mj_rnePostConstraint(m, d)
        w = probe.interaction_wrench(d, wing, d.xpos[wing])
        R = d.xmat[unit].reshape(3, 3)
        rows.append([d.time-t0, crank(sim) % (2*np.pi), *(R.T@d.cvel[unit][:3]), *(R.T@w[:3]), *(R.T@w[3:])])
    return np.asarray(rows)


def rotation_sensing():
    """Phase template from a still run; regress the moving-run residual on roll rate."""
    still = rotation_run(0., 2., 1.5)
    moving = rotation_run(.2, 2., 1.5)
    def bins(x):
        return np.minimum((x[:, 1]/(2*np.pi)*BINS).astype(int), BINS-1)
    b0, b1 = bins(still), bins(moving)
    load = slice(5, 11)
    template = np.array([still[b0 == k, load].mean(axis=0) for k in range(BINS)])
    residual = moving[:, load]-template[b1]
    omega = moving[:, 2]  # thrust-unit x is the roll axis
    alpha = np.gradient(omega, moving[:, 0])
    gains, fitted = np.zeros((BINS, 6)), np.zeros_like(residual)
    for k in range(BINS):
        rows = b1 == k
        X = np.c_[omega[rows], alpha[rows], np.ones(rows.sum())]
        coef, *_ = np.linalg.lstsq(X, residual[rows], rcond=None)
        gains[k], fitted[rows] = coef[0], X@coef
    explained = 1-np.var(residual-fitted, axis=0)/np.var(residual, axis=0)
    carrier = still[:, load].std(axis=0)
    sensitivity = np.sqrt(np.mean(gains**2, axis=0))
    return dict(
        roll_rate_peak_rad_s=float(np.max(np.abs(omega))),
        per_component={name: dict(flapping_load_rms=float(carrier[i]),
                                  rotation_gain_rms_per_rad_s=float(sensitivity[i]),
                                  ratio_per_rad_s=float(sensitivity[i]/carrier[i]),
                                  residual_variance_explained=float(explained[i]))
                       for i, name in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])},
        units='wing-root load on left_wing (parent on child), thrust-unit axes; N and N.m about the wing body origin',
        model='residual = G(phase)*roll_rate + H(phase)*roll_acceleration + c(phase), 72 phase bins')


def mean_wrench(voltage, *, amplitude=0., offset=0., pitch=0., roll=0., seconds=.6, settle=.3):
    """Whole-cycle mean aerodynamic wrench about the whole-robot COM, world axes."""
    sim, probe = make(voltage)
    dt = sim.model.opt.timestep
    sim.set_gimbal(pitch=pitch, roll=roll)
    def drive():
        if amplitude:
            v = voltage*(1+amplitude*math.sin(crank(sim)+offset))
            sim.set_command([float(np.clip(v, 0., BASE.actuator.max_voltage))])
    for _ in range(round(settle/(BLOCK*dt))):
        drive()
        sim.step(BLOCK)
    rows = []
    for _ in range(round(seconds/(BLOCK*dt))):
        drive()
        sim.step(BLOCK)
        d = sim.data
        aero = probe.sum_wrenches(d, sim.fluid.forces(d))
        arm = d.subtree_com[probe.body]-d.xpos[probe.body]
        rows.append([d.time, crank(sim), *aero[:3], *(aero[3:]-np.cross(arm, aero[:3])),
                     float(d.joint('motor_joint').qvel[0])])
    rows = np.asarray(rows)
    edges = np.flatnonzero(np.diff(np.floor(rows[:, 1]/(2*np.pi))) > 0)+1
    rows = rows[edges[0]:edges[-1]+1]
    span = rows[-1, 0]-rows[0, 0]
    mean = np.sum(.5*(rows[1:, 2:]+rows[:-1, 2:])*np.diff(rows[:, 0])[:, None], axis=0)/span
    return dict(force_N=mean[:3].tolist(), moment_Nm=mean[3:6].tolist(), shaft_rad_s=float(mean[6]),
                flap_hz=float((rows[-1, 1]-rows[0, 1])/(2*np.pi*span)), cycles=int(len(edges)-1))


def split_cycle(voltage=3.6, amplitude=.15):
    baseline = mean_wrench(voltage)
    def delta(r):
        return dict(dF_mN=(1e3*(np.array(r['force_N'])-baseline['force_N'])).round(3).tolist(),
                    dM_uNm=(1e6*(np.array(r['moment_Nm'])-baseline['moment_Nm'])).round(1).tolist(),
                    flap_hz=round(r['flap_hz'], 3))
    return dict(
        baseline=baseline, voltage=voltage, amplitude=amplitude,
        split_cycle={f'{45*k}deg': delta(mean_wrench(voltage, amplitude=amplitude, offset=k*np.pi/4))
                     for k in range(8)},
        gimbal_residual={'pitch+0.035rad': delta(mean_wrench(voltage, pitch=.035)),
                         'roll+0.035rad': delta(mean_wrench(voltage, roll=.035))},
        units='dF in mN, dM in micro N.m about the whole-robot COM, world axes; voltage = V0*(1+a*sin(crank+offset))')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('part', choices=['rotation', 'split'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = rotation_sensing() if a.part == 'rotation' else split_cycle()
    sim, _ = make(4.0)
    result.update(wing_mass_g=1e3*float(sim.model.body('left_wing').mass[0]),
                  scope='fixed mount; quasi-steady blade elements; ideal noiseless sensors; feasibility only')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
