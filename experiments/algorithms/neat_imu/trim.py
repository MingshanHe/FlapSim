"""Calibrate full V2 fixed-base lift and moment balance for residual NEAT."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from flapsim import Simulator, load_config
from flapsim.rl.env import PRESET
from flapsim.rl.native import accelerate
from flapsim.rl.native_fast import accelerate_fast


def measure(command, settle=.8, duration=.6, sample_dt=.0005, model_path=None):
    speed, alpha, beta = map(float, command)
    source = load_config(PRESET)
    if model_path is not None:
        source=replace(source,model_path=Path(model_path))
    sim = accelerate_fast(accelerate(Simulator(replace(source,
        scene=replace(source.scene, fixed_base=True, orientation=(1., 0., 0., 0.)),
        gimbal=replace(source.gimbal, angles=(alpha, beta))))))
    sim.set_speed(speed)
    sim.set_gimbal(pitch=alpha, roll=beta)
    sim.step(round(settle/sim.model.opt.timestep))
    samples = []
    dt = sample_dt
    block = round(dt/sim.model.opt.timestep)
    mass = sim.model.body_mass
    for _ in range(round(duration/dt)+1):
        d = sim.data
        force = sim.fluid.forces(d)
        com = np.sum(mass[:, None]*d.xipos, axis=0)/mass.sum()
        torque = np.sum(force[:, 3:]+np.cross(d.xipos-com, force[:, :3]), axis=0)
        samples.append([d.time, d.joint('carrier_2_joint').qpos[0],
                        *force[:, :3].sum(axis=0), *torque,
                        d.joint('motor_joint').qvel[0],
                        d.joint('pitch_joint').qpos[0], d.joint('roll_joint').qpos[0]])
        sim.step(block)
    values = np.asarray(samples)
    if not np.isfinite(values).all() or any(w.number for w in sim.data.warning):
        raise RuntimeError('Numerical failure in trim measurement')
    t, phase = values[:, 0], values[:, 1]/(2*np.pi)
    if not np.all(np.diff(phase) > 0):
        raise RuntimeError('Carrier did not rotate monotonically')
    ticks = np.arange(np.ceil(phase[0]), np.floor(phase[-1])+1)
    boundaries = np.interp(ticks, phase, t)
    if len(boundaries) < 4:
        raise RuntimeError('Not enough complete cycles for calibration')
    cycles = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        grid = np.r_[start, t[(t > start) & (t < end)], end]
        block_values = np.column_stack([np.interp(grid, t, values[:, j]) for j in range(2, values.shape[1])])
        cycles.append(np.trapezoid(block_values, grid, axis=0)/(end-start))
    mean = np.average(cycles, axis=0, weights=np.diff(boundaries))
    weight = float(mass.sum()*np.linalg.norm(sim.model.opt.gravity))
    return dict(command=list(map(float, command)), lift_gf=float(mean[2]/.00980665),
                force_N=mean[:3].tolist(), torque_com_Nm=mean[3:6].tolist(),
                flapping_hz=float((len(boundaries)-1)/(boundaries[-1]-boundaries[0])),
                actual_motor_rad_s=float(mean[6]), actual_gimbal_rad=mean[7:9].tolist(),
                weight_N=weight, weight_gf=weight/.00980665, mass_g=float(mass.sum()*1000),
                complete_cycles=len(cycles), sample_seconds=float(boundaries[-1]-boundaries[0]),
                cycle_force_z_std_N=float(np.std(np.asarray(cycles)[:, 2])),
                late_force_z_change_N=float(np.mean(np.asarray(cycles)[len(cycles)//2:, 2])-
                                            np.mean(np.asarray(cycles)[:len(cycles)//2, 2])),
                late_torque_change_Nm=(np.mean(np.asarray(cycles)[len(cycles)//2:, 3:6], axis=0)-
                                      np.mean(np.asarray(cycles)[:len(cycles)//2, 3:6], axis=0)).tolist(),
                sample_dt=sample_dt,
                physics_steps=round((settle+duration+dt)/sim.model.opt.timestep))


def precise_measure(command):
    return measure(command, duration=1.2, sample_dt=.00005)


def residual(row):
    return np.r_[np.linalg.norm(row['force_N'])/row['weight_N']-1,
                 np.asarray(row['torque_com_Nm'])[:2]/(row['weight_N']*.02)]


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--precise', action='store_true')
    p.add_argument('--model-path', type=Path, help='independent experimental MJCF mass variant')
    args = p.parse_args()
    if args.model_path is not None:
        args.model_path = args.model_path.resolve()
        if not args.model_path.is_file():
            p.error('model-path must be an existing MJCF file')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    history = []
    # Serial construction primes the native build cache before workers start.
    measurement = partial(measure, model_path=args.model_path,
                          duration=1.2 if args.precise else .6,
                          sample_dt=.00005 if args.precise else .0005)
    initial = measurement([3400., 0., 0.])
    history.append(initial)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scan = list(pool.map(measurement, [[s, 0., 0.] for s in (2800., 3100., 3700., 4000., 4400.)]))
        history.extend(scan)
        scan = sorted([initial]+scan, key=lambda row: row['command'][0])
        write(args.output/'speed_scan.json', scan)
        for row in scan:
            print(json.dumps(row), flush=True)
        speeds = [r['command'][0] for r in scan]
        lifts = [r['force_N'][2] for r in scan]
        if not all(b > a for a, b in zip(lifts[:-1], lifts[1:])):
            raise RuntimeError('Lift curve is not monotonic; inspect scan')
        if not lifts[0] <= initial['weight_N'] <= lifts[-1]:
            raise RuntimeError('Weight is not bracketed by scan')
        neutral_speed = float(np.interp(initial['weight_N'], lifts, speeds))
        neutral = measurement([neutral_speed, 0., 0.])
        history.append(neutral)
        write(args.output/'neutral_lift_trim.json', neutral)
        x = np.array([neutral_speed, 0., 0.])
        best = neutral
        # Numerical Newton steps balance |F|=mg and roll/pitch moment=0.
        # Result is checked explicitly; no claim of a solved trim without it.
        for iteration in range(7):
            base = best if np.array_equal(x, best['command']) else measurement(x)
            if base is not best:
                history.append(base)
            errors = residual(base)
            print(json.dumps(dict(iteration=iteration, command=x.tolist(), residual=errors.tolist())), flush=True)
            if abs(errors[0]) < .001 and np.linalg.norm(base['torque_com_Nm'][:2]) < 1e-7:
                best = base
                break
            delta = np.array([40., .01, .01])
            probes = list(pool.map(measurement, [x+np.eye(3)[i]*delta[i] for i in range(3)]))
            history.extend(probes)
            jacobian = np.column_stack([(residual(row)-errors)/delta[i] for i, row in enumerate(probes)])
            step = np.linalg.lstsq(jacobian, -errors, rcond=1e-5)[0]
            step /= max(1., abs(step[0])/250., abs(step[1])/.06, abs(step[2])/.06)
            candidates = [np.clip(x+factor*step, [2400., -.25, -.25], [4700., .25, .25])
                          for factor in (1., .5)]
            rows = list(pool.map(measurement, candidates))
            history.extend(rows)
            chosen = min([base]+rows, key=lambda r: np.linalg.norm(residual(r)))
            best = chosen
            write(args.output/'measurements.json', history)
            if np.array_equal(x, chosen['command']):
                break
            x = np.array(chosen['command'])
    force = np.array(best['force_N'])
    # R_y(pitch) R_x(roll) sends the measured force vector to world +z.
    roll = math.atan2(force[1], force[2])
    pitch = -math.atan2(force[0], math.hypot(force[1], force[2]))
    result = dict(**best, hover_attitude_rad=[roll, pitch, 0.],
                  normalized_residual=residual(best).tolist(),
                  converged=bool(abs(residual(best)[0]) < .001 and np.linalg.norm(best['torque_com_Nm'][:2]) < 1e-7),
                  tolerances=dict(relative_force=.001, roll_pitch_torque_Nm=1e-7),
                  neutral_lift_only=neutral,
                  scope='Full two-wing V2 simulation trim, not a hardware calibration',
                  source_config=str(PRESET),
                  model_path=str(args.model_path or load_config(PRESET).path),
                  model_sha256=hashlib.sha256((args.model_path or load_config(PRESET).path).read_bytes()).hexdigest(),
                  wall_seconds=time.monotonic()-start,
                  total_physics_steps=sum(r['physics_steps'] for r in history))
    write(args.output/'measurements.json', history)
    write(args.output/'trim.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
