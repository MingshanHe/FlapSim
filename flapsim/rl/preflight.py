"""Fixed-base frequency/lift scan before spending time on PPO training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import math
import numpy as np
from flapsim import Simulator, load_config
from flapsim.actuator import DCMotor
from .env import PRESET


class VoltageDrive(DCMotor):
    """Diagnostic adapter: controller command is terminal voltage, not shaft speed."""
    def torque(self, position, velocity, target, xp=np):
        return self.torque_from_voltage(velocity, target, xp)


def scan(config, speeds, *, settle=.2, duration=.6, mode='speed'):
    results = []
    if settle < 0 or duration <= 0 or not math.isfinite(settle+duration):
        raise ValueError('settle must be nonnegative and duration positive')
    if mode not in ('speed', 'voltage'):
        raise ValueError('mode must be speed or voltage')
    if config.actuator.kind != 'dcmotor':
        raise ValueError('electrical scan requires a DC motor')
    for speed in speeds:
        maximum = config.actuator.max_voltage if mode == 'voltage' else config.actuator.max_velocity
        if not 0 < speed <= maximum:
            raise ValueError('scan commands must lie within the configured motor limits')
        actuator = VoltageDrive(config.actuator) if mode == 'voltage' else DCMotor(config.actuator)
        sim = Simulator(replace(config, scene=replace(config.scene, fixed_base=True)), actuator=actuator)
        sim.set_command([speed])
        sim.step(math.ceil(settle / sim.model.opt.timestep))
        block = max(1, round(.0005 / sim.model.opt.timestep))
        samples = []
        for _ in range(math.ceil(duration/(block*sim.model.opt.timestep))):
            sim.step(block)
            shaft = float(sim.data.joint('motor_joint').qvel[0])
            voltage = speed if mode == 'voltage' else float(actuator.voltage_for_speed(shaft, speed))
            current = np.clip((voltage-config.actuator.back_emf*shaft)/config.actuator.resistance,
                              -config.actuator.max_current, config.actuator.max_current)
            samples.append([float(sim.data.time), float(sim.data.joint('carrier_2_joint').qpos[0]),
                            shaft, *sim.fluid.forces(sim.data)[:, :3].sum(axis=0), voltage, current])
        values = np.asarray(samples)
        if not np.isfinite(values).all() or any(w.number for w in sim.data.warning):
            raise RuntimeError(f'unstable physics at shaft command {speed}')
        crossings = np.flatnonzero(np.diff(np.floor(values[:, 1]/(2*np.pi))) > 0) + 1
        if len(crossings) < 2:
            raise ValueError(f'not enough complete cycles at {speed}; increase --duration')
        values = values[crossings[0]:crossings[-1]+1]
        elapsed = values[-1, 0] - values[0, 0]
        integrals = np.sum(.5*(values[1:, 2:]+values[:-1, 2:]) * np.diff(values[:, 0])[:, None], axis=0)
        mean = integrals / elapsed
        mass = float(sim.model.body_mass.sum())
        weight = mass * np.linalg.norm(sim.model.opt.gravity)
        lift = float(mean[3])
        row = {'mode': mode, 'command': float(speed), 'shaft_rad_s': float(mean[0]),
               'flapping_hz': float((values[-1, 1]-values[0, 1])/(2*np.pi*elapsed)),
               'mean_force_N': mean[1:4].tolist(), 'lift_gf': lift / .00980665,
               'mean_voltage_V': float(mean[4]), 'mean_current_A': float(mean[5]),
               'mass_g': mass*1000, 'lift_to_weight': float(lift/weight),
               'sample_seconds': float(elapsed), 'complete_cycles': int(len(crossings)-1),
               'max_voltage': config.actuator.max_voltage}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PRESET)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--speeds', type=float, nargs='+')
    group.add_argument('--voltages', type=float, nargs='+', help='direct terminal voltage, bypasses speed loop')
    parser.add_argument('--settle', type=float, default=.2)
    parser.add_argument('--duration', type=float, default=.6)
    parser.add_argument('--output', type=Path, default=Path('runs/preflight.json'))
    args = parser.parse_args()
    commands = args.voltages or args.speeds or [3000., 4000., 5000.]
    rows = scan(load_config(args.config), commands, settle=args.settle, duration=args.duration,
                mode='voltage' if args.voltages else 'speed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
