"""Full-physics motor and gimbal identification, including cycle-average moments."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from flapsim import Simulator, load_config
from flapsim.rl.env import PRESET
from flapsim.rl.native import accelerate


def measure(speed, pitch=0., roll=0., settle=.2, duration=.4):
    config = load_config(PRESET)
    sim = accelerate(Simulator(replace(config, scene=replace(config.scene, fixed_base=True))))
    sim.set_speed(speed)
    sim.set_gimbal(pitch=pitch, roll=roll)
    sim.step(round(settle/sim.model.opt.timestep))
    values = []
    start = time.monotonic()
    for _ in range(round(duration/.0005)):
        sim.step(round(.0005/sim.model.opt.timestep))
        d = sim.data
        f = sim.fluid.forces(d)
        com = d.subtree_com[sim.model.body('base_link').id]
        torque = np.sum(f[:, 3:] + np.cross(d.xipos-com, f[:, :3]), axis=0)
        values.append([d.time, d.joint('carrier_2_joint').qpos[0],
                       d.joint('motor_joint').qvel[0], *np.sum(f[:, :3], axis=0), *torque])
    values = np.asarray(values)
    crossings = np.flatnonzero(np.diff(np.floor(values[:, 1]/(2*np.pi))) > 0)+1
    if len(crossings) < 2 or any(w.number for w in sim.data.warning):
        raise RuntimeError('invalid scan')
    v = values[crossings[0]:crossings[-1]+1]
    elapsed = v[-1, 0]-v[0, 0]
    mean = np.sum((v[1:, 2:]+v[:-1, 2:])*.5*np.diff(v[:, 0])[:, None], axis=0)/elapsed
    row = dict(speed=speed, pitch=pitch, roll=roll, shaft=float(mean[0]),
               force=mean[1:4].tolist(), torque=mean[4:].tolist(),
               mass=float(sim.model.body_mass.sum()), wall_seconds=time.monotonic()-start)
    print(json.dumps(row), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('runs/identification.json'))
    args = p.parse_args()
    rows = [measure(speed, pitch, roll) for speed, pitch, roll in
            [(2400,0,0),(2600,0,0),(2800,0,0),(3000,0,0),(4000,0,0),
             (2800,.08,0),(2800,-.08,0),(2800,0,.08),(2800,0,-.08)]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
