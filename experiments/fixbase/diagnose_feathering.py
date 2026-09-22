"""Compare CAD/upright poses over complete strokes; no forced pitch motion."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import numpy as np
from flapsim import Simulator, SimulationConfig, SceneConfig, ControllerConfig

out = Path(__file__).parent / 'results'
out.mkdir(exist_ok=True)
results = {}
for label, quat, speed, duration in (
    ('old_100', (1, 0, 0, 0), 100, 1.3),
    ('upright_100', (0, 1, 0, 0), 100, 1.3),
    ('upright_500', (0, 1, 0, 0), 500, .65),
):
    sim = Simulator(SimulationConfig(scene=SceneConfig(orientation=quat),
                                    controller=ControllerConfig(speed=(speed,))))
    rows = []
    names = ['motor_joint', 'left_flapping_joint', 'right_flapping_joint',
             'left_feathering_joint', 'right_feathering_joint']
    for _ in range(round(duration/.001)):
        sim.step(200)
        rows.append([sim.data.time] + [sim.data.joint(n).qpos[0] for n in names]
                    + [sim.data.joint(n).qvel[0] for n in names])
    values = np.asarray(rows)
    np.savez(out / f'diagnosis_{label}.npz', data=values)
    steady = values[values[:, 0] > .1]
    results[label] = {'min_pitch_deg': np.rad2deg(steady[:, 4:6].min(axis=0)).tolist(),
                      'max_pitch_deg': np.rad2deg(steady[:, 4:6].max(axis=0)).tolist(),
                      'motor_speed': float(steady[:, 6].mean()),
                      'warnings': sim.data.warning.number.tolist()}
    print(label, results[label], flush=True)
(out / 'diagnosis.json').write_text(json.dumps(results, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for ax, label in zip(axes, results):
    a = np.load(out / f'diagnosis_{label}.npz')['data']
    ax.plot(a[:, 0], np.rad2deg(a[:, 2]), label='left flap')
    ax.plot(a[:, 0], np.rad2deg(a[:, 4]), label='left pitch')
    ax.plot(a[:, 0], np.rad2deg(a[:, 5]), '--', label='right pitch')
    ax.set_title(label)
    ax.set_ylabel('deg')
    ax.grid(alpha=.3)
    ax.legend(loc='upper right')
axes[-1].set_xlabel('Simulation time [s]')
fig.tight_layout()
fig.savefig(out / 'diagnosis.png', dpi=130)
