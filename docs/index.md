# FlapSim

**Open simulation platform for flapping-wing robots, built on MuJoCo.**

Bring your own robot as URDF or MJCF, declare its wings in a small manifest,
pick aerodynamic models and other components from a plugin registry, and run
the simulation headless, in the viewer, or as a Gymnasium environment.

<div class="grid cards" markdown>

- **Import any robot** — URDF, MJCF, a programmatic `MjSpec`, or the
  parametric flat-plate builder. A `robot.yaml` names the wings, their
  frames, joints, planform and mass distribution. [Import your robot](robot_manifest.md)
- **Wings compute their own lift** — quasi-steady blade-element forces at
  the centre of pressure, with rotational lift, added mass, wake capture and
  ground effect as additive models. [Aerodynamic models](aero_models.md)
- **Everything is a plugin** — actuators, sensors, controllers, atmosphere,
  recorders and your own models share one step pipeline and a signal bus.
  [Write a plugin](plugins.md)
- **Reproducible science** — regression tests pin the validated numbers,
  `flapsim sweep` identifies cycle-averaged parameters, and every example in
  the docs runs in CI. [Getting started](getting_started.md)

</div>

```bash
pip install flapsim
flapsim run SingleThrustFlapper                   # viewer, fixed base, force arrows
flapsim run SingleThrustFlapper --config hover    # floating base, hover PID
flapsim inspect my_robot.urdf                     # bodies and joints for your manifest
```

```python
from flapsim import Simulation

sim = Simulation("SingleThrustFlapper")
sim.add(
    {
        "type": "actuator/sinusoidal_stroke",
        "id": "stroke",
        "joints": ["left_flapping_joint", "right_wing_flapping"],
        "directions": [1, -1],
        "frequency_hz": 40.0,
        "amplitude_rad": 1.5,
    }
)
sim.add({"type": "aero/stack", "models": ["quasi_steady", "rotational_lift"]})
sim.add({"type": "recorder/memory", "signals": ["aero/F_world"], "every": 1e-3})
sim.run(1.0)
print(sim["recorder"]["aero/F_world"][:, 2].mean(), "N mean lift")
```

FlapSim is Apache-2.0 licensed. The quasi-steady model follows Whitney &
Wood (2010) as implemented in the FlapX simulator; see the
[derivations](derivations/quasi_steady_model.md).
