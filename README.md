# FlapSim

**Open simulation platform for flapping-wing robots, built on [MuJoCo](https://mujoco.org).**

[![CI](https://github.com/MingshanHe/FlapSim/actions/workflows/ci.yml/badge.svg)](https://github.com/MingshanHe/FlapSim/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Bring your own robot as URDF or MJCF, declare its wings in a small manifest,
pick aerodynamic models and other components from a plugin registry, and run
the simulation headless, in the viewer, or as a Gymnasium environment.

<p align="center">
  <img src="docs/assets/single_thrust_fixed.gif" width="360" alt="Fixed-base flapping with force arrows">
  <img src="docs/assets/flat_plate_fixed.gif" width="360" alt="Parametric flat-plate flapper">
</p>

## Highlights

- **Import any robot.** URDF, MJCF or a programmatic `MjSpec`; a
  `robot.yaml` names the wings, their frames, stroke and feathering joints,
  planform and mass distribution. `flapsim inspect` lists what you need.
  No CAD? `flat_plate_flapper()` builds a parametric two-wing robot.
- **Wings compute their own lift.** Quasi-steady blade-element forces at
  the centre of pressure (Whitney & Wood 2010), plus rotational lift, added
  mass, wake capture and ground effect as additive models you can stack.
- **Everything is a plugin.** Actuators (sinusoidal stroke, PD, servo, DC
  motor + crank), sensors (IMU, range finder, encoder, load cell with
  noise/delay), controllers (hover PID, schedules, Python callbacks),
  atmosphere (density, wind, gusts, turbulence), body drag, recorders.
  Third-party packages register through an entry point.
- **Identification and learning.** `flapsim sweep` fits a cycle-averaged
  hover model from the per-step one; `flapsim.envs.SimulationEnv` turns any
  setup into a Gymnasium environment; `flapsim.study` runs parallel parameter
  sweeps.
- **Reproducible.** Regression tests pin the validated lift sweep, every
  example runs in CI on Linux, macOS and Windows, and the physics
  conventions are documented and self-checked at load time.

## Install

```bash
pip install flapsim                  # core
pip install "flapsim[rl]"            # Gymnasium (add stable-baselines3 for PPO)
pip install "flapsim[video,hdf5]"    # video export, HDF5 recordings
```

From a checkout: `pip install -e ".[dev]"` and `pytest -q`.

## Quick start

```bash
flapsim run SingleThrustFlapper                    # viewer, fixed base, force arrows
flapsim run SingleThrustFlapper --config hover     # floating base with the hover PID
flapsim run SingleThrustFlapper --headless --duration 2 --frequency 40
flapsim inspect my_robot.urdf                      # bodies and joints for your manifest
flapsim plugins                                    # everything you can add
```

```python
from flapsim import Simulation

sim = Simulation("SingleThrustFlapper")  # built-in robot, fixed base
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
sim.add({"type": "sensor/imu", "rate_hz": 500.0})
sim.add(
    {"type": "recorder/memory", "signals": ["aero/F_world", "imu/*"], "every": 1e-3}
)
sim.run(1.0)
print(sim["recorder"]["aero/F_world"][:, 2].mean(), "N mean lift")
```

Setups can also be YAML files (`schema: flapsim.sim/1`); see
`flapsim/assets/robots/single_thrust_flapper/sim.yaml`.

## Your robot in four steps

1. `flapsim inspect robot.urdf` to see bodies and joints.
2. Write `robot.yaml`: model file, root body, wings (`span`, `chord`,
   `stroke_joint`, `feathering_joint`, `planform`, `mass_model`), optional
   joint/equality/mass patches.
3. `flapsim inspect robot.yaml`: the loader warns if a joint sign or chord
   direction looks reversed.
4. Copy a `sim.yaml`, point it at your manifest, `flapsim run sim.yaml`.

Full tutorial: [docs/robot_manifest.md](docs/robot_manifest.md).

## Documentation

`docs/` holds the MkDocs site (`pip install -e ".[docs]" && mkdocs serve`):
getting started, robot manifests, [aerodynamic models](docs/aero_models.md),
[components](docs/components.md), [writing plugins](docs/plugins.md),
[learning and studies](docs/rl.md), the model
[derivations](docs/derivations/quasi_steady_model.md) and the API reference.
The Chinese originals of the derivations live under `docs/zh/`.

## Layout

| Directory | Contents |
|---|---|
| `flapsim/core` | `Simulation`: step pipeline with phases and a signal bus; `flapsim.sim/1` setup files |
| `flapsim/plugins` | Plugin base classes, registry, entry-point discovery |
| `flapsim/robot`, `flapsim/wings` | Manifests, URDF/MJCF/MjSpec import, wing frames, planform integrals, parametric builder |
| `flapsim/aero`, `flapsim/fluid` | Aero engine, models, atmosphere, body drag, cycle-averaged model and identification; the blade-element math |
| `flapsim/actuators`, `flapsim/sensors`, `flapsim/controllers`, `flapsim/io` | Components and recording/visualisation |
| `flapsim/envs`, `flapsim/study` | Gymnasium wrapper and parameter studies |
| `flapsim/assets/robots` | Example robot: URDF, meshes, `robot.yaml`, `sim.yaml`, `hover.yaml` |
| `examples/` | Scripts (hover, sensors, lift sweep, custom aero model, video, parameter study, RL) and notebooks |
| `tests/` | 150+ tests: unit, regression (lift sweep), conformance, examples |

## Contributing and citing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](ROADMAP.md). Please
follow the [code of conduct](CODE_OF_CONDUCT.md). If FlapSim helps your
research, cite it with [CITATION.cff](CITATION.cff).

## License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The
quasi-steady aerodynamic model reproduces the formulation of the FlapX
simulator (Whitney & Wood 2010 with the Dickinson et al. coefficient fits).
