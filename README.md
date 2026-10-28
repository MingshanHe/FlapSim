# FlapSim

**Open simulation platform for flapping-wing robots, built on [MuJoCo](https://mujoco.org).**

Import your own robot (URDF or MJCF), tell FlapSim which bodies are wings, pick
aerodynamic models and other components from a plugin registry, and run the
simulation headless or in the viewer.

> Status: the platform is being built on the `open-platform` branch following
> [ROADMAP.md](ROADMAP.md). The current release is the validated single-robot
> simulator described below; the plugin API lands phase by phase.

## Install

```bash
git clone https://github.com/MingshanHe/FlapSim.git
cd FlapSim
pip install -e ".[dev]"
```

Requires Python 3.10 or newer and MuJoCo 3.3 or newer (installed automatically).

## Run the example robot

```bash
# viewer, fixed base, quasi-steady blade-element aerodynamics
python flapsim.py --robot SingleThrustFlapper

# headless for 2 simulated seconds
python flapsim.py --robot SingleThrustFlapper --headless --duration 2

# floating base with the cycle-averaged hover model and the altitude controller
python flapsim.py --robot SingleThrustFlapper --fix false
```

From Python, assemble a simulation from plugins:

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
sim.add("aero/quasi_steady")
sim.add(
    {
        "type": "recorder/memory",
        "signals": ["aero/F_world", "wings/*/alpha"],
        "every": 1e-3,
    }
)
sim.run(1.0)  # headless
lift = sim["recorder"]["aero/F_world"][:, 2].mean()
```

or run a shipped setup file (`flapsim run SingleThrustFlapper --config hover`
opens the floating-base hover study). Your own robot needs a manifest
(`robot.yaml`) that names its wings; `flapsim inspect my_robot.urdf` lists the
bodies and joints to fill it in.

## What is in the box

| Directory | Contents |
|---|---|
| `flapsim/core` | `Simulation`: step pipeline with phases and a signal bus; `flapsim.sim/1` config files |
| `flapsim/plugins` | Plugin base classes, registry, entry-point discovery |
| `flapsim/robot`, `flapsim/wings` | Robot manifests, URDF/MJCF/MjSpec import, wing frames and planform integrals, parametric builder |
| `flapsim/aero`, `flapsim/fluid` | Quasi-steady and cycle-averaged aero plugins; the blade-element math |
| `flapsim/actuators`, `flapsim/controllers` | Sinusoidal stroke, PD position, servo, DC motor + crank; hover PID, set-point schedule, Python callback |
| `flapsim/sensors`, `flapsim/io` | IMU, range finder, encoder, load cell with noise/delay models; memory recorder (npz/csv/hdf5), force-arrow overlay, GIF/MP4/PNG rendering |
| `flapsim/assets/robots` | Example robot: URDF, meshes, `robot.yaml`, `sim.yaml`, `hover.yaml` |
| `flapsim/testing` | Conformance check for third-party plugins |
| `docs/` | Derivations and engineering notes (`docs/zh` holds the Chinese originals) |
| `examples/` | Notebooks on robot import, wing parameters and aerodynamics |
| `tests/` | Unit, symmetry and regression tests (`pytest`) |

## Testing

```bash
pytest -q
ruff check . && ruff format --check .
```

`tests/test_lift_sweep.py` pins the mean lift of the example robot at 20 to
50 Hz. If you change force conventions or wing parameters on purpose, update
the numbers there and say why in the pull request.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Everyone interacting with the project
is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## License and attribution

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The
quasi-steady aerodynamic model reproduces the formulation used by the FlapX
simulator; see `docs/zh/derivations` for the full derivation.

## Citing

See [CITATION.cff](CITATION.cff).
