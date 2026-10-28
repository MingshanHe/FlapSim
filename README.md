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

From Python:

```python
from flapsim import SimCore

sim = SimCore("SingleThrustFlapper", fix=True)
sim.set_thrust_command(frequency=40.0, amplitude=1.5)
sim.run(viewer=False, duration=1.0)
print(sim.aero.diagnostics["F_world"])  # world-frame aerodynamic force [N]
```

## What is in the box

| Directory | Contents |
|---|---|
| `flapsim/environment` | MuJoCo scene, URDF import through `MjSpec`, step loop |
| `flapsim/fluid` | Quasi-steady blade-element aerodynamics (pure functions) and the MuJoCo adapter; cycle-averaged hover model |
| `flapsim/actuator`, `flapsim/controller` | PD position and sinusoidal stroke actuators; plate, thrust and hover controllers |
| `flapsim/config` | JSON configuration and typed parameter objects |
| `flapsim/robot/SingleThrustFlapper` | Example robot: URDF and meshes of a 30 g single-thrust flapper |
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
