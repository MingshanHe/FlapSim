# Getting started

## Install

```bash
pip install flapsim                 # core: MuJoCo, NumPy, PyYAML
pip install "flapsim[rl]"           # + Gymnasium (add stable-baselines3 for PPO)
pip install "flapsim[video,hdf5]"   # + imageio/ffmpeg video, h5py export
```

From a checkout: `pip install -e ".[dev]"` then `pytest -q`.

Python 3.10 or newer; MuJoCo 3.3 or newer is installed automatically. On a
headless machine set `MUJOCO_GL=osmesa` or `MUJOCO_GL=egl` for rendering.

## Run the example robot

The package ships a 30 g single-thrust flapper (two passive-feathering wings
on a two-axis gimbal) with two setups:

```bash
flapsim run SingleThrustFlapper                       # sim.yaml: fixed base, quasi-steady aero
flapsim run SingleThrustFlapper --config hover        # hover.yaml: floating base, hover PID
flapsim run SingleThrustFlapper --headless --duration 2 --frequency 40
flapsim run SingleThrustFlapper --describe            # the assembled plugins and signals
```

## Assemble a simulation in Python

```python
from flapsim import Simulation

sim = Simulation("SingleThrustFlapper")  # or a robot.yaml / URDF / MJCF path
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
sim.add({"type": "sensor/imu", "rate_hz": 500.0})
sim.add(
    {"type": "recorder/memory", "signals": ["aero/F_world", "imu/*"], "every": 1e-3}
)

sim.run(1.0)  # headless
rec = sim["recorder"]
print(rec["aero/F_world"][:, 2].mean())
rec.to_csv("run.csv")
```

Plugins are addressed by `kind/name` (`flapsim plugins` lists them). They
communicate through named signals: the aero plugin publishes
`wings/left/alpha`, the IMU publishes `imu/gyro`, a controller publishes
`cmd/stroke/frequency_hz` and the stroke actuator obeys it.

## Or write a setup file

```yaml
# my_sim.yaml
schema: flapsim.sim/1
robot: SingleThrustFlapper          # or a path to robot.yaml / model file
plugins:
  - type: actuator/sinusoidal_stroke
    id: stroke
    joints: [left_flapping_joint, right_wing_flapping]
    directions: [1, -1]
    frequency_hz: 40.0
    amplitude_rad: 1.5
  - type: atmosphere/standard
    wind: [0.5, 0.0, 0.0]
  - type: aero/stack
    models: [quasi_steady, rotational_lift, added_mass]
  - type: recorder/memory
    signals: ["aero/F_world", "wings/*/alpha"]
    every: 1.0e-3
run:
  duration: 2.0
```

```bash
flapsim run my_sim.yaml --headless
flapsim record my_sim.yaml --output clip.gif --duration 0.1 --slow 20
```

## Next steps

* [Import your robot](robot_manifest.md): write a manifest for your URDF.
* [Aerodynamic models](aero_models.md) and [Components](components.md).
* [Write a plugin](plugins.md) and ship it as a pip package.
* [Learning and studies](rl.md): Gymnasium wrapper and parameter sweeps.
