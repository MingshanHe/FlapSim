# Write a plugin

A plugin is a Python class bound to one `Simulation`. It takes part in the
step phases it overrides and talks to other plugins through named signals.

```
pre_control  sensors read the state and publish readings
control      controllers read signals, publish cmd/<plugin>/<name>
actuate      actuators drive joints (torques or kinematic targets)
forces       force models write xfrc_applied
mj_step
post_step    kinematic actuators re-impose targets (return True if qpos changed)
record       recorders sample the signal bus
```

## A minimal force plugin

```python
import numpy as np
from flapsim.plugins import ForcePlugin, register


@register
class ConstantThrust(ForcePlugin):
    name = "constant_thrust"  # addressed as force/constant_thrust
    provides = ("force",)  # signals you promise to publish
    example_config = {"body": "base_link", "newtons": 0.1}

    def __init__(self, *, body, newtons=0.1, id=None):
        super().__init__(id=id, body=body, newtons=newtons)
        self.body_name, self.newtons = body, float(newtons)

    def setup(self):  # resolve MuJoCo ids once
        import mujoco

        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name
        )
        if self.body_id < 0:
            raise ValueError(f"body {self.body_name!r} not found")

    def forces(self):
        force = np.array([0.0, 0.0, self.command("newtons", self.newtons)])
        self.data.xfrc_applied[self.body_id, :3] += force
        self.publish("force", force)
```

Use it with `sim.add(ConstantThrust(body="base_link"))`, by name
`sim.add({"type": "force/constant_thrust", "body": "base_link"})`, or from a
setup file. `self.command("newtons")` reads `cmd/<id>/newtons`, so any
controller can drive it.

## Plugin kinds

`Plugin`, `ForcePlugin`, `AeroPlugin`, `AtmospherePlugin`, `ActuatorPlugin`,
`SensorPlugin`, `ControllerPlugin`, `RecorderPlugin` set the `kind`. Aero
*models* (additive per-wing loads) are not plugins but `AeroModel`
subclasses registered with `flapsim.aero.register_model`; see
[Aerodynamic models](aero_models.md#writing-a-model).

## Ship it as a package

Declare an entry point so `pip install my-flapsim-plugins` makes your
plugins available by name:

```toml
[project.entry-points."flapsim.plugins"]
my_plugins = "my_package.plugins"      # a module that registers on import,
                                       # a Plugin class, or a callable
```

## Check it

```python
from flapsim.testing import check_plugin

check_plugin(lambda: ConstantThrust(body="base_link"))
```

The conformance check attaches the plugin to the example robot, steps,
resets, steps again, verifies declared signals and that `describe()` is
JSON-serialisable. Run it in your own test suite.
