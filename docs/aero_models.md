# Aerodynamic models

FlapSim's aerodynamics are additive per-wing models evaluated by one engine
(`flapsim.aero.AeroEngine`). The engine computes each wing's relative-flow
kinematics once per step, asks every model for its loads (a force at a point
plus a pure moment, all in the wing aero frame), multiplies the sum by the
modifiers' factors and writes the wrench to MuJoCo. Plugins expose the engine:
`aero/quasi_steady` is the single-model stack, `aero/stack` composes models.

```yaml
- type: aero/stack
  models:
    - quasi_steady
    - rotational_lift
    - {type: added_mass, include_rotational_inertia: true}
    - {type: wake_capture, C_wc: 0.8}
  modifiers:
    - {type: ground_effect, model: exponential, gain: 0.3}
```

Signals: `aero/F_world`, `aero/lift_total`, `aero/rho`, per wing
`wings/<name>/F_N` (total normal force), `alpha`, `U_i`, `U_o`, `F_world`,
`torque_world`, `cop_world`, and per model `wings/<name>/<model>/<key>`.

## Models

| Model | Physics | Parameters | Reference |
|---|---|---|---|
| `quasi_steady` | Translational normal force `1/2 rho c C_N(alpha) (U^2 integrals)` at the centre of pressure, pitching moment through the chordwise CoP, rotational damping `M_rd` | `options` (AeroOptions overrides) | Whitney & Wood 2010; Dickinson et al. 1999 fits |
| `rotational_lift` | `C_rot rho theta_dot integral c^2 U dr`, positive when pitching up into the flow, at the half chord | `C_rot` (default `pi (0.75 - x0)`), `point_fraction` | Sane & Dickinson 2002 |
| `added_mass` | `integral (rho pi c^2 / 4) dU_n/dt dr` with the normal acceleration from a one-step finite difference; rotational added inertia `rho pi c^4 / 128` | `include_rotational_inertia` | Sedov; Sane & Dickinson 2002 |
| `wake_capture` | Empirical transient `C_wc 1/2 rho S U_rev^2 exp(-t/tau)` after each stroke reversal | `C_wc`, `tau_chords`, `point_fraction`, `min_speed` | Dickinson et al. 1999 (phenomenon) |

## Modifiers

| Modifier | Factor | Parameters |
|---|---|---|
| `ground_effect` | `1 + gain exp(-h / (length_scale R))` or Cheeseman-Bennett `1 / (1 - (R / 4h)^2)` | `model`, `gain`, `length_scale`, `ground_height`, `max_factor` |

## Related plugins

* `atmosphere/standard`: density (ISA at `altitude_m` or `rho`), steady
  `wind`, a `gust` (`one_minus_cos`, `sine`, `step`) and Ornstein-Uhlenbeck
  `turbulence`. Publishes `atmosphere/rho` and `atmosphere/wind_world`, which
  the aero stack and body drag read.
* `force/body_drag`: `-1/2 rho C_d A |v| v` per body axis relative to the wind.
* `aero/cycle_averaged`: rigid-body hover model, `lift = k f^2` along the
  gimbal-tilted thrust axis plus control moments and damping. Parameters come
  from `flapsim sweep` (`flapsim.aero.identify`), which runs the fixed-base
  per-step model as a virtual wind tunnel: frequency sweep for `k`, gimbal
  deflections for thrust tilt and moments per radian, imposed body velocities
  and rates for damping.

## Writing a model

```python
from flapsim.aero import AeroModel, WingLoads, register_model
import numpy as np


@register_model
class MyModel(AeroModel):
    name = "my_model"

    def compute(self, wing, state, kin, flow, options):
        force_n = ...  # along the wing normal
        point = np.array([0.0, state.side * kin["span_CoP"], -0.5 * wing.mean_chord])
        return WingLoads(
            self.name, np.array([force_n, 0, 0]), point, np.zeros(3), {"k": 1.0}
        )
```

`kin` holds the shared kinematics (`U_i1`, `U_i0`, `U_o1`, `U_o0`, `alpha`,
`span_CoP`, `theta_dot`, ...), `flow` the air density, wind and torso
velocity, `wing.params` the planform (`R`, `c`, integrals, hinge geometry).
Models shipped in another package register through the same decorator and
become available by name in `aero/stack` configs.
