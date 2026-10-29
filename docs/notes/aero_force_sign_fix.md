# Record of the aerodynamic force direction bug fix

*Translated from the Chinese original (`docs/zh/notes/aero_force_sign_fix.md`). It documents the implementation as it stood before 2026-10-28; the pitching-moment sign and the removed force flip are described in [quasi_steady_model.md](../derivations/quasi_steady_model.md).*

## Symptom

With `base_link` fixed, raising the flapping frequency clearly increased the absolute normal force of the left and right wings, yet the mean world-frame `Fz` was always negative:

```text
20 Hz: -0.067 mN
30 Hz: -0.300 mN
40 Hz: -0.823 mN
50 Hz: -1.685 mN
60 Hz: -2.774 mN
```

This means that the higher the frequency, the larger the mean downward aerodynamic force on the robot. The hovering controller kept raising the frequency as the altitude dropped, forming a wrong positive feedback loop.

Note: the sum of the left and right `|F_N|` exceeding the robot's weight does not mean that enough world-frame `+Z` lift is produced. The transformed mean `F_world[2]` must be checked.

## Comparison with FlapX

Running the FlapX Purdue Flapper in the parent directory directly, with

```text
frequency = 30 Hz
amplitude = 1.5 rad
fluid = aero
torso = clamped
```

gives

```text
L_stroke / R_stroke correlation = 1.0
L_pitch  / R_pitch  correlation = 1.0
mean Fz = +0.1029 N
```

The FlapX joint axes are:

```text
L_stroke: +Z
R_stroke: -Z
L_pitch:  +Y
R_pitch:  +Y
```

The left and right stroke actuators receive the same position command, and the mirrored motion is realised by opposite joint axes. The left and right passive-pitch joint coordinates have the same sign.

## Differences between FlapSim and FlapX

### 1. Stroke joint coordinate

After the joint-origin rotation in the current URDF, the world axes of both strokes are close to `+Z`. FlapSim must therefore use opposite joint coordinates to obtain a mirrored stroke:

```text
q_right_stroke = -q_left_stroke
```

Before being passed into the FlapX aerodynamic formulas, the right-wing stroke coordinate is converted back to the common aerodynamic convention through `stroke_sign=-1`.

### 2. Wrong right-wing pitch sign

The adapter once contained an experimental

```python
pitch_sign = -1.0
```

But in the FlapX Purdue model both pitch axes are `+Y`, so the left and right pitch coordinates should use the same sign. That sign inversion has been removed.

### 3. Inconsistent neutral aero frames of the left and right wings

In FlapX, the rotation matrices of `L_wing` and `R_wing` at the neutral pose are both the identity, i.e. the left and right aero frames coincide exactly.

After URDF import, the left and right wing bodies use different local mesh frames. If the same fixed body-to-aero transform is applied directly to both, the chord/normal of the right wing points in the wrong direction.

The fix is to calibrate at the model's neutral pose:

```python
neutral_aero_rotation = left_body_rotation @ aero_from_body_left

aero_from_body_right = right_body_rotation.T @ neutral_aero_rotation
```

After calibration, the maximum matrix error between the left and right neutral aero frames is about

```text
1.3e-15
```

### 4. STL normal opposite to the FlapX oriented normal

A phase-by-phase comparison of FlapX and FlapSim showed that the stroke and feathering phases already agreed, but the world `Fz` had exactly the opposite sign.

Example phase data:

```text
FlapX:   +0.256, -0.052, +0.254, -0.051 ... N
FlapSim: -0.210, +0.201, -0.212, +0.200 ... N
```

This shows that the passive feathering phase is not inverted; rather, the oriented normal of the imported STL is opposite to the FlapX force convention.

The final treatment was to invert only the translational normal force applied to the wing body, in `MuJoCoAeroAdapter.apply_forces()`:

```python
left_wrench = result["xfrc_applied_L"].copy()
right_wrench = result["xfrc_applied_R"].copy()

left_wrench[:3] *= -1.0
right_wrench[:3] *= -1.0
```

The existing aerodynamic torque is left unchanged, because it already produces the correct reversal phase of passive feathering. Multiplying the whole wrench by `-1` would also flip the feathering torque and destroy the passive-rotation dynamics that were already correct. [Translator's note: this force flip was removed on 2026-10-28 together with the pitching-moment sign correction; see [quasi_steady_model.md](../derivations/quasi_steady_model.md).]

`F_world` in the diagnostics is inverted as well, so that the telemetry represents the force actually applied:

```python
self.diagnostics["F_world"] = -np.asarray(result["diagnostics"]["F_world"])
```

## Feathering mirror constraint

The local frames of the left and right wing meshes in the URDF are not strict mirror images. A check of the spatial poses showed that the wing surfaces are mirror images in world coordinates only when the feathering joint coordinates have the same sign.

A passive joint equality was therefore added to the model:

```text
q_left_feathering = q_right_feathering
```

This equality does not prescribe a feathering trajectory, and no feathering actuator was added. The angle is still determined by aerodynamic force, inertia, spring, damping and the joint limits.

## Mass and numerical parameters

The robot mass was redistributed as:

```text
base_link + link1                         = 20 g
link2 + left/right arm + left/right wing = 10 g
total                                     = 30 g
```

The mass of each wing was reduced from about `4.4 g` to `0.0626 g`, with the inertia scaled in proportion to the mass. Otherwise the FlapX-scale aerodynamic moments cannot overcome the wing's gravitational moment and rotational inertia, and the feathering stays pinned at one limit for long periods.

The simulation uses:

```text
timestep   = 5e-5 s
integrator = implicitfast
```

Passive feathering parameters:

```text
range        = ±0.785398 rad
stiffness    = 2e-7 N·m/rad
damping      = 1e-9 N·m·s/rad
frictionloss = 1e-8 N·m
armature     = 1e-10 kg·m²
```

## Lift validation after the fix

Fixed-base frequency sweep at amplitude `1.5 rad`:

```text
20 Hz: mean Fz = +0.0662 N, L/W = 0.225
30 Hz: mean Fz = +0.1486 N, L/W = 0.505
40 Hz: mean Fz = +0.2638 N, L/W = 0.896
50 Hz: mean Fz = +0.4107 N, L/W = 1.396
```

[Translator's note: these are historical measurements from the pre-2026-10-28 implementation; the current regression values are in `tests/test_lift_sweep.py`.]

It now holds that:

```text
frequency increases -> mean +Z lift increases
```

The weight of the 30 g robot is:

```text
0.03 * 9.81 = 0.2943 N
```

From the sweep results the hover frequency is estimated at about `42 Hz`.

## Subsequent state of the hovering controller

The sign of the aerodynamic force in the altitude direction has been fixed, and the floating-base model produces a mean lift close to the weight at about 42 Hz.

However, the attitude loop may still flip through large angles within 0.1--0.2 s. The remaining issues are:

- the periodic aerodynamic moment fluctuates strongly at 42 Hz;
- the attitude control uses instantaneous states directly, without low-pass filtering or averaging over a flap cycle;
- the plate-to-body moment gain changes with frequency;
- there is no independent yaw control channel yet;
- Euler angles become singular near large attitudes.

So the aerodynamic force sign bug is fixed, but fully stable hovering still needs cycle-averaged attitude estimation, gain scheduling based on the actual control effectiveness, and a more reliable quaternion attitude controller.

## Files involved

```text
flapsim/fluid/mujoco_adapter.py
flapsim/fluid/aero.py
flapsim/environment/environment.py
flapsim/actuator/SingleThrustFlapper/trajectory_controller.py
flapsim/controller/SingleThrustFlapper/hovering_controller.py
flapsim/robot/SingleThrustFlapper/urdf/SingleThrustFlapper.urdf
```

[Translator's note: paths as of the original. Current locations: `flapsim/fluid/mujoco_adapter.py` is now a thin compatibility wrapper around `flapsim/aero/base.py` (`AeroEngine`) and the `aero/quasi_steady` plugin in `flapsim/aero/stack.py`; `flapsim/fluid/aero.py` is unchanged; `flapsim/environment/environment.py` became `flapsim/core/simulation.py` (class `Simulation`); `flapsim/actuator/SingleThrustFlapper/trajectory_controller.py` became `flapsim/actuators/sinusoidal_stroke.py`; `flapsim/controller/SingleThrustFlapper/hovering_controller.py` became `flapsim/controllers/hover_pid.py`; the URDF moved to `flapsim/assets/robots/single_thrust_flapper/urdf/SingleThrustFlapper.urdf`.]
