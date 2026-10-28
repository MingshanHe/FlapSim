# Record of the PID hover and waypoint control implementation

*Translated from the Chinese original (`docs/zh/notes/pid_hover_control.md`). It documents the implementation as it stood before 2026-10-28; the pitching-moment sign and the removed force flip are described in [quasi_steady_model.md](../derivations/quasi_steady_model.md).*

## Goal

Implement free-base hover for the `SingleThrustFlapper`, whose mass is about `30 g`:

- take off from `z = 0.5 m` and hold the target altitude;
- stabilise the body roll/pitch;
- control the horizontal position `x/y`;
- switch to a new waypoint every `3 s`;
- keep the visible motion of the stroke and plate joints.

How to run:

```bash
python flapsim.py --robot SingleThrustFlapper --fix false
```

`--fix true` still uses the fixed base and the detailed per-step aerodynamics, without hover control.

## Why the per-step aerodynamics were not used directly in free flight

The fixed-base sweep confirmed that the mean lift around `42 Hz` can balance the weight, but with a free base the per-step model produces very large periodic aerodynamic moments. At the same time, the stroke trajectory overwrites the joint `qpos/qvel` directly, which injects energy into the lightweight multi-body system. Combined, this led to:

- the body flipping over within a very short time;
- `QACC` diverging numerically;
- the attitude leaving the linear control range before the PID had time to respond.

The floating mode therefore uses the `AveragedHoverAero` cycle-averaged model (now `flapsim/aero/cycle_averaged.py`). The fixed mode keeps the detailed `MuJoCoAeroAdapter` for checking the single-wing aerodynamics and passive feathering.

## Cycle-averaged aerodynamic model

Fixed-base measurement:

```text
30 Hz -> mean Fz = 0.14858349 N
```

The lift is fitted with a quadratic relation:

```text
L = k_f * frequency^2
k_f = 0.14858349 / 30^2
```

The weight of the 30 g robot is:

```text
W = 0.03 * 9.81 = 0.2943 N
```

so the theoretical hover frequency is about:

```text
frequency_hover = sqrt(W / k_f) ~= 42.2 Hz
```

The plate command also changes the thrust direction:

```text
direction_body = normalize([-plate_roll, -plate_pitch, 1])
```

The fixed-base identification gives the approximate plate-to-body-moment gains:

```text
roll moment  / plate_pitch = 0.01055 / 20 N*m/rad
pitch moment / plate_roll  = 0.01144 / 20 N*m/rad
```

The gains scale with the square of frequency:

```text
scale = (frequency / 42)^2
```

To represent the cycle-averaged drag of a real flapping wing, and to suppress the undamped horizontal oscillation in free flight, the following were also added:

```text
angular damping    = [2e-4, 2e-4, 1e-4]
horizontal damping = 0.06 N/(m/s)
```

## Controller structure

The controller is a three-level cascade:

```text
waypoint (x, y, z)
        |
        +-- z altitude PID -----> flapping frequency ------> lift
        |
        +-- x/y position PD ----> desired body tilt
                                      |
                                      +-- Roll/Pitch PD --> plate angles --> moments
```

The state is not the instantaneous value of a single simulation step; it is accumulated and averaged over one flapping cycle. The control update period is:

```text
control_period = 1 / current_frequency
```

This prevents the controller from tracking the 42 Hz periodic oscillation.

## Altitude PID

Altitude error:

```text
e_z = target_z - z
```

Control law:

```text
frequency = 42
          + Kp_z * e_z
          - Kd_z * velocity_z
          + Ki_z * integral(e_z)
```

Current parameters:

```text
Kp_z = 8.0
Kd_z = 4.0
Ki_z = 2.0
integral limit = +/-0.5
frequency limit = [20, 55] Hz
```

The integral term removes the altitude steady-state error caused by the aerodynamic fit error, and the integral limit prevents windup.

## Horizontal position outer loop

The x/y position control does not produce a horizontal force directly; it produces a desired body tilt:

```text
desired_pitch_tilt = Kp_xy * (target_x - x) - Kd_xy * velocity_x
desired_roll_tilt  = -Kp_xy * (target_y - y) + Kd_xy * velocity_y
```

Current parameters:

```text
Kp_xy = 0.10
Kd_xy = 0.50
tilt limit = +/-0.10 rad
```

The sign for roll differs from that for pitch because, for small angles, the horizontal components of the body z axis in world coordinates satisfy:

```text
acceleration_x proportional to +pitch_tilt
acceleration_y proportional to -roll_tilt
```

## Attitude inner loop

The attitude control does not use Euler roll/pitch directly as feedback; it uses the tilt of the body z axis relative to the world z axis:

```text
roll_tilt  = -body_z_world[y]
pitch_tilt =  body_z_world[x]
```

This suits small-angle hover better than Euler angles directly, and avoids some rotation-order problems.

Current attitude PD parameters:

```text
Kp_attitude = 0.80
Kd_attitude = 0.06
plate limit = +/-0.10 rad
```

An early version used `Kp=0.04`, `Kd=0.003` and unscaled moment gains. The body could be stabilised, but the plate deflected by at most about `0.229 deg`, so the thrust module looked permanently vertical in the viewer. The joint command is now amplified 20 times and the plate-to-moment gain divided by 20, so the magnitude of the control moment is essentially unchanged while the visible maximum plate deflection is about `4.62 deg`. The plate deflects noticeably at waypoint switches and returns close to vertical as the target is approached, which is the normal behaviour of a converged position PID.

In the URDF, the names and the physical axes of the two plate joints are crossed:

```text
pitch_joint -> about +X -> mainly controls body Roll
roll_joint  -> about -Y -> mainly controls body Pitch
```

The control law is therefore:

```text
plate_pitch = -Kp * (roll_tilt  - desired_roll_tilt)  - Kd * roll_rate
plate_roll  =  Kp * (pitch_tilt - desired_pitch_tilt) + Kd * pitch_rate
```

## Waypoint update

Default waypoints:

```text
0: (0.00, 0.00, 0.50)
1: (1.00, 0.00, 0.50)
2: (1.00, 1.00, 0.80)
3: (0.00, 1.00, 0.80)
```

They switch every `3 s`, cycling back to the first point after the last one. They can also be set from Python:

```python
simulation.set_hover_waypoints(
    [(0.0, 0.0, 0.5), (0.2, 0.0, 0.6)],
    interval=3.0,
)
```

## Why the plate appearance once did not change

Initially, to rule out numerical divergence in free flight, the floating mode disconnected the physical `PositionController`. The result was:

- the plate commands in the hovering controller changed normally;
- the averaged aerodynamics also used these commands;
- but the visible `pitch_joint/roll_joint` in MuJoCo did not follow the commands;
- the joints sagged under gravity and appeared to stay at a fixed angle.

The floating mode now uses a kinematic position mode: at every step the plate `qpos` is synchronised to the control command and the corresponding `qvel` is set to 0, but the PD torque of up to `0.5 N*m` is not applied. The display is thus consistent with the control quantities, while no energy is injected again into the lightweight internal joints.

Normal start-up has no artificial initial attitude disturbance. The previously used

```text
initial body Roll  = +10 deg
initial body Pitch = -8 deg
```

exists only in a standalone validation script; it is a perturbation of the `floating_base` quaternion, not an initial plate joint angle.

## Validation results

Running 20 s from a level initial state:

```text
position ~= (0.0000, 0.0000, 0.4999) m
body Roll/Pitch ~= 0 deg
frequency ~= 42.22 Hz
```

Running 20 s from body roll `+10 deg`, pitch `-8 deg`:

```text
position = (-0.0061, -0.0098, 0.4999) m
body Roll = -0.02 deg
body Pitch = +0.01 deg
frequency = 42.22 Hz
```

In the waypoint switching test, the target updated correctly at `3/6/9/12 s`, and the altitude and attitude remained stable.

To make the attitude control easier to observe in the viewer, the default horizontal waypoint spacing was later enlarged from `0.10 m` to `1.00 m`, and the altitude change from `0.05 m` to `0.30 m`. In one 15 s closed-loop test:

```text
maximum |body Roll|  ~= 2.1 deg
maximum |body Pitch| ~= 2.1 deg
maximum |plate pitch_joint| = 4.62 deg
maximum |plate roll_joint|  = 4.62 deg
```

The tilt command is still limited to `+/-0.10 rad` (about `+/-5.7 deg`), so with the enlarged path the tilt is clearly visible, while no dangerous attitude is demanded because the waypoints are far apart.

## Current model boundaries

- floating hover uses the cycle-averaged aerodynamic model, not the per-step wing-surface aerodynamics;
- there is no independent yaw actuator yet; yaw relies mainly on damping and cannot track a commanded yaw;
- the plate commands are usually only about `1e-4` to `1e-3 rad`, so the visual change is small, but the joint state is consistent with the command;
- if the detailed instantaneous aerodynamics are restored later, the periodic flapping moments need system identification, and energy injection into the free multi-body system by directly overwriting `qpos/qvel` must be avoided.

## Files involved

```text
flapsim/controller/SingleThrustFlapper/hovering_controller.py
flapsim/fluid/averaged_hover.py
flapsim/actuator/SingleThrustFlapper/position_controller.py (now flapsim/actuators/pd_position.py)
flapsim/actuator/SingleThrustFlapper/trajectory_controller.py
flapsim/environment/environment.py
flapsim/sim_core.py
flapsim.py
```

[Translator's note: paths as of the original. Current locations: `flapsim/controller/SingleThrustFlapper/hovering_controller.py` became `flapsim/controllers/hover_pid.py`; `flapsim/fluid/averaged_hover.py` became `flapsim/aero/cycle_averaged.py`; `flapsim/actuator/SingleThrustFlapper/trajectory_controller.py` became `flapsim/actuators/sinusoidal_stroke.py`; `flapsim/environment/environment.py` became `flapsim/core/simulation.py` (class `Simulation`).]
