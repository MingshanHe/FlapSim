# Component plugins

All components are plugins (see `flapsim plugins`). They exchange named
signals: a sensor publishes `imu/gyro`, a controller reads it and publishes
`cmd/stroke/frequency_hz`, the stroke actuator consumes the command. Plugins
run in insertion order within each phase (`pre_control` for sensors,
`control` for controllers, `actuate` for actuators, `forces` for aero).

## Actuators

| Plugin | What it does | Commands | Signals |
|---|---|---|---|
| `actuator/sinusoidal_stroke` | Kinematic `q = d A sin(2 pi f t)` on stroke joints (`kinematic` writes velocities, `position_only` does not) | `frequency_hz`, `amplitude_rad` | `frequency_hz`, `amplitude_rad`, `phase` |
| `actuator/pd_position` | Gravity-compensated PD with torque limit, or kinematic hold; joints carry a `role` label | `<joint>` target | `<joint>` |
| `actuator/servo` | `pd_position` with a rate-limited, first-order lagged target | `<joint>` desired | `<joint>`, `<joint>/desired` |
| `actuator/dc_motor_crank` | DC motor (`i = (V - k_e w)/R`, `tau = k_t i`), gearbox `N`, crank `q = d A sin(theta_c)`; loaded by the aero torque about the stroke joints | `voltage_v` | `speed_rad_s`, `frequency_hz`, `current_a`, `torque_load`, `power_w`, ... |

## Sensors

Every sensor samples at `rate_hz` through a `Conditioner`: first-order
low-pass `tau`, pure delay `delay_s`, Gaussian `noise_std`, a bias drawn per
episode from `bias_range`, and quantisation `resolution`.

| Plugin | Measures | Defaults |
|---|---|---|
| `sensor/imu` | `euler` (low-passed attitude), `gyro` (body rates), `accel` (specific force `R^T (a - g)`), `quat` | 500 Hz, 20 ms attitude lag, 0.02 deg / 0.002 rad/s / 0.02 m/s^2 noise, 0.01 m/s^2 bias |
| `sensor/rangefinder` | slant `range` along a body axis to the ground plane or by ray cast, `range_rate` | 500 Hz, 20 ms delay, 5 mm noise, 4 m range |
| `sensor/encoder` | joint `position` (quantised by `counts_per_rev`) and `velocity` | continuous |
| `sensor/wrench` | interaction wrench with the parent body (`cfrc_int`): a load cell under a fixed-base robot | averaging `average_s` |

## Controllers

| Plugin | Role |
|---|---|
| `controller/hover_pid` | Period-averaged altitude (frequency) and attitude (gimbal) loops with waypoints |
| `controller/schedule` | Time tables of set points (`[[t, value], ...]`), stepped or interpolated |
| `controller/callback` | Any Python function `callback(sim, controller) -> {command: value}` at a rate |

## Recording and visualisation

* `recorder/memory`: samples glob-matched signals every `every` seconds;
  `to_dict`, `to_npz`, `to_csv`, `to_hdf5`.
* `flapsim.io.ForceArrows`: viewer or renderer overlay with each wing's force
  arrow at its centre of pressure (`flapsim run ... --overlay forces`).
* `flapsim.io.record_video` / `flapsim record`: offscreen rendering to GIF,
  PNG frames or MP4 (`MUJOCO_GL=osmesa` or `egl` on headless machines).
