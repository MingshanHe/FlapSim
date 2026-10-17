# Airborne PID experiment

This round targets sustained height and roll/pitch stability for the unchanged
7 g upper / 8 g lower robot. Yaw and horizontal position are uncontrolled. It
does not train NEAT or PPO, and its changed initialization and control frequency
make it a new experiment rather than an optimizer-only comparison.

The controller is a two-axis attitude PID plus a height PID. The attitude
frontend filters base-link Euler angles at every physics step; the derivative
term uses a further filtered finite difference. Height and vertical velocity
come from the existing causal IMU inertial integration unless explicitly set
to the `truth` diagnostic. That diagnostic reads both true height and vertical
velocity; it is not a validated ToF sensor implementation.

The final release starts the whole mechanism at rest, at approximately 0.8 m,
with alpha/beta at their calibrated neutral positions. Powered phase-bank joint
positions are retained for the other joints, all velocities start at zero, and
motor speed ramps over 100 ms. These state assignments happen only at reset.
There is no flight-time pose reset, artificial lift, external stabilizing torque,
changed gravity, or modified mass. This is an airborne release experiment, not
ground takeoff.

Physics runs at 5 microseconds; control at 2 ms. The selected policy also tunes
the existing simulated gimbal position servos; their 0.02 Nm torque cap remains
unchanged. Physical angle targets stay within +/-0.3 rad. After the startup ramp,
motor target limits remain 3115.224079–3807.496097 rad/s (the original +/-10%
range). The height loop's feedforward motor target is independently retuned.

The IMU model retains the prior round's optimistic high-rate orientation and
specific-force sampling. Sensor bandwidth, saturation and long-term drift on
real hardware have not been validated. Collision masks are disabled in the CAD
source; the audit instead checks a conservative geometric ground-clearance bound
over every robot geometry at 500 Hz. See the report for the resulting scope.

## Artifacts and replay

The frozen policy, protocol, all logged searches, held-out tests and report are
in `artifacts/pid_airborne_20261008/`. The test source archive is independent of
the old NEAT/PPO records. Previous experiments are not overwritten.

From the repository root, using the environment with MuJoCo, neat-python, NumPy,
SciPy, Matplotlib, Pillow, imageio and imageio-ffmpeg installed:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m experiments.algorithms.pid_airborne.results \
  --root artifacts/pid_airborne_20261008 \
  --video /tmp/pid-hover-replay.mp4
```

The replay temporarily relocates mesh paths to the current checkout and checks
every recorded flight/audit trace value against the frozen first nominal case
(seed 310900000; tolerance 1e-12). The camera follows horizontal motion only;
ground height remains fixed, and horizontal drift is displayed. Playback is 1x;
25 fps undersamples the wings' roughly 25 Hz motion. Use the recorded telemetry
to study fast wing motion rather than inferring frequency from the video.

```sh
python -m pytest -q tests/test_pid_airborne.py
```

## Repeating the search

Use a fresh output directory, then run `search --stage screen`, followed by
`search --stage evolve` for at least four completed generations. The exploratory
run here was stopped after seven logged generations when the alternative motor
startup worked. `initialization`, `startup_probe`, and `height_search` record the
follow-up tests. `height_search` deliberately uses the saved generation-3 gains;
it does not silently consume an arbitrary later best candidate.

Run `audit --stage register` before held-out evaluation, `audit --stage validate`
to select among the three height candidates, then `audit --stage test` once the
policy is frozen. These commands write result files: use a new output directory
for repetitions, preserving the published data. The test seeds are separate from
all logged development and validation seeds. The exact executed search parameters
and early-stop accounting are documented in the report.
