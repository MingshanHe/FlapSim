# FlapSim open-platform plan

**Branch:** `open-platform` (cut from `master` on 2026-10-28, all training code and data removed)
**Status:** Phases 0-7 complete; release 0.2.0 prepared (see section 9)

FlapSim becomes an open-source simulation platform for flapping-wing robots built
on MuJoCo. A user brings their own robot description, tells FlapSim which bodies
are wings, picks aerodynamic models and other components from a plugin registry,
and runs the simulation headless or in the viewer. Everything specific to one
robot lives in that robot's description package, not in `flapsim/`.

---

## 1. Goals and non-goals

### Goals

1. **Import any robot.** URDF (today), MJCF, a programmatic `mujoco.MjSpec`, or a
   parametric builder for users without CAD.
2. **Wings as first-class objects.** A robot manifest declares which bodies are
   wings, their aerodynamic frame, stroke and feathering joints, and planform
   geometry. FlapSim derives wing kinematics from the MuJoCo state and computes
   lift, drag and moments.
3. **Aerodynamics as plugins.** Quasi-steady blade-element (the current FlapX
   derived model), cycle-averaged, rotational lift, added mass, wake capture,
   body drag, ground effect, wind and gusts. Users can stack built-in models or
   ship their own in a separate pip package.
4. **Other components as plugins.** Actuators, sensors, controllers, recorders
   and viewer overlays use the same plugin API and the same step pipeline.
5. **Reproducible science.** Regression tests pin the validated numbers (the
   fixed-base lift sweep, FlapX parity). Every result a paper cites can be
   regenerated with one command.
6. **Usable by outsiders.** README, tutorials, API docs, CI, PyPI package,
   contribution guide, permissive license, citation file.

### Non-goals (for now)

- CFD or vortex-lattice fluid solvers. The plugin interface allows them later.
- GPU batch simulation (MJX). Measured earlier: per-step cost with the joint
  equality constraints was far worse than CPU MuJoCo. Keep CPU-first.
- Hardware-in-the-loop. Firmware lives outside this repo.

---

## 2. Baseline audit (what the branch contains today)

| Area | File(s) | State |
|---|---|---|
| Step loop | `flapsim/environment/environment.py` | Already a proto pipeline: supervisors, control pairs, force models, `mj_step`, `post_step`. Good seed for plugin phases. |
| URDF import | `environment.py` via `MjSpec.from_file` | Works, but patches robot-specific joints (`left_wing_feathering`), adds a mirror equality and overrides the floating-base mass to 30 g in code. |
| Scene | `flapsim/environment/scene.py` | Generic, reusable. |
| Aero (pure) | `flapsim/fluid/aero.py` | Quasi-steady blade-element model, pure functions, SI units, well documented in `tutorial/aero.md`. Hard-coded to exactly two wings (left/right). |
| Aero (MuJoCo) | `flapsim/fluid/mujoco_adapter.py` | Hard-codes body names `link2`, `left_wing`, `right_wing`, four joint names, the aero-frame rotation and the STL normal flip. |
| Cycle-averaged | `flapsim/fluid/averaged_hover.py` | Hard-codes identified constants for one robot at one amplitude. |
| Config | `flapsim/config/*.json`, `loader.py` | `SingleThrustRobot.json` already has an `aero_frames` section with bodies, axes and signs, but nothing reads it. |
| Actuators, controllers | `flapsim/actuator/SingleThrustFlapper`, `flapsim/controller/SingleThrustFlapper` | Reusable logic (PD position, sinusoidal trajectory, cascaded hover PID) in robot-named packages. |
| Entry point | `sim_core.py`, `flapsim.py` | `ROBOT_URDFS` dict with one entry; `fix=True/False` selects the aero model. |
| Robot assets | `flapsim/robot/SingleThrustFlapper` | URDF + 7 STL meshes, becomes the first example robot package. |
| Docs | `tutorial/`, `log/` | Three notebooks, a full aero derivation (zh-CN), two engineering notes. No README, LICENSE, CONTRIBUTING. |
| Tests | none | The `tests/` directory held only bytecode. |

Measured on this branch (fixed base, 5e-5 s timestep): 1000 steps in 0.12 s,
so one simulated second costs about 2.4 s of wall time. Treat this as the
performance floor; the refactor must not make it worse.

---

## 3. Target architecture

### 3.1 User-facing workflow

```python
import flapsim
from flapsim.aero import QuasiSteadyBladeElement, RotationalLift, AddedMass
from flapsim.actuators import SinusoidalStroke
from flapsim.sensors import IMU
from flapsim.io import Recorder

robot = flapsim.load_robot("my_robot/robot.yaml")  # URDF or MJCF + wing manifest
sim = flapsim.Simulation(robot, timestep=5e-5, base="fixed")

sim.add(QuasiSteadyBladeElement(wings="all") + RotationalLift() + AddedMass())
sim.add(
    SinusoidalStroke(
        joints=["left_stroke", "right_stroke"],
        frequency_hz=35.0,
        amplitude_rad=1.5,
        mirror=True,
    )
)
sim.add(IMU(body="torso", rate_hz=500.0, noise="default"))
sim.add(Recorder(["wings/*/force_world", "imu/*"], every=1e-3))

sim.run(duration=2.0, viewer=False)
table = sim["recorder"].to_numpy()
```

The same assembly can be written as YAML and run with
`flapsim run my_robot/sim.yaml --duration 2 --headless`.

### 3.2 Package layout

```
flapsim/
  core/        Simulation, step pipeline, phases, state views (BodyState, JointState)
  robot/       load URDF/MJCF/MjSpec, RobotManifest schema, patches, parametric builders
  wings/       Wing runtime object: frame, kinematics, planform integrals, CoP
  aero/        AeroModel base, AeroStack, built-in models, force application
  plugins/     Plugin base classes, registry, entry-point discovery, lifecycle
  actuators/   PD position, sinusoidal trajectory, DC motor + crank, servo
  sensors/     IMU, TOF, encoder, force/torque; noise and delay models
  controllers/ PID hover, waypoint tracker, user callback controller
  io/          Recorder (npz, csv, hdf5), telemetry, viewer overlays, video
  config/      schema, YAML/JSON loading, deep merge, validation errors
  cli.py       flapsim run | inspect | convert | sweep
flapsim/assets/robots/   example robot packages shipped with the wheel (single_thrust_flapper/)
examples/      runnable scripts and notebooks (hover, lift sweep, write a plugin)
docs/          mkdocs site: tutorials, concepts, API, derivations (moved from tutorial/, log/)
tests/         unit, regression (golden numbers), plugin-conformance suite
```

### 3.3 Core abstractions

**Simulation and step pipeline.** One `Simulation` owns `MjModel`, `MjData` and
an ordered list of plugins. Each step runs the phases
`pre_control -> control -> actuate -> forces -> mj_step -> post_step -> record`.
A plugin implements only the phases it needs. This is the existing
`SimulationEnvironment.step()` made explicit and extensible.

**RobotManifest.** A YAML or JSON file next to the robot model that holds what
URDF cannot express. Example for the current robot:

```yaml
schema: flapsim.robot/1
name: SingleThrustFlapper
model:
  file: urdf/SingleThrustFlapper.urdf          # .urdf, .xml (MJCF) or python:module:function
  root_body: base_link
  spawn: {position: [0.0, 0.0, 0.5]}
  patches:                                     # applied on the MjSpec before compile
    joints:
      left_wing_feathering:  {stiffness: 2.0e-7, armature: 1.0e-10, solref_limit: [4.0e-4, 1.0]}
      right_wing_feathering: {stiffness: 2.0e-7, armature: 1.0e-10, solref_limit: [4.0e-4, 1.0]}
    equalities:
      - {type: joint, joints: [left_wing_feathering, right_wing_feathering], polycoef: [0, 1, 0, 0, 0]}
    exclude_self_collision: true
torso: {body: link2}
wings:
  - name: left
    body: left_wing
    side: left
    frame: {normal: [1, 0, 0], span: [0, 0, -1], chord: [0, 1, 0]}   # aero axes in body frame
    stroke_joint:     {name: left_flapping_joint,  sign: +1}
    feathering_joint: {name: left_wing_feathering, sign: -1}
    planform: {length: 0.07, mean_chord: 0.02, profile: rectangular}
    force_sign: -1                              # mesh normal is opposite to the aero convention
  - name: right
    mirror_of: left                             # frame calibrated at the neutral pose, as today
    body: right_wing
    stroke_joint:     {name: right_wing_flapping,   sign: -1}
    feathering_joint: {name: right_wing_feathering, sign: +1}
```

Every hard-coded fact in `environment.py` and `mujoco_adapter.py` moves into
this file. The `aero_frames` block already in `SingleThrustRobot.json` is the
starting point.

**Wing.** Built from one manifest entry. Resolves MuJoCo ids once, exposes
`psi, psi_dot, theta, theta_dot`, the world rotation of the aero frame, the
body origin and centre of mass, and the planform integrals
(`r00..r33`, `z_cp0..z_cp2`, `z_rd`). Integrals come from one of three inputs:
the raw numbers (today's `Wing.json`), a chord polyline `c(r)` (what tutorial 2
computes), or a mesh outline (stretch goal).

**AeroModel plugin.** Pure computation, no MuJoCo calls:

```python
class AeroModel(ForcePlugin):
    def compute(self, wing: WingState, body: BodyState, air: Atmosphere) -> WingLoads:
        """Return force and moment in the wing aero frame plus the centre of pressure."""
```

`AeroStack` sums the loads of several models for each wing. The core transforms
the total to the world frame and applies it at the centre of pressure with
`mj_applyFT` (today the force is applied at the body origin with a torque
correction). `Atmosphere` is itself a plugin: density, wind vector, gust model,
ground-effect factor as a function of height.

**Component plugins.** `ActuatorPlugin` (writes `qfrc_applied`, `ctrl`, or sets
kinematic joint state), `SensorPlugin` (reads state, applies noise and delay,
publishes a named signal), `ControllerPlugin` (reads signals, writes commands),
`RecorderPlugin` (subscribes to signals). Signals are addressed by path such as
`wings/left/alpha` or `imu/gyro`, so plugins compose without importing each other.

**Registry and discovery.** `flapsim.plugins.register(kind, name, cls)` for
in-tree classes, plus `importlib.metadata` entry points in the group
`flapsim.plugins` so a third-party `pip install flapsim-myaero` is found
automatically. Config files refer to plugins by `kind/name`.

**Identification tools.** The cycle-averaged hover model is reframed as a tool:
`flapsim sweep` runs a fixed-base frequency and amplitude sweep with any aero
stack, fits lift and control-moment coefficients, and writes a
`CycleAveragedAero` parameter file for that robot. No constants in code.

---

## 4. Phases

Each phase ends with a tagged commit and green CI. Acceptance criteria are
written so they can be checked mechanically.

### Phase 0: clean baseline (done 2026-10-28)

- Branch `open-platform` cut from `master`.
- Removed: `flapsim/rl` (PPO code and trained models), `MUJOCO_LOG.TXT`,
  tracked bytecode, untracked run logs, checkpoints and videos.
- `.gitignore` now blocks `runs/`, `artifacts/`, `checkpoints/`, model zips,
  videos and MuJoCo logs. `pyproject.toml` lost the dangling `rl` extra and
  gained a `dev` extra.

### Phase 1: foundations and safety net (done 2026-10-28)

Goal: lock current behaviour in tests before touching any physics code.

- `README.md` (English, with a zh-CN section link), `LICENSE`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, `CHANGELOG.md`.
- Move `log/*.md` to `docs/notes/`, `tutorial/aero*.md` to `docs/derivations/`,
  notebooks to `examples/notebooks/`.
- `tests/`: unit tests for `normal_force_coefficient`, `center_of_pressure_fraction`,
  `wing_aero`, `world_wrench`; left/right mirror symmetry; a regression test
  that reproduces the fixed-base lift sweep from `docs/notes/aero_force_sign_fix.md`
  (20, 30, 40, 50 Hz) within 1 percent; a headless smoke run for both base modes.
- GitHub Actions: pytest on Ubuntu, macOS and Windows for Python 3.10 to 3.13;
  `ruff check` and `ruff format --check`.
- Pin `mujoco>=3.3,<4` (MjSpec API) and declare `numpy>=1.24`.

Acceptance: CI green on all nine matrix cells; the lift-sweep regression test
fails if any force sign in the adapter is flipped.

### Phase 2: robot description and import (done 2026-10-28)

Goal: zero robot-specific names inside `flapsim/`.

- `RobotManifest` schema (dataclasses plus JSON Schema export) and loader for
  YAML and JSON.
- `flapsim.robot.load()` accepting a manifest, a bare URDF or MJCF path, or an
  `MjSpec`. Bare files load with no wings and a warning.
- Manifest patches: joint dynamics, equalities, collision exclusions, free or
  fixed base, spawn pose. Replaces the inline code in `environment.py`.
- `Wing` runtime object; `mirror_of` calibration at the neutral pose (the
  current `aero_from_body_right` logic).
- Move `flapsim/robot/SingleThrustFlapper` to
  `flapsim/assets/robots/single_thrust_flapper/` with `robot.yaml`, meshes and
  URDF, packaged as data so `pip install flapsim` ships the example.
- `flapsim inspect <model>`: prints bodies, joints, axes and limits so a user
  can write a manifest in minutes.
- `flapsim.robot.builders.FlatPlateFlapper(span, chord, ...)`: a parametric
  two-wing robot generated in Python, for users without CAD and for tests.
- Add a xacro preprocessing hook as an optional extra. SDF stays out of scope.

Acceptance: `grep -r "left_wing\|link2\|right_wing_flapping" flapsim/ --include=*.py`
returns nothing outside `flapsim/robot/builders.py` (which defines its own
robot); the Phase 1 regression tests pass unchanged through the
manifest path; the builder robot loads, flaps and produces positive mean lift.
Result: all three hold. Porting to the manifest first reproduced the Phase 1
lift-sweep values to the last digit, which validated the frame plumbing. The
parametric builder robot then exposed a physics defect inherited from the
original code: the pitching moment carried a `sign(alpha)` factor that made
it independent of flow direction, so the centre of pressure switched sides of
the hinge at each reversal and a force-only flip in the adapter masked the
lift sign. The moment was restored to `-d_cp c F_N`, the flip removed, wing
frames redefined by `span` and `chord`, and a per-wing mass model added
(the URDF's uniform-solid centroid 22 mm behind the hinge gives anti-phase
feathering and negative lift; a spar-plate distribution flies). The
regression values were re-measured: 0.074, 0.167, 0.297, 0.464 N at 20, 30,
40, 50 Hz, a clean frequency-squared law with hover near 40 Hz for the 30 g
robot. Details in `docs/derivations/quasi_steady_model.md`.

### Phase 3: plugin system (done 2026-10-28)

Goal: the existing simulation is assembled from plugins and config, not from
`SimCore.__init__`.

- `Plugin` base class with lifecycle (`bind(sim)`, `reset`, phase hooks,
  `signals`), `Simulation` with ordered phases.
- Registry, entry-point discovery, `kind/name` resolution from config.
- Port `PositionController`, `TrajectoryController`, `PlateController`,
  `ThrustController`, `HoveringController`, `MuJoCoAeroAdapter` and
  `AveragedHoverAero` to plugins. Keep thin compatibility shims in `sim_core.py`
  and `flapsim.py` for one release, with deprecation warnings.
- `flapsim run sim.yaml` builds everything from a file.
- Plugin-conformance test suite that third-party authors can run against their
  own plugin (`pytest --pyargs flapsim.testing.conformance`).

Acceptance: `python flapsim.py --robot SingleThrustFlapper --headless --duration 1`
still works and produces the same lift-sweep numbers; a dummy plugin installed
from a separate directory is discovered by entry point and runs.
Result: both hold. `SimCore` now builds the shipped `sim.yaml` / `hover.yaml`
through `Simulation.from_config`; the regression values are unchanged.
Entry-point discovery is covered by a test with a fake distribution entry
point. The old `flapsim.environment`, `flapsim.actuator` and
`flapsim.controller` packages were removed rather than shimmed: the branch is
pre-release and `SimCore` plus `flapsim.fluid` remain as the compatibility
surface.

### Phase 4: aerodynamics library (done 2026-10-28)

Goal: a documented, tested set of aero plugins that work for any number of
wings and any frame.

- `QuasiSteadyBladeElement`: generalise `aero.py` to N wings, arbitrary frame
  and stroke-plane definitions from the manifest, forces applied at the CoP.
- `CycleAveragedAero`: parameter file produced by `flapsim sweep`; no built-in
  constants; mass handling declared in the manifest instead of overriding to 30 g.
- New plugins, each with a unit test against an analytic or published case:
  `RotationalLift` (Sane and Dickinson rotational circulation),
  `AddedMass`, `WakeCapture` (empirical), `BodyDrag` (per-body drag
  coefficients), `GroundEffect`, `Atmosphere` with steady wind and gust models.
- `WingPlanform` utilities: integrals from a chord polyline, from a mesh
  outline, and the reverse (export the planform used).
- Diagnostics as signals: `alpha`, `C_N`, `F_N`, CoP, per-model contributions,
  so users can plot which effect produced which force.
- Performance guard: benchmark test records steps per second; fail if slower
  than the Phase 0 floor by more than 20 percent. Optional `numba` path if needed.

Acceptance: FlapX parity test still passes for the two-wing robot; a four-wing
builder robot runs with the same plugin; each effect toggles independently.
Result: the lift regression holds through the engine (single-model stack equals
the legacy adapter to machine precision); models are additive and toggle
independently; `flapsim sweep` identifies the cycle-averaged parameters and
`aero/cycle_averaged` loads them. The four-wing builder is deferred to the
examples in Phase 7. See `docs/aero_models.md`.

### Phase 5: component plugins (done 2026-10-28)

- Actuators: PD position, sinusoidal trajectory, `DCMotorCrank` (motor model
  plus crank or gear kinematics, from `dc_motor.json`), servo with rate limit.
- Sensors: IMU, TOF range, joint encoder, body force and torque; noise, bias
  and delay models that reproduce `rl_imu_hover/sensors.py` from the `ppo`
  branch.
- Controllers: `HoverPID` (generalised from `HoveringController`), waypoint
  tracker, `CallbackController` for user Python functions.
- IO: `Recorder` (npz, csv, hdf5), viewer overlays drawing force arrows and CoP
  markers per wing, offscreen video rendering.

Acceptance: an `examples/hover_pid.py` holds altitude with the example robot;
`examples/sensors.py` logs IMU and TOF to a file and plots them.
Result: both scripts run in the test suite. The DC motor model loads itself
from the aero plugin's wing torques, so stroke frequency responds to voltage
and aerodynamic load; the load cell reads MuJoCo's constraint wrench on the
root body. Video export renders offscreen (OSMesa/EGL) to GIF, PNG frames or
MP4 when imageio is installed.

### Phase 6: learning and optimisation integration (done 2026-10-28)

- `flapsim[rl]`: a generic Gymnasium wrapper over `Simulation` with
  observation and action spaces assembled from named signals and commands.
- Recreate the `ppo` branch's IMU plus TOF hover task as `examples/rl/` using
  only public plugin APIs. The old code stays in git history.
- `flapsim sweep` and a `ParameterStudy` helper for design optimisation
  (wing length, frequency, amplitude) with multiprocessing.

Acceptance: the example PPO task trains for a smoke run in CI without touching
`flapsim/core`.
Result: `flapsim.envs.SimulationEnv` (Gymnasium) with signal observations and
command actions, `hover_reward`/`hover_terminated`, `flapsim.study` for
parallel parameter studies; `examples/rl/` rebuilds the IMU + TOF hover task
and trains PPO; a dedicated CI job installs the `rl` extra and runs the smoke
training.

### Phase 7: documentation, examples, release (done 2026-10-28)

- MkDocs Material site with mkdocstrings: Getting started, Import your robot,
  Define wings, Choose aerodynamics, Write a plugin, Run RL, Derivations
  (translated from `tutorial/aero.md`), API reference.
- Example gallery with rendered videos.
- `v0.2.0` on PyPI (`flapsim` is unclaimed as of 2026-10-28), GitHub release
  notes, Zenodo DOI, issue and PR templates.
- Optional: JOSS paper draft once two external users have imported a robot.

Result: MkDocs Material site (`mkdocs.yml`, `docs/`) with getting started,
manifest tutorial, aero models, components, plugin guide, CLI, learning,
gallery (rendered GIFs), derivations (English conventions page plus
translations of the Chinese originals), engineering notes and an
mkdocstrings API reference; `docs.yml` builds it on every push and deploys
to GitHub Pages from `master`; `release.yml` builds and publishes on `v*`
tags. Version 0.2.0 set in `pyproject.toml`, `CITATION.cff` and the package.
The JOSS paper is left for later.

---

## 5. Validation strategy

- **Golden numbers.** Fixed-base lift sweep at 1.5 rad amplitude: 20, 30, 40,
  50 Hz mean lift, plus the FlapX phase-resolved comparison. Any refactor must
  reproduce them.
- **Symmetry.** Mirrored wings produce mirrored forces at every phase.
- **Analytic limits.** Flat plate in steady flow matches `C_N(alpha)`; pure
  feathering rotation produces opposing damping moment; zero velocity produces
  zero force.
- **Conservation checks.** Fixed-base reaction force equals the sum of wing
  forces plus gravity, read back from MuJoCo.
- **Cross-robot.** The same aero plugin passes on the CAD robot and on the
  parametric builder robot.
- **Plugin conformance.** Third-party plugins run the shared conformance suite.

---

## 6. Decisions needed from the maintainer

1. **License.** Recommended: Apache-2.0 (patent grant, common in robotics) or
   MIT. Must be compatible with FlapX, whose quasi-steady model and empirical
   `C_N` and `d_cp` fits this code reproduces. Verify the FlapX license and add
   attribution in `README.md` and `aero.py` before the first public release.
2. **Documentation language.** Proposed: English primary, zh-CN translations
   of tutorials and derivations kept in `docs/zh/`.
3. **Package and repository names.** Keep `flapsim` for both.
4. **RL code.** Return as `examples/rl/` inside this repo (proposed) or as a
   separate `flapsim-rl` repository.
5. **Support matrix.** Proposed: Linux, macOS, Windows; Python 3.10 to 3.13;
   MuJoCo 3.3 and newer.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Coordinate and sign conventions regress while generalising (the fixes in `docs/notes/aero_force_sign_fix.md` were hard won). | Phase 1 regression tests before any refactor; manifest fields `sign` and `force_sign` make conventions explicit and testable. |
| Per-step Python aero becomes the bottleneck with more wings and more models. | Benchmark gate in CI; vectorise across wings with NumPy; optional numba. |
| MjSpec API changes between MuJoCo releases. | Pin `mujoco>=3.3,<4`; CI runs the newest and the oldest supported version. |
| Scope creep in aerodynamic models. | Ship quasi-steady and cycle-averaged first; everything else is a plugin with its own test. |
| Users cannot write a manifest for their URDF. | `flapsim inspect`, a documented worked example, and a validator with precise error messages. |

---

## 8. Decisions taken

- 2026-10-28: Apache-2.0 license; English documentation with the Chinese
  originals kept under `docs/zh/`; execute all phases in order.
- Example robots live inside the package (`flapsim/assets/robots/`) so that a
  plain `pip install flapsim` ships them; the parametric builder robots live in
  `flapsim.robot.builders`.


## 9. Release checklist for 0.2.0 (maintainer actions)

1. Merge `open-platform` into `master` (the CI, docs and release workflows
   watch `master`).
2. Enable GitHub Pages for the repository (source: the `gh-pages` branch
   that `mkdocs gh-deploy` creates on the first push to `master`).
3. On pypi.org, register a *trusted publisher* for `MingshanHe/FlapSim`,
   workflow `release.yml`, environment `pypi`, and create the `pypi`
   environment in the repository settings. The project name `flapsim` was
   unclaimed on 2026-10-28.
4. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`. The workflow
   builds the wheel and sdist, publishes to PyPI and creates the GitHub
   release with generated notes.
5. Optional: link the repository to Zenodo before tagging so the release
   receives a DOI, then add the DOI badge and `doi` field to `CITATION.cff`.
6. Announce; the first external robot imports will show where the manifest
   tutorial needs more detail.
