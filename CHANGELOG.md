# Changelog

All notable changes to FlapSim are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Aerodynamics library: `AeroEngine` evaluating additive per-wing models
  with shared kinematics; models `quasi_steady`, `rotational_lift`,
  `added_mass`, `wake_capture`; modifier `ground_effect`; `aero/stack`
  plugin; `atmosphere/standard` (ISA density, wind, gusts, turbulence);
  `force/body_drag`; `aero/cycle_averaged` generalized to identified
  thrust-tilt and moment matrices with parameter files; `flapsim sweep`
  identification (frequency, gimbal and damping sweeps). `docs/aero_models.md`.
- Plugin system: `flapsim.core.Simulation` with the phases `pre_control`,
  `control`, `actuate`, `forces`, `mj_step`, `post_step`, `record`; a signal
  bus (`wings/left/alpha`, `cmd/stroke/frequency_hz`, ...); plugin base
  classes; a registry addressed by `kind/name` with entry-point discovery
  (`flapsim.plugins` group); `flapsim.sim/1` YAML/JSON simulation configs
  with per-plugin overrides.
- Built-in plugins: `actuator/sinusoidal_stroke`, `actuator/pd_position`
  (with joint roles), `aero/quasi_steady`, `aero/cycle_averaged`,
  `controller/hover_pid`, `recorder/memory` (npz/csv export).
- `flapsim.testing.check_plugin` conformance check for third-party plugins;
  `flapsim plugins` and `flapsim run <config.yaml> --describe` commands;
  `flat_plate_simulation()` helper.
- Example robot ships `sim.yaml` (fixed base) and `hover.yaml` (floating
  base) next to its manifest.
- Robot manifests (`robot.yaml`): wings with aero frames, stroke and
  feathering joints, planforms, mirror relations; model patches for joint
  dynamics, equalities, mass overrides and collision exclusion; `when: free`
  / `when: fixed` conditional patches.
- `flapsim.robot.load()` accepting built-in names, manifests, bare URDF/MJCF
  files and `MjSpec` objects; `builtin_robots()`; `RobotModel` with `Wing`
  objects.
- `flapsim.wings`: planform integrals from analytic profiles, chord polylines,
  planform polygons and STL mesh outlines; runtime `Wing` state.
- `flapsim.robot.flat_plate_flapper()` parametric two-wing robot built in
  Python (no CAD needed).
- `flapsim inspect` and `flapsim run` command line (`python -m flapsim.cli`).
- The quasi-steady adapter now handles any number of wings from the manifest.
- Per-wing `mass_model` (`spar_plate`, `thin_plate`, `explicit`) declared in
  wing coordinates and converted to each body frame, so mirrored wings share
  one description; hinge geometry (`pitch_axis_fraction`,
  `leading_edge_offset`) in the planform and the aerodynamic moment.
- Geometry self-checks at load time: warnings when a wing's chord direction
  or a joint sign looks reversed.
- `docs/derivations/quasi_steady_model.md`: frame conventions, equations,
  hinge geometry, the passive-feathering regime and the model correction.
- Apache-2.0 license, NOTICE, contributing guide, code of conduct, citation file.
- Test suite: aerodynamic unit tests, left/right symmetry test, fixed-base
  lift-sweep regression test, headless smoke tests.
- GitHub Actions CI (ruff + pytest on Linux, macOS, Windows; Python 3.10-3.13).
- `ROADMAP.md` describing the open-platform architecture and phases.

### Fixed
- Quasi-steady pitching moment: the earlier implementation multiplied the
  moment by `sign(alpha)`, which made it independent of the flow direction and
  moved the centre of pressure to the opposite side of the hinge at each
  stroke reversal; the MuJoCo adapter compensated with a translational force
  flip. The moment is now `-(x_le + (d_cp - x0) c) F_N` with the centre of
  pressure fixed on the plate, the flip is gone, and the wing frame is
  defined by `span` and `chord` so the plate side is explicit. See
  `docs/derivations/quasi_steady_model.md`. Regression values were
  re-measured.

### Changed
- `SimCore` is a thin wrapper that loads the robot's shipped setup through
  `Simulation.from_config`; its method names are unchanged.
- Example robot moved to `flapsim/assets/robots/single_thrust_flapper/` with
  its manifest and joint configuration; no robot names remain in package code.
- `SimCore` loads any robot that provides `extras.joints`; `PlateController`,
  `HoveringController` and `AveragedHoverAero` take joint and body names from
  configuration instead of hard-coding them.
- Documentation moved to `docs/` (Chinese originals under `docs/zh/`),
  notebooks and scripts to `examples/`.
- `pyproject.toml`: MuJoCo pinned to `>=3.3,<4`, project metadata, ruff config.

### Removed
- `flapsim.environment`, `flapsim.actuator`, `flapsim.controller` and the
  example's `joints.json`, replaced by `flapsim.core`, `flapsim.actuators`,
  `flapsim.controllers` and the `sim.yaml` / `hover.yaml` setups.
- Unused configuration bundle loader and JSON files superseded by the manifest.
- Reinforcement-learning training code and trained policies (`flapsim/rl`).
  They return as examples built on the plugin API in a later phase.
- Committed bytecode and MuJoCo log file.

## [0.1.0] - 2026-10-19

Initial single-robot simulator: URDF import through `MjSpec`, quasi-steady
blade-element aerodynamics, cycle-averaged hover model, PID hover controller.
