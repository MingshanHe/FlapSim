# Changelog

All notable changes to FlapSim are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Apache-2.0 license, NOTICE, contributing guide, code of conduct, citation file.
- Test suite: aerodynamic unit tests, left/right symmetry test, fixed-base
  lift-sweep regression test, headless smoke tests.
- GitHub Actions CI (ruff + pytest on Linux, macOS, Windows; Python 3.10-3.13).
- `ROADMAP.md` describing the open-platform architecture and phases.

### Changed
- Documentation moved to `docs/` (Chinese originals under `docs/zh/`),
  notebooks and scripts to `examples/`.
- `pyproject.toml`: MuJoCo pinned to `>=3.3,<4`, project metadata, ruff config.

### Removed
- Reinforcement-learning training code and trained policies (`flapsim/rl`).
  They return as examples built on the plugin API in a later phase.
- Committed bytecode and MuJoCo log file.

## [0.1.0] - 2026-10-19

Initial single-robot simulator: URDF import through `MjSpec`, quasi-steady
blade-element aerodynamics, cycle-averaged hover model, PID hover controller.
