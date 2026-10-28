# Contributing to FlapSim

Thank you for helping build an open simulation platform for flapping-wing
robots. This page explains how to set up a development environment, what we
expect from a change, and how the project is organised.

## Development setup

```bash
git clone https://github.com/MingshanHe/FlapSim.git
cd FlapSim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Making a change

1. Open an issue first for anything larger than a bug fix, so the design can
   be discussed before you write code. The [roadmap](ROADMAP.md) lists what is
   planned and in which order.
2. Branch from `open-platform` (or `master` once the platform work is merged).
3. Keep pull requests focused. One behaviour change per pull request.
4. Add or update tests. Physics changes must keep
   `tests/test_lift_sweep.py` passing, or update its reference values and
   explain the physical reason in the pull request description.
5. Run `ruff check .`, `ruff format .` and `pytest -q` before pushing. CI runs
   the same commands on Linux, macOS and Windows for Python 3.10 to 3.13.
6. Update documentation when behaviour or public API changes.

## Code conventions

- Python 3.10+, type hints on public functions, docstrings in plain English.
- SI units everywhere. Angles in radians unless the name ends in `_deg`.
- MuJoCo world-frame wrenches are ordered `[Fx, Fy, Fz, Tx, Ty, Tz]`.
- No robot-specific names (bodies, joints) inside the `flapsim` package. They
  belong in the robot's manifest.
- Never commit simulation outputs, trained models or videos. `.gitignore`
  blocks the usual locations.

## Writing a plugin

The plugin API is introduced in roadmap Phase 3. Until then, aerodynamic
models follow the `apply_forces()` / `reset()` / `diagnostics` protocol used by
`flapsim.fluid.MuJoCoAeroAdapter`, and command sources and actuators follow the
`targets` / `apply(model, data, targets)` pair used in `flapsim.actuator`.

## Reporting bugs

Use the issue templates. For import problems, include your model file (or a
minimal version of it) and the output of the inspection command once it is
available.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0.
