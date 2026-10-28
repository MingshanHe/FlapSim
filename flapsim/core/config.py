"""Simulation configuration files (``schema: flapsim.sim/1``).

::

    schema: flapsim.sim/1
    robot: robot.yaml            # built-in name, manifest or model file
    base: fixed                  # optional override of the manifest's base
    plugins:
      - type: actuator/sinusoidal_stroke
        id: stroke
        joints: [left_stroke, right_stroke]
        frequency_hz: 30.0
      - type: aero/quasi_steady
    run:
      duration: 2.0
      viewer: false
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SIM_SCHEMA = "flapsim.sim/1"
TOP_LEVEL_KEYS = (
    "schema",
    "robot",
    "base",
    "timestep",
    "position",
    "scene",
    "plugins",
    "run",
)
RUN_KEYS = ("duration", "viewer", "realtime", "fps")


class SimConfigError(ValueError):
    """Raised when a simulation configuration is malformed."""


def load_sim_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a simulation config; relative paths are kept as is."""
    path = Path(path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SimConfigError(f"cannot read {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise SimConfigError(f"{path} must contain a mapping")
    data = dict(data)
    data["_base_dir"] = str(path.parent)
    validate_sim_config(data)
    return data


def is_sim_config(path: str | Path) -> bool:
    """True when the file declares the ``flapsim.sim/1`` schema."""
    path = Path(path)
    if path.suffix.lower() not in (".yaml", ".yml", ".json") or not path.is_file():
        return False
    try:
        head = path.read_text(encoding="utf-8")[:4096]
    except OSError:
        return False
    return SIM_SCHEMA in head


def validate_sim_config(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - set(TOP_LEVEL_KEYS) - {"_base_dir"})
    if unknown:
        raise SimConfigError(
            f"unknown top-level field(s) {unknown}; allowed: {TOP_LEVEL_KEYS}"
        )
    if config.get("schema", SIM_SCHEMA) != SIM_SCHEMA:
        raise SimConfigError(
            f"unsupported schema {config.get('schema')!r}; expected {SIM_SCHEMA!r}"
        )
    if "robot" not in config:
        raise SimConfigError("a simulation config needs a 'robot'")
    if config.get("base") not in (None, "fixed", "free"):
        raise SimConfigError("base must be 'fixed' or 'free'")
    plugins = config.get("plugins", [])
    if not isinstance(plugins, list):
        raise SimConfigError("'plugins' must be a list")
    for index, entry in enumerate(plugins):
        if isinstance(entry, str):
            continue
        if not isinstance(entry, Mapping) or "type" not in entry:
            raise SimConfigError(
                f"plugins[{index}] needs a 'type' such as 'aero/quasi_steady'"
            )
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        raise SimConfigError("'run' must be a mapping")
    unknown_run = sorted(set(run) - set(RUN_KEYS))
    if unknown_run:
        raise SimConfigError(f"unknown run field(s) {unknown_run}; allowed: {RUN_KEYS}")


def resolve_robot_source(config: Mapping[str, Any]) -> str | Path:
    """A robot path relative to the config file, or a built-in name."""
    robot = config["robot"]
    if isinstance(robot, Mapping):
        raise SimConfigError("inline robot manifests are not supported; give a path")
    base_dir = config.get("_base_dir")
    candidate = Path(str(robot)).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        local = Path(base_dir) / candidate
        if local.exists():
            return local
    return str(robot) if not candidate.exists() else candidate


def build_simulation(
    config: str | Path | Mapping[str, Any],
    *,
    base: str | None = None,
    overrides: Iterable[Mapping[str, Any]] | None = None,
):
    """Create a :class:`~flapsim.core.Simulation` from a config file or mapping."""
    from .simulation import Simulation

    if isinstance(config, str | Path):
        config = load_sim_config(config)
    else:
        config = dict(config)
        validate_sim_config(config)
    source = resolve_robot_source(config)
    sim = Simulation(
        source,
        base=base or config.get("base"),
        timestep=config.get("timestep"),
        position=config.get("position"),
        scene=bool(config.get("scene", True)),
    )
    plugins = [
        dict(p) if isinstance(p, Mapping) else p for p in config.get("plugins", [])
    ]
    for override in overrides or ():
        target = override.get("id")
        for entry in plugins:
            if isinstance(entry, dict) and entry.get("id") == target:
                entry.update({k: v for k, v in override.items() if k != "id"})
                break
        else:
            raise SimConfigError(f"override targets unknown plugin id {target!r}")
    for entry in plugins:
        sim.add(entry)
    sim.run_config = dict(config.get("run", {}))
    return sim


__all__ = [
    "SIM_SCHEMA",
    "SimConfigError",
    "build_simulation",
    "is_sim_config",
    "load_sim_config",
    "resolve_robot_source",
    "validate_sim_config",
]
