"""Small JSON loader used by the first readable FlapSim prototype.

Configuration intentionally stays as ordinary dictionaries.  That makes the
JSON files and the Python code look the same, and avoids a second hierarchy of
schema classes while the simulator is still evolving.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .AeroOptions import AeroOptions
from .WingParams import WingParams

CONFIG_DIR = Path(__file__).resolve().parent
FILES = {
    "environment": "environment.json",
    "robot": "SingleThrustRobot.json",
    "wing": "Wing.json",
    "actuator": "flapping_actuator.json",
    "controller": "pid_controller.json",
}


class ConfigError(ValueError):
    """Raised when a configuration file cannot be used."""


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"config root must be an object: {path}")
    if value.get("schema_version", 1) != 1:
        raise ConfigError(f"unsupported schema_version in {path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` without changing either input."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(slots=True)
class ConfigBundle:
    environment: dict[str, Any]
    robot: dict[str, Any]
    wing: dict[str, Any]
    actuator: dict[str, Any]
    controller: dict[str, Any]

    @property
    def pid_controller(self) -> dict[str, Any]:
        return self.controller


def _resolve_urdf(config: dict[str, Any], base_dir: Path) -> None:
    structure = config.get("structure", {})
    value = structure.get("urdf_path")
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        structure["urdf_path"] = str(path.resolve())


def load_config_bundle(
    path: str | Path | None = None,
    *,
    robot: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ConfigBundle:
    """Load defaults, then one optional bundle file, then CLI overrides."""

    sections = {
        name: read_json(CONFIG_DIR / filename) for name, filename in FILES.items()
    }
    _resolve_urdf(sections["robot"], CONFIG_DIR)

    if path is not None:
        bundle_path = Path(path).expanduser().resolve()
        bundle = read_json(bundle_path)
        for name in FILES:
            external_name = {
                "actuator": "flapping_actuator",
                "controller": "pid_controller",
            }.get(name, name)
            if external_name in bundle:
                layer = deepcopy(bundle[external_name])
                if name == "robot":
                    _resolve_urdf(layer, bundle_path.parent)
                sections[name] = deep_merge(sections[name], layer)

    if overrides:
        for name in FILES:
            external_name = {
                "actuator": "flapping_actuator",
                "controller": "pid_controller",
            }.get(name, name)
            if external_name in overrides:
                layer = deepcopy(dict(overrides[external_name]))
                if name == "robot":
                    _resolve_urdf(layer, Path.cwd())
                sections[name] = deep_merge(sections[name], layer)

    aliases = {
        sections["robot"].get("model_name"),
        *sections["robot"].get("aliases", []),
    }
    if robot is not None and robot not in aliases:
        raise ConfigError(f"unknown robot {robot!r}; available: {sorted(aliases)}")
    if float(sections["environment"].get("dt", 0.0)) <= 0.0:
        raise ConfigError("environment.dt must be positive")
    urdf = Path(sections["robot"]["structure"]["urdf_path"])
    if not urdf.is_file():
        raise ConfigError(f"URDF does not exist: {urdf}")
    return ConfigBundle(**sections)


def load_wing_parameters(path: str | Path | None = None) -> WingParams:
    values = read_json(path or CONFIG_DIR / FILES["wing"])
    fields = WingParams.__dataclass_fields__
    return WingParams(**{name: values[name] for name in fields})


def wing_options(values: Mapping[str, Any]) -> AeroOptions:
    return AeroOptions(**dict(values.get("options", {})))


# Small convenience wrappers kept for notebooks.
def _section(name: str, path: str | Path | None) -> dict[str, Any]:
    return read_json(path or CONFIG_DIR / FILES[name])


def load_environment_config(path: str | Path | None = None) -> dict[str, Any]:
    return _section("environment", path)


def load_robot_config(path: str | Path | None = None) -> dict[str, Any]:
    config = _section("robot", path)
    _resolve_urdf(config, Path(path).resolve().parent if path else CONFIG_DIR)
    return config


def load_wing_config(path: str | Path | None = None) -> dict[str, Any]:
    return _section("wing", path)


def load_dc_motor_config(path: str | Path | None = None) -> dict[str, Any]:
    return read_json(path or CONFIG_DIR / "dc_motor.json")


def load_flapping_actuator_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    return _section("actuator", path)


def load_pid_controller_config(path: str | Path | None = None) -> dict[str, Any]:
    return _section("controller", path)


load_motor_config = load_dc_motor_config
load_controller_config = load_pid_controller_config
load_config = load_config_bundle
Config = ConfigBundle
