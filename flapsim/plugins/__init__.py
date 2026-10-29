"""Plugin API: base classes, registry and discovery."""

from .base import (
    ActuatorPlugin,
    AeroPlugin,
    AtmospherePlugin,
    ControllerPlugin,
    ForcePlugin,
    Plugin,
    RecorderPlugin,
    SensorPlugin,
)
from .registry import (
    ENTRY_POINT_GROUP,
    available,
    build,
    discover_entry_points,
    get,
    parse_type,
    register,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "ActuatorPlugin",
    "AeroPlugin",
    "AtmospherePlugin",
    "ControllerPlugin",
    "ForcePlugin",
    "Plugin",
    "RecorderPlugin",
    "SensorPlugin",
    "available",
    "build",
    "discover_entry_points",
    "get",
    "parse_type",
    "register",
]
