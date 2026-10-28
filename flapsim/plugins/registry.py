"""Plugin registry and discovery.

Built-in plugins register themselves with :func:`register` when
``flapsim.plugins.builtin`` is imported.  Third-party packages declare an
entry point in the ``flapsim.plugins`` group pointing at a plugin class, a
module that registers plugins on import, or a callable that does so::

    [project.entry-points."flapsim.plugins"]
    my_aero = "my_package.aero:MyAero"

Plugins are addressed by ``"kind/name"``, for example ``"aero/quasi_steady"``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from .base import Plugin

ENTRY_POINT_GROUP = "flapsim.plugins"

_REGISTRY: dict[str, dict[str, type[Plugin]]] = {}
_BUILTINS_LOADED = False
_ENTRY_POINTS_LOADED = False


class PluginError(ValueError):
    """Raised for unknown plugin types or bad plugin specifications."""


def register(
    cls: type[Plugin] | None = None, *, kind: str | None = None, name: str | None = None
):
    """Class decorator (or function) adding a plugin class to the registry."""

    def _register(plugin_cls: type[Plugin]) -> type[Plugin]:
        if not (inspect.isclass(plugin_cls) and issubclass(plugin_cls, Plugin)):
            raise PluginError(f"{plugin_cls!r} is not a Plugin subclass")
        plugin_kind = kind or plugin_cls.kind
        plugin_name = name or plugin_cls.name
        if not plugin_name:
            raise PluginError(
                f"{plugin_cls.__name__} has no 'name'; set the class attribute"
            )
        bucket = _REGISTRY.setdefault(plugin_kind, {})
        existing = bucket.get(plugin_name)
        if existing is not None and existing is not plugin_cls:
            raise PluginError(
                f"plugin {plugin_kind}/{plugin_name} is already registered by "
                f"{existing.__module__}.{existing.__name__}"
            )
        bucket[plugin_name] = plugin_cls
        return plugin_cls

    return _register(cls) if cls is not None else _register


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        _BUILTINS_LOADED = True
        importlib.import_module("flapsim.plugins.builtin")


def discover_entry_points(
    group: str = ENTRY_POINT_GROUP, *, force: bool = False
) -> list[str]:
    """Load plugins advertised by installed packages; return what was loaded."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return []
    _ENTRY_POINTS_LOADED = True
    loaded: list[str] = []
    try:
        entry_points = metadata.entry_points(group=group)
    except TypeError:  # Python < 3.10 API
        entry_points = metadata.entry_points().get(group, [])
    for entry_point in entry_points:
        target = entry_point.load()
        if inspect.isclass(target) and issubclass(target, Plugin):
            register(target)
        elif callable(target) and not inspect.ismodule(target):
            target()
        loaded.append(entry_point.name)
    return loaded


def parse_type(type_name: str) -> tuple[str, str]:
    if not isinstance(type_name, str) or type_name.count("/") != 1:
        raise PluginError(f"plugin type must look like 'kind/name', got {type_name!r}")
    kind, name = type_name.split("/")
    if not kind or not name:
        raise PluginError(f"plugin type must look like 'kind/name', got {type_name!r}")
    return kind, name


def get(kind: str, name: str | None = None) -> type[Plugin]:
    """Return the plugin class for ``kind, name`` or ``"kind/name"``."""
    if name is None:
        kind, name = parse_type(kind)
    _ensure_builtins()
    discover_entry_points()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise PluginError(
            f"unknown plugin {kind}/{name}; available: {sorted(available(kind))}"
            if kind in _REGISTRY
            else f"unknown plugin kind {kind!r}; kinds: {sorted(_REGISTRY)}"
        ) from None


def available(kind: str | None = None) -> dict[str, list[str]] | list[str]:
    """Registered plugin names, per kind or for one kind."""
    _ensure_builtins()
    discover_entry_points()
    if kind is not None:
        return sorted(_REGISTRY.get(kind, {}))
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def build(spec: str | Mapping[str, Any] | type[Plugin], **overrides: Any) -> Plugin:
    """Instantiate a plugin from ``"kind/name"``, a config mapping or a class."""
    if inspect.isclass(spec) and issubclass(spec, Plugin):
        return spec(**overrides)
    if isinstance(spec, str):
        return get(spec)(**overrides)
    if isinstance(spec, Mapping):
        config = dict(spec)
        try:
            type_name = config.pop("type")
        except KeyError:
            raise PluginError(
                "plugin config needs a 'type' such as 'aero/quasi_steady'"
            ) from None
        config.update(overrides)
        return get(type_name).from_config(config)
    raise PluginError(f"cannot build a plugin from {spec!r}")


__all__ = [
    "ENTRY_POINT_GROUP",
    "PluginError",
    "available",
    "build",
    "discover_entry_points",
    "get",
    "parse_type",
    "register",
]
