"""Plugin conformance checks that third-party authors can run.

::

    from flapsim.testing import check_plugin
    check_plugin(lambda: MyAero(gain=2.0))

The check builds a simulation with a built-in robot, attaches the plugin,
steps, resets and steps again, and verifies the metadata contract.  It
raises ``AssertionError`` with a plain-language message on failure and
returns a report dictionary on success.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from flapsim.plugins.base import Plugin


def check_plugin(
    factory: Callable[[], Plugin] | type[Plugin],
    *,
    robot: str = "SingleThrustFlapper",
    base: str | None = None,
    steps: int = 50,
) -> dict[str, Any]:
    from flapsim.core import Simulation

    plugin = factory() if not isinstance(factory, type) else _instantiate(factory)
    assert isinstance(plugin, Plugin), f"{plugin!r} is not a flapsim Plugin"
    assert plugin.name, f"{type(plugin).__name__} must set a class attribute 'name'"
    assert "/" not in plugin.name and plugin.kind, "name must not contain '/'"
    assert plugin.sim is None, "a fresh plugin must not be bound"

    sim = Simulation(robot, base=base)
    sim.add(plugin)
    assert plugin.sim is sim, "bind() must store the simulation"
    before = sim.time
    sim.advance(steps * sim.timestep)
    assert sim.time > before, "the simulation did not advance"
    published = [n for n in sim.signals.names() if n.startswith(f"{plugin.id}/")]
    for name in plugin.provides:
        assert f"{plugin.id}/{name}" in sim.signals, (
            f"{plugin.type_name()} declares provides={plugin.provides!r} but did not "
            f"publish {plugin.id}/{name}"
        )
    for name in plugin.requires:
        assert name in sim.signals, f"required signal {name!r} is missing"

    sim.reset()
    assert sim.steps == 0 and sim.time == 0.0, "reset must rewind time"
    sim.advance(steps * sim.timestep)

    description = plugin.describe()
    assert (
        isinstance(description, dict) and description.get("type") == plugin.type_name()
    )
    json.dumps(description)  # must be serialisable

    sim.remove(plugin)
    assert plugin.sim is None, "remove() must unbind the plugin"
    return {
        "type": plugin.type_name(),
        "id": plugin.id,
        "published": published,
        "steps": steps,
    }


def _instantiate(cls: type[Plugin]) -> Plugin:
    config = cls.example_config
    if config is None:
        return cls()
    return cls(**config)


def example_plugin_factories() -> dict[str, Callable[[], Plugin]]:
    """Factories for every registered plugin that ships an ``example_config``."""
    from flapsim.plugins.registry import available, get

    factories: dict[str, Callable[[], Plugin]] = {}
    for kind, names in available().items():
        for name in names:
            cls = get(kind, name)
            if cls.example_config is not None:
                factories[f"{kind}/{name}"] = lambda c=cls: _instantiate(c)
    return factories


__all__ = ["check_plugin", "example_plugin_factories"]
