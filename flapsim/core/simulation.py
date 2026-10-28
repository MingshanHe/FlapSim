"""The simulation: a robot, a signal bus and an ordered list of plugins."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import mujoco

from flapsim.robot.loader import RobotModel, load

from .signals import SignalBus


class Simulation:
    """Own the MuJoCo model and data and step them with attached plugins.

    ``robot`` is a :class:`~flapsim.robot.RobotModel` or anything
    :func:`flapsim.robot.load` accepts.  Plugins are added with :meth:`add`
    (instances, ``"kind/name"`` strings or config mappings) and run in the
    order they were added inside each phase; see :mod:`flapsim.core.phases`.
    """

    def __init__(
        self,
        robot,
        *,
        base: str | None = None,
        timestep: float | None = None,
        position=None,
        scene: bool = True,
        check: bool = True,
    ) -> None:
        if not isinstance(robot, RobotModel):
            robot = load(
                robot,
                base=base,
                timestep=timestep,
                position=position,
                scene=scene,
                check=check,
            )
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.signals = SignalBus()
        self.plugins: list = []
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.signals.publish("time", self.data.time)

    # -- properties ---------------------------------------------------------

    @property
    def fixed_base(self) -> bool:
        return self.robot.fixed_base

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    # -- plugins ------------------------------------------------------------

    def add(self, *plugins):
        """Add plugins (instances, ``"kind/name"`` or config mappings)."""
        from flapsim.plugins.base import Plugin
        from flapsim.plugins.registry import build

        added = []
        for item in plugins:
            plugin = item if isinstance(item, Plugin) else build(item)
            if any(existing.id == plugin.id for existing in self.plugins):
                raise ValueError(
                    f"a plugin with id {plugin.id!r} is already attached; pass id=..."
                )
            plugin.bind(self)
            self.plugins.append(plugin)
            added.append(plugin)
        return added[0] if len(added) == 1 else tuple(added)

    def remove(self, plugin_or_id) -> None:
        plugin = (
            self.get(plugin_or_id) if isinstance(plugin_or_id, str) else plugin_or_id
        )
        self.plugins.remove(plugin)
        plugin.unbind()

    def get(self, plugin_id: str):
        for plugin in self.plugins:
            if plugin.id == plugin_id:
                return plugin
        raise KeyError(
            f"no plugin with id {plugin_id!r}; attached: {[p.id for p in self.plugins]}"
        )

    def find(self, plugin_id: str, default=None):
        try:
            return self.get(plugin_id)
        except KeyError:
            return default

    def __getitem__(self, plugin_id: str):
        return self.get(plugin_id)

    def __contains__(self, plugin_id: str) -> bool:
        return any(plugin.id == plugin_id for plugin in self.plugins)

    def plugins_of(self, kind: str) -> list:
        return [plugin for plugin in self.plugins if plugin.kind == kind]

    # -- stepping -----------------------------------------------------------

    def reset(self) -> None:
        self.robot.reset()
        self.signals.clear()
        self.signals.publish("time", self.data.time)
        self.steps = 0
        for plugin in self.plugins:
            plugin.reset()

    def step(self) -> None:
        data = self.data
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        plugins = self.plugins
        for plugin in plugins:
            plugin.pre_control()
        for plugin in plugins:
            plugin.control()
        for plugin in plugins:
            plugin.actuate()
        for plugin in plugins:
            plugin.forces()
        mujoco.mj_step(self.model, data)
        forwarded = False
        for plugin in plugins:
            if plugin.post_step():
                forwarded = True
        if forwarded:
            mujoco.mj_forward(self.model, data)
        self.steps += 1
        self.signals.publish("time", data.time)
        for plugin in plugins:
            plugin.record()

    def advance(self, seconds: float) -> int:
        """Step headless for ``seconds`` of simulated time; return the step count."""
        end = self.data.time + float(seconds)
        steps = 0
        while self.data.time < end:
            self.step()
            steps += 1
        return steps

    def run(
        self,
        duration: float | None = None,
        *,
        viewer: bool = False,
        realtime: bool = True,
        fps: float = 60.0,
        on_frame=None,
    ) -> int:
        """Run for ``duration`` seconds, headless or in the passive viewer.

        ``on_frame(sim, viewer_handle)`` is called once per rendered frame.
        """
        if not viewer:
            if duration is None:
                raise ValueError("headless runs need a duration")
            return self.advance(duration)

        import mujoco.viewer

        steps = 0
        end_time = None if duration is None else self.data.time + duration
        with mujoco.viewer.launch_passive(self.model, self.data) as window:
            while window.is_running() and (
                end_time is None or self.data.time < end_time
            ):
                wall_start = time.perf_counter()
                frame_end = self.data.time + 1.0 / fps
                while self.data.time < frame_end:
                    self.step()
                    steps += 1
                if on_frame is not None:
                    on_frame(self, window)
                window.sync()
                if realtime:
                    remaining = 1.0 / fps - (time.perf_counter() - wall_start)
                    if remaining > 0.0:
                        time.sleep(remaining)
        return steps

    # -- config -------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: str | Path | Mapping[str, Any],
        *,
        base: str | None = None,
        overrides: Iterable[Mapping[str, Any]] | None = None,
    ) -> Simulation:
        """Build a simulation from a ``flapsim.sim/1`` YAML/JSON file or mapping."""
        from .config import build_simulation

        return build_simulation(config, base=base, overrides=overrides)

    def describe(self) -> dict[str, Any]:
        return {
            "robot": self.robot.manifest.name,
            "base": self.robot.base_type,
            "timestep": self.timestep,
            "plugins": [plugin.describe() for plugin in self.plugins],
            "signals": self.signals.names(),
        }

    def __repr__(self) -> str:
        return (
            f"Simulation({self.robot.manifest.name!r}, base={self.robot.base_type}, "
            f"plugins={[p.id for p in self.plugins]})"
        )


__all__ = ["Simulation"]
