"""Plugin base classes.

A plugin is bound to one :class:`~flapsim.core.Simulation` and participates
in its step phases.  Subclasses override the phases they need::

    class MyAero(AeroPlugin):
        name = "my_aero"
        def setup(self):            # resolve MuJoCo ids once
            ...
        def forces(self):           # write xfrc_applied / qfrc_applied
            ...

Outputs are published as signals under ``<id>/<name>``; commands are read
from ``cmd/<id>/<name>`` so controllers and actuators stay decoupled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from flapsim.core.simulation import Simulation


class Plugin:
    kind: ClassVar[str] = "plugin"
    name: ClassVar[str] = ""
    provides: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[tuple[str, ...]] = ()
    example_config: ClassVar[dict[str, Any] | None] = None

    def __init__(self, *, id: str | None = None, **config: Any) -> None:
        self.id = id or self.name or type(self).__name__.lower()
        self.config: dict[str, Any] = dict(config)
        self.sim: Simulation | None = None

    # -- binding ------------------------------------------------------------

    def bind(self, sim: Simulation) -> None:
        self.sim = sim
        self.setup()

    def unbind(self) -> None:
        self.sim = None

    def setup(self) -> None:
        """Resolve MuJoCo ids and validate configuration (called by ``bind``)."""

    @property
    def model(self):
        return self._sim().model

    @property
    def data(self):
        return self._sim().data

    @property
    def robot(self):
        return self._sim().robot

    @property
    def signals(self):
        return self._sim().signals

    def _sim(self) -> Simulation:
        if self.sim is None:
            raise RuntimeError(f"plugin {self.id!r} is not attached to a simulation")
        return self.sim

    # -- phases -------------------------------------------------------------

    def reset(self) -> None:
        """Called after the simulation state was reset."""

    def pre_control(self) -> None:
        """Sensors: read the state."""

    def control(self) -> None:
        """Controllers: publish commands."""

    def actuate(self) -> None:
        """Actuators: apply joint forces or kinematic targets."""

    def forces(self) -> None:
        """Force models: write ``xfrc_applied`` / ``qfrc_applied``."""

    def post_step(self) -> bool:
        """After ``mj_step``; return True if ``qpos`` was changed."""
        return False

    def record(self) -> None:
        """Recorders: sample the signal bus."""

    # -- signals ------------------------------------------------------------

    def publish(self, name: str, value: Any) -> None:
        self.signals.publish(f"{self.id}/{name}", value)

    def command(self, name: str, default: Any = None) -> Any:
        return self.signals.get(f"cmd/{self.id}/{name}", default)

    def send_command(self, plugin_id: str, name: str, value: Any) -> None:
        self.signals.publish(f"cmd/{plugin_id}/{name}", value)

    # -- metadata -----------------------------------------------------------

    @classmethod
    def type_name(cls) -> str:
        return f"{cls.kind}/{cls.name}"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Plugin:
        return cls(**dict(config))

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.type_name(),
            "id": self.id,
            "config": _plain(self.config),
            "provides": [f"{self.id}/{n}" for n in self.provides],
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {f: _plain(getattr(value, f)) for f in value.__dataclass_fields__}
    return value


class ForcePlugin(Plugin):
    kind = "force"


class AeroPlugin(ForcePlugin):
    kind = "aero"


class AtmospherePlugin(Plugin):
    kind = "atmosphere"


class ActuatorPlugin(Plugin):
    kind = "actuator"


class SensorPlugin(Plugin):
    kind = "sensor"


class ControllerPlugin(Plugin):
    kind = "controller"


class RecorderPlugin(Plugin):
    kind = "recorder"


__all__ = [
    "ActuatorPlugin",
    "AeroPlugin",
    "AtmospherePlugin",
    "ControllerPlugin",
    "ForcePlugin",
    "Plugin",
    "RecorderPlugin",
    "SensorPlugin",
]
