"""Quasi-steady blade-element aerodynamics for every wing in the manifest."""

from __future__ import annotations

from collections.abc import Mapping

from flapsim.config import AeroOptions
from flapsim.fluid.mujoco_adapter import MuJoCoAeroAdapter
from flapsim.plugins.base import AeroPlugin
from flapsim.plugins.registry import register


@register
class QuasiSteadyAero(AeroPlugin):
    """Per-step normal force at each wing's centre of pressure.

    Publishes ``aero/F_world`` (total force), ``aero/lift_total`` and, for
    every wing, ``wings/<name>/F_N``, ``alpha``, ``C_N``, ``M_aero``, ``M_rd``,
    ``chord_CoP``, ``span_CoP``, ``cop_world``, ``F_world`` and
    ``torque_world``.  ``options`` are :class:`~flapsim.config.AeroOptions`
    fields (``force_limit``, ``lift_scale``, ``read_only``, ...).
    """

    name = "quasi_steady"
    provides = ("F_world", "lift_total")
    example_config: dict = {}

    def __init__(
        self, *, options: Mapping | AeroOptions | None = None, id: str | None = "aero"
    ) -> None:
        if isinstance(options, AeroOptions):
            option_values = {
                f: getattr(options, f) for f in AeroOptions.__dataclass_fields__
            }
        else:
            option_values = dict(options or {})
        super().__init__(id=id, options=option_values)
        self._options = AeroOptions(**option_values)
        self.adapter: MuJoCoAeroAdapter | None = None

    def setup(self) -> None:
        self.adapter = MuJoCoAeroAdapter(self.robot, self._options)

    @property
    def options(self) -> AeroOptions:
        return self._options

    @options.setter
    def options(self, value: AeroOptions) -> None:
        self._options = value
        if self.adapter is not None:
            self.adapter.options = value

    @property
    def diagnostics(self) -> dict:
        return {} if self.adapter is None else self.adapter.diagnostics

    @property
    def wings(self):
        return self.robot.wings

    def forces(self) -> None:
        adapter = self.adapter
        adapter.apply_forces()
        diagnostics = adapter.diagnostics
        publish = self.signals.publish
        self.publish("F_world", diagnostics["F_world"])
        self.publish("lift_total", diagnostics["lift_total"])
        for wing in adapter.wings:
            prefix = wing.name
            for key in (
                "F_N",
                "alpha",
                "C_N",
                "M_aero",
                "M_rd",
                "chord_CoP",
                "span_CoP",
                "cop_world",
                "F_world",
                "torque_world",
            ):
                publish(f"wings/{prefix}/{key}", diagnostics[f"{prefix}/{key}"])

    def reset(self) -> None:
        if self.adapter is not None:
            self.adapter.reset()


__all__ = ["QuasiSteadyAero"]
