"""Aero plugins built on :class:`~flapsim.aero.base.AeroEngine`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flapsim.config import AeroOptions
from flapsim.plugins.base import AeroPlugin
from flapsim.plugins.registry import register

from .base import AeroEngine

PUBLISHED_WING_KEYS = (
    "F_N",
    "F_aero",
    "F_world",
    "torque_world",
    "cop_world",
    "alpha",
    "U_i",
    "U_o",
)


@register
class AeroStack(AeroPlugin):
    """Sum several additive aero models on every wing, with modifiers.

    ``models`` lists model names or ``{type: name, ...}`` mappings (see
    ``flapsim.aero.base.available_models()``), ``modifiers`` likewise.
    Reads ``atmosphere/rho`` and ``atmosphere/wind_world`` when an atmosphere
    plugin publishes them.  Publishes ``aero/F_world``, ``aero/lift_total``,
    ``aero/rho`` and per wing ``wings/<name>/<key>`` totals plus
    ``wings/<name>/<model>/<key>`` contributions.  ``imposed_body_velocity``
    and ``imposed_body_rates`` replace the torso velocity seen by the models
    (identification and testing only).
    """

    name = "stack"
    provides = ("F_world", "lift_total", "rho")
    example_config = {"models": ["quasi_steady", "rotational_lift", "added_mass"]}

    def __init__(
        self,
        *,
        models: Sequence[str | Mapping[str, Any]] = ("quasi_steady",),
        modifiers: Sequence[str | Mapping[str, Any]] = (),
        options: Mapping[str, Any] | AeroOptions | None = None,
        wings: Sequence[str] | None = None,
        imposed_body_velocity: Sequence[float] | None = None,
        imposed_body_rates: Sequence[float] | None = None,
        id: str | None = "aero",
    ) -> None:
        if isinstance(options, AeroOptions):
            option_values = {
                f: getattr(options, f) for f in AeroOptions.__dataclass_fields__
            }
        else:
            option_values = dict(options or {})
        super().__init__(
            id=id,
            models=[m if isinstance(m, str) else dict(m) for m in models],
            modifiers=[m if isinstance(m, str) else dict(m) for m in modifiers],
            options=option_values,
            wings=None if wings is None else list(wings),
            imposed_body_velocity=(
                None if imposed_body_velocity is None else list(imposed_body_velocity)
            ),
            imposed_body_rates=None
            if imposed_body_rates is None
            else list(imposed_body_rates),
        )
        self._model_specs = list(models)
        self._modifier_specs = list(modifiers)
        self._options = AeroOptions(**option_values)
        self._wing_names = None if wings is None else list(wings)
        self.imposed_body_velocity = imposed_body_velocity
        self.imposed_body_rates = imposed_body_rates
        self.engine: AeroEngine | None = None

    def setup(self) -> None:
        self.engine = AeroEngine(
            self.robot,
            self._model_specs,
            self._modifier_specs,
            options=self._options,
            wings=self._wing_names,
        )

    # -- access -------------------------------------------------------------

    @property
    def options(self) -> AeroOptions:
        return self._options

    @options.setter
    def options(self, value: AeroOptions) -> None:
        self._options = value
        if self.engine is not None:
            self.engine.options = value

    @property
    def models(self) -> list:
        return [] if self.engine is None else self.engine.models

    @property
    def modifiers(self) -> list:
        return [] if self.engine is None else self.engine.modifiers

    @property
    def diagnostics(self) -> dict:
        return {} if self.engine is None else self.engine.diagnostics

    @property
    def wings(self):
        return self.robot.wings if self.engine is None else self.engine.wings

    # -- phases -------------------------------------------------------------

    def forces(self) -> None:
        engine = self.engine
        signals = self.signals
        flow = engine.flow_state(
            rho=signals.get("atmosphere/rho"),
            wind_world=signals.get("atmosphere/wind_world"),
            imposed_v_body=self.imposed_body_velocity,
            imposed_w_body=self.imposed_body_rates,
        )
        diagnostics = engine.apply(flow)
        self.publish("F_world", diagnostics["F_world"])
        self.publish("lift_total", diagnostics["lift_total"])
        self.publish("rho", diagnostics["rho"])
        publish = signals.publish
        for wing in engine.wings:
            prefix = wing.name
            for key in PUBLISHED_WING_KEYS:
                publish(f"wings/{prefix}/{key}", diagnostics[f"{prefix}/{key}"])
            for model in engine.models:
                model_prefix = f"{prefix}/{model.name}/"
                for key, value in diagnostics.items():
                    if key.startswith(model_prefix):
                        publish(f"wings/{key}", value)

    def reset(self) -> None:
        if self.engine is not None:
            self.engine.reset()

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        if self.engine is not None:
            description["engine"] = self.engine.describe()
        return description


@register
class QuasiSteadyAero(AeroStack):
    """Per-step quasi-steady blade-element forces (the single-model stack).

    Publishes ``aero/F_world``, ``aero/lift_total`` and, for every wing,
    ``wings/<name>/F_N``, ``alpha``, ``F_world``, ``torque_world``,
    ``cop_world`` and the ``quasi_steady/`` contributions (``C_N``,
    ``M_aero``, ``M_rd``, ``chord_CoP``, ``span_CoP``).  ``options`` are
    :class:`~flapsim.config.AeroOptions` fields.
    """

    name = "quasi_steady"
    example_config: dict = {}

    def __init__(
        self,
        *,
        options: Mapping[str, Any] | AeroOptions | None = None,
        wings: Sequence[str] | None = None,
        id: str | None = "aero",
    ) -> None:
        super().__init__(models=("quasi_steady",), options=options, wings=wings, id=id)
        self.config = {"options": self.config["options"], "wings": self.config["wings"]}


__all__ = ["AeroStack", "QuasiSteadyAero"]
