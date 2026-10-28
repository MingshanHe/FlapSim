"""Composable aerodynamics: flow state, per-wing kinematics, model loads.

An :class:`AeroModel` turns one wing's kinematics into :class:`WingLoads`
(a force at a point plus a pure moment, all in the wing aero frame).  An
:class:`AeroEngine` evaluates the shared kinematics once per wing, sums the
loads of every model, applies :class:`AeroModifier` factors (for example
ground effect) and writes the resulting wrench to MuJoCo.  Plugins wrap the
engine; see :mod:`flapsim.aero.stack`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

import mujoco
import numpy as np

from flapsim.config import AeroOptions
from flapsim.fluid.aero import wing_kinematics

if TYPE_CHECKING:
    from flapsim.robot.loader import RobotModel
    from flapsim.wings import Wing, WingState


@dataclass
class FlowState:
    """Torso velocity relative to the air and the air properties, per step."""

    rho: float
    v_body: np.ndarray
    w_body: np.ndarray
    wind_world: np.ndarray
    time: float
    dt: float
    imposed: bool = False


@dataclass
class WingLoads:
    """Loads of one model on one wing, in the wing aero frame.

    ``force`` acts at ``point`` (relative to the wing body origin);
    ``moment`` is a pure couple.  ``diagnostics`` are scalars published as
    ``wings/<wing>/<model>/<key>``.
    """

    source: str
    force: np.ndarray
    point: np.ndarray
    moment: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def zero(cls, source: str) -> WingLoads:
        return cls(source, np.zeros(3), np.zeros(3), np.zeros(3), {})


class AeroModel:
    """Base class of additive per-wing aerodynamic models."""

    name: ClassVar[str] = ""

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def reset(self) -> None:
        """Forget any history (called when the simulation resets)."""

    def compute(
        self,
        wing: Wing,
        state: WingState,
        kin: dict,
        flow: FlowState,
        options: AeroOptions,
    ) -> WingLoads | None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "config": _plain(self.config)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AeroModifier:
    """Multiplies the summed loads of a wing by a factor (ground effect ...)."""

    name: ClassVar[str] = ""

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def reset(self) -> None:
        pass

    def factor(self, wing: Wing, state: WingState, kin: dict, flow: FlowState) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"modifier": self.name, "config": _plain(self.config)}


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


# --------------------------------------------------------------------------- #
# Model and modifier registries                                               #
# --------------------------------------------------------------------------- #

MODELS: dict[str, type[AeroModel]] = {}
MODIFIERS: dict[str, type[AeroModifier]] = {}


def register_model(cls: type[AeroModel]) -> type[AeroModel]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a 'name'")
    MODELS[cls.name] = cls
    return cls


def register_modifier(cls: type[AeroModifier]) -> type[AeroModifier]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a 'name'")
    MODIFIERS[cls.name] = cls
    return cls


def _ensure_models() -> None:
    import flapsim.aero.models  # noqa: F401  (registers the built-in models)


def available_models() -> list[str]:
    _ensure_models()
    return sorted(MODELS)


def available_modifiers() -> list[str]:
    _ensure_models()
    return sorted(MODIFIERS)


def build_model(spec: str | Mapping[str, Any] | AeroModel) -> AeroModel:
    _ensure_models()
    if isinstance(spec, AeroModel):
        return spec
    if isinstance(spec, str):
        name, config = spec, {}
    elif isinstance(spec, Mapping):
        config = dict(spec)
        name = config.pop("type", None) or config.pop("model", None)
        if name is None:
            raise ValueError("aero model config needs a 'type'")
    else:
        raise TypeError(f"cannot build an aero model from {spec!r}")
    try:
        return MODELS[name](**config)
    except KeyError:
        raise ValueError(
            f"unknown aero model {name!r}; available: {sorted(MODELS)}"
        ) from None


def build_modifier(spec: str | Mapping[str, Any] | AeroModifier) -> AeroModifier:
    _ensure_models()
    if isinstance(spec, AeroModifier):
        return spec
    if isinstance(spec, str):
        name, config = spec, {}
    elif isinstance(spec, Mapping):
        config = dict(spec)
        name = config.pop("type", None) or config.pop("modifier", None)
        if name is None:
            raise ValueError("aero modifier config needs a 'type'")
    else:
        raise TypeError(f"cannot build an aero modifier from {spec!r}")
    try:
        return MODIFIERS[name](**config)
    except KeyError:
        raise ValueError(
            f"unknown aero modifier {name!r}; available: {sorted(MODIFIERS)}"
        ) from None


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #


class AeroEngine:
    """Evaluate a stack of models on every wing of a robot and apply them."""

    def __init__(
        self,
        robot: RobotModel,
        models: Sequence[AeroModel | str | Mapping[str, Any]],
        modifiers: Sequence[AeroModifier | str | Mapping[str, Any]] = (),
        *,
        options: AeroOptions | None = None,
        wings: Sequence[str] | None = None,
    ) -> None:
        if not robot.wings:
            raise ValueError(
                f"robot {robot.manifest.name!r} declares no wings; add a wings section "
                "to its manifest"
            )
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.wings = [w for w in robot.wings if wings is None or w.name in wings]
        if wings is not None and len(self.wings) != len(wings):
            missing = sorted(set(wings) - {w.name for w in self.wings})
            raise ValueError(
                f"unknown wings {missing}; wings are {[w.name for w in robot.wings]}"
            )
        self.models = [build_model(m) for m in models]
        self.modifiers = [build_modifier(m) for m in modifiers]
        if not self.models:
            raise ValueError("an aero engine needs at least one model")
        self.options = options or AeroOptions()
        self.torso_body = robot.torso_body
        self._velocity = np.zeros(6)
        self.diagnostics: dict[str, Any] = {}

    # -- flow ---------------------------------------------------------------

    def flow_state(
        self,
        *,
        rho: float | None = None,
        wind_world=None,
        dt: float | None = None,
        imposed_v_body=None,
        imposed_w_body=None,
    ) -> FlowState:
        data = self.data
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.torso_body,
            self._velocity,
            0,
        )
        rotation = data.xmat[self.torso_body].reshape(3, 3)
        wind = (
            np.zeros(3) if wind_world is None else np.asarray(wind_world, dtype=float)
        )
        v_body = rotation.T @ (self._velocity[3:] - wind)
        w_body = rotation.T @ self._velocity[:3]
        imposed = False
        if imposed_v_body is not None:
            v_body = np.asarray(imposed_v_body, dtype=float)
            imposed = True
        if imposed_w_body is not None:
            w_body = np.asarray(imposed_w_body, dtype=float)
            imposed = True
        magnitude = float(np.linalg.norm(w_body))
        if magnitude > self.options.angular_velocity_limit:
            w_body = w_body * (self.options.angular_velocity_limit / magnitude)
        return FlowState(
            rho=float(rho)
            if rho is not None
            else float(self.robot.manifest.air_density),
            v_body=v_body,
            w_body=w_body,
            wind_world=wind,
            time=float(data.time),
            dt=float(dt if dt is not None else self.model.opt.timestep),
            imposed=imposed,
        )

    # -- evaluation ---------------------------------------------------------

    def apply(self, flow: FlowState | None = None) -> dict[str, Any]:
        """Compute all loads for the current state and write them to MuJoCo."""
        data = self.data
        flow = flow or self.flow_state()
        options = self.options
        total_force = np.zeros(3)
        lift_total = 0.0
        diagnostics: dict[str, Any] = {}
        for wing in self.wings:
            wing.clear(data)
            state = wing.state(data)
            params = wing.params
            if abs(params.rho - flow.rho) > 1e-12:
                params = replace(params, rho=flow.rho)
            kin = wing_kinematics(
                params,
                int(state.side),
                state.psi,
                state.psi_dot,
                state.theta,
                state.theta_dot,
                flow.v_body,
                flow.w_body,
                options,
            )
            rotation = state.R_world_from_aero
            force_aero = np.zeros(3)
            torque_world = np.zeros(3)
            weighted_point = np.zeros(3)
            weight = 0.0
            prefix = wing.name
            loads_list = []
            for model in self.models:
                loads = model.compute(wing, state, kin, flow, options)
                if loads is None:
                    continue
                loads_list.append(loads)
            factor = 1.0
            for modifier in self.modifiers:
                factor *= float(modifier.factor(wing, state, kin, flow))
            for loads in loads_list:
                force = factor * loads.force
                moment = factor * loads.moment
                force_world = rotation @ force
                point_world = state.origin_world + rotation @ loads.point
                arm = point_world - state.com_world
                torque_world += np.cross(arm, force_world) + rotation @ moment
                force_aero += force
                magnitude = float(np.linalg.norm(force))
                weighted_point += magnitude * loads.point
                weight += magnitude
                diagnostics[f"{prefix}/{loads.source}/F_N"] = float(force[0])
                for key, value in loads.diagnostics.items():
                    diagnostics[f"{prefix}/{loads.source}/{key}"] = value
            force_world = rotation @ force_aero
            if not options.read_only:
                wing.apply_wrench(data, force_world, torque_world)
            total_force += force_world
            lift_total += abs(float(force_aero[0]))
            cop_local = weighted_point / weight if weight > 0.0 else np.zeros(3)
            diagnostics[f"{prefix}/F_N"] = float(force_aero[0])
            diagnostics[f"{prefix}/F_aero"] = force_aero
            diagnostics[f"{prefix}/F_world"] = force_world
            diagnostics[f"{prefix}/torque_world"] = torque_world
            diagnostics[f"{prefix}/cop_world"] = (
                state.origin_world + rotation @ cop_local
            )
            diagnostics[f"{prefix}/alpha"] = kin["alpha"]
            diagnostics[f"{prefix}/U_i"] = kin["U_i_at_CoP"]
            diagnostics[f"{prefix}/U_o"] = kin["U_o_at_CoP"]
            diagnostics[f"{prefix}/modifier_factor"] = factor
        diagnostics["F_world"] = total_force
        diagnostics["lift_total"] = lift_total
        diagnostics["rho"] = flow.rho
        diagnostics["read_only"] = options.read_only
        names = {wing.name for wing in self.wings}
        if names == {"left", "right"}:
            diagnostics["F_N_L"] = diagnostics["left/F_N"]
            diagnostics["F_N_R"] = diagnostics["right/F_N"]
            diagnostics["alpha_L"] = diagnostics["left/alpha"]
            diagnostics["alpha_R"] = diagnostics["right/alpha"]
        self.diagnostics = diagnostics
        return diagnostics

    def reset(self) -> None:
        for wing in self.wings:
            wing.clear(self.data)
        for model in self.models:
            model.reset()
        for modifier in self.modifiers:
            modifier.reset()
        self.diagnostics = {}

    def describe(self) -> dict[str, Any]:
        return {
            "wings": [w.name for w in self.wings],
            "models": [m.describe() for m in self.models],
            "modifiers": [m.describe() for m in self.modifiers],
            "options": _plain(self.options),
        }


__all__ = [
    "MODELS",
    "MODIFIERS",
    "AeroEngine",
    "AeroModel",
    "AeroModifier",
    "FlowState",
    "WingLoads",
    "available_models",
    "available_modifiers",
    "build_model",
    "build_modifier",
    "register_model",
    "register_modifier",
]
