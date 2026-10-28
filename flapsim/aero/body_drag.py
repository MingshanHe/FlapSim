"""Quadratic drag on bodies (fuselage, legs, tail)."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from flapsim.plugins.base import ForcePlugin
from flapsim.plugins.registry import register


@register
class BodyDrag(ForcePlugin):
    """``F = -1/2 rho C_d A |v| v`` per body axis, relative to the wind.

    ``bodies`` maps body names to ``{cd, area}`` where each may be a scalar
    (isotropic) or a 3-vector in the body frame.  Density comes from
    ``atmosphere/rho`` when published, else ``rho`` or the manifest.  Forces
    are added to ``xfrc_applied`` at the body centre of mass, so place this
    plugin after any plugin that overwrites a body's wrench.  Publishes
    ``<id>/<body>/F_world``.
    """

    name = "body_drag"
    example_config = {"bodies": {"link2": {"cd": 1.0, "area": 8.0e-4}}}

    def __init__(
        self,
        *,
        bodies: Mapping[str, Mapping[str, float] | None],
        rho: float | None = None,
        id: str | None = "drag",
    ) -> None:
        super().__init__(
            id=id, bodies={k: dict(v or {}) for k, v in bodies.items()}, rho=rho
        )
        if not bodies:
            raise ValueError("BodyDrag needs at least one body")
        self.settings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, settings in bodies.items():
            settings = dict(settings or {})
            cd = np.broadcast_to(
                np.asarray(settings.get("cd", 1.0), dtype=float), (3,)
            ).copy()
            area = np.broadcast_to(
                np.asarray(settings.get("area", 0.0), dtype=float), (3,)
            ).copy()
            if np.any(cd < 0.0) or np.any(area < 0.0):
                raise ValueError(f"body {name!r}: cd and area must be non-negative")
            self.settings[name] = (cd, area)
        self.rho = None if rho is None else float(rho)
        self._ids: dict[str, int] = {}
        self._velocity = np.zeros(6)

    def setup(self) -> None:
        for name in self.settings:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                names = [self.model.body(i).name for i in range(self.model.nbody)]
                raise ValueError(f"body {name!r} not found; bodies are {names}")
            self._ids[name] = int(body)

    def forces(self) -> None:
        signals = self.signals
        rho = signals.get("atmosphere/rho")
        if rho is None:
            rho = self.rho if self.rho is not None else self.robot.manifest.air_density
        wind = signals.get("atmosphere/wind_world")
        wind = np.zeros(3) if wind is None else np.asarray(wind, dtype=float)
        data = self.data
        for name, (cd, area) in self.settings.items():
            body = self._ids[name]
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, self._velocity, 0
            )
            rotation = data.xmat[body].reshape(3, 3)
            v_body = rotation.T @ (self._velocity[3:] - wind)
            force_body = -0.5 * rho * cd * area * np.abs(v_body) * v_body
            force_world = rotation @ force_body
            data.xfrc_applied[body, :3] += force_world
            self.publish(f"{name}/F_world", force_world)


__all__ = ["BodyDrag"]
