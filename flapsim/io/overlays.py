"""Viewer and renderer overlays: force arrows and centre-of-pressure markers."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np


def _scene_of(target) -> mujoco.MjvScene:
    scene = getattr(target, "user_scn", None)
    return scene if scene is not None else target


def add_marker(scene: mujoco.MjvScene, position, radius: float, rgba) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def add_arrow(scene: mujoco.MjvScene, start, end, width: float, rgba) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


class ForceArrows:
    """Draw each wing's aerodynamic force as an arrow from its centre of pressure.

    Use as ``sim.run(viewer=True, on_frame=ForceArrows())`` or pass it to the
    video recorder.  ``scale`` is metres of arrow per newton.  The callable
    accepts a viewer handle (uses ``user_scn``) or an ``MjvScene``.
    """

    def __init__(
        self,
        *,
        scale: float = 0.5,
        width: float = 0.002,
        color=(1.0, 0.35, 0.1, 0.9),
        cop_color=(0.1, 0.9, 0.3, 0.9),
        cop_radius: float = 0.003,
        wings: Sequence[str] | None = None,
        total: bool = False,
        total_color=(0.2, 0.4, 1.0, 0.9),
    ) -> None:
        self.scale = float(scale)
        self.width = float(width)
        self.color = tuple(color)
        self.cop_color = tuple(cop_color)
        self.cop_radius = float(cop_radius)
        self.wings = None if wings is None else list(wings)
        self.total = bool(total)
        self.total_color = tuple(total_color)

    def __call__(self, sim, target) -> None:
        scene = _scene_of(target)
        scene.ngeom = 0
        signals = sim.signals
        wings = self.wings or [w.name for w in sim.robot.wings]
        for name in wings:
            force = signals.get(f"wings/{name}/F_world")
            cop = signals.get(f"wings/{name}/cop_world")
            if force is None or cop is None:
                continue
            add_marker(scene, cop, self.cop_radius, self.cop_color)
            add_arrow(
                scene,
                cop,
                np.asarray(cop) + self.scale * np.asarray(force),
                self.width,
                self.color,
            )
        if self.total:
            total = signals.get("aero/F_world")
            if total is not None:
                origin = sim.data.xipos[sim.robot.torso_body]
                add_arrow(
                    scene,
                    origin,
                    origin + self.scale * np.asarray(total),
                    self.width * 1.5,
                    self.total_color,
                )


__all__ = ["ForceArrows", "add_arrow", "add_marker"]
