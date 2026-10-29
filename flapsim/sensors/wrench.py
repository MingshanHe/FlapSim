"""Load cell: the interaction wrench between a body and its parent."""

from __future__ import annotations

import mujoco
import numpy as np

from flapsim.plugins.base import SensorPlugin
from flapsim.plugins.registry import register

from .noise import Conditioner


@register
class BodyWrench(SensorPlugin):
    """Force and torque transmitted from a body's parent (MuJoCo ``cfrc_int``).

    For the root body of a fixed-base robot this is the reaction of the
    mounting, that is the force a load cell under the robot would read (its
    weight minus the lift).  Values are in the world frame about the body's
    centre of mass, averaged over ``average_s`` seconds when given.
    Publishes ``<id>/force`` and ``<id>/torque``.
    """

    name = "wrench"
    provides = ("force", "torque")
    example_config = {"average_s": 0.01}

    def __init__(
        self,
        *,
        body: str | None = None,
        average_s: float = 0.0,
        noise: float = 0.0,
        seed: int = 0,
        id: str | None = "loadcell",
    ) -> None:
        super().__init__(id=id, body=body, average_s=average_s, noise=noise, seed=seed)
        self.body_name = body
        self.average_s = float(average_s)
        self.force = Conditioner(size=3, tau=average_s, noise_std=noise, seed=seed)
        self.torque = Conditioner(size=3, tau=average_s, noise_std=noise, seed=seed + 1)
        self.body_id = -1

    def setup(self) -> None:
        name = self.body_name or self.robot.root_body_name
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if self.body_id < 0:
            raise ValueError(f"wrench body {name!r} not found")
        self.reset()

    def reset(self) -> None:
        self.force.reset(np.zeros(3), reseed=True)
        self.torque.reset(np.zeros(3), reseed=True)

    def pre_control(self) -> None:
        data = self.data
        mujoco.mj_rnePostConstraint(self.model, data)
        wrench = data.cfrc_int[self.body_id]
        time = float(data.time)
        self.publish("force", self.force.update(wrench[3:], time))
        self.publish("torque", self.torque.update(wrench[:3], time))


__all__ = ["BodyWrench"]
