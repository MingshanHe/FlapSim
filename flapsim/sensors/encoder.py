"""Joint position and velocity measurements."""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np

from flapsim.plugins.base import SensorPlugin
from flapsim.plugins.registry import register

from .noise import Conditioner


@register
class JointEncoder(SensorPlugin):
    """Quantised, noisy joint positions and velocities at ``rate_hz``.

    ``counts_per_rev`` sets the position resolution of hinge joints (``None``
    for continuous).  Publishes ``<id>/<joint>/position`` and
    ``<id>/<joint>/velocity``.
    """

    name = "encoder"
    example_config = {"joints": ["left_flapping_joint"], "counts_per_rev": 4096}

    def __init__(
        self,
        *,
        joints: Sequence[str],
        rate_hz: float | None = None,
        counts_per_rev: int | None = None,
        position_noise: float = 0.0,
        velocity_noise: float = 0.0,
        velocity_tau: float = 0.0,
        seed: int = 0,
        id: str | None = "enc",
    ) -> None:
        super().__init__(
            id=id,
            joints=list(joints),
            rate_hz=rate_hz,
            counts_per_rev=counts_per_rev,
            position_noise=position_noise,
            velocity_noise=velocity_noise,
            velocity_tau=velocity_tau,
            seed=seed,
        )
        if not joints:
            raise ValueError("JointEncoder needs at least one joint")
        self.joint_names = list(joints)
        self.rate_hz = None if rate_hz is None else float(rate_hz)
        resolution = (
            0.0 if counts_per_rev is None else 2.0 * math.pi / float(counts_per_rev)
        )
        n = len(self.joint_names)
        self.position = Conditioner(
            size=n,
            rate_hz=rate_hz,
            noise_std=position_noise,
            resolution=resolution,
            seed=seed,
        )
        self.velocity = Conditioner(
            size=n,
            rate_hz=rate_hz,
            noise_std=velocity_noise,
            tau=velocity_tau,
            seed=seed + 1,
        )
        self._addresses: list[tuple[int, int]] = []
        self._next_time = -math.inf

    def setup(self) -> None:
        model = self.model
        self._addresses = []
        for name in self.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"encoder joint {name!r} not found")
            self._addresses.append(
                (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))
            )
        self.reset()

    def reset(self) -> None:
        self._next_time = -math.inf
        q = np.array([self.data.qpos[a] for a, _ in self._addresses])
        v = np.array([self.data.qvel[d] for _, d in self._addresses])
        self.position.reset(q, reseed=True)
        self.velocity.reset(v, reseed=True)

    def pre_control(self) -> None:
        time = float(self.data.time)
        if time + 1e-12 < self._next_time:
            return
        self._next_time = time + (1.0 / self.rate_hz if self.rate_hz else 0.0)
        q = np.array([self.data.qpos[a] for a, _ in self._addresses])
        v = np.array([self.data.qvel[d] for _, d in self._addresses])
        positions = self.position.update(q, time)
        velocities = self.velocity.update(v, time)
        for index, name in enumerate(self.joint_names):
            self.publish(f"{name}/position", positions[index])
            self.publish(f"{name}/velocity", velocities[index])


__all__ = ["JointEncoder"]
