"""Ground-effect lift augmentation as a multiplicative factor."""

from __future__ import annotations

import math

from flapsim.aero.base import AeroModifier, register_modifier

MODELS = ("exponential", "cheeseman_bennett")


@register_modifier
class GroundEffect(AeroModifier):
    """Scale a wing's loads by a factor that grows near the ground.

    ``exponential``: ``1 + gain * exp(-h / (length_scale * R))`` with ``h``
    the height of the wing's origin above ``ground_height``.
    ``cheeseman_bennett``: the rotor-in-ground-effect formula
    ``1 / (1 - (R / (4 h))^2)``, capped at ``max_factor``.  Both are
    empirical and tuned by the user.
    """

    name = "ground_effect"

    def __init__(
        self,
        *,
        model: str = "exponential",
        gain: float = 0.3,
        length_scale: float = 0.5,
        ground_height: float = 0.0,
        max_factor: float = 2.0,
    ) -> None:
        super().__init__(
            model=model,
            gain=gain,
            length_scale=length_scale,
            ground_height=ground_height,
            max_factor=max_factor,
        )
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.model = model
        self.gain = float(gain)
        self.length_scale = float(length_scale)
        self.ground_height = float(ground_height)
        self.max_factor = float(max_factor)

    def factor(self, wing, state, kin, flow) -> float:
        height = max(float(state.origin_world[2]) - self.ground_height, 1e-6)
        R = wing.length
        if self.model == "exponential":
            value = 1.0 + self.gain * math.exp(-height / (self.length_scale * R))
        else:
            ratio = R / (4.0 * height)
            value = 1.0 / (1.0 - ratio**2) if ratio < 1.0 else self.max_factor
        return float(min(value, self.max_factor))


__all__ = ["GroundEffect"]
