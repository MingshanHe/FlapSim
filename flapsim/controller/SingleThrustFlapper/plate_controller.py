"""Command source for the two plate joints (angles are in radians)."""

from __future__ import annotations


class PlateController:
    def __init__(self, pitch: float = 0.0, roll: float = 0.0) -> None:
        self._targets = {
            "pitch_joint": float(pitch),
            "roll_joint": float(roll),
        }

    @property
    def targets(self) -> dict[str, float]:
        return self._targets.copy()

    def set_command(
        self, *, pitch: float | None = None, roll: float | None = None
    ) -> None:
        if pitch is not None:
            self._targets["pitch_joint"] = float(pitch)
        if roll is not None:
            self._targets["roll_joint"] = float(roll)
