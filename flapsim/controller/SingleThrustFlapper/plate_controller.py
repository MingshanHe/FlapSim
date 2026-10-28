"""Command source for the two gimbal plate joints (angles in radians).

Which joints play the pitch and roll roles comes from the robot's joint
configuration (entries with ``"controller": "position_controller"`` and a
``"role"`` of ``"pitch"`` or ``"roll"``).
"""

from __future__ import annotations

import json
from pathlib import Path


class PlateController:
    def __init__(
        self,
        pitch: float = 0.0,
        roll: float = 0.0,
        *,
        config_path: str | Path | None = None,
        pitch_joint: str | None = None,
        roll_joint: str | None = None,
    ) -> None:
        if config_path is not None:
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            for joint_name, settings in config.items():
                if settings.get("controller") != "position_controller":
                    continue
                role = settings.get("role")
                if role == "pitch" and pitch_joint is None:
                    pitch_joint = joint_name
                elif role == "roll" and roll_joint is None:
                    roll_joint = joint_name
        if pitch_joint is None or roll_joint is None:
            raise ValueError(
                "PlateController needs a pitch and a roll joint: pass pitch_joint/"
                "roll_joint or a joints config with 'role': 'pitch'/'roll' entries"
            )
        self.pitch_joint = pitch_joint
        self.roll_joint = roll_joint
        self._targets = {pitch_joint: float(pitch), roll_joint: float(roll)}

    @property
    def targets(self) -> dict[str, float]:
        return self._targets.copy()

    @property
    def pitch(self) -> float:
        return self._targets[self.pitch_joint]

    @property
    def roll(self) -> float:
        return self._targets[self.roll_joint]

    def set_command(
        self, *, pitch: float | None = None, roll: float | None = None
    ) -> None:
        if pitch is not None:
            self._targets[self.pitch_joint] = float(pitch)
        if roll is not None:
            self._targets[self.roll_joint] = float(roll)
