import numpy as np
from .base import Controller


class SpeedController(Controller):
    """Direct shaft-speed command; suitable for a fixed-base mechanism test."""
    def command(self):
        return np.asarray(self.config.speed, dtype=float)

    def update(self, data, state, command, xp=np):
        return command, state
