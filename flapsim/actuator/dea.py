"""Lumped antagonistic DEA pair: signed V² torque, spring and damping.

This is a joint-level approximation, not a finite-element dielectric model.
Voltage/coefficient/stiffness require calibration to the actual transmission.
"""
import numpy as np
from .base import Actuator


class DEA(Actuator):
    def torque(self, position, velocity, target, xp=np):
        c = self.config
        passive = c.stiffness * position + c.damping * velocity
        requested = c.speed_gain * (target - velocity) + passive
        voltage = xp.minimum(xp.sqrt(xp.abs(requested) / c.dea_coefficient), c.max_voltage)
        return xp.sign(requested) * c.dea_coefficient * voltage**2 - passive
