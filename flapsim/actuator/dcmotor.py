"""Quasi-static brushed DC motor with voltage, back-EMF and current limits."""
import numpy as np
from .base import Actuator


class DCMotor(Actuator):
    def torque(self, position, velocity, target, xp=np):
        c = self.config
        desired = c.speed_gain * (target - velocity)
        voltage = c.resistance * desired / c.torque_constant + c.back_emf * velocity
        voltage = xp.clip(voltage, -c.max_voltage, c.max_voltage)
        current = xp.clip((voltage - c.back_emf * velocity) / c.resistance,
                          -c.max_current, c.max_current)
        return c.torque_constant * current
