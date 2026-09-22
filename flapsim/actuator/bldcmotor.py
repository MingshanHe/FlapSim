"""Averaged BLDC q-axis current loop; commutation and inductance omitted."""
import numpy as np
from .base import Actuator


class BLDCMotor(Actuator):
    def torque(self, position, velocity, target, xp=np):
        c = self.config
        iq = c.speed_gain * (target - velocity) / c.torque_constant
        iq = xp.clip(iq, -c.max_current, c.max_current)
        voltage = xp.clip(c.resistance * iq + c.back_emf * velocity,
                          -c.max_voltage, c.max_voltage)
        iq = xp.clip((voltage - c.back_emf * velocity) / c.resistance,
                     -c.max_current, c.max_current)
        return c.torque_constant * iq
