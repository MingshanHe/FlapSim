"""Quasi-static brushed DC motor with voltage, back-EMF and current limits."""
import numpy as np
from .base import Actuator


class DCMotor(Actuator):
    def voltage_for_speed(self, velocity, target, xp=np):
        """Requested terminal voltage after the speed loop and supply saturation."""
        c = self.config
        desired = c.speed_gain * (target - velocity)
        current = desired / c.torque_constant + c.no_load_current * xp.tanh(velocity)
        return xp.clip(c.resistance * current + c.back_emf * velocity,
                       -c.max_voltage, c.max_voltage)

    def torque_from_voltage(self, velocity, voltage, xp=np):
        """Direct terminal-voltage drive for comparison with bench measurements.

        I0 represents approximately constant internal loss torque, smoothed
        near zero speed. Kt and Ke use SI units (N.m/A and V.s/rad).
        """
        c = self.config
        voltage = xp.clip(voltage, -c.max_voltage, c.max_voltage)
        current = xp.clip((voltage-c.back_emf*velocity)/c.resistance,
                          -c.max_current, c.max_current)
        return c.torque_constant * (current - c.no_load_current*xp.tanh(velocity))

    def torque(self, position, velocity, target, xp=np):
        return self.torque_from_voltage(velocity, self.voltage_for_speed(velocity, target, xp), xp)
