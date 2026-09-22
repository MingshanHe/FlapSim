"""Interchangeable joint actuators."""
from .base import Actuator, BaseActuator
from .dcmotor import DCMotor
from .bldcmotor import BLDCMotor
from .dea import DEA


def create_actuator(config):
    return {'dcmotor': DCMotor, 'bldcmotor': BLDCMotor, 'dea': DEA}[config.kind](config)


__all__ = ['Actuator', 'BaseActuator', 'DCMotor', 'BLDCMotor', 'DEA', 'create_actuator']
