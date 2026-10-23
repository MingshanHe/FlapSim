from .base import Controller, BaseController, SixDOFCommand
from .speed import SpeedController
from .move import MoveController
from .hovering import HoveringController
from .allocation import LinearAllocator


def create_controller(config):
    return {'speed': SpeedController, 'hover': HoveringController,
            'move': MoveController}[config.kind](config)


__all__ = ['Controller', 'BaseController', 'SixDOFCommand', 'SpeedController',
           'MoveController', 'HoveringController', 'LinearAllocator', 'create_controller']
