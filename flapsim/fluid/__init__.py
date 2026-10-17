"""Flapping-wing aerodynamics and MuJoCo integration."""

from .base import FluidModel, BaseFluid, NoFluid
from .blade_element import BladeElement
from .mujoco import MuJoCoFluid


def create_fluid(config):
    return {'blade_element': BladeElement, 'mujoco': MuJoCoFluid,
            'none': NoFluid}[config.kind](config)

__all__ = [
    "FluidModel", "BaseFluid", "NoFluid", "BladeElement", "MuJoCoFluid", "create_fluid",
]
