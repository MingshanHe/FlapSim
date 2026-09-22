"""Flapping-wing aerodynamics and MuJoCo integration."""

from .aero import (
    aero_step,
    center_of_pressure_fraction,
    load_wing_parameters,
    normal_force_coefficient,
    save_wing_parameters,
    wing_aero,
    world_wrench,
)
from .base import FluidModel, BaseFluid, NoFluid
from .blade_element import BladeElement
from .mujoco import MuJoCoFluid


def create_fluid(config):
    return {'blade_element': BladeElement, 'mujoco': MuJoCoFluid,
            'none': NoFluid}[config.kind](config)

__all__ = [
    "FluidModel", "BaseFluid", "NoFluid", "BladeElement", "MuJoCoFluid", "create_fluid",
    "aero_step",
    "center_of_pressure_fraction",
    "load_wing_parameters",
    "normal_force_coefficient",
    "save_wing_parameters",
    "wing_aero",
    "world_wrench",
]
