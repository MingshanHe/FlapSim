"""Fluid forces use world-frame [Fx,Fy,Fz,Tx,Ty,Tz] about each body's COM."""
from abc import ABC, abstractmethod
import numpy as np


class FluidModel(ABC):
    def __init__(self, config):
        self.config = config

    def configure(self, spec):
        # Prevent built-in and blade-element forces from being counted twice.
        spec.option.density = 0
        spec.option.viscosity = 0
        spec.option.wind = self.config.wind

    def bind(self, model):
        self.nbody = model.nbody

    @abstractmethod
    def forces(self, data, xp=np):
        """Pure array calculation returning an (nbody, 6) array."""


class NoFluid(FluidModel):
    def forces(self, data, xp=np):
        return xp.zeros((self.nbody, 6))


BaseFluid = FluidModel
