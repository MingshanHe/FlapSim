"""Smooth periodic references in world coordinates, with analytic derivatives."""
from dataclasses import dataclass
import math
import numpy as np


NAMES = ('hover', 'circle', 'figure8', 'ellipse', 'line', 'helix', 'lissajous')


@dataclass(frozen=True)
class Trajectory:
    name: str = 'circle'
    center: tuple = (0., 0., .8)
    radius: float = .2
    period: float = 8.
    height: float = .1
    phase: float = 0.

    def __post_init__(self):
        if self.name not in NAMES:
            raise ValueError(f'unknown trajectory {self.name!r}; choose from {NAMES}')
        if np.shape(self.center) != (3,) or not np.isfinite(self.center).all():
            raise ValueError('center must be three finite world coordinates')
        if (not all(math.isfinite(v) for v in (self.radius, self.period, self.height, self.phase))
                or self.radius <= 0 or self.period <= 0 or self.height < 0):
            raise ValueError('radius/period must be positive; height must be nonnegative')

    def sample(self, time):
        """Return position [m], velocity [m/s], acceleration [m/s²].

        Helix is a closed up/down helix, avoiding an altitude jump on reset.
        All paths have continuous position, velocity and acceleration.
        """
        w = 2 * np.pi / self.period
        theta = w * time + self.phase
        r, h = self.radius, self.height
        if self.name == 'hover':
            amplitudes, harmonics, offsets = (0, 0, 0), (1, 1, 1), (0, 0, 0)
        elif self.name == 'circle':
            amplitudes, harmonics, offsets = (r, r, 0), (1, 1, 1), (np.pi/2, 0, 0)
        elif self.name == 'figure8':
            amplitudes, harmonics, offsets = (r, r/2, 0), (1, 2, 1), (0, 0, 0)
        elif self.name == 'ellipse':
            amplitudes, harmonics, offsets = (1.2*r, .7*r, 0), (1, 1, 1), (np.pi/2, 0, 0)
        elif self.name == 'line':
            amplitudes, harmonics, offsets = (r, 0, 0), (1, 1, 1), (0, 0, 0)
        elif self.name == 'helix':
            amplitudes, harmonics, offsets = (r, r, h), (2, 2, 1), (np.pi/2, 0, 0)
        else:
            amplitudes, harmonics, offsets = (r, r, h), (1, 2, 3), (np.pi/2, 0, 0)
        a, k = np.array(amplitudes), np.array(harmonics)
        angle = k * theta + offsets
        position = np.array(self.center) + a * np.sin(angle)
        velocity = a * k * w * np.cos(angle)
        acceleration = -a * (k*w)**2 * np.sin(angle)
        return position, velocity, acceleration
