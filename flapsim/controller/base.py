"""Controller interface: commands + state -> actuator angular velocities."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SixDOFCommand:
    """World xyz [m], ZYX roll/pitch/yaw [rad], world linear/angular velocity."""
    position: tuple = (0.0, 0.0, 0.5)
    orientation: tuple = (0.0, 0.0, 0.0)
    linear_velocity: tuple = (0.0, 0.0, 0.0)
    angular_velocity: tuple = (0.0, 0.0, 0.0)

    def as_array(self):
        parts = [np.asarray(v, dtype=float) for v in
                 (self.position, self.orientation, self.linear_velocity, self.angular_velocity)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in parts):
            raise ValueError('each command component must have three finite values')
        return np.concatenate(parts)


class Controller(ABC):
    def __init__(self, config):
        self.config = config

    def bind(self, model, actuator, root_body):
        self.body_id = model.body(root_body).id
        self.root_id = int(model.body_rootid[self.body_id])
        self.mass = float(model.body_subtreemass[self.body_id])
        self.gravity = model.opt.gravity.copy()
        self.dt = float(model.opt.timestep)
        self.actuator = actuator

    def initial_state(self, xp=np):
        return xp.zeros(6)

    @abstractmethod
    def command(self):
        """Initial command array; updated through Simulator.set_command."""

    @abstractmethod
    def update(self, data, state, command, xp=np):
        """Return (shaft velocities [rad/s], new controller state), without mutation."""


BaseController = Controller
