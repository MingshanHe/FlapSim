"""Local six-axis pose PID, followed by an explicit robot-specific allocator."""
import warnings
import numpy as np
from .base import Controller, SixDOFCommand
from .allocation import LinearAllocator
from flapsim.kinematics import point_velocity, euler_matrix, rotation_error


class MoveController(Controller):
    def __init__(self, config, allocator=None):
        super().__init__(config)
        self.allocator = allocator

    def bind(self, model, actuator, root_body):
        super().bind(model, actuator, root_body)
        if not np.any((model.jnt_bodyid == self.body_id) & (model.jnt_type == 0)):
            raise ValueError('hover/move control requires a floating base')
        if self.allocator is None:
            if self.config.allocation is None:
                raise ValueError('hover/move requires a calibrated allocation matrix or custom allocator; '
                                 'IST_robot has only one independent motor')
            self.allocator = LinearAllocator(self.config.allocation, self.config.speed,
                                             actuator.config.max_velocity)
        if getattr(self.allocator, 'rank', 6) < 6:
            warnings.warn('Allocation rank < 6: six-axis commands are accepted, but the '
                          'mechanism cannot independently track every axis.', UserWarning)

    def command(self):
        return SixDOFCommand(self.config.target[:3], self.config.target[3:]).as_array()

    def update(self, data, state, command, xp=np):
        c, body = self.config, self.body_id
        rotation = data.xmat[body].reshape(3, 3)
        desired = euler_matrix(command[3:6], xp)
        error = xp.concatenate((command[:3] - data.xpos[body],
                                rotation_error(rotation, desired, xp)))
        velocity = xp.concatenate((point_velocity(data, body, self.root_id, data.xpos[body], xp),
                                   data.cvel[body, :3]))
        integral = xp.clip(state + self.dt * error, -c.integral_limit, c.integral_limit)
        correction = (xp.asarray(c.kp) * error + xp.asarray(c.ki) * integral
                      + xp.asarray(c.kd) * (command[6:] - velocity))
        # Allocation is linearized around trim speed; compensate only the change
        # in body-frame gravity from a level hover (trim already balances mg).
        gravity_world = -self.mass * xp.asarray(self.gravity)
        force_body = rotation.T @ (correction[:3] + gravity_world) - gravity_world
        wrench = xp.concatenate((force_body, rotation.T @ correction[3:]))
        return self.allocator.allocate(wrench, xp), integral
