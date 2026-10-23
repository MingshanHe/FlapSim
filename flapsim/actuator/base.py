"""Angular-velocity commands become joint torques, never prescribed qpos."""
from abc import ABC, abstractmethod
import mujoco
import numpy as np


class Actuator(ABC):
    def __init__(self, config):
        self.config = config

    def bind(self, model):
        joints = [model.joint(name) for name in self.config.joints]
        if any(j.type[0] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('angular-velocity actuators require hinge joints')
        self.qpos_ids = np.array([j.qposadr[0] for j in joints], dtype=int)
        self.dof_ids = np.array([j.dofadr[0] for j in joints], dtype=int)
        self.mapping = np.eye(model.nv)[self.dof_ids]

    def forces(self, data, target, xp=np):
        c = self.config
        target = xp.clip(target, -c.max_velocity, c.max_velocity)
        torque = self.torque(data.qpos[self.qpos_ids], data.qvel[self.dof_ids], target, xp)
        torque = xp.clip(torque, -c.max_torque, c.max_torque)
        return torque @ xp.asarray(self.mapping)

    @abstractmethod
    def torque(self, position, velocity, target, xp=np):
        """Pure array operation; xp is numpy or jax.numpy. Returns N.m."""


BaseActuator = Actuator
