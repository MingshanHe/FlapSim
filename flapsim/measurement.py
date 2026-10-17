"""World-frame six-axis wrenches, referenced to an explicit body origin."""
import mujoco
import numpy as np


class FixedBaseWrenchProbe:
    """Read a rigid fixture's load without introducing a compliant weld.

    Order is [Fx, Fy, Fz, Mx, My, Mz], in N and N.m. All moments are about
    the root body origin; all components use world axes. ``fixture`` means
    robot ON fixture. ``support`` is the opposite force, fixture ON robot.
    Call mj_forward with current applied forces before read().
    """

    def __init__(self, model, body='base_link'):
        self.model = model
        self.body = model.body(body).id
        if model.body_parentid[self.body] != 0 or model.body_dofnum[self.body]:
            raise ValueError('probe requires a fixed, top-level body')
        self.bodies = np.flatnonzero(model.body_rootid == self.body)

    def sum_wrenches(self, data, wrenches, bodies=None):
        """Sum COM-based body wrenches at the fixture origin, preserving arms."""
        ids = self.bodies if bodies is None else np.asarray(bodies, dtype=int)
        forces = np.asarray(wrenches)[ids, :3]
        moments = np.asarray(wrenches)[ids, 3:]
        arms = data.xipos[ids] - data.xpos[self.body]
        return np.r_[forces.sum(axis=0), (moments + np.cross(arms, forces)).sum(axis=0)]

    def read(self, data, aerodynamic_wrenches):
        mujoco.mj_rnePostConstraint(self.model, data)
        support = self.interaction_wrench(data, self.body, data.xpos[self.body])
        gravity = np.zeros((self.model.nbody, 6))
        gravity[:, :3] = self.model.body_mass[:, None] * self.model.opt.gravity
        weight = self.sum_wrenches(data, gravity)
        fixture = -support
        return {
            'aero': self.sum_wrenches(data, aerodynamic_wrenches),
            'support': support,
            'fixture': fixture,
            'gravity': weight,
            'fixture_dynamic': fixture - weight,
        }

    def interaction_wrench(self, data, body, point):
        """Parent ON child, world axes, moment about point (after read/RNE)."""
        body = self.model.body(body).id if isinstance(body, str) else body
        internal = data.cfrc_int[body]
        force = internal[3:].copy()
        root = self.model.body_rootid[body]
        moment = internal[:3] + np.cross(data.subtree_com[root] - point, force)
        return np.r_[force, moment]
