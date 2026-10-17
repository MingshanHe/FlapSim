"""Centroidal flight features, computed from the unmodified articulated model."""
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from .env import TrajectoryTrackingEnv


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class CentroidalTrackingEnv(TrajectoryTrackingEnv):
    def _observation(self):
        original = super()._observation()
        m, d = self.sim.model, self.sim.data
        angular_map = np.zeros((3, m.nv))
        mujoco.mj_angmomMat(m, d, angular_map, self.body)
        address = self.free_dof+3
        # The free angular coordinates are body local. This is the rigid-body
        # equivalent rate of total momentum, including all moving wing/gear links.
        equivalent_rate = np.linalg.solve(angular_map[:, address:address+3], angular_map@d.qvel)
        base_rotation = d.xmat[self.body].reshape(3, 3)
        unit_rotation = d.xmat[m.body('thrust_unit').id].reshape(3, 3)
        # IST's fixed 180-degree installation makes thrust-unit local -Z upward.
        thrust_axis = -unit_rotation[:, 2] @ base_rotation
        return np.r_[original, equivalent_rate, thrust_axis].astype(np.float32)
