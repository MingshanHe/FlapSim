"""Array-only helpers shared by native MuJoCo and MJX."""
import numpy as np


def point_velocity(data, body_id, root_id, points, xp=np):
    """World velocity at points; cvel is based at the root subtree COM."""
    angular = data.cvel[body_id, :3]
    return data.cvel[body_id, 3:] + xp.cross(
        angular, points - data.subtree_com[root_id])


def euler_matrix(rpy, xp=np):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = xp.cos(r), xp.sin(r), xp.cos(p), xp.sin(p), xp.cos(y), xp.sin(y)
    return xp.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def rotation_error(current, desired, xp=np):
    """SO(3) attitude error in world axes (local controller, avoids yaw wrap)."""
    skew = desired @ current.T - current @ desired.T
    return .5 * xp.array([skew[2, 1], skew[0, 2], skew[1, 0]])
