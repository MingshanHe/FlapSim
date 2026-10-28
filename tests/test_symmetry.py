"""Left and right wings must produce mirrored forces.

The example robot's wings span the body x axis, so the mirror plane is x = 0:
Fx flips sign between the wings while Fy and Fz are equal.  The imported
meshes have slightly different local frames, hence the few-percent tolerance.
"""

from __future__ import annotations

import mujoco
import numpy as np

from flapsim import SimCore


def test_wing_forces_are_mirrored_every_step():
    sim = SimCore("SingleThrustFlapper", fix=True)
    env = sim.environment
    left = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_wing")
    right = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wing")
    while env.data.time < 0.1:
        env.step()
    mirror = np.array([-1.0, 1.0, 1.0])
    forces_left, forces_right = [], []
    end = env.data.time + 2.0 / 35.0
    while env.data.time < end:
        env.step()
        forces_left.append(env.data.xfrc_applied[left, :3].copy())
        forces_right.append(env.data.xfrc_applied[right, :3].copy())
    forces_left = np.array(forces_left)
    forces_right = np.array(forces_right)
    scale = np.abs(forces_left).max()
    assert scale > 0.0
    np.testing.assert_allclose(forces_right, forces_left * mirror, atol=0.05 * scale)
    # The spanwise (x) force of the pair cancels on average.
    assert abs((forces_left + forces_right)[:, 0].mean()) < 0.01 * scale
