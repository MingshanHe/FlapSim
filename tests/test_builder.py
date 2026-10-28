"""The parametric flat-plate flapper flies with the generic aero adapter."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from flapsim.fluid import MuJoCoAeroAdapter
from flapsim.robot import flat_plate_flapper, load


def _kinematic_stroke(robot, frequency: float, amplitude: float):
    joints = [
        mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in robot.manifest.extras["stroke_joints"]
    ]
    qpos = [int(robot.model.jnt_qposadr[j]) for j in joints]
    dof = [int(robot.model.jnt_dofadr[j]) for j in joints]
    omega = 2.0 * math.pi * frequency

    def apply(data):
        angle = omega * data.time
        for q, d in zip(qpos, dof, strict=True):
            data.qpos[q] = amplitude * math.sin(angle)
            data.qvel[d] = amplitude * omega * math.cos(angle)

    return apply


def test_flat_plate_flapper_builds_loads_and_lifts():
    spec, manifest = flat_plate_flapper()
    robot = load(spec, manifest=manifest)
    assert robot.manifest.name == "flat_plate_flapper"
    assert [w.name for w in robot.wings] == ["left", "right"]
    assert robot.model.njnt == 4
    assert float(np.sum(robot.model.body_mass)) == pytest.approx(0.03)

    aero = MuJoCoAeroAdapter(robot)
    stroke = _kinematic_stroke(robot, frequency=30.0, amplitude=1.2)
    model, data = robot.model, robot.data
    left, right = robot.wings
    lift, forces_left, forces_right = [], [], []
    while data.time < 0.3 + 10 / 30.0:
        stroke(data)
        mujoco.mj_forward(model, data)
        aero.apply_forces()
        if data.time >= 0.3:
            lift.append(aero.diagnostics["F_world"][2])
            forces_left.append(data.xfrc_applied[left.body_id, :3].copy())
            forces_right.append(data.xfrc_applied[right.body_id, :3].copy())
        mujoco.mj_step(model, data)
        stroke(data)
    mean_lift = float(np.mean(lift))
    assert mean_lift > 0.02, f"mean lift {mean_lift:.4f} N should be clearly positive"
    forces_left = np.array(forces_left)
    forces_right = np.array(forces_right)
    scale = np.abs(forces_left).max()
    # wings span x, so the mirror plane is x = 0
    np.testing.assert_allclose(
        forces_right, forces_left * np.array([-1.0, 1.0, 1.0]), atol=0.02 * scale
    )


def test_flat_plate_flapper_parameters_are_validated():
    with pytest.raises(ValueError, match="positive"):
        flat_plate_flapper(span=-1.0)
    with pytest.raises(ValueError, match="body_mass"):
        flat_plate_flapper(body_mass=1e-5)
    spec, manifest = flat_plate_flapper(span=0.1, chord=0.03)
    assert manifest.wings[0].planform.length == 0.1
    robot = load(spec, manifest=manifest, base="free")
    assert not robot.fixed_base
