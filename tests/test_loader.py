"""Loading robots from built-in names, manifests, bare model files and MjSpec."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from flapsim.robot import ManifestError, builtin_robots, load, resolve_manifest

URDF = (
    builtin_robots()["SingleThrustFlapper"].parent / "urdf" / "SingleThrustFlapper.urdf"
)


def test_builtin_names_and_aliases():
    robots = builtin_robots()
    assert "single_thrust_flapper" in robots
    assert robots["SingleThrustFlapper"] == robots["SingleThrustRobot"]
    assert resolve_manifest("SingleThrustFlapper").name == "SingleThrustFlapper"
    with pytest.raises(ManifestError, match="Built-in robots"):
        resolve_manifest("NoSuchRobot")


def test_load_builtin_fixed_and_free():
    fixed = load("SingleThrustFlapper")
    assert fixed.fixed_base
    assert fixed.model.nq == 6
    assert [w.name for w in fixed.wings] == ["left", "right"]
    assert fixed.torso_body_name == "link2"
    assert fixed.root_body_name == "base_link"
    assert fixed.model.opt.timestep == pytest.approx(5e-5)
    assert fixed.model.body(fixed.root_body).mass[0] == pytest.approx(0.019)

    free = load("SingleThrustFlapper", base="free")
    assert not free.fixed_base
    assert free.model.nq == 6 + 7
    # the cycle-averaged mass model lumps 30 g into the root body; the wings
    # keep their declared mass model, every other child body is a placeholder
    assert free.model.body(free.root_body).mass[0] == pytest.approx(0.03)
    assert free.model.body("left_arm").mass[0] == pytest.approx(1e-9)
    assert free.model.body("left_wing").mass[0] == pytest.approx(6.26e-5)
    assert (
        mujoco.mj_name2id(free.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base") >= 0
    )


def test_patches_applied_to_compiled_model():
    robot = load("SingleThrustFlapper")
    model = robot.model
    joint = model.joint("left_wing_feathering")
    assert model.jnt_stiffness[joint.id] == pytest.approx(2e-7)
    assert model.dof_armature[model.jnt_dofadr[joint.id]] == pytest.approx(1e-10)
    assert model.neq == 1 and model.equality(0).name == "mirror_feathering"
    assert model.nexclude == 6 * 7 // 2


def test_load_bare_urdf_without_wings():
    robot = load(URDF, position=(0.0, 0.0, 1.0))
    assert robot.wings == []
    assert robot.manifest.name == "SingleThrustFlapper"
    np.testing.assert_allclose(robot.data.xpos[robot.root_body], [0.0, 0.0, 1.0])


def test_load_mjcf_and_mjspec(tmp_path):
    mjcf = tmp_path / "pendulum.xml"
    mjcf.write_text(
        """<mujoco model="pendulum"><worldbody>
             <body name="base" pos="0 0 1"><geom type="sphere" size="0.02" mass="0.1"/>
               <body name="arm"><joint name="hinge" axis="0 1 0"/>
                 <geom type="capsule" size="0.01 0.1" mass="0.1"/></body>
             </body>
           </worldbody></mujoco>"""
    )
    robot = load(mjcf, scene=False)
    assert robot.manifest.name == "pendulum"
    assert robot.model.njnt == 1

    spec = mujoco.MjSpec.from_file(str(mjcf))
    robot2 = load(spec, base="free", timestep=1e-3)
    assert robot2.model.opt.timestep == pytest.approx(1e-3)
    assert robot2.model.njnt == 2  # free joint added

    jointed_root = tmp_path / "jointed.xml"
    jointed_root.write_text(
        """<mujoco><worldbody><body name="arm"><joint name="h"/>
             <geom type="sphere" size="0.02" mass="0.1"/></body></worldbody></mujoco>"""
    )
    with pytest.raises(ManifestError, match="already has joints"):
        load(jointed_root, base="free", scene=False)


def test_wing_frames_match_the_validated_adapter_convention():
    robot = load("SingleThrustFlapper")
    left, right = robot.wings
    # columns: X (normal) = +x, Y (span) = -z, Z = -chord = +y in the left_wing frame
    expected_left = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(left.aero_from_body, expected_left, atol=1e-12)
    world_left = robot.data.xmat[left.body_id].reshape(3, 3) @ left.aero_from_body
    world_right = robot.data.xmat[right.body_id].reshape(3, 3) @ right.aero_from_body
    np.testing.assert_allclose(world_left, world_right, atol=1e-12)
    state = right.state(robot.data)
    assert state.side == -1.0
    assert right.params.R == pytest.approx(0.07)
    assert right.params.d0 == pytest.approx(0.01395)


def test_geometry_checks_flag_reversed_signs(tmp_path):
    import warnings

    from flapsim.robot import RobotManifest

    manifest = RobotManifest.from_file(builtin_robots()["SingleThrustFlapper"])
    assert load(manifest, check=True).geometry_warnings() == []

    # a reversed stroke sign on the left wing
    data = manifest.to_dict()
    data["wings"][0]["stroke_joint"]["sign"] = -1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load(RobotManifest.from_dict(data, base_dir=manifest.base_dir))
    messages = " ".join(str(w.message) for w in caught)
    assert "wing 'left': positive stroke_joint rate" in messages

    # a reversed chord with the URDF's own centroid (no declared mass model):
    # the centre of mass lands on the +Z side and the stroke axis flips too
    data = manifest.to_dict()
    data["wings"][0]["frame"]["chord"] = [0.0, 1.0, 0.0]
    for wing in data["wings"]:
        wing.pop("mass_model", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        robot = load(RobotManifest.from_dict(data, base_dir=manifest.base_dir))
    messages = " ".join(str(w.message) for w in caught)
    assert "'chord' probably points the wrong way" in messages
    assert "stroke_joint" in messages
    assert len(robot.geometry_warnings()) >= 2


def test_unknown_wing_body_is_reported(tmp_path):
    bad = tmp_path / "robot.yaml"
    bad.write_text(
        f"""name: bad
model: {{file: "{URDF}"}}
wings:
  - name: w
    body: nope
    frame: {{span: [0, 1, 0], chord: [1, 0, 0]}}
    planform: {{profile: rectangular, length: 0.1, mean_chord: 0.02}}
"""
    )
    with pytest.raises(ManifestError, match="wing body 'nope' not found"):
        load(bad)
