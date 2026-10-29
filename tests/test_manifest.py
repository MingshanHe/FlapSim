"""Robot manifest parsing, validation and round trips."""

from __future__ import annotations

import pytest

from flapsim.robot import ManifestError, RobotManifest, builtin_robots

MINIMAL = {
    "name": "demo",
    "model": {"file": "demo.urdf"},
    "wings": [
        {
            "name": "left",
            "body": "lw",
            "side": "left",
            "frame": {"span": [1, 0, 0], "chord": [0, 0, -1]},
            "planform": {"profile": "rectangular", "length": 0.05, "mean_chord": 0.01},
        },
        {
            "name": "right",
            "body": "rw",
            "side": "right",
            "mirror_of": "left",
            "planform": {"profile": "rectangular", "length": 0.05, "mean_chord": 0.01},
        },
    ],
}


def test_example_manifest_loads_and_round_trips(tmp_path):
    manifest = RobotManifest.from_file(builtin_robots()["SingleThrustFlapper"])
    assert manifest.name == "SingleThrustFlapper"
    assert manifest.wing_names == ("left", "right")
    assert manifest.wing("right").mirror_of == "left"
    assert manifest.model.file.is_file()
    assert manifest.model.patches.exclude_self_collision
    assert "left_wing_feathering" in manifest.model.patches.joints
    assert manifest.model.patches.bodies["base_link"].when == "free"

    saved = manifest.save(tmp_path / "copy.yaml")
    again = RobotManifest.from_file(saved)
    assert again.to_dict()["wings"] == manifest.to_dict()["wings"]
    assert again.model.patches == manifest.model.patches

    as_json = manifest.save(tmp_path / "copy.json")
    assert RobotManifest.from_file(as_json).wing_names == manifest.wing_names


def test_minimal_manifest_from_dict(tmp_path):
    (tmp_path / "demo.urdf").write_text("<robot name='x'/>")
    manifest = RobotManifest.from_dict(MINIMAL, base_dir=tmp_path)
    assert manifest.model.file == (tmp_path / "demo.urdf").resolve()
    assert manifest.wing("left").side_sign == 1.0
    assert manifest.wing("right").side_sign == -1.0
    assert manifest.with_base("free").model.base.type == "free"
    assert manifest.model.base.type == "fixed"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(schema="flapsim.robot/9"), "unsupported manifest schema"),
        (lambda d: d.update(bogus=1), "unknown field"),
        (lambda d: d["wings"][0].pop("frame"), "needs a frame or mirror_of"),
        (lambda d: d["wings"][0]["frame"].update(span=[0, 0, 1]), "perpendicular"),
        (lambda d: d["wings"][0]["frame"].pop("chord"), "needs 'span' and 'chord'"),
        (
            lambda d: d["wings"][0]["frame"].update(normal=[1, 0, 0]),
            "normal must be perpendicular",
        ),
        (lambda d: d["wings"][1].update(mirror_of="right"), "must name another wing"),
        (lambda d: d["wings"][0].update(name="right"), "duplicate wing names"),
        (
            lambda d: d["wings"][0]["planform"].pop("length"),
            "needs length and mean_chord",
        ),
        (
            lambda d: d["wings"][0].update(stroke_joint={"name": "j", "sign": 2}),
            r"\+1 or -1",
        ),
        (lambda d: d["wings"][0].update(side="up"), "must be one of"),
        (lambda d: d["model"].update(base={"type": "floating"}), "must be one of"),
        (
            lambda d: d["model"].update(patches={"joints": {"j": {"stiffness": -1}}}),
            "non-negative",
        ),
        (
            lambda d: d["model"].update(patches={"equalities": [{"type": "joint"}]}),
            "needs 'targets'",
        ),
        (
            lambda d: d["model"].update(
                patches={"equalities": [{"type": "connect", "targets": ["a", "b"]}]}
            ),
            "needs an anchor",
        ),
    ],
)
def test_invalid_manifests_are_rejected_with_a_clear_message(mutate, message):
    import copy

    data = copy.deepcopy(MINIMAL)
    mutate(data)
    with pytest.raises(ManifestError, match=message):
        RobotManifest.from_dict(data)


def test_missing_model_file_is_reported(tmp_path):
    (tmp_path / "robot.yaml").write_text("name: x\nmodel: {file: nope.urdf}\n")
    with pytest.raises(ManifestError, match="does not exist"):
        RobotManifest.from_file(tmp_path / "robot.yaml")


def test_bare_model_manifest():
    path = (
        builtin_robots()["SingleThrustFlapper"].parent
        / "urdf"
        / "SingleThrustFlapper.urdf"
    )
    manifest = RobotManifest.for_model_file(path)
    assert manifest.wings == ()
    assert manifest.model.file == path.resolve()
