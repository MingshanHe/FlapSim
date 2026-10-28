"""Wing mass models declared in wing coordinates land correctly on each body."""

from __future__ import annotations

import numpy as np
import pytest

from flapsim.robot import ManifestError, RobotManifest, builtin_robots, load


def _manifest_with(mass_model):
    manifest = RobotManifest.from_file(builtin_robots()["SingleThrustFlapper"])
    data = manifest.to_dict()
    for wing in data["wings"]:
        wing["mass_model"] = mass_model
    return RobotManifest.from_dict(data, base_dir=manifest.base_dir)


def test_example_robot_declares_a_spar_plate_mass_model():
    robot = load("SingleThrustFlapper")
    left, right = robot.wings
    assert left.spec.mass_model.type == "spar_plate"
    assert robot.model.body("left_wing").mass[0] == pytest.approx(6.26e-5)
    assert robot.model.body("right_wing").mass[0] == pytest.approx(6.26e-5)
    # the centre of mass sits near the leading edge (2 mm aft of a 20 mm chord)
    # and mid span, on the chord side of the hinge for both wings
    for wing in (left, right):
        com_body = robot.model.body_ipos[wing.body_id]
        com_aero = wing.aero_from_body.T @ com_body
        assert com_aero[2] == pytest.approx(-0.002)  # -Z is the chord side
        assert abs(com_aero[1]) == pytest.approx(0.035)
        assert np.sign(com_aero[1]) == wing.side
    assert robot.geometry_warnings() == []


def test_thin_plate_and_explicit_models():
    thin = load(_manifest_with({"type": "thin_plate", "mass": 1.0e-4}))
    wing = thin.wings[0]
    com_aero = wing.aero_from_body.T @ thin.model.body_ipos[wing.body_id]
    assert com_aero[2] == pytest.approx(-0.01)
    inertia = thin.model.body_inertia[wing.body_id]
    # principal inertias of a 70 x 20 mm plate of 0.1 g with 1 % thickness
    t2 = (0.01 * 0.02) ** 2
    expected = sorted(
        [
            1e-4 * (0.07**2 + 0.02**2) / 12,
            1e-4 * (0.02**2 + t2) / 12,
            1e-4 * (0.07**2 + t2) / 12,
        ]
    )
    np.testing.assert_allclose(sorted(inertia), expected, rtol=1e-9)

    explicit = load(
        _manifest_with(
            {
                "type": "explicit",
                "mass": 5.0e-5,
                "com": [0.0, 0.03, 0.004],
                "diaginertia": [2.6e-8, 2e-9, 2.5e-8],
            }
        )
    )
    wing = explicit.wings[1]
    com_aero = wing.aero_from_body.T @ explicit.model.body_ipos[wing.body_id]
    np.testing.assert_allclose(com_aero, [0.0, wing.side * 0.03, -0.004], atol=1e-12)


def test_mass_model_validation():
    with pytest.raises(ManifestError, match="explicit mass needs"):
        _manifest_with({"type": "explicit", "mass": 1e-4})
    with pytest.raises(ManifestError, match="takes no com"):
        _manifest_with({"type": "thin_plate", "mass": 1e-4, "com": [0, 0, 0]})
    with pytest.raises(ManifestError, match="spar_fraction must lie"):
        _manifest_with({"type": "spar_plate", "mass": 1e-4, "spar_fraction": 1.0})
    with pytest.raises(ManifestError, match="must be positive"):
        _manifest_with({"type": "spar_plate", "mass": -1.0})
    with pytest.raises(ManifestError, match="triangle inequality"):
        _manifest_with(
            {
                "type": "explicit",
                "mass": 1e-4,
                "com": [0, 0.03, 0.004],
                "diaginertia": [3e-8, 2e-9, 2.5e-8],
            }
        )
