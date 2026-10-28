"""Planform integrals from profiles, polylines, polygons and meshes."""

from __future__ import annotations

import math

import pytest

from flapsim.robot import builtin_robots
from flapsim.wings import (
    RECTANGULAR,
    from_chord_distribution,
    from_mesh,
    from_polygon,
    from_polyline,
    from_profile,
    wing_params,
)

MESH = builtin_robots()["SingleThrustFlapper"].parent / "meshes" / "left_wing.STL"


def test_rectangular_profile_matches_closed_form():
    assert from_profile("rectangular") == RECTANGULAR
    shifted = from_profile("rectangular", pitch_axis_fraction=0.5)
    assert shifted.z_rd == pytest.approx(2 * 0.5**4)
    assert shifted.r22 == RECTANGULAR.r22


def test_elliptic_profile_has_unit_mean_chord_and_inboard_loading():
    e = from_profile("elliptic")
    assert e.r00 == pytest.approx(1.0, abs=1e-6)
    # an elliptic planform concentrates area inboard: lower radial moments
    assert e.r22 < RECTANGULAR.r22
    assert e.r33 < RECTANGULAR.r33
    # closed form: integral (4/pi) sqrt(1 - r^2) r^2 dr over [0, 1] = 1/4
    assert e.r22 == pytest.approx(0.25, rel=1e-4)
    assert e.r11 == pytest.approx(4.0 / (3.0 * math.pi), rel=1e-4)


def test_polyline_recovers_rectangle_and_dimensions():
    length, chord, integrals = from_polyline([(0.0, 0.02), (0.07, 0.02)])
    assert length == pytest.approx(0.07)
    assert chord == pytest.approx(0.02)
    assert integrals.r11 == pytest.approx(0.5)
    assert integrals.z_cp2 == pytest.approx(1 / 3)
    tapered = from_polyline([(0.0, 0.03), (0.07, 0.01)])[2]
    assert tapered.r22 < RECTANGULAR.r22


def test_polygon_rectangle_hinged_at_leading_edge_is_exact():
    result = from_polygon(
        [(0, 0), (0.07, 0), (0.07, 0.02), (0, 0.02)], (0, 0), (1, 0), (0, 1)
    )
    assert result.length == pytest.approx(0.07)
    assert result.mean_chord == pytest.approx(0.02)
    assert result.area == pytest.approx(0.07 * 0.02)
    for name, value in RECTANGULAR.as_dict().items():
        assert getattr(result.integrals, name) == pytest.approx(value, abs=1e-9), name
    assert result.pitch_axis_exact


def test_polygon_rejects_bad_input():
    with pytest.raises(ValueError, match="perpendicular"):
        from_polygon([(0, 0), (1, 0), (1, 1)], (0, 0), (1, 0), (1, 1))
    with pytest.raises(ValueError, match="behind the hinge"):
        from_polygon([(-1, 0), (1, 0), (1, -1), (-1, -1)], (0, 0), (1, 0), (0, 1))


def test_chord_distribution_validation():
    with pytest.raises(ValueError, match="increasing"):
        from_chord_distribution([0, 0.5, 0.5, 1], [1, 1, 1, 1])
    with pytest.raises(ValueError, match="zero area"):
        from_chord_distribution([0, 1], [0, 0])


def test_mesh_outline_of_the_example_wing():
    planform = from_mesh(MESH, span_axis=(0, 0, -1), chord_axis=(0, -1, 0), bins=40)
    # the mesh includes the root spar, so it is longer than the 70 mm aero length
    assert 0.07 <= planform.length <= 0.12
    assert 0.005 < planform.mean_chord < 0.06
    assert planform.integrals.r00 == pytest.approx(1.0, abs=1e-6)
    assert planform.integrals.z_rd > 0.0
    assert len(planform.polyline) >= 10
    for r, c in planform.polyline:
        assert 0.0 <= r <= planform.length and c > 0.0


def test_leading_edge_offset_scales_rotational_damping():
    from flapsim.robot import Planform
    from flapsim.wings import resolve_planform

    plain = resolve_planform(
        Planform(profile="rectangular", length=0.07, mean_chord=0.02)
    )
    offset = resolve_planform(
        Planform(
            profile="rectangular",
            length=0.07,
            mean_chord=0.02,
            leading_edge_offset=0.02,
        )
    )
    assert plain[2].z_rd == pytest.approx(1.0)
    # chord spans [1, 2] chords from the hinge: 2^4 - 1^4 = 15
    assert offset[2].z_rd == pytest.approx(15.0)
    assert offset[2].r22 == plain[2].r22


def test_wing_params_assembly():
    params = wing_params(0.07, 0.02, RECTANGULAR, rho=1.1, d0=0.01, ds=0.02)
    assert params.R == 0.07 and params.c == 0.02
    assert params.rho == 1.1 and params.d0 == 0.01 and params.ds == 0.02
    assert params.r33 == RECTANGULAR.r33
