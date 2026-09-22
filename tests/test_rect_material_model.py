"""Geometry, assembly and mass consistency for the user-supplied rectangular STL."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT/'flapsim/robot/IST_robot'
sys.path[:0] = [str(ROBOT), str(ROOT/'experiments/fixbase_curved_real')]
from build_rectangular import rectangular_parts
from rect_material_experiment import config_for, OUTPUT
from rect_simulation import ProfileBladeElement
from flapsim import Simulator


def test_film_is_included_once_and_root_is_retained():
    _, _, meta = rectangular_parts()
    raw = trimesh.load_mesh(ROBOT/'meshes/Rectangular.STL')
    material = meta['materials']
    expected = (raw.volume*1e-9*material['pet_frame_density_kg_m3']
                + meta['membrane_area_m2']*meta['membrane_thickness_m']
                *(material['kapton_density_kg_m3']-material['pet_frame_density_kg_m3'])
                + meta['root_connector_mass_kg'])
    np.testing.assert_allclose(meta['mass_kg'], expected, rtol=1e-13)
    assert 7.9e-5 < meta['root_connector_mass_kg'] < 8e-5
    assert .00208 < meta['membrane_area_m2'] < .00209


def test_assembly_and_nonwing_inertials_preserved():
    original = ET.parse(ROBOT/'urdf/IST_robot.urdf').getroot()
    rectangular = ET.parse(ROBOT/'urdf/IST_robot_rectangular.urdf').getroot()
    for joint in original.findall('joint'):
        other = rectangular.find(f"joint[@name='{joint.get('name')}']")
        assert ET.tostring(joint) == ET.tostring(other)
    for link in original.findall('link'):
        if link.get('name') not in ('left_wing','right_wing'):
            other = rectangular.find(f"link[@name='{link.get('name')}']")
            assert ET.tostring(link) == ET.tostring(other)
    model = mujoco.MjModel.from_xml_path(str(ROBOT/'urdf/IST_robot_rectangular.xml'))
    left, right = model.body('left_wing'), model.body('right_wing')
    np.testing.assert_allclose(left.mass, right.mass)
    np.testing.assert_allclose(left.inertia, right.inertia)
    np.testing.assert_allclose(left.ipos*np.array([-1,1,1]), right.ipos)
    meta=json.loads((ROBOT/'rectangular_inertials.json').read_text())
    np.testing.assert_allclose(left.mass[0], meta['mass_kg'],rtol=1e-13)
    np.testing.assert_allclose(left.ipos, meta['com_m'],atol=1e-15)


def test_single_wing_aerodynamic_area_and_quadrature():
    config = config_for(strips=24)
    fluid = ProfileBladeElement(config.fluid, OUTPUT/'rect_profile.npz')
    sim = Simulator(config, fluid=fluid)
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wing') == -1
    assert sim.model.body('right_flapping').id > 0
    np.testing.assert_allclose(np.degrees(sim.model.joint('left_feathering_joint').range),[-26.1,26.1],atol=1e-8)
    meta=json.loads((ROBOT/'rectangular_inertials.json').read_text())
    area=fluid.profiles[0][-2].sum()
    assert abs(area/meta['membrane_area_m2']-1) < .01
    sim.set_speed(100.)
    sim.step(500)
    assert np.isfinite(sim.data.qpos).all() and not np.any(sim.data.warning.number)
