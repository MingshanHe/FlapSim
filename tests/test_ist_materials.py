"""Analytic film inertia and consistency of the material-corrected robot assets."""
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'flapsim/robot/IST_robot'
spec = importlib.util.spec_from_file_location('ist_inertials', ROOT/'recalculate_inertials.py')
calc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc)


def test_film_matches_offset_rectangular_slab():
    a, b, t, rho = .08, .03, 1e-5, 1420.
    triangles = np.array([[[0, 0, 0], [0, a, 0], [0, a, b]],
                          [[0, 0, 0], [0, a, b], [0, 0, b]]])
    offset = np.array([.004, -.03, .008])
    mass, com, inertia, area = calc.film_properties(triangles+offset, rho, t)
    np.testing.assert_allclose(mass, rho*a*b*t)
    np.testing.assert_allclose(com, offset+[0, a/2, b/2], atol=1e-15)
    np.testing.assert_allclose(area, a*b)
    expected = mass/12*np.diag([a*a+b*b, t*t+b*b, t*t+a*a])
    np.testing.assert_allclose(inertia, expected, atol=1e-22)


def test_material_assets_and_base_preservation(tmp_path):
    report = calc.calculate()
    baseline = json.loads((ROOT/'cad_inertials_original.json').read_text())['links']
    u = ET.parse(ROOT/'urdf/IST_robot.urdf').getroot()
    xml = ET.parse(ROOT/'urdf/IST_robot.xml').getroot()
    model = mujoco.MjModel.from_xml_path(str(ROOT/'urdf/IST_robot.xml'))
    for name in baseline:
        result = baseline[name] if name == 'base_link' else report['links'][name]
        inertial = u.find(f"link[@name='{name}']/inertial")
        a = inertial.find('inertia').attrib
        urdf_i = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])],
                           [float(a['ixy']), float(a['iyy']), float(a['iyz'])],
                           [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
        np.testing.assert_allclose(float(inertial.find('mass').get('value')), result['mass_kg'], rtol=1e-13)
        np.testing.assert_allclose(urdf_i, result['inertia_kg_m2'], rtol=1e-13, atol=1e-25)
        xml_i = xml.find(f".//body[@name='{name}']/inertial")
        np.testing.assert_allclose(
            np.fromstring(xml_i.get('fullinertia'), sep=' '),
            [*urdf_i.diagonal(), urdf_i[0,1], urdf_i[0,2], urdf_i[1,2]],
            rtol=1e-13, atol=1e-25)
        body = model.body(name)
        np.testing.assert_allclose(body.mass[0], result['mass_kg'], rtol=1e-13)
        np.testing.assert_allclose(body.ipos, result['com_m'], atol=1e-15)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        actual = rotation@np.diag(body.inertia)@rotation.T
        # MuJoCo's eigenvector solver is approximate (up to 8.2 ppm here);
        # check at tensor scale. The serialized tensor is checked above at 1e-13.
        expected = np.array(result['inertia_kg_m2'])
        assert np.linalg.norm(actual-expected) < 1e-5*np.linalg.norm(expected)
    for name in ('left_wing', 'right_wing'):
        wing = report['links'][name]
        assert wing['membrane_unique_triangles'] == 54
        np.testing.assert_allclose(wing['membrane_mass_kg'], 2.40070244349283e-5)
        assert wing['com_m'][1] > baseline[name]['com_m'][1]
    # Repeated writes must not compound scaling; no changes outside inertials.
    (tmp_path/'urdf').mkdir()
    for path in (ROOT/'urdf').iterdir():
        shutil.copyfile(path, tmp_path/'urdf'/path.name)
    calc.update_models(report, tmp_path)
    first = {p.name: p.read_bytes() for p in (tmp_path/'urdf').iterdir()}
    calc.update_models(report, tmp_path)
    for path in (tmp_path/'urdf').iterdir():
        assert path.read_bytes() == first[path.name] == (ROOT/'urdf'/path.name).read_bytes()
