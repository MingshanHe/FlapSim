"""Build a rectangular-wing URDF/MJCF from the supplied millimetre STL.

The STL already contains a 10 um film. Assign PET to the solid, then correct
the exposed film volume to Kapton density, without counting its volume twice.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from recalculate_inertials import combine, film_properties

ROOT = Path(__file__).resolve().parent


def tensor_values(tensor):
    return [*tensor.diagonal(), tensor[0, 1], tensor[0, 2], tensor[1, 2]]


def fmt(values):
    return ' '.join(f'{v:.15g}' for v in values)


def rectangular_parts(root=ROOT, retain_root=True):
    materials = json.loads((root/'materials.json').read_text())
    raw = trimesh.load_mesh(root/'meshes/Rectangular.STL')
    if not raw.is_volume:
        raise ValueError('Rectangular.STL must be a consistently oriented closed solid')
    thickness = materials['kapton_thickness_m']
    zmin = raw.bounds[0, 2]
    top = zmin+thickness*1000
    film_mask = np.all(np.abs(raw.triangles[:, :, 2]-top) < 1e-5, axis=1)
    film_mask &= raw.face_normals[:, 2] > .999
    if film_mask.sum() != 10:
        raise ValueError('Expected the 10 top film triangles in the supplied rectangular STL')
    # Shared 2 x 2 mm root feature is at STL (0..2,0..2), formerly STEP (-2..0,-2..0).
    # Keep the previous assembly datum: body=(STEP Z,STEP X,STEP Y)+(-.35,2.25,4) mm.
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    translation = np.array([-.00035-zmin*.001, .00025, .002])
    vertices = raw.vertices@rotation.T*.001+translation
    mesh = trimesh.Trimesh(vertices=vertices, faces=raw.faces.copy(), process=False)
    film_triangles = raw.triangles[film_mask]@rotation.T*.001+translation
    film_triangles[:, :, 0] -= thickness/2  # STL top surface -> film midplane
    pet, kapton = materials['pet_frame_density_kg_m3'], materials['kapton_density_kg_m3']
    film = film_properties(film_triangles, kapton, thickness)
    geometric_film = film_properties(film_triangles, 1., thickness)
    total_m, total_c, total_i = mesh.volume*pet, mesh.center_mass, mesh.moment_inertia*pet
    # Use raw moments for subtraction, then combine at the final COM.
    correction = (geometric_film[0]*(kapton-pet), geometric_film[1], geometric_film[2]*(kapton-pet))
    parts = [(total_m, total_c, total_i), correction]
    visuals = [mesh]
    root_mass = 0.
    if retain_root:
        old = trimesh.load_mesh(root/'meshes/left_wing.stl')
        candidates = [c for c in old.split(only_watertight=True)
                      if c.volume > 5e-8 and c.bounds[0, 1] < 1e-8 and c.bounds[1, 1] < .00501]
        if len(candidates) != 1:
            raise ValueError('Cannot uniquely identify the existing wing-root connector')
        connector = candidates[0]
        root_mass = connector.volume*pet
        parts.append((root_mass, connector.center_mass, connector.moment_inertia*pet))
        visuals.append(connector)
    mass, com, inertia = combine(parts)
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError('Invalid rectangular wing inertia')
    combined_mesh = trimesh.util.concatenate(visuals)
    metadata = dict(materials=materials, retain_original_PET_root=retain_root,
                    input_stl_units='mm', scale_to_m=.001,
                    source_to_body_rotation=rotation.tolist(), source_to_body_translation_m=translation.tolist(),
                    mounting_basis='Shared 2 x 2 mm root feature aligned with previous CAD assembly datum; not fitted to force data',
                    input_solid_volume_mm3=float(raw.volume),
                    frame_volume_mm3=float((mesh.volume-geometric_film[0])*1e9),
                    frame_mass_kg=float((mesh.volume-geometric_film[0])*pet),
                    root_connector_mass_kg=float(root_mass), membrane_mass_kg=film[0],
                    membrane_area_m2=film[3], membrane_thickness_m=thickness,
                    film_assignment='Exposed STL 10 um top layer is Kapton; remaining solid is PET. Hidden film under ribs is not separately resolved.',
                    mass_kg=float(mass), com_m=com.tolist(), inertia_kg_m2=inertia.tolist(),
                    source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                                   ['meshes/Curve.STL', 'meshes/Rectangular.STL', 'meshes/left_wing.stl',
                                    'materials.json', 'urdf/IST_robot.urdf', 'urdf/IST_robot.xml']})
    return combined_mesh, film_triangles, metadata


def build(root=ROOT, retain_root=True):
    mesh, film_triangles, meta = rectangular_parts(root, retain_root)
    urdf = ET.parse(root/'urdf/IST_robot.urdf')
    mjcf = ET.parse(root/'urdf/IST_robot.xml')
    urdf.getroot().set('name', 'IST_robot_rectangular')
    mjcf.getroot().set('model', 'IST_robot_rectangular')
    for tree in (urdf, mjcf):
        tree.getroot().insert(0, ET.Comment(' Rectangular STL wing: PET frame/root + existing 10 um Kapton layer. See ../rectangular_inertials.json. '))
    for side, sign in [('left', 1), ('right', -1)]:
        mirror = np.diag([sign, 1., 1.])
        m = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = mirror
        m.apply_transform(transform)
        filename = f'{side}_rectangular_wing.stl'
        m.export(root/'meshes'/filename)
        com = mirror@np.array(meta['com_m'])
        inertia = mirror@np.array(meta['inertia_kg_m2'])@mirror.T
        link = urdf.getroot().find(f"link[@name='{side}_wing']")
        link.find('inertial/origin').set('xyz', fmt(com))
        link.find('inertial/mass').set('value', f"{meta['mass_kg']:.15g}")
        for key, value in zip(('ixx','iyy','izz','ixy','ixz','iyz'), tensor_values(inertia)):
            link.find('inertial/inertia').set(key, f'{value:.15g}')
        for element in link.findall('.//mesh'):
            element.set('filename', '../meshes/'+filename)
        body = mjcf.getroot().find(f".//body[@name='{side}_wing']")
        body.find('inertial').attrib = dict(pos=fmt(com), mass=f"{meta['mass_kg']:.15g}", fullinertia=fmt(tensor_values(inertia)))
        mjcf.getroot().find(f".//mesh[@name='{side}_wing_mesh']").set('file', filename)
        air = body.find(f"geom[@name='{side}_wing_air']")
        points = film_triangles.reshape(-1, 3)@mirror.T
        lo, hi = points.min(0), points.max(0)
        size = (hi-lo)/2
        size[0] = meta['membrane_thickness_m']/2
        air.set('pos', fmt((lo+hi)/2))
        air.set('size', fmt(size))
    for tree, suffix in [(urdf,'urdf'), (mjcf,'xml')]:
        ET.indent(tree, space='  ')
        tree.write(root/f'urdf/IST_robot_rectangular.{suffix}', encoding='utf-8', xml_declaration=True)
    (root/'rectangular_inertials.json').write_text(json.dumps(meta, indent=2)+'\n')
    # Membrane-only projection for the experiment's blade-element quadrature.
    film_mesh = trimesh.Trimesh(vertices=film_triangles.reshape(-1, 3),
                               faces=np.arange(len(film_triangles)*3).reshape(-1, 3), process=False)
    film_mesh.export(root/'meshes/rectangular_membrane.stl')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-root', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(retain_root=not args.without_root), indent=2))
