"""Recompute IST inertials from the immutable CAD baseline and material settings.

Run with --write to update URDF, MJCF and material_inertials.json. Without --write,
only print the calculation. Requires numpy (no mesh/CAD runtime dependency).
"""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent


def read_triangles(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], 'little')
    if len(raw) != 84 + 50 * count:
        raise ValueError(f'Expected binary STL: {path}')
    dtype = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)['v'].astype(float)


def membrane_triangles(triangles, plane_abs_x):
    """Select each two-sided membrane triangle once, excluding solid contacts.

    In this CAD export the membrane is the paired planar surface at x=+/-0.35 mm.
    Other reversed face pairs are internal solid interfaces, not extra film.
    Mesh hashes below guard this model-specific selection against geometry edits.
    """
    groups = defaultdict(list)
    for triangle in triangles:
        groups[tuple(sorted(map(tuple, triangle)))].append(triangle)
    selected = []
    for faces in groups.values():
        if len(faces) != 2:
            continue
        a, b = faces
        if np.ptp(a[:, 0]) > 1e-12 or abs(abs(a[0, 0]) - plane_abs_x) > 1e-9:
            continue
        na = np.cross(a[1]-a[0], a[2]-a[0])
        nb = np.cross(b[1]-b[0], b[2]-b[0])
        if np.dot(na, nb) >= 0:
            raise ValueError('Membrane duplicate faces must have opposite winding')
        selected.append(a)
    if not selected:
        raise ValueError('No two-sided membrane found')
    return np.asarray(selected)


def film_properties(triangles, density, thickness):
    """Exact triangular lamina integrals plus symmetric finite-thickness inertia."""
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)/2
    if np.any(area <= 0) or density <= 0 or thickness <= 0:
        raise ValueError('Film requires positive triangle areas, density and thickness')
    weights = area*density*thickness
    mass = weights.sum()
    com = np.einsum('n,ni->i', weights, triangles.mean(axis=1))/mass
    centered = triangles-com
    sums = centered.sum(axis=1)
    # E[rr^T] = (sum(v_i v_i^T) + sum(v_i)sum(v_i)^T)/12.
    second = (np.einsum('nvi,nvj->nij', centered, centered)
              + np.einsum('ni,nj->nij', sums, sums))/12
    normals = cross/(2*area[:, None])
    second += thickness**2/12*np.einsum('ni,nj->nij', normals, normals)
    second = np.einsum('n,nij->ij', weights, second)
    inertia = np.trace(second)*np.eye(3)-second
    return float(mass), com, inertia, float(area.sum())


def combine(parts):
    mass = sum(m for m, _, _ in parts)
    com = sum(m*c for m, c, _ in parts)/mass
    inertia = np.zeros((3, 3))
    for m, c, tensor in parts:
        d = c-com
        inertia += tensor + m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    return mass, com, inertia


def calculate(root=ROOT):
    settings = json.loads((root/'materials.json').read_text())
    baseline = json.loads((root/'cad_inertials_original.json').read_text())
    for key in ('peek_density_kg_m3', 'pet_frame_density_kg_m3',
                'kapton_density_kg_m3', 'kapton_thickness_m'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    results = {}
    for name, original in baseline['links'].items():
        if name == 'base_link':
            continue
        mesh_path = root/'meshes'/f'{name}.stl'
        if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != original['mesh_sha256']:
            raise ValueError(f'{name}: geometry changed; regenerate CAD baseline first')
        wing = name in ('left_wing', 'right_wing')
        density = settings['pet_frame_density_kg_m3' if wing else 'peek_density_kg_m3']
        scale = density/baseline['original_density_kg_m3']
        solid_mass = original['mass_kg']*scale
        mass, com, inertia = solid_mass, np.array(original['com_m']), np.array(original['inertia_kg_m2'])*scale
        extra = {}
        if wing:
            triangles = membrane_triangles(read_triangles(mesh_path), settings['membrane_plane_abs_x_m'])
            film_m, film_c, film_i, area = film_properties(
                triangles, settings['kapton_density_kg_m3'], settings['kapton_thickness_m'])
            mass, com, inertia = combine([(mass, com, inertia), (film_m, film_c, film_i)])
            extra = dict(frame_mass_kg=solid_mass, membrane_mass_kg=film_m,
                         membrane_area_m2=area, membrane_com_m=film_c.tolist(),
                         membrane_unique_triangles=len(triangles))
        principal = np.linalg.eigvalsh(inertia)
        if not np.isfinite(inertia).all() or principal[0] <= 0 or principal[2] > principal[:2].sum():
            raise ValueError(f'Nonphysical inertia for {name}')
        results[name] = dict(material='PET frame + Kapton film' if wing else 'PEEK',
                             old_mass_kg=original['mass_kg'], mass_kg=float(mass),
                             com_m=com.tolist(), inertia_kg_m2=inertia.tolist(), **extra)
    return dict(settings=settings, solid_geometry_basis=baseline['density_basis'], links=results)


def update_models(report, root=ROOT):
    """Replace only moving-link inertial elements; preserve all other XML bytes."""
    files = [(root/'urdf/IST_robot.urdf', 'link'), (root/'urdf/IST_robot.xml', 'body')]
    updated = []
    fmt = lambda values: ' '.join(f'{x:.15g}' for x in values)
    for path, tag in files:
        text = path.read_text()
        for name, result in report['links'].items():
            c, m = fmt(result['com_m']), f"{result['mass_kg']:.15g}"
            i = np.array(result['inertia_kg_m2'])
            if tag == 'link':
                element = (f'<inertial>\n      <origin xyz="{c}" rpy="0 0 0" />\n'
                           f'      <mass value="{m}" />\n'
                           f'      <inertia ixx="{i[0,0]:.15g}" ixy="{i[0,1]:.15g}" ixz="{i[0,2]:.15g}" '
                           f'iyy="{i[1,1]:.15g}" iyz="{i[1,2]:.15g}" izz="{i[2,2]:.15g}" />\n    </inertial>')
                ending = r'<inertial>.*?</inertial>'
            else:
                values = fmt([*i.diagonal(), i[0,1], i[0,2], i[1,2]])
                element = f'<inertial pos="{c}" mass="{m}" fullinertia="{values}" />'
                ending = r'<inertial\b[^>]*/>'
            pattern = rf'(<{tag}\b[^>]*\bname="{re.escape(name)}"[^>]*>\s*){ending}'
            text, count = re.subn(pattern, lambda match: match[1]+element, text, flags=re.S)
            if count != 1:
                raise ValueError(f'Expected one inertial element for {name} in {path}')
        ET.fromstring(text)
        updated.append((path, text))
    for path, text in updated:
        path.write_text(text)
    (root/'material_inertials.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    report = calculate()
    if args.write:
        update_models(report)
    for name, result in report['links'].items():
        print(f"{name:16s} {result['old_mass_kg']*1e6:10.5f} -> {result['mass_kg']*1e6:10.5f} mg")
