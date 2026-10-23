"""Extract the curved wing's projected outline for blade-element presets.

The CAD membrane is planar; 'curved' refers to its curved outline. Slice the
whole wing, including frame, in its local YZ plane. At each span station the
outer chord envelope is used, treating internal frame/film seams as covered.
This provides geometry, not measured aerodynamic coefficients.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from recalculate_inertials import read_triangles

ROOT = Path(__file__).resolve().parent


def save_json(path, values):
    # Keep numeric vectors on one line so the generated presets remain reviewable.
    text = json.dumps(values, indent=2)
    number = r'(-?\d[\d.eE+-]*)'
    pattern = r'\[\n\s*' + number + r',\n\s*' + number + r',\n\s*' + number + r'\n\s*\]'
    text = re.sub(pattern, lambda match: '['+', '.join(match.groups())+']', text)
    path.write_text(text+'\n')


def outline(triangles, span, fractions):
    a = triangles.reshape(-1, 3)
    b = np.roll(triangles, -1, axis=1).reshape(-1, 3)
    rows = []
    for fraction in fractions:
        radius = min(fraction*span, span-1e-10)
        selected = ((a[:, 1] <= radius) & (b[:, 1] > radius)
                    | (b[:, 1] <= radius) & (a[:, 1] > radius))
        aa, bb = a[selected], b[selected]
        z = aa[:, 2] + (radius-aa[:, 1])/(bb[:, 1]-aa[:, 1])*(bb[:, 2]-aa[:, 2])
        if not len(z):
            raise ValueError(f'empty projected wing section at {radius}')
        rows.append([float(fraction), float(z.min()), float(z.max())])
    return rows


def main():
    wings, geometry = [], {}
    for name in ('left_wing', 'right_wing'):
        path = ROOT/'meshes'/f'{name}.stl'
        triangles = read_triangles(path)
        span = float(triangles[:, :, 1].max())
        profile = outline(triangles, span, np.linspace(0, 1, 97))
        table = np.array(profile)
        area = float(np.sum(np.diff(table[:, 0]) * span
                     * ((table[1:, 2]-table[1:, 1])+(table[:-1, 2]-table[:-1, 1]))/2))
        wings.append({'body': name, 'span': span, 'chord': area/span, 'planform': profile})
        geometry[name] = {'mesh_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'projected_area_m2': area, 'span_m': span}
    fluid = {'kind': 'blade_element', 'strips': 24, 'wings': wings}
    report = {'source': 'curved wing STL YZ outline; positive Y from wing root',
              'assumption': 'outer envelope includes membrane and frame; local force at mid-chord',
              'geometry': geometry, 'fluid': fluid}
    save_json(ROOT/'curved_wing_aerodynamics.json', report)
    config = ROOT.parents[1]/'config'
    for path in (config/'robots/IST_robot.json', config/'robots/IST_robot-V2.json',
                 config/'training/ist_v2_flight.json'):
        values = json.loads(path.read_text())
        values['fluid'] = fluid
        save_json(path, values)
    print(json.dumps(geometry, indent=2))


if __name__ == '__main__':
    main()
