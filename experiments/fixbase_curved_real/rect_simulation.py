"""Independent CAD/rectangular validation; no coefficients fitted to force data.

Run with --prepare to build geometry, or --run to measure all 12 cases.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import sys
import json
import hashlib
import argparse
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/fixbase')]
from single_wing_sweep import build_single_wing_model, single_config
from run_15hz import cycle_statistics
from inspect_planform import planform
from geometry_sensitivity import CADBladeElement
from flapsim.fluid.blade_element import BladeElement
from flapsim import Simulator
from flapsim.measurement import FixedBaseWrenchProbe
OUTPUT = FOLDER/'rect_validation'
GEOMETRY = OUTPUT/'geometry'
AXES = ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')


class ProfileBladeElement(CADBladeElement):
    """Use a saved spanwise profile with the existing CAD force law."""
    def __init__(self, config, profile):
        super().__init__(config, 'full_planform')
        self.profile = Path(profile)

    def bind(self, model):
        BladeElement.bind(self, model)
        p = np.load(self.profile)
        self.profiles = []
        for wing, body, root, points, selector in self.elements:
            radius = (np.arange(self.config.strips)+.5)*wing.span/self.config.strips
            width = np.interp(radius, p['radius'], p['width'])
            center = np.interp(radius, p['radius'], p['center'])
            local = (np.asarray(wing.origin) + radius[:, None]*np.asarray(wing.span_axis)
                     + center[:, None]*np.asarray(wing.chord_axis))
            dr = wing.span/self.config.strips
            self.profiles.append((wing, body, root, local, selector, width*dr, float((width**4).sum()*dr)))


def prepare():
    import trimesh
    from inspect_step import read_step, properties
    GEOMETRY.mkdir(parents=True, exist_ok=True)
    curve_path = build_single_wing_model(GEOMETRY/'curve')
    rect_path = build_single_wing_model(GEOMETRY/'rect')
    r, c, z, dr = planform(ROOT/'flapsim/robot/IST_robot/meshes/left_wing.stl')
    np.savez(GEOMETRY/'curve_profile.npz', radius=r, width=c, center=z)
    # Membrane dimensions supplied by the user. STEP frame reaches Y=28 mm;
    # place 26 mm film at Y=2..28 mm. Keep inferred original mounting offsets.
    span, chord, thickness = .080, .026, .000025
    np.savez(GEOMETRY/'rect_profile.npz', radius=np.array([0, span]),
             width=np.array([chord, chord]), center=np.array([.019, .019]))
    # Proper axis permutation: body (x,y,z) = STEP (z,x,y), all in metres.
    R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    offset = np.array([-.00035, .00225, .004])
    step = read_step(FOLDER/'0831_rect_capton/Rectangular.STEP')
    prop = properties(step)
    curve_prop = properties(read_step(FOLDER/'0831_curve_capton/Curve.STEP'))
    tree = ET.parse(rect_path)
    body = tree.getroot().find(".//body[@name='left_wing']")
    inertial = body.find('inertial')
    old_mass = float(inertial.get('mass'))
    curve_area = float(c.sum()*dr)
    density = old_mass/(curve_prop['volume_mm3']*1e-9+curve_area*thickness)
    frame_mass = prop['volume_mm3']*1e-9*density
    film_mass = span*chord*thickness*density
    frame_com = R@np.array(prop['com_mm'])*.001+offset
    film_com = np.array([0, .00225+span/2, .019])
    mass = frame_mass+film_mass
    com = (frame_mass*frame_com+film_mass*film_com)/mass
    frame_I = R@np.array(prop['inertia_mm5'])@R.T * density*1e-15
    dims = np.array([thickness, span, chord])
    film_I = np.diag(film_mass/12*(np.sum(dims**2)-dims**2))
    inertia = frame_I+film_I
    for m, p in ((frame_mass, frame_com), (film_mass, film_com)):
        d = p-com
        inertia += m*(np.dot(d, d)*np.eye(3)-np.outer(d, d))
    assert np.linalg.eigvalsh(inertia).min() > 0
    inertial.set('mass', str(mass))
    inertial.set('pos', ' '.join(map(str, com)))
    inertial.set('fullinertia', ' '.join(map(str, [*inertia.diagonal(), inertia[0,1], inertia[0,2], inertia[1,2]])))
    # Replace curved visual mesh and built-in aerodynamic proxy, too.
    meshes = [trimesh.load(p) for p in sorted(GEOMETRY.glob('rect_solid_*.stl'))]
    frame = trimesh.util.concatenate(meshes)
    frame.vertices = frame.vertices@R.T*.001+offset
    frame.export(GEOMETRY/'rect'/'rect_frame.stl')
    compiler = tree.getroot().find('compiler')
    original_meshdir = (rect_path.parent/compiler.get('meshdir')).resolve()
    for asset in tree.getroot().find('asset').findall('mesh'):
        asset.set('file', str((original_meshdir/asset.get('file')).resolve()))
    tree.getroot().find(".//mesh[@name='left_wing_mesh']").set('file', str(GEOMETRY/'rect'/'rect_frame.stl'))
    compiler.set('meshdir', '.')
    ET.SubElement(body, 'geom', name='rect_membrane_visual', type='box',
                  pos=' '.join(map(str, film_com)), size=f'{thickness/2} {span/2} {chord/2}',
                  mass='0', rgba='.3 .7 .95 .6', contype='0', conaffinity='0')
    air = body.find("geom[@name='left_wing_air']")
    air.set('pos', ' '.join(map(str, film_com)))
    air.set('size', f'{thickness/2} {span/2} {chord/2}')
    tree.getroot().set('model', 'IST_robot_single_rect_wing')
    ET.indent(tree, space='  ')
    tree.write(rect_path, encoding='utf-8', xml_declaration=True)
    meta = dict(rect_span_m=span, rect_chord_m=chord, rect_area_m2=span*chord,
                curve_area_m2=curve_area, rect_origin_m=[0,.00225,0],
                step_to_body_rotation=R.tolist(), step_to_body_translation_m=offset.tolist(),
                film_assumed_thickness_m=thickness, effective_density_kg_m3=density,
                density_basis='Existing curved wing mass / (Curve.STEP volume + curved membrane area * assumed film thickness)',
                rect_estimated_mass_kg=mass, rect_com_m=com.tolist(), rect_inertia_kg_m2=inertia.tolist(),
                mass_and_mounting_status='Estimated; film thickness, material distribution, and attachment datum unconfirmed',
                assumed_membrane_STEP_extent_mm=[[0,2],[80,28]],
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [FOLDER/'0831_curve_capton/Curve.STEP', FOLDER/'0831_rect_capton/Rectangular.STEP',
                                         ROOT/'flapsim/robot/IST_robot/meshes/left_wing.stl']})
    (GEOMETRY/'model_assumptions.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def config_for(kind, *, strips=12, mass_scale=1.):
    config = single_config(GEOMETRY/kind/'IST_robot_single_left_wing.xml')
    wing = config.fluid.wings[0]
    if kind == 'rect':
        wing = replace(wing, span=.080, chord=.026, origin=(0,.00225,0), chord_offset=.019)
    return replace(config, fluid=replace(config.fluid, strips=strips, wings=(wing,)))


def mean_between(t, values, start, end):
    inside = (t > start) & (t < end)
    points = np.r_[start, t[inside], end]
    return np.trapezoid(np.interp(points, t, values), points)/(end-start)


def run_case(kind, hz, target, *, strips=12, mass_scale=1., timestep=5e-6, label=None,
             config=None, profile_path=None, output=OUTPUT):
    out = output/'simulation'/(label or kind)/f'{hz}hz'
    out.mkdir(parents=True, exist_ok=True)
    config = config if config is not None else config_for(kind, strips=strips)
    profile_path = Path(profile_path) if profile_path is not None else GEOMETRY/f'{kind}_profile.npz'
    config = replace(config, scene=replace(config.scene, timestep=timestep))
    command = 2*np.pi*16*target+.35*target**2
    if label:
        prior_path = output/'simulation'/kind/f'{hz}hz'/'calibration.json'
        if prior_path.is_file():
            command = json.loads(prior_path.read_text())[-1]['command_rad_s']
    # Prior curve calibration seeds the speed command only; each case is measured.
    calibration = []
    for attempt in range(1, 6):
        sim = Simulator(config, fluid=ProfileBladeElement(config.fluid, profile_path))
        if mass_scale != 1:
            body = sim.model.body('left_wing').id
            sim.model.body_mass[body] *= mass_scale
            sim.model.body_inertia[body] *= mass_scale
            mujoco.mj_setConst(sim.model, sim.data)
            sim.reset()
        probe = FixedBaseWrenchProbe(sim.model)
        measurement = mujoco.MjData(sim.model)
        root = sim.model.body('base_link').id
        base = sim.data.xpos[root].copy()
        records = []
        for i in range(4001):
            if i:
                phase = min((i-1)/2000/.15, 1)
                speed = command*(.5-.5*np.cos(np.pi*phase))
                sim.set_speed(speed)
                sim.step(round(1/2000/timestep))
            else:
                speed = 0.
            mujoco.mj_copyData(measurement, sim.model, sim.data)
            aero = sim.fluid.forces(measurement)
            measurement.xfrc_applied[:] = aero
            measurement.qfrc_applied[:] = sim.actuator.forces(measurement, np.array([speed]))
            mujoco.mj_forward(sim.model, measurement)
            w = probe.read(measurement, aero)
            velocity = measurement.cvel[:,3:]+np.cross(
                measurement.cvel[:,:3], measurement.xipos-measurement.subtree_com[sim.model.body_rootid])
            momentum = np.sum(sim.model.body_mass[:,None]*velocity,axis=0)
            records.append([sim.data.time, sim.data.joint('motor_joint').qpos[0],
                            sim.data.joint('left_feathering_joint').qpos[0],
                            sim.data.joint('left_flapping_joint').qpos[0],
                            *w['aero'], *w['fixture_dynamic'], *momentum,
                            float(np.max(np.abs(sim.data.xpos[root]-base)))])
        a = np.asarray(records)
        assert np.isfinite(a).all() and not np.any(sim.data.warning.number)
        columns = ['time_s', 'motor_joint_angle_rad', 'pitch_rad', 'flap_rad']
        columns += [f'{g}_{axis}_{"N" if axis[0]=="F" else "Nm"}' for g in ('aero','fixture_dynamic') for axis in AXES]
        columns += ['momentum_x', 'momentum_y', 'momentum_z', 'base_pose_error']
        data = dict(zip(columns, a.T))
        stats = cycle_statistics(data, target_hz=target)
        np.savez_compressed(out/f'run_{attempt}.npz', **data)
        actual = stats['measured_wing_frequency_hz']
        calibration.append(dict(command_rad_s=float(command), actual_hz=actual, mean_lift_mN=stats['mean_lift_mN']))
        (out/'calibration.json').write_text(json.dumps(calibration, indent=2))
        print(kind, hz, 'attempt', attempt, 'Hz', round(actual,5), 'lift mN', round(stats['mean_lift_mN'],3), flush=True)
        if abs(actual-target) < .01:
            start, end = stats['averaging_interval_s']
            means = {k: float(mean_between(a[:,0], v, start, end)) for k,v in data.items()
                     if k.startswith(('aero_', 'fixture_dynamic_'))}
            result = dict(kind=kind, nominal_hz=hz, **stats, mean_wrenches=means,
                          source=str((out/f'run_{attempt}.npz').relative_to(output)),
                          simulation=asdict(config), mass_scale=mass_scale,
                          model_sha256=hashlib.sha256(config.path.read_bytes()).hexdigest(),
                          profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                          feathering_limits_deg=np.degrees(sim.model.joint('left_feathering_joint').range).tolist(),
                          pitch_minmax_deg=np.degrees([a[a[:,0]>.6,2].min(),a[a[:,0]>.6,2].max()]).tolist(),
                          max_base_pose_error=float(a[:,-1].max()), numerical_warnings=sim.data.warning.number.tolist())
            (out/'result.json').write_text(json.dumps(result, indent=2))
            return result
        slope = 1/(2*np.pi*16)
        if len(calibration) > 1:
            slope = (actual-calibration[-2]['actual_hz'])/(command-calibration[-2]['command_rad_s'])
        assert slope > 0
        command += (target-actual)/slope
        assert 0 < command <= 2000
    raise RuntimeError('Frequency calibration failed')


def run_all(momentum=False):
    grouped = json.loads((OUTPUT/'real_grouped.json').read_text())
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_case, r['kind'], r['hz'], r['encoder_hz'],
                               label=r['kind']+'_momentum' if momentum else None) for r in grouped]
        for f in as_completed(futures):
            results.append(f.result())
            (OUTPUT/('momentum_summary.json' if momentum else 'simulation_summary.json')).write_text(json.dumps(results, indent=2))


def sensitivity():
    target = next(r['encoder_hz'] for r in json.loads((OUTPUT/'real_grouped.json').read_text())
                  if r['kind']=='rect' and r['hz']==15)
    cases = [dict(label='rect_mass_075',mass_scale=.75),
             dict(label='rect_mass_125',mass_scale=1.25),
             dict(label='rect_strips_24',strips=24)]
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_case,'rect',15,target,**case) for case in cases]
        for future in as_completed(futures):
            results.append(future.result())
            (OUTPUT/'sensitivity_summary.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--momentum', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run_all()
    if args.sensitivity:
        sensitivity()
    if args.momentum:
        run_all(momentum=True)
