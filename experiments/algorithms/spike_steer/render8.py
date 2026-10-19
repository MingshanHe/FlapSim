"""Plan v6 videos: the classical opponent (2-D set-point schedule tuned per direction class, runs/spike_steer/v6/
baseline.json) flying nod (4 directions) and dash (8 directions). One grid video per task, one panel per direction,
each panel a fixed oblique 3D view (render_v4 style: white background, grid floor, Kapton wings, full body trail) with
an arrow on the floor showing the commanded direction. Panels share the clock.

    MUJOCO_GL=osmesa python -m experiments.algorithms.spike_steer.render8 --workers 12
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
import mujoco                                                            # noqa: E402
import imageio.v2 as imageio                                             # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402
from . import bench as B                                                 # noqa: E402
from . import bench8 as E                                                # noqa: E402
from .render_v3 import add_geom, font                                    # noqa: E402
from .render_v4 import style_model, GRID, GRID_MAJOR, TRAIL              # noqa: E402

FPS = 30
PW, PH = 640, 420
SW, SH = 1280, 800
CLASS = {0: 'pitch', 180: 'pitch', 90: 'roll', 270: 'roll', 45: 'diag', 135: 'diag', 225: 'diag', 315: 'diag'}
NAME = {0: 'forward', 45: 'forward-left', 90: 'left', 135: 'back-left', 180: 'back', 225: 'back-right', 270: 'right',
        315: 'forward-right'}
INK, INK2 = (24, 30, 38), (90, 98, 108)
ARROW = [.05, .62, .30, .9]


def record(args):
    task, c, law, sched = args
    env = B._ENV
    d = env.sim.data
    frames, orig = [], env.step

    def step(action):
        out = orig(action)
        frames.append((float(d.time), d.qpos.copy()))
        return out
    env.step = step
    try:
        res = E.run_episode8(env, task, c, E.Classical2(law, sched))
    finally:
        del env.step
    return res, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='runs/spike_steer/v6/baseline.json')
    p.add_argument('--case-base', type=int, default=E.BASES['val']+90_000, help='fresh cases (not tune / train / test)')
    p.add_argument('--slow', type=float, default=2.)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--out', default='artifacts/spike_steer/plan_v6/videos')
    p.add_argument('--single', action='store_true', help='one 1280x800 video per direction instead of a grid')
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    base = json.loads(Path(a.baseline).read_text())
    law = base['law']
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for task in ('nod', 'dash'):
        dirs = E.TASK_DIRS[task]
        for k, phi in enumerate(dirs):
            c = E.cases8(task, a.case_base, 1, offset=k*10, dirs=(phi,))[0]
            jobs.append((task, c, law, base['opponent'][task][CLASS[phi]]))
    with E.pool(a.workers) as ex:
        recs = list(ex.map(record, jobs, chunksize=1))
    B.worker_init(str(E.SOURCE), .15)
    env = B._ENV
    m, d = env.sim.model, env.sim.data
    body = env.air.env.body
    style_model(m)
    for gi in range(m.ngeom):
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_group[gi] = 5
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    rr = mujoco.Renderer(m, height=SH-70 if a.single else PH, width=SW if a.single else PW, max_geom=3000)
    f_title, f_lab, f_small = font(26, True), font(17, True), font(14)
    manifest = []
    for task in ('nod', 'dash'):
        idx = [i for i, j in enumerate(jobs) if j[0] == task]
        panels = []
        for i in idx:
            res, frames = recs[i]
            phi = jobs[i][1]['dir_deg']
            times = np.array([f[0] for f in frames])
            path = []
            for (_, q) in frames:
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                path.append(d.xpos[body].copy())
            path = np.array(path)
            i0 = min(int(np.searchsorted(times, B.T0)), len(frames)-1)
            d.qpos[:] = frames[i0][1]
            mujoco.mj_forward(m, d)
            start = d.xpos[body].copy()
            fwd = d.xmat[body].reshape(3, 3)[:, 0].copy()
            fwd[2] = 0.
            fwd /= np.linalg.norm(fwd)+1e-12
            left = np.array([-fwd[1], fwd[0], 0.])
            dvec = math.cos(math.radians(phi))*fwd+math.sin(math.radians(phi))*left
            heading = math.degrees(math.atan2(fwd[1], fwd[0]))
            lo, hi = path.min(0), path.max(0)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (lo+hi)/2+np.array([0., 0., .02])
            cam.azimuth, cam.elevation = heading+128., -30.        # same camera for every direction
            cam.distance = max(.5, .35+1.1*float(np.linalg.norm(hi-lo)))
            az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
            view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
            u = np.cross(view, [0., 0., 1.])
            u /= np.linalg.norm(u)
            panels.append(dict(res=res, frames=frames, times=times, path=path, start=start, fwd=fwd, left=left, dvec=dvec,
                               rot=np.column_stack([fwd, left, [0., 0., 1.]]), cam=cam, phi=phi,
                               bd=(np.asarray(cam.lookat)+view*3., np.column_stack([u, np.cross(u, view), view]))))
            manifest.append(dict(task=task, dir_deg=phi, direction=NAME[int(phi)], case_seed=jobs[i][1]['seed'],
                                 schedule=jobs[i][3], score=res['score'], flown=res['flown'], goal_t=res['goal_t'],
                                 **{k: res.get(k) for k in ('t30', 't_recover', 't_reach', 't_settle', 'overshoot_m', 'cross_m', 'drift_m')}))
        title0 = 'Nod 30° in 4 directions' if task == 'nod' else 'Dash 0.15 m and stop in 8 directions'
        groups = [[P] for P in panels] if a.single else [panels]
        for group in groups:
            render_group(a, task, group, title0, out, rr, m, d, opt, f_title, f_lab, f_small)
    rr.close()
    (out/'manifest_classical.json').write_text(json.dumps(manifest, indent=1))
    for e in manifest:
        print(json.dumps(e), flush=True)


def render_group(a, task, panels, title0, out, rr, m, d, opt, f_title, f_lab, f_small):
        single = len(panels) == 1 and a.single
        pw, ph = (SW, SH-70) if single else (PW, PH)
        cols = 1 if single else (2 if task == 'nod' else 4)
        rows_n = int(math.ceil(len(panels)/cols))
        W, H = cols*pw, rows_n*ph+70
        horizon = min(float(p_['times'][-1]) for p_ in panels)
        n_frames = int(round(horizon*FPS*a.slow))
        if single:
            phi = int(panels[0]['phi'])
            vpath = out/f'classical_{task}_{phi:03d}_{NAME[phi]}.mp4'
            title = ('Nod 30° ' if task == 'nod' else 'Dash 0.15 m ')+f'{NAME[phi]} ({phi}°)'
        else:
            vpath = out/f'classical_{task}_{len(panels)}dir.mp4'
            title = title0
        writer = imageio.get_writer(str(vpath), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
        try:
            for k in range(n_frames):
                t = (k+1)/(FPS*a.slow)
                img = Image.new('RGB', (W, H), (255, 255, 255))
                dr = ImageDraw.Draw(img)
                dr.text((16, 12), f'{title}  ·  classical opponent (2-D set-point schedule)', font=f_title, fill=INK)
                dr.text((16, 44), f'{1/a.slow:g}x speed  ·  t = {t:4.2f} s' + ('  ·  command on' if t >= B.T0 else ''), font=f_small,
                        fill=(12, 140, 70) if t >= B.T0 else INK2)
                for n, P in enumerate(panels):
                    i = min(int(np.searchsorted(P['times'], t)), len(P['frames'])-1)
                    d.qpos[:] = P['frames'][i][1]
                    mujoco.mj_forward(m, d)
                    rr.update_scene(d, camera=P['cam'], scene_option=opt)
                    scn = rr.scene
                    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [6., 6., .01], P['bd'][0], [.985, .99, 1., 1.], P['bd'][1])
                    scn.geoms[scn.ngeom-1].emission = 1.
                    st, fw, lf, rot = P['start'], P['fwd'], P['left'], P['rot']
                    for gx in np.arange(-1., 1.001, .05):
                        major = abs(round(gx/.25)*.25-gx) < 1e-6
                        col, wdt = (GRID_MAJOR, .0013) if major else (GRID, .0008)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [wdt, 1., .00004], np.r_[st[:2]+fw[:2]*gx, .0003], col, rot)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [1., wdt, .00004], np.r_[st[:2]+lf[:2]*gx, .0003], col, rot)
                    L = .15 if task == 'dash' else .08                   # direction arrow on the floor (dash: target length)
                    dv = P['dvec']
                    mid = st[:2]+dv[:2]*L/2
                    arot = np.column_stack([dv, np.cross([0., 0., 1.], dv), [0., 0., 1.]])
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [L/2, .005, .0004], np.r_[mid, .0008], ARROW, arot)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.011, 0, 0], np.r_[st[:2]+dv[:2]*L, .001], ARROW)
                    for j in range(0, i, 8):
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0018, 0, 0], P['path'][j], TRAIL)
                    x0, y0 = (n % cols)*pw, 70+(n//cols)*ph
                    img.paste(Image.fromarray(rr.render()), (x0, y0))
                    dr.rectangle((x0, y0, x0+pw-1, y0+ph-1), outline=(225, 229, 235))
                    r = P['res']
                    dr.text((x0+12, y0+8), f"{NAME[int(P['phi'])]} ({int(P['phi'])}°)", font=f_lab, fill=INK)
                    status = 'reached' if (r['goal_t'] is not None and t-B.T0 >= r['goal_t']) else ''
                    if status:
                        dr.text((x0+12, y0+30), f"target reached at +{r['goal_t']:.2f} s", font=f_small, fill=(12, 140, 70))
                    if not r['flown'] and t >= r['elapsed']:
                        dr.text((x0+12, y0+50), f"crash ({r['termination']})", font=f_small, fill=(194, 59, 59))
                writer.append_data(np.asarray(img))
            for _ in range(FPS):
                writer.append_data(np.asarray(img))
            img.save(vpath.with_name(vpath.stem+'_end.png'))
        finally:
            writer.close()
        print('wrote', vpath, flush=True)


if __name__ == '__main__':
    main()
