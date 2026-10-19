"""Single-panel 3D videos of the learned controller (plan-v4 round 5, M1 controllers) for several nod angles and dash
distances: fixed oblique top-down camera, light background, grid floor, bright light without shadows, translucent
Kapton-amber wings, and a fading trail of the body path.

- The camera is fixed for the whole clip: placed above and to the side-front of the start point (relative to the body
  heading at the command onset), looking down obliquely at the middle of the motion.
- Targets are overridden for rendering only: nod goal angle (bench uses 30 deg) and dash distance (bench.DASH_M). The
  controllers were trained for nod 30 deg / dash 0.15 m; other targets are generalisation tests (labelled on the frame).
- Every frame is a recorded simulator state at control rate; nothing is interpolated.

    MUJOCO_GL=osmesa python -m experiments.algorithms.spike_steer.render_v4 --nod 20,30,40,50 --dash 0.10,0.15,0.25,0.35
"""
import argparse
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
from . import confirm_m1 as C                                            # noqa: E402
from .render_v3 import _MathNod, record, add_geom, font, TRAINED         # noqa: E402

FPS = 30
W, H = 1280, 800
SKY_TOP, SKY_BOT = np.array([255, 255, 255]), np.array([243, 247, 251])
KAPTON = [.86, .55, .14, .55]
GROUND = [.965, .97, .978, 1.]
GRID = [.64, .70, .79, 1.]
GRID_MAJOR = [.48, .56, .68, 1.]
TRAIL = [.16, .47, .84, .9]
TARGET_RGBA = [.05, .62, .30, .32]
INK, INK2 = (24, 30, 38), (90, 98, 108)


def style_model(m):
    for gi in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ''
        if name in ('left_wing_visual', 'right_wing_visual'):
            m.geom_rgba[gi] = KAPTON
    m.light_castshadow[:] = 0
    m.light_diffuse[:] = .75
    m.light_ambient[:] = .25
    m.vis.headlight.ambient[:] = .45
    m.vis.headlight.diffuse[:] = .55
    m.vis.headlight.specular[:] = .1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nod', default='20,30,40,50')
    p.add_argument('--dash', default='0.10,0.15,0.25,0.35')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--slow', type=float, default=2.)
    p.add_argument('--out', default='artifacts/spike_steer/videos_v4')
    a = p.parse_args()
    C.install()
    gen, spec_path = C.specs()[a.seed]
    spec = json.loads(spec_path.read_text())
    baseline = json.loads(Path('artifacts/spike_steer/confirm_m1/claude_ea/baseline.json').read_text())
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    B.worker_init(str(source), baseline['prereg']['dash_m'] or .25)
    env = B._ENV
    m, d = env.sim.model, env.sim.data
    body = env.air.env.body
    style_model(m)
    for gi in range(m.ngeom):                                          # simulator ground plane -> geom group 5, hidden when rendering
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_group[gi] = 5
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    far = float(m.vis.map.zfar*m.stat.extent)
    rgb_r = mujoco.Renderer(m, height=H, width=W, max_geom=4000)
    dep_r = mujoco.Renderer(m, height=H, width=W, max_geom=4000)
    dep_r.enable_depth_rendering()
    t_ = np.linspace(0., 1., H)[:, None, None]
    sky = (SKY_TOP*(1-t_)+SKY_BOT*t_).astype(np.uint8).repeat(W, 1)
    f_title, f_small = font(26, True), font(17)
    jobs = [('nod', float(x)) for x in a.nod.split(',') if x] + [('dash', float(x)) for x in a.dash.split(',') if x]
    manifest = []
    for task, target in jobs:
        B.math = _MathNod(target if task == 'nod' else 30.)
        B.DASH_M['value'] = target if task == 'dash' else TRAINED['dash']
        c = B.cases(task, 'test', None)[a.case]
        res, frames = record(env, task, c, 'learned', spec)
        tr = np.array(res.pop('trace'))
        B.math = math
        times = np.array([f[0] for f in frames])
        i0 = min(int(np.searchsorted(times, B.T0)), len(frames)-1)
        d.qpos[:] = frames[i0][1]
        mujoco.mj_forward(m, d)
        start = d.xpos[body].copy()
        fwd = d.xmat[body].reshape(3, 3)[:, 0].copy()
        fwd[2] = 0.
        fwd /= np.linalg.norm(fwd)+1e-12
        side = np.array([-fwd[1], fwd[0], 0.])
        rot = np.column_stack([fwd, side, [0., 0., 1.]])
        heading = math.degrees(math.atan2(fwd[1], fwd[0]))
        reach = .09 if task == 'nod' else max(.15, target)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = start+fwd*(.5*target if task == 'dash' else 0.)-np.array([0., 0., .03])
        cam.azimuth = heading+90.+38.                                    # side-front, oblique
        cam.elevation = -30.
        cam.distance = .2+1.15*reach
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
        u = np.cross(view, [0., 0., 1.])
        u /= np.linalg.norm(u)
        v = np.cross(u, view)
        backdrop_rot = np.column_stack([u, v, view])
        backdrop_pos = np.asarray(cam.lookat)+view*3.
        # body path for the trail
        path_xyz = []
        for (_, q) in frames:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            path_xyz.append(d.xpos[body].copy())
        path_xyz = np.array(path_xyz)
        trained = target == TRAINED[task]
        title = f'Nod to {target:g} deg' if task == 'nod' else f'Dash {target:.2f} m forward and stop'
        sub = 'trained target' if trained else f'generalisation test (trained: {"30 deg" if task == "nod" else "0.15 m"})'
        tag = f'{task}_{int(target)}deg' if task == 'nod' else f'{task}_{int(round(target*100))}cm'
        vpath = out/f'r5s{a.seed}_{tag}.mp4'
        writer = imageio.get_writer(str(vpath), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
        horizon = float(times[-1])
        n_frames = int(round(horizon*FPS*a.slow))
        still_k = int(round((B.T0+.55)*FPS*a.slow)) if task == 'nod' else int(round((B.T0+1.2)*FPS*a.slow))
        try:
            for k in range(n_frames):
                t = (k+1)/(FPS*a.slow)
                i = min(int(np.searchsorted(times, t)), len(frames)-1)
                d.qpos[:] = frames[i][1]
                mujoco.mj_forward(m, d)
                for rr in (rgb_r,):
                    rr.update_scene(d, camera=cam, scene_option=opt)
                    scn = rr.scene
                    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [6., 6., .01], backdrop_pos, [.985, .99, 1., 1.], backdrop_rot)   # white backdrop
                    scn.geoms[scn.ngeom-1].emission = 1.
                    for gx in np.arange(-1., 1.001, .05):                    # floor grid, 5 cm; 25 cm lines darker
                        major = abs(round(gx/.25)*.25-gx) < 1e-6
                        col, wdt = (GRID_MAJOR, .0013) if major else (GRID, .0008)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [wdt, 1., .00004], np.r_[start[:2]+fwd[:2]*gx, .0003], col, rot)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [1., wdt, .00004], np.r_[start[:2]+side[:2]*gx, .0003], col, rot)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [.012, .0004, 0], np.r_[start[:2], .0006], [.35, .4, .48, .9])  # start mark
                    if task == 'dash':
                        tp = start+fwd*target
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.0025, .1, .00008], np.r_[tp[:2], .0006], [.05, .6, .28, 1.], rot)       # target line
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.0006, .06, start[2]/2], np.r_[tp[:2], start[2]/2], [.05, .62, .30, .16], rot)  # faint target plane
                    for j in range(max(0, i-600), i, 6):                     # trail of the last ~1.2 s, fading
                        alpha = .15+.75*(j-(i-600))/600
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0016, 0, 0], path_xyz[j], TRAIL[:3]+[max(.12, alpha)])
                rgb = rgb_r.render()
                pic = Image.fromarray(rgb)
                draw = ImageDraw.Draw(pic)
                draw.text((24, 18), title, font=f_title, fill=INK)
                draw.text((24, 54), f'{sub}  ·  learned controller (round 5)  ·  {1/a.slow:g}x speed', font=f_small, fill=INK2)
                draw.text((W-150, 22), f't = {t:4.2f} s', font=f_small, fill=INK2)
                if t >= B.T0:
                    draw.text((W-150, 46), 'command on', font=f_small, fill=(12, 140, 70))
                writer.append_data(np.asarray(pic))
                if k == min(still_k, n_frames-1):
                    pic.save(out/f'r5s{a.seed}_{tag}_still.png')
            for _ in range(FPS):
                writer.append_data(np.asarray(pic))
        finally:
            writer.close()
        entry = dict(task=task, target=target, trained_target=trained, video=str(vpath), case_seed=c['seed'],
                     controller=f'round-5 seed {a.seed} gen {gen}', score=res['score'], flown=res['flown'], termination=res['termination'],
                     goal_t=res.get('goal_t'))
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    rgb_r.close()
    dep_r.close()
    B.DASH_M['value'] = TRAINED['dash']
    (out/f'manifest_r5s{a.seed}.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
