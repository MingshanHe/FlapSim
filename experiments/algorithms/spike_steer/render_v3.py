"""Fixed-camera comparison videos with a light background: learned controller (plan-v4 round 5, M1) vs the classical
opponent on the SAME case, for several nod angles and dash distances.

- Fixed third-person SIDE view: the camera is placed perpendicular to the body heading at the command onset and does
  not move, so the pitch excursion (nod) and the forward travel (dash) are visible in the image plane.
- Light background: colour + depth are rendered per frame and empty pixels are replaced by a white-to-light-blue
  gradient; a light ground grid, a start line and the dash target plane are added as scene geoms.
- Variable targets (rendering only, nothing archived is changed): the nod goal angle (bench uses 30 deg) and the dash
  distance (bench.DASH_M) are overridden for the run. The controllers were trained for nod 30 deg and dash 0.15 m;
  other targets are generalisation tests and are labelled as such on the frame.
Every frame is a recorded simulator state at control rate; nothing is interpolated.

    MUJOCO_GL=osmesa python -m experiments.algorithms.spike_steer.render_v3 --nod 20,30,40,50 --dash 0.10,0.15,0.25,0.35
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
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
from . import bench as B                                                 # noqa: E402
from . import descend_r5 as R5                                           # noqa: E402
from . import confirm_m1 as C                                            # noqa: E402

FPS = 30
PANEL = (800, 560)
PLOT_H = 230
HEADER_H = 92
FOOT_H = 34
INK, INK2, INK3 = (24, 30, 38), (74, 82, 92), (130, 138, 148)
LEARNED, CLASSIC, TARGET, GOOD, BAD = (42, 120, 214), (226, 104, 52), (12, 150, 70), (12, 150, 70), (208, 59, 59)
SKY_TOP, SKY_BOT = np.array([255, 255, 255]), np.array([214, 232, 248])
TRAINED = dict(nod=30., dash=.15)


def font(size, bold=False):
    names = ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf') if bold else \
            ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


class _MathNod:
    """bench.run_episode decides 'nod goal reached' with math.radians(30.); map that one call to the requested angle."""

    def __init__(self, deg):
        self.deg = deg

    def radians(self, x):
        return math.radians(self.deg if x == 30. else x)

    def __getattr__(self, k):
        return getattr(math, k)


def record(env, task, c, kind, arg):
    d = env.sim.data
    frames, orig = [], env.step

    def step(action):
        out = orig(action)
        frames.append((float(d.time), d.qpos.copy()))
        return out
    env.step = step
    try:
        res = B.run_episode(env, task, c, B.make_policy(kind, arg), trace=True)
    finally:
        del env.step
    return res, frames


def add_geom(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel() if mat is None else np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def sky(h, w):
    t = np.linspace(0., 1., h)[:, None, None]
    return (SKY_TOP*(1-t)+SKY_BOT*t).astype(np.uint8).repeat(w, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nod', default='20,30,40,50')
    p.add_argument('--dash', default='0.10,0.15,0.25,0.35')
    p.add_argument('--seed', type=int, default=4, help='round-5 seed (M1 controllers)')
    p.add_argument('--case', type=int, default=0, help='index into the M1 fresh blind set')
    p.add_argument('--slow', type=float, default=2.)
    p.add_argument('--out', default='artifacts/spike_steer/videos_v3')
    a = p.parse_args()
    C.install()                                                         # round-5 dispatch; M1 blind set
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
    for gi in range(m.ngeom):                                           # the simulator's ground plane: light instead of dark blue-grey
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_rgba[gi] = [.955, .962, .972, 1.]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = PANEL
    renderer = mujoco.Renderer(m, height=H, width=W)
    renderer_d = mujoco.Renderer(m, height=H, width=W)
    renderer_d.enable_depth_rendering()
    f_title, f_big, f_mid, f_small = font(24, True), font(20, True), font(17), font(14)
    jobs = [('nod', float(x)) for x in a.nod.split(',') if x] + [('dash', float(x)) for x in a.dash.split(',') if x]
    manifest = []
    for task, target in jobs:
        nod_deg = target if task == 'nod' else 30.
        B.math = _MathNod(nod_deg)
        B.DASH_M['value'] = target if task == 'dash' else TRAINED['dash']
        c = B.cases(task, 'test', None)[a.case]
        oname = baseline['opponent'][task]
        runs = [('Learned (bio-inspired SNN layer)', 'learned', spec, LEARNED), (f'Classical ({oname})', 'classical', baseline['arms_full'][task][oname], CLASSIC)]
        recs = []
        for label, kind, arg, col in runs:
            res, fr = record(env, task, c, kind, arg)
            tr = np.array(res.pop('trace'))
            recs.append((label, col, res, tr, fr))
        B.math = math
        # fixed camera from the learned run's state at the command onset (identical start in both runs)
        tr0 = recs[0][3]
        i0 = int(np.searchsorted(tr0[:, 0], B.T0))
        start = tr0[i0, 2:5].copy()
        fr0 = recs[0][4]
        d.qpos[:] = fr0[min(i0, len(fr0)-1)][1]
        mujoco.mj_forward(m, d)
        R = d.xmat[body].reshape(3, 3)
        fwd = R[:, 0].copy()
        fwd[2] = 0.
        fwd /= np.linalg.norm(fwd)+1e-12
        heading = math.degrees(math.atan2(fwd[1], fwd[0]))
        span = max(.18, (target if task == 'dash' else .12))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = start+fwd*(span*.45 if task == 'dash' else 0.)+np.array([0., 0., -.01])
        cam.azimuth, cam.elevation = heading+90., -12.
        cam.distance = .32+1.15*span
        ground_z = .0002                                               # real ground plane (z = 0)
        # per-run timing metrics for the overlay (from the recorded trace)
        def metrics(res, tr):
            t = tr[:, 0]-B.T0
            if task == 'nod':
                pitch = np.degrees(tr[:, 1])
                hit = np.flatnonzero((t >= 0) & (np.abs(pitch) >= nod_deg))
                return dict(series=(t, pitch), reach=float(t[hit[0]]) if len(hit) else None, peak=float(np.abs(pitch[t >= 0]).max()))
            fwdx = (tr[:, 2:4]-start[:2])@fwd[:2]
            hit = np.flatnonzero((t >= 0) & (fwdx >= target))
            return dict(series=(t, fwdx), reach=float(t[hit[0]]) if len(hit) else None, peak=float(fwdx[t >= 0].max()))
        mets = [metrics(r[2], r[3]) for r in recs]
        horizon = max(r[4][-1][0] for r in recs)
        tag = f'{task}_{int(round(target*100)) if task == "dash" else int(target)}{"cm" if task == "dash" else "deg"}'
        path = out/f'r5s{a.seed}_{tag}.mp4'
        writer = imageio.get_writer(str(path), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
        FW, FH = 2*W, HEADER_H+H+PLOT_H+FOOT_H
        n_frames = int(round(horizon*FPS*a.slow))
        trained = target == TRAINED[task]
        unit = 'deg' if task == 'nod' else 'm'
        tgt_txt = f'{target:g} {unit}' if task == 'nod' else f'{target:.2f} m'
        title = (f'Nod: pitch to {target:g} deg, hold 0.1 s, return level' if task == 'nod'
                 else f'Dash: move {target:.2f} m forward and stop')
        sub = ('trained target' if trained else f'generalisation test (trained on {TRAINED[task]:g} {"deg" if task == "nod" else "m"})')
        try:
            for k in range(n_frames):
                t = (k+1)/(FPS*a.slow)
                pic = Image.new('RGB', (FW, FH), (255, 255, 255))
                draw = ImageDraw.Draw(pic)
                draw.text((20, 14), title, font=f_title, fill=INK)
                draw.text((20, 48), f'{sub}  ·  same case and disturbances in both panels  ·  {1/a.slow:g}x speed', font=f_small, fill=INK2)
                rel = t-B.T0
                phase = 'command ON' if rel >= 0 else 'hover (reflex layer only)'
                draw.text((FW-330, 18), f't = {t:5.2f} s', font=f_big, fill=INK)
                draw.text((FW-330, 48), phase, font=f_mid, fill=TARGET if rel >= 0 else INK3)
                for col, ((label, colr, res, tr, fr), met) in enumerate(zip(recs, mets)):
                    x0 = col*W
                    i = min(int(np.searchsorted([f[0] for f in fr], t)), len(fr)-1)
                    d.qpos[:] = fr[i][1]
                    mujoco.mj_forward(m, d)
                    for rr in (renderer, renderer_d):
                        rr.update_scene(d, camera=cam)
                        scn = rr.scene
                        side = np.array([-fwd[1], fwd[0], 0.])
                        rot = np.column_stack([fwd, side, [0, 0, 1]])
                        for gx in np.arange(-.6, .61, .05):                       # light ground grid, 5 cm
                            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.0005, .6, .00005], np.r_[start[:2]+fwd[:2]*gx, ground_z], [.80, .84, .89, 1], rot)
                            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.6, .0005, .00005], np.r_[start[:2]+side[:2]*gx, ground_z], [.80, .84, .89, 1], rot)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.0008, .04, (start[2]-ground_z)/2], np.r_[start[:2], (start[2]+ground_z)/2], [.55, .6, .68, .35], rot)
                        if task == 'dash':
                            tp = start+fwd*target
                            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [.0012, .05, (start[2]-ground_z)/2+.04], np.r_[tp[:2], (start[2]+ground_z)/2+.04], [.05, .6, .28, .4], rot)
                    rgb = renderer.render()
                    depth = renderer_d.render()
                    bgmask = depth >= depth.max()-1e-6
                    img = np.where(bgmask[..., None], sky(H, W), rgb)
                    pic.paste(Image.fromarray(img), (x0, HEADER_H))
                    draw = ImageDraw.Draw(pic)
                    draw.text((x0+16, HEADER_H+12), label, font=f_big, fill=colr)
                    if not res['flown'] and t > res['elapsed']+.02:
                        draw.text((x0+16, HEADER_H+40), f'stopped at {res["elapsed"]:.2f} s ({res["termination"]})', font=f_mid, fill=BAD)
                    j = min(int(np.searchsorted(tr[:, 0], t)), len(tr)-1)
                    pitch_now = math.degrees(tr[j, 1])
                    fwd_now = float((tr[j, 2:4]-start[:2])@fwd[:2])
                    live = f'pitch {pitch_now:+5.1f} deg' if task == 'nod' else f'forward {fwd_now:+.3f} m'
                    draw.text((x0+16, HEADER_H+H-34), live, font=f_mid, fill=INK)
                    r = met['reach']
                    reach_txt = f'reached {tgt_txt} at {r:.2f} s' if (r is not None and rel >= r) else (f'target {tgt_txt}')
                    draw.text((x0+W-300, HEADER_H+H-34), reach_txt, font=f_mid, fill=TARGET if (r is not None and rel >= r) else INK2)
                    if col == 1:
                        draw.line([(x0, HEADER_H), (x0, HEADER_H+H)], fill=(220, 224, 230), width=2)
                # time-series plot of both runs
                py0, px0, px1 = HEADER_H+H+18, 70, FW-30
                ph = PLOT_H-50
                tmax = max(horizon-B.T0, .5)
                if task == 'nod':
                    lo, hi = -max(60., nod_deg+15.), max(60., nod_deg+15.)
                    ylabel, ticks = 'pitch (deg)', [-nod_deg, 0, nod_deg]
                else:
                    lo, hi = -.05, max(.2, target*1.4)
                    ylabel, ticks = 'forward (m)', [0, target]
                X = lambda tt: px0+(px1-px0)*min(max(tt, -.3), tmax)/(tmax+.3)+ (px1-px0)*.3/(tmax+.3)
                Y = lambda v: py0+ph*(hi-min(max(v, lo), hi))/(hi-lo)
                draw.rectangle([px0, py0, px1, py0+ph], outline=(222, 226, 232), fill=(250, 251, 252))
                for v in ticks:
                    yv = Y(v)
                    draw.line([(px0, yv), (px1, yv)], fill=TARGET if v != 0 else (205, 210, 216), width=2 if v != 0 else 1)
                    draw.text((8, yv-9), f'{v:g}', font=f_small, fill=TARGET if v != 0 else INK3)
                draw.line([(X(0), py0), (X(0), py0+ph)], fill=(205, 210, 216), width=1)
                for tk in np.arange(0., tmax+1e-9, .5):
                    draw.line([(X(tk), py0+ph), (X(tk), py0+ph+5)], fill=INK3, width=1)
                    draw.text((X(tk)-8, py0+ph+6), f'{tk:g}', font=f_small, fill=INK3)
                draw.text((px1-420, py0+4), f'{ylabel} vs time since command (s); green = target', font=f_small, fill=INK2)
                for (label, colr, res, tr, fr), met in zip(recs, mets):
                    tt, vv = met['series']
                    sel = (tt >= -.3) & (tt <= rel)
                    if sel.sum() > 1:
                        pts = [(X(a_), Y(b_)) for a_, b_ in zip(tt[sel][::2], vv[sel][::2])]
                        draw.line(pts, fill=colr, width=3)
                draw.line([(X(min(rel, tmax)), py0), (X(min(rel, tmax)), py0+ph)], fill=(160, 166, 174), width=1)
                draw.text((20, FH-26), 'Recorded simulator states at control rate (no interpolation). Fixed side camera. Blue = learned layer, orange = classical opponent.',
                          font=f_small, fill=INK3)
                writer.append_data(np.asarray(pic))
                if k == min(n_frames-1, int(round((B.T0+(mets[0]['reach'] or .6)+.15)*FPS*a.slow))):
                    pic.save(out/f'r5s{a.seed}_{tag}_still.png')
            for _ in range(FPS):
                writer.append_data(np.asarray(pic))
        finally:
            writer.close()
        entry = dict(task=task, target=target, trained_target=trained, video=str(path), case_seed=c['seed'], controller=f'round-5 seed {a.seed} gen {gen}',
                     learned=dict(score=recs[0][2]['score'], flown=recs[0][2]['flown'], reach_s=mets[0]['reach'], peak=mets[0]['peak']),
                     classical=dict(arm=oname, score=recs[1][2]['score'], flown=recs[1][2]['flown'], reach_s=mets[1]['reach'], peak=mets[1]['peak']))
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    renderer.close()
    renderer_d.close()
    B.DASH_M['value'] = TRAINED['dash']
    (out/f'manifest_r5s{a.seed}.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
