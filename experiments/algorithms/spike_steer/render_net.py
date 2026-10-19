"""Network-overview video: the learned descending SNN (round-5 seed 4, the M1 controller) running in closed loop,
synchronised with the flight.

Left: fixed oblique 3D view of the flight (same style as render_v4) and the manoeuvre curve (nod: pitch; dash: forward
travel) with every motor pulse marked (learned layer vs classical law). Right: the network -- 19 input nodes
(rectified IMU halves, bias, nod / dash commands; brightness = input), input->hidden edges (top weights, summed over
the 11 delays; lit by presynaptic activity), 8 hidden LIF neurons (fill = membrane potential, ring = spike in this
frame), the up / down spiking outputs, the graded unit (pitch reference), the motor pulse and the command gate; below,
a scrolling spike raster. Every value is logged from the actual closed-loop run (DescendingNet.tick is wrapped to
record its state; nothing is recomputed).

    MUJOCO_GL=osmesa python -m experiments.algorithms.spike_steer.render_net --jobs nod:30:1,dash:0.15:15
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
from .descend_ann import _SNN                                            # noqa: E402  (B.DescendingNet is a dispatcher after install)
from .render_v3 import _MathNod, record, add_geom, font, TRAINED         # noqa: E402
from .render_v4 import style_model, GRID, GRID_MAJOR, TRAIL              # noqa: E402

FPS = 30
W, H = 1920, 1080
VW, VH = 860, 560                       # 3D view
INK, INK2, MUTE = (24, 30, 38), (90, 98, 108), (200, 205, 212)
POS, NEG = (222, 120, 30), (40, 110, 200)          # excitatory / inhibitory
SPIKE, LEARN, CLASSIC = (230, 60, 50), (222, 120, 30), (140, 148, 160)
CH = ['roll', 'pitch', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
IN_NODES = [(i, f'{c}+') for i, c in enumerate(CH)] + [(8+i, f'{c}−') for i, c in enumerate(CH)] + \
           [(16, 'bias'), (18, 'nod cmd'), (19, 'dash cmd')]


class NetLog:
    """Wraps DescendingNet.tick: per tick input x, hidden spikes (either substep), membrane potentials, output spikes, g."""

    def __init__(self):
        self.rows = []
        self.orig = _SNN.tick
        log = self

        def tick(net, x):
            v_before = net.spikes
            hs = np.zeros(net.H, bool)
            p0 = net.p
            fired, ref = log.orig(net, x)
            for k in range(2):
                hs |= net.hist[(p0+k) % net.L].astype(bool)
            log.rows.append(dict(x=np.asarray(x, float).copy(), hs=hs, vh=net.v_h.copy(), vo=net.v_o.copy(),
                                 fo=np.asarray(fired, bool).copy(), ref=float(ref), n=net.spikes-v_before))
            return fired, ref
        self.tick = tick

    def __enter__(self):
        self.rows = []
        _SNN.tick = self.tick
        return self

    def __exit__(self, *exc):
        _SNN.tick = self.orig


def lerp(c0, c1, a):
    a = float(np.clip(a, 0., 1.))
    return tuple(int(round(c0[k]+(c1[k]-c0[k])*a)) for k in range(3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', default='nod:30:1,dash:0.15:15', help='task:target:case_index, comma separated')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--slow', type=float, default=10.)
    p.add_argument('--before', type=float, default=.4)
    p.add_argument('--after', type=float, default=1.6)
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
    for gi in range(m.ngeom):
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_group[gi] = 5
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rr = mujoco.Renderer(m, height=VH, width=VW, max_geom=4000)
    f_title, f_mid, f_small, f_tiny = font(30, True), font(20, True), font(17), font(14)

    W_in, W_out, W_g = np.asarray(spec['W_in']), np.asarray(spec['W_out']), np.asarray(spec['W_g'])
    S = float(spec.get('spike_scale', B.SPIKE_SCALE))
    w_ih = W_in.sum(2)                                        # H x 20, summed over delays
    w_ho = S*W_out.sum(2)                                     # 2 x H
    w_hg = S*W_g.sum(1)                                       # H
    cand = [(abs(w_ih[h, i]), h, i) for h in range(w_ih.shape[0]) for (i, _) in IN_NODES]
    top = sorted(cand, reverse=True)[:56]
    w_top = top[0][0]

    # network layout (right half)
    X0 = 940
    in_x, hid_x, out_x = X0+140, X0+460, X0+690
    in_y = {i: 110+k*31 for k, (i, _) in enumerate(IN_NODES)}
    hid_y = [170+k*62 for k in range(8)]
    out_y = dict(up=250, down=360, g=520)

    manifest = []
    for job in a.jobs.split(','):
        task, target, ci = job.split(':')
        target, ci = float(target), int(ci)
        B.math = _MathNod(target if task == 'nod' else 30.)
        B.DASH_M['value'] = target if task == 'dash' else TRAINED['dash']
        c = B.cases(task, 'test', None)[ci]
        with NetLog() as log:
            res, frames = record(env, task, c, 'learned', spec)
        B.math = math
        tr = np.array(res.pop('trace'))
        net = log.rows
        n = min(len(net), len(tr))
        assert abs(len(net)-len(tr)) <= 2, (len(net), len(tr))
        net, tr = net[-n:], tr[-n:]
        tt = tr[:, 0]
        fire = tr[:, 10] > .5
        cmd_on = tr[:, 11:13].max(1) > .5
        gate = np.zeros(n, bool)
        off_t = None
        for k in range(n):                                     # same rule as descend_gated.GatedLearned
            if k and cmd_on[k-1] and not cmd_on[k]:
                off_t = tt[k]
            gate[k] = cmd_on[k] or (off_t is not None and tt[k]-off_t < 1.)
        hs = np.array([r['hs'] for r in net])
        fo = np.array([r['fo'] for r in net])
        vh = np.array([r['vh'] for r in net])
        xs = np.array([r['x'] for r in net])
        ref = np.array([r['ref'] for r in net])
        pitch = np.degrees(tr[:, 1])
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
        ta, tb = B.T0-a.before, min(B.T0+a.after, float(times[-1]))
        path = []
        for (_, q) in frames:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            path.append(d.xpos[body].copy())
        path = np.array(path)
        win = (times >= ta) & (times <= tb)
        lo, hi = path[win].min(0), path[win].max(0)
        span = float(np.linalg.norm(hi-lo))
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (lo+hi)/2
        cam.azimuth, cam.elevation = heading+128., -30.
        cam.distance = max(.42, .3+1.1*span)
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
        u = np.cross(view, [0., 0., 1.])
        u /= np.linalg.norm(u)
        bd_rot, bd_pos = np.column_stack([u, np.cross(u, view), view]), np.asarray(cam.lookat)+view*3.
        travel = (tr[:, 2:4]-start[:2])@fwd[:2]

        # manoeuvre curve geometry (left, below the 3D view)
        PX0, PX1, PY0, PY1 = 70, 900, 690, 960
        if task == 'nod':
            curve, ylab, lim, goal = pitch, 'pitch (deg)', (-45., 45.), [target, -target]
        else:
            curve, ylab, lim, goal = travel*100, 'forward travel (cm)', (-5., max(25., target*100+8)), [target*100]
        sel = (tt >= ta) & (tt <= tb)

        def px(t):
            return PX0+(t-ta)/(tb-ta)*(PX1-PX0)

        def py(v):
            return PY1-(np.clip(v, *lim)-lim[0])/(lim[1]-lim[0])*(PY1-PY0)
        pts = [(px(t), py(v)) for t, v in zip(tt[sel], curve[sel])]

        tag = f'{task}_{int(target)}deg' if task == 'nod' else f'{task}_{int(round(target*100))}cm'
        vpath = out/f'r5s{a.seed}_{tag}_network.mp4'
        writer = imageio.get_writer(str(vpath), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
        n_frames = int(round((tb-ta)*FPS*a.slow))
        prev_k = int(np.searchsorted(tt, ta))
        title = f'Nod to {target:g} deg' if task == 'nod' else f'Dash {target:.2f} m forward and stop'
        try:
            for f in range(n_frames):
                t = ta+(f+1)/(FPS*a.slow)
                k = min(int(np.searchsorted(tt, t)), n-1)
                kf = slice(prev_k, k+1)                            # ticks shown by this frame
                prev_k = k+1
                hs_f, fo_f, fire_f = hs[kf].any(0), fo[kf].any(0), bool(fire[kf].any())
                i = min(int(np.searchsorted(times, t)), len(frames)-1)
                d.qpos[:] = frames[i][1]
                mujoco.mj_forward(m, d)
                rr.update_scene(d, camera=cam, scene_option=opt)
                scn = rr.scene
                scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [6., 6., .01], bd_pos, [.985, .99, 1., 1.], bd_rot)
                scn.geoms[scn.ngeom-1].emission = 1.
                for gx in np.arange(-1., 1.001, .05):
                    major = abs(round(gx/.25)*.25-gx) < 1e-6
                    col, wdt = (GRID_MAJOR, .0013) if major else (GRID, .0008)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [wdt, 1., .00004], np.r_[start[:2]+fwd[:2]*gx, .0003], col, rot)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [1., wdt, .00004], np.r_[start[:2]+side[:2]*gx, .0003], col, rot)
                for j in range(int(np.searchsorted(times, ta)), i, 6):
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0016, 0, 0], path[j], TRAIL)
                img = Image.new('RGB', (W, H), (255, 255, 255))
                img.paste(Image.fromarray(rr.render()), (40, 100))
                dr = ImageDraw.Draw(img)
                dr.text((40, 22), f'{title}  ·  learned descending SNN in closed loop', font=f_title, fill=INK)
                dr.text((40, 62), f'round-5 seed {a.seed} (M1 controller)  ·  {1/a.slow:g}x speed  ·  t = {t:5.3f} s'
                        f'  ({"+" if t >= B.T0 else "−"}{abs(t-B.T0)*1000:4.0f} ms from command)', font=f_small, fill=INK2)
                dr.rectangle((40, 100, 40+VW, 100+VH), outline=MUTE)

                # manoeuvre curve with motor pulses
                dr.text((PX0, PY0-28), ylab, font=f_small, fill=INK2)
                for g in goal:
                    dr.line((PX0, py(g), PX1, py(g)), fill=(160, 200, 170), width=1)
                    dr.text((PX0-50, py(g)-9), f'{g:g}', font=f_tiny, fill=(12, 140, 70))
                dr.text((PX0-50, py(0.)-9), '0', font=f_tiny, fill=INK2)
                dr.line((PX0, py(0.), PX1, py(0.)), fill=MUTE, width=1)
                if pts:
                    dr.line(pts, fill=MUTE, width=2)
                shown = [p_ for p_, t_ in zip(pts, tt[sel]) if t_ <= t]
                if len(shown) > 1:
                    dr.line(shown, fill=(40, 110, 200), width=3)
                for kk in np.flatnonzero(sel & (tt <= t) & fire):
                    dr.line((px(tt[kk]), PY1+8, px(tt[kk]), PY1+24), fill=LEARN if gate[kk] else CLASSIC, width=2)
                dr.line((px(B.T0), PY0, px(B.T0), PY1+24), fill=(12, 140, 70), width=1)
                dr.line((px(t), PY0, px(t), PY1+24), fill=INK, width=1)
                dr.text((PX0, PY1+30), 'motor pulses:', font=f_tiny, fill=INK2)
                dr.text((PX0+110, PY1+30), '▮ learned layer', font=f_tiny, fill=LEARN)
                dr.text((PX0+250, PY1+30), '▮ classical law (gate off)', font=f_tiny, fill=CLASSIC)
                dr.text((px(B.T0)+4, PY0), 'command', font=f_tiny, fill=(12, 140, 70))

                # network: edges
                xk = xs[k]
                for (wa, h, ii) in top:
                    act = min(1., xk[ii]/1.)*wa/w_top
                    base = POS if w_ih[h, ii] > 0 else NEG
                    col = lerp((236, 239, 243), base, .15+.85*act)
                    dr.line((in_x+10, in_y[ii], hid_x-18, hid_y[h]), fill=col, width=1+int(3*act))
                for h in range(8):
                    for (key, w) in (('up', w_ho[0, h]), ('down', w_ho[1, h]), ('g', w_hg[h])):
                        a_ = min(1., abs(w)/max(abs(w_ho).max(), abs(w_hg).max()))
                        lit = hs_f[h]
                        col = lerp((236, 239, 243), POS if w > 0 else NEG, (.25+.75*a_) if lit else .12+.2*a_)
                        dr.line((hid_x+18, hid_y[h], out_x-22, out_y[key]), fill=col, width=3 if lit else 1)
                # input nodes
                dr.text((in_x-110, 70), 'inputs (IMU, rectified)', font=f_mid, fill=INK)
                for (ii, lab) in IN_NODES:
                    v = float(np.clip(xk[ii], 0., 1.))
                    y = in_y[ii]
                    fill = lerp((245, 246, 248), (40, 110, 200) if ii < 16 else (12, 140, 70), v)
                    dr.ellipse((in_x-9, y-9, in_x+9, y+9), fill=fill, outline=INK2)
                    dr.text((in_x-110, y-9), lab, font=f_tiny, fill=INK if v > .05 else INK2)
                # hidden LIF neurons
                dr.text((hid_x-80, 110), '8 hidden LIF neurons', font=f_mid, fill=INK)
                for h in range(8):
                    y = hid_y[h]
                    v = float(np.clip(vh[k, h], 0., 1.))
                    dr.ellipse((hid_x-18, y-18, hid_x+18, y+18), fill=lerp((245, 246, 248), (250, 200, 120), v), outline=INK2, width=2)
                    hh = int(32*v)
                    dr.rectangle((hid_x+24, y+16-hh, hid_x+30, y+16), fill=(250, 170, 60))
                    dr.rectangle((hid_x+24, y-16, hid_x+30, y+16), outline=MUTE)
                    if hs_f[h]:
                        dr.ellipse((hid_x-25, y-25, hid_x+25, y+25), outline=SPIKE, width=4)
                    dr.text((hid_x-10, y-9), f'h{h+1}', font=f_tiny, fill=INK)
                dr.text((hid_x-150, hid_y[-1]+30), 'fill / bar = membrane potential (threshold 1)\nred ring = spike in this frame',
                        font=f_tiny, fill=INK2)
                # outputs
                dr.text((out_x-40, 170), 'outputs', font=f_mid, fill=INK)
                for key, lab in (('up', 'up (pitch-up timing)'), ('down', 'down (pitch-down timing)')):
                    o = 0 if key == 'up' else 1
                    y = out_y[key]
                    on = fo_f[o]
                    dr.ellipse((out_x-22, y-22, out_x+22, y+22), fill=SPIKE if on else (245, 246, 248), outline=INK2, width=2)
                    dr.text((out_x+30, y-10), lab, font=f_small, fill=INK)
                y = out_y['g']
                dr.rectangle((out_x-22, y-22, out_x+22, y+22), fill=(245, 246, 248), outline=INK2, width=2)
                dr.text((out_x-8, y-11), 'g', font=f_mid, fill=INK)
                dr.text((out_x+30, y-22), 'graded unit → pitch ref', font=f_small, fill=INK)
                rv = math.degrees(ref[k])
                gx0, gx1 = out_x+30, out_x+200
                dr.rectangle((gx0, y+6, gx1, y+20), outline=MUTE)
                mid = (gx0+gx1)/2
                xr = mid+rv/math.degrees(.7)*(gx1-gx0)/2
                dr.rectangle((min(mid, xr), y+6, max(mid, xr), y+20), fill=(40, 110, 200))
                dr.line((mid, y+2, mid, y+24), fill=INK2)
                dr.text((gx1+8, y+2), f'{rv:+5.1f}°', font=f_small, fill=INK)
                # motor pulse + gate
                yb = 610
                dr.rectangle((out_x-40, yb, out_x+250, yb+60), fill=(255, 236, 215) if fire_f else (247, 248, 250), outline=INK2, width=2)
                dr.text((out_x-28, yb+8), 'motor pulse (6 ms, 3807 rad/s)', font=f_small, fill=INK)
                dr.text((out_x-28, yb+32), 'FIRING' if fire_f else '—', font=f_mid, fill=SPIKE if fire_f else INK2)
                dr.line((out_x, out_y['down']+22, out_x, out_y['g']-24), fill=MUTE, width=1)
                dr.text((out_x+6, out_y['down']+40), 'either spike → motor pulse', font=f_tiny, fill=INK2)
                gk = gate[k]
                dr.rectangle((X0+60, 700, X0+400, 748), fill=(255, 236, 215) if gk else (240, 242, 245), outline=INK2, width=2)
                dr.text((X0+74, 705), 'gate: LEARNED layer in charge' if gk else 'gate: classical law in charge',
                        font=f_small, fill=LEARN if gk else INK2)
                dr.text((X0+74, 727), 'command on / < 1 s after' if gk else 'no command (hover reflex)', font=f_tiny, fill=INK2)

                # raster, last 0.4 s
                RX0, RX1, RY0 = X0+60, W-60, 805
                span_t = .4
                dr.text((RX0, RY0-30), f'spike raster (last {span_t*1000:.0f} ms)', font=f_mid, fill=INK)
                rows_lab = [f'h{h+1}' for h in range(8)]+['up', 'down', 'motor']
                rh = 21
                for r_, lab in enumerate(rows_lab):
                    dr.text((RX0, RY0+r_*rh), lab, font=f_tiny, fill=INK2)
                    dr.line((RX0+50, RY0+r_*rh+rh//2, RX1, RY0+r_*rh+rh//2), fill=(240, 242, 245))
                ks = np.flatnonzero((tt > t-span_t) & (tt <= t))
                for kk in ks:
                    xpx = RX0+50+(tt[kk]-(t-span_t))/span_t*(RX1-RX0-50)
                    for h in np.flatnonzero(hs[kk]):
                        dr.line((xpx, RY0+h*rh+3, xpx, RY0+h*rh+rh-3), fill=INK, width=2)
                    for o in np.flatnonzero(fo[kk]):
                        dr.line((xpx, RY0+(8+o)*rh+3, xpx, RY0+(8+o)*rh+rh-3), fill=SPIKE, width=2)
                    if fire[kk]:
                        dr.line((xpx, RY0+10*rh+3, xpx, RY0+10*rh+rh-3), fill=LEARN if gate[kk] else CLASSIC, width=2)
                if t-span_t < B.T0 <= t:
                    xc = RX0+50+(B.T0-(t-span_t))/span_t*(RX1-RX0-50)
                    dr.line((xc, RY0-4, xc, RY0+11*rh), fill=(12, 140, 70), width=1)
                writer.append_data(np.asarray(img))
            for _ in range(FPS):
                writer.append_data(np.asarray(img))
            img.save(out/f'r5s{a.seed}_{tag}_network_still.png')
        finally:
            writer.close()
        entry = dict(task=task, target=target, case_index=ci, case_seed=c['seed'], video=str(vpath), window_s=[ta, tb],
                     slow=a.slow, hidden_spikes=int(hs[sel].sum()), output_spikes=int(fo[sel].sum()),
                     learned_pulse_ticks=int((fire & gate & sel).sum()), score=res['score'])
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    rr.close()
    B.DASH_M['value'] = TRAINED['dash']
    (out/f'manifest_network_r5s{a.seed}.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
