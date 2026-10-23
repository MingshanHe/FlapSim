"""plan v11 单任务讲解视频：左边是任务的三维飞行画面，右边是脉冲下行网络的实时可视化，下面是脉冲栅格、姿态与设定点、
执行器指令、任务进度四条时间曲线。所有量都来自同一次闭环飞行的记录（Net8.tick 被包一层来记录状态，不重算）。

    MUJOCO_GL=osmesa python -W ignore -m experiments.algorithms.spike_steer.render_v11 --task dash --case 0 --seed 4

输入：artifacts/spike_steer/plan_v11/homepc_snn/s<seed>/selected.json（种子 4、5）或 plan_v11/snn/s<seed>/selected.json（种子 1–3），
      plan_v11/baseline.json（经典对手，用来给出同一案例的分数比值）。
输出：artifacts/spike_steer/plan_v11/videos/v11_<task>_s<seed>_case<case>.mp4、同名 _still.png（到达目标时的一帧）和 .json（摘要）。
视频用 ffmpeg 管道写出（libx264），不依赖 imageio。
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
import mujoco                                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
from . import bench as B                                                 # noqa: E402
from . import bench8 as E                                                # noqa: E402
from . import bench9 as T                                                # noqa: E402
from . import bench10 as T10                                             # noqa: E402
from .descend8 import Learned8, Net8, N_G                                # noqa: E402
from ..neuro_gates import gates                                          # noqa: E402
from ..snn_pid_compare.environment import CENTER, HALF                   # noqa: E402

FPS = 30
W, H = 1920, 1080
VX, VY, VW, VH = 30, 84, 880, 520                 # 三维视图位置与大小
NX0 = 950                                          # 网络面板左边界
INK, INK2, MUTE, PALE = (24, 30, 38), (92, 100, 110), (200, 205, 212), (240, 242, 245)
POS, NEG = (222, 120, 30), (40, 110, 200)          # 兴奋 / 抑制连接
SPIKE, LEARN, CLASSIC, GOOD, BAD = (230, 60, 50), (222, 120, 30), (140, 148, 160), (12, 140, 70), (194, 59, 59)
ROLL_C, PITCH_C = (40, 110, 200), (200, 60, 120)
KAPTON, GRID, GRID_MAJOR, TRAIL = [.86, .55, .14, .55], [.64, .70, .79, 1.], [.48, .56, .68, 1.], [.16, .47, .84, .9]
ARROW1, ARROW2 = [.05, .62, .30, .95], [.84, .45, .08, .95]
CH = ('roll', 'pitch', 'gx', 'gy', 'gz', 'ax', 'ay', 'az')
CMD = ('evade·cos', 'evade·sin', 'dash·cos', 'dash·sin')
IN_NODES = [(i, f'{c} +') for i, c in enumerate(CH)] + [(8+i, f'{c} −') for i, c in enumerate(CH)] + [(16, 'bias')] + \
           [(17+i, f'{c} +') for i, c in enumerate(CMD)] + [(21+i, f'{c} −') for i, c in enumerate(CMD)]
NAME = {0: 'forward', 45: 'fwd-left', 90: 'left', 135: 'back-left', 180: 'back', 225: 'back-right', 270: 'right', 315: 'fwd-right'}
TITLE = dict(evade='Evade 0.10 m', redirect='Redirect: 0.08 m, then 90° turn and 0.10 m', dash='Dash 0.15 m and stop')
GOAL_CM = dict(evade=10., redirect=10., dash=15.)


def font(size, bold=False):
    for n in (('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf') if bold else
              ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_geom(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel() if mat is None else np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def style_model(m):
    for gi in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ''
        if name in ('left_wing_visual', 'right_wing_visual'):
            m.geom_rgba[gi] = KAPTON
        if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_group[gi] = 5
    m.light_castshadow[:] = 0
    m.light_diffuse[:] = .75
    m.light_ambient[:] = .25
    m.vis.headlight.ambient[:] = .45
    m.vis.headlight.diffuse[:] = .55
    m.vis.headlight.specular[:] = .1


def lerp(c0, c1, a):
    a = float(np.clip(a, 0., 1.))
    return tuple(int(round(c0[k]+(c1[k]-c0[k])*a)) for k in range(3))


class NetLog:
    """包住 Net8.tick：每个 tick 记录输入 x、隐藏层是否发放（两个子步之一）、膜电位、输出发放、分级单元。"""

    def __init__(self):
        self.rows, self.orig = [], Net8.tick
        log = self

        def tick(net, x):
            p0 = net.p
            fired, ref = log.orig(net, x)
            hs = net.hist[p0 % net.L].astype(bool) | net.hist[(p0+1) % net.L].astype(bool)
            log.rows.append(dict(x=np.asarray(x, float).copy(), hs=hs, vh=net.v_h.copy(), vo=net.v_o.copy(),
                                 fo=np.asarray(fired, bool).copy(), g=net.g.copy()))
            return fired, ref
        self.tick = tick

    def __enter__(self):
        self.rows = []
        Net8.tick = self.tick
        return self

    def __exit__(self, *exc):
        Net8.tick = self.orig


def record(env, task, c, policy):
    """闭环飞一局，记录每个 2 ms 步的 (time, qpos, 执行器指令 [alpha, beta, omega])，并返回带 trace 的结果。"""
    d, frames, orig = env.sim.data, [], env.step

    def step(action):
        out = orig(action)
        frames.append((float(d.time), d.qpos.copy(), np.asarray(action, float)*HALF+CENTER))
        return out
    env.step = step
    try:
        res = T10.run_episode10(env, task, c, policy, trace=True)
    finally:
        del env.step
    return res, frames


class Panel:
    """下方时间曲线面板：x 轴是时间 [ta, tb]，y 轴是 [lo, hi]。"""

    def __init__(self, dr, x0, y0, x1, y1, ta, tb, lo, hi, title, f_small, f_tiny):
        self.dr, self.x0, self.y0, self.x1, self.y1, self.ta, self.tb, self.lo, self.hi = dr, x0, y0, x1, y1, ta, tb, lo, hi
        dr.rectangle((x0, y0, x1, y1), outline=MUTE)
        dr.text((x0, y0-26), title, font=f_small, fill=INK)
        self.f_tiny = f_tiny

    def px(self, t):
        return self.x0+(t-self.ta)/(self.tb-self.ta)*(self.x1-self.x0)

    def py(self, v):
        return self.y1-(np.clip(v, self.lo, self.hi)-self.lo)/(self.hi-self.lo)*(self.y1-self.y0)

    def hline(self, v, col=PALE, label=None, width=1):
        self.dr.line((self.x0, self.py(v), self.x1, self.py(v)), fill=col, width=width)
        if label:
            self.dr.text((self.x0+4, self.py(v)-14), label, font=self.f_tiny, fill=INK2)

    def curve(self, tt, vv, t, col, width=2, dash=False):
        sel = (tt >= self.ta) & (tt <= min(t, self.tb))
        pts = [(self.px(a), self.py(b)) for a, b in zip(tt[sel][::2], vv[sel][::2])]
        if len(pts) < 2:
            return
        if dash:
            for i in range(0, len(pts)-1, 4):
                self.dr.line(pts[i:i+3], fill=col, width=width)
        else:
            self.dr.line(pts, fill=col, width=width)

    def cursor(self, t, t0):
        self.dr.line((self.px(t0), self.y0, self.px(t0), self.y1), fill=GOOD, width=1)
        self.dr.line((self.px(t), self.y0, self.px(t), self.y1), fill=INK, width=1)
        for s in np.arange(math.ceil(self.ta*2)/2, self.tb+1e-9, .5):
            self.dr.text((self.px(s)-12, self.y1+4), f'{s:.1f}', font=self.f_tiny, fill=INK2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=T.TASKS, default='dash')
    p.add_argument('--case', type=int, default=0, help='plan v11 盲测集中该任务的案例序号（0–63）')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--slow', type=float, default=8., help='放慢倍数')
    p.add_argument('--before', type=float, default=.3, help='指令前显示的秒数')
    p.add_argument('--after', type=float, default=2.2, help='指令后显示的秒数')
    p.add_argument('--out', default='artifacts/spike_steer/plan_v11/videos')
    a = p.parse_args()
    A11 = Path('artifacts/spike_steer/plan_v11')
    sdir = A11/(f'homepc_snn/s{a.seed}' if a.seed >= 4 else f'snn/s{a.seed}')
    spec = json.loads((sdir/'selected.json').read_text())
    meta = json.loads((sdir/'meta.json').read_text())
    base = json.loads((A11/'baseline.json').read_text())
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        B.prepare_source(E.SOURCE)
    B.worker_init(str(E.SOURCE), .15)
    E.load_signs()
    env = B._ENV
    m, d, body = env.sim.model, env.sim.data, env.air.env.body
    c = T.cases9(a.task, 2_099_100_000, 64)[a.case]
    t0 = c['t0']

    # ---- 飞两局：学习层（记录网络状态）和经典对手（只要分数）
    with NetLog() as log:
        res, frames = record(env, a.task, c, Learned8(spec))
    net = log.rows
    res_cls = T10.run_episode10(env, a.task, c, T.Classical9(base['law'], base['opponent'][a.task][T.klass(c)]))
    tr = np.array(res.pop('trace'))
    n = min(len(net), len(tr))
    assert abs(len(net)-len(tr)) <= 2, (len(net), len(tr))
    net, tr = net[-n:], tr[-n:]
    tt = tr[:, 0]
    pitch, roll = np.degrees(tr[:, 1]), np.degrees(tr[:, 2])
    ref_roll, ref_pitch = np.degrees(tr[:, 10]), np.degrees(tr[:, 11])
    fire = tr[:, 12] > .5
    cmd_on = ~np.isnan(tr[:, 13])
    gate, off_t = np.zeros(n, bool), None
    for k in range(n):                                        # Learned8 的门：指令开着，或关掉后不到 1 s
        if k and cmd_on[k-1] and not cmd_on[k]:
            off_t = tt[k]
        gate[k] = cmd_on[k] or (off_t is not None and tt[k]-off_t < spec.get('tail_s', 1.))
    xs = np.array([r['x'] for r in net])
    hs = np.array([r['hs'] for r in net])
    vh = np.array([r['vh'] for r in net])
    fo = np.array([r['fo'] for r in net])
    gg = np.array([r['g'] for r in net])
    times = np.array([f[0] for f in frames])
    acts = np.array([f[2] for f in frames])                   # alpha, beta, omega
    i0 = min(int(np.searchsorted(times, t0)), len(frames)-1)
    d.qpos[:] = frames[i0][1]
    mujoco.mj_forward(m, d)
    start = d.xpos[body].copy()
    yaw = float(math.atan2(d.xmat[body].reshape(3, 3)[1, 0], d.xmat[body].reshape(3, 3)[0, 0]))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.])
    left = np.array([-fwd[1], fwd[0], 0.])
    rot = np.column_stack([fwd, left, [0., 0., 1.]])
    path = []
    for (_, q, _) in frames:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        path.append(d.xpos[body].copy())
    path = np.array(path)
    ta, tb = t0-a.before, min(t0+a.after, float(times[-1]))
    d1 = math.cos(math.radians(c['dir_deg']))*fwd+math.sin(math.radians(c['dir_deg']))*left
    along = (tr[:, 3:6]-start)@d1*100.                        # 沿第一指令方向的位移，cm
    if a.task == 'redirect':
        d2 = math.cos(math.radians(c['dir2_deg']))*fwd+math.sin(math.radians(c['dir2_deg']))*left
        along2 = (tr[:, 3:6]-start)@d2*100.
    speed = np.linalg.norm(tr[:, 6:8], axis=1)*100.
    height = (tr[:, 5]-start[2])*100.

    # ---- 相机：看住窗口内的航迹
    win = (times >= ta) & (times <= tb)
    lo, hi = path[win].min(0), path[win].max(0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (lo+hi)/2
    cam.azimuth, cam.elevation = math.degrees(yaw)+128., -30.
    cam.distance = max(.34, .22+1.0*float(np.linalg.norm(hi-lo)))
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    u = np.cross(view, [0., 0., 1.])
    u /= np.linalg.norm(u)
    bd_rot, bd_pos = np.column_stack([u, np.cross(u, view), view]), np.asarray(cam.lookat)+view*3.
    dirs = [(c['dir_deg'], ARROW1)]+([(c['dir2_deg'], ARROW2)] if a.task == 'redirect' else [])
    style_model(m)
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    rr = mujoco.Renderer(m, height=VH, width=VW, max_geom=4000)
    f_title, f_mid, f_small, f_tiny, f_micro = font(28, True), font(19, True), font(16), font(13), font(11)

    # ---- 网络面板的几何
    W_in, W_out, W_g = np.asarray(spec['W_in']), np.asarray(spec['W_out']), np.asarray(spec['W_g'])
    S = float(spec['spike_scale'])
    w_ih, w_ho, w_hg = W_in.sum(2), S*W_out.sum(2), S*W_g.sum(2)             # H×25, 2×H, G×H
    top = sorted([(abs(w_ih[h, i]), h, i) for h in range(8) for i in range(25)], reverse=True)[:64]
    w_top = top[0][0]
    w_omax = max(abs(w_ho).max(), abs(w_hg).max())
    in_x, hid_x, out_x = NX0+150, NX0+470, NX0+700
    in_y = {i: 104+k*21 for k, (i, _) in enumerate(IN_NODES)}
    hid_y = [150+k*58 for k in range(8)]
    out_y = dict(up=180, down=270, g0=380, g1=450)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'v11_{a.task}_s{a.seed}_case{a.case}'
    vpath = out/f'{stem}.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(vpath)], stdin=subprocess.PIPE)
    n_frames = int(round((tb-ta)*FPS*a.slow))
    prev_k = int(np.searchsorted(tt, ta))
    ratio = res['score']/res_cls['score']
    lab_dir = NAME[int(c['dir_deg'])]+(f" → {NAME[int(c['dir2_deg'])]}" if a.task == 'redirect' else '')
    still_done = False
    try:
        for f in range(n_frames):
            t = ta+(f+1)/(FPS*a.slow)
            k = min(int(np.searchsorted(tt, t)), n-1)
            kf = slice(prev_k, k+1)
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
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [1., wdt, .00004], np.r_[start[:2]+left[:2]*gx, .0003], col, rot)
            for (phi, colr) in dirs:
                pr = math.radians(phi)
                dv = math.cos(pr)*fwd+math.sin(pr)*left
                L = .12
                arot = np.column_stack([dv, np.cross([0., 0., 1.], dv), [0., 0., 1.]])
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [L/2, .005, .0004], np.r_[start[:2]+dv[:2]*L/2, .0008], colr, arot)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.011, 0, 0], np.r_[start[:2]+dv[:2]*L, .001], colr)
            for j in range(int(np.searchsorted(times, ta)), i, 6):
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0016, 0, 0], path[j], TRAIL)
            img = Image.new('RGB', (W, H), (255, 255, 255))
            img.paste(Image.fromarray(rr.render()), (VX, VY))
            dr = ImageDraw.Draw(img)

            # ---- 标题
            dr.text((VX, 14), f'{TITLE[a.task]}  ·  {lab_dir}  ·  spiking descending layer in closed loop', font=f_title, fill=INK)
            dr.text((VX, 52), f"plan v11 SNN + ES, seed {a.seed}, generation {meta['generation']} (HomePC)  ·  blind case {a.case} "
                    f"(seed {c['seed']})  ·  {1/a.slow:g}x speed  ·  t = {t:5.3f} s  ({'+' if t >= t0 else '−'}{abs(t-t0)*1000:4.0f} ms from command)",
                    font=f_small, fill=INK2)
            dr.rectangle((VX, VY, VX+VW, VY+VH), outline=MUTE)
            if t0 <= t < t0+.4:
                dr.text((VX+12, VY+10), 'COMMAND', font=f_mid, fill=GOOD)
            if a.task == 'redirect' and res.get('t_switch') is not None and t-t0 >= res['t_switch']:
                dr.text((VX+12, VY+10), f"turn at +{res['t_switch']:.2f} s", font=f_small, fill=(214, 115, 20))
            if res['goal_t'] is not None and t-t0 >= res['goal_t']:
                dr.text((VX+12, VY+34), f"goal at +{res['goal_t']:.2f} s", font=f_small, fill=GOOD)
            if not res['flown'] and t >= res['elapsed']:
                dr.text((VX+12, VY+58), 'crash', font=f_small, fill=BAD)
            dr.text((VX, VY+VH+8), f"score {res['score']:.3f}  ·  classical opponent on the same case {res_cls['score']:.3f}  ·  ratio {ratio:.2f} "
                    f"(lower is better)  ·  motor pulses {res['pulses']}", font=f_small, fill=GOOD if ratio < 1 else INK2)
            gk = gate[k]
            dr.rectangle((VX, VY+VH+34, VX+VW, VY+VH+58), fill=(255, 236, 215) if gk else PALE, outline=MUTE)
            dr.text((VX+8, VY+VH+37), 'gate: LEARNED LAYER in charge (command on, or < 1 s after)' if gk else 'gate: classical hover law in charge',
                    font=f_small, fill=LEARN if gk else INK2)

            # ---- 网络：连接
            xk = xs[k]
            for (wa, h, ii) in top:
                act = min(1., xk[ii])*wa/w_top
                col = lerp((236, 239, 243), POS if w_ih[h, ii] > 0 else NEG, .12+.88*act)
                dr.line((in_x+8, in_y[ii], hid_x-18, hid_y[h]), fill=col, width=1+int(3*act))
            for h in range(8):
                for key, w in (('up', w_ho[0, h]), ('down', w_ho[1, h]), ('g0', w_hg[0, h]), ('g1', w_hg[1, h])):
                    a_ = min(1., abs(w)/w_omax)
                    lit = hs_f[h]
                    col = lerp((236, 239, 243), POS if w > 0 else NEG, (.3+.7*a_) if lit else .1+.2*a_)
                    dr.line((hid_x+18, hid_y[h], out_x-22, out_y[key]), fill=col, width=3 if lit else 1)
            # 输入节点
            dr.text((in_x-150, 76), '25 inputs', font=f_mid, fill=INK)
            dr.text((in_x+20, 76), 'IMU halves (rectified), bias, command channels', font=f_tiny, fill=INK2)
            for (ii, lab) in IN_NODES:
                v = float(np.clip(xk[ii], 0., 1.))
                y = in_y[ii]
                col = (40, 110, 200) if ii < 16 else ((120, 120, 120) if ii == 16 else (12, 140, 70))
                dr.ellipse((in_x-7, y-7, in_x+7, y+7), fill=lerp((245, 246, 248), col, v), outline=INK2)
                dr.text((in_x-150, y-7), lab, font=f_micro, fill=INK if v > .05 else INK2)
            # 隐藏层
            dr.text((hid_x-70, 108), '8 hidden LIF', font=f_mid, fill=INK)
            for h in range(8):
                y = hid_y[h]
                v = float(np.clip(vh[k, h], 0., 1.))
                dr.ellipse((hid_x-17, y-17, hid_x+17, y+17), fill=lerp((245, 246, 248), (250, 200, 120), v), outline=INK2, width=2)
                hh = int(30*v)
                dr.rectangle((hid_x+23, y+15-hh, hid_x+29, y+15), fill=(250, 170, 60))
                dr.rectangle((hid_x+23, y-15, hid_x+29, y+15), outline=MUTE)
                if hs_f[h]:
                    dr.ellipse((hid_x-24, y-24, hid_x+24, y+24), outline=SPIKE, width=4)
                dr.text((hid_x-9, y-8), f'h{h+1}', font=f_micro, fill=INK)
            dr.text((hid_x-120, hid_y[-1]+30), 'fill / bar = membrane potential (threshold 1)\nred ring = spike in this frame\n'
                    'edges: orange = excitatory, blue = inhibitory\n(top 64 input weights, summed over the 11 delays)', font=f_micro, fill=INK2)
            # 输出
            dr.text((out_x-30, 130), 'outputs', font=f_mid, fill=INK)
            for key, lab, o in (('up', 'up (pitch-up pulse)', 0), ('down', 'down (pitch-down pulse)', 1)):
                y = out_y[key]
                on = fo_f[o]
                dr.ellipse((out_x-20, y-20, out_x+20, y+20), fill=SPIKE if on else (245, 246, 248), outline=INK2, width=2)
                dr.text((out_x+28, y-9), lab, font=f_small, fill=INK)
            for key, lab, gi, colr in (('g0', 'roll set-point', 0, ROLL_C), ('g1', 'pitch set-point', 1, PITCH_C)):
                y = out_y[key]
                dr.rectangle((out_x-20, y-20, out_x+20, y+20), fill=(245, 246, 248), outline=INK2, width=2)
                dr.text((out_x-7, y-10), f'g{gi+1}', font=f_small, fill=INK)
                rv = math.degrees(B.REF_MAX*math.tanh(gg[k, gi]))
                gx0, gx1 = out_x+28, out_x+190
                dr.text((gx0, y-20), f'graded → {lab}', font=f_small, fill=INK)
                dr.rectangle((gx0, y+4, gx1, y+18), outline=MUTE)
                mid = (gx0+gx1)/2
                xr = mid+rv/math.degrees(B.REF_MAX)*(gx1-gx0)/2
                dr.rectangle((min(mid, xr), y+4, max(mid, xr), y+18), fill=colr)
                dr.line((mid, y, mid, y+22), fill=INK2)
                dr.text((gx1+6, y+1), f'{rv:+5.1f}°', font=f_small, fill=INK)
            dr.line((out_x, out_y['down']+20, out_x, out_y['g0']-22), fill=MUTE, width=1)
            yb = 520
            dr.rectangle((out_x-30, yb, out_x+230, yb+52), fill=(255, 236, 215) if (fire_f and gk) else PALE, outline=INK2, width=2)
            dr.text((out_x-20, yb+6), 'motor pulse (3 ticks, 3807 rad/s)', font=f_small, fill=INK)
            dr.text((out_x-20, yb+28), ('FIRING (learned)' if gk else 'firing (hover law)') if fire_f else '—',
                    font=f_mid, fill=(SPIKE if gk else CLASSIC) if fire_f else INK2)
            dr.text((out_x-30, yb+60), 'either output spike → pulse; set-points → PID gimbal', font=f_micro, fill=INK2)

            # ---- 下方四个面板
            PY0, PY1 = 690, 1010
            cols = [(30, 480), (520, 950), (990, 1420), (1460, 1890)]
            # (1) 脉冲栅格
            x0, x1 = cols[0]
            pa = Panel(dr, x0+40, PY0, x1, PY1, ta, tb, 0, 1, 'spikes: hidden h1–h8, outputs, motor pulses', f_small, f_tiny)
            rows_lab = [f'h{h+1}' for h in range(8)]+['up', 'down', 'motor']
            rh = (PY1-PY0)/len(rows_lab)
            for g0, g1 in _runs(tt, gate & (tt >= ta) & (tt <= min(t, tb))):
                dr.rectangle((pa.px(g0), PY0+1, pa.px(g1), PY1-1), fill=(255, 244, 230))
            for r_, lab in enumerate(rows_lab):
                dr.text((x0, PY0+r_*rh+4), lab, font=f_tiny, fill=INK2)
                dr.line((pa.x0, PY0+(r_+1)*rh, pa.x1, PY0+(r_+1)*rh), fill=PALE)
            sel = (tt >= ta) & (tt <= min(t, tb))
            for kk in np.flatnonzero(sel):
                xpx = pa.px(tt[kk])
                for h in np.flatnonzero(hs[kk]):
                    dr.line((xpx, PY0+h*rh+3, xpx, PY0+(h+1)*rh-3), fill=INK, width=1)
                for o in np.flatnonzero(fo[kk]):
                    dr.line((xpx, PY0+(8+o)*rh+3, xpx, PY0+(9+o)*rh-3), fill=SPIKE, width=2)
                if fire[kk]:
                    dr.line((xpx, PY0+10*rh+3, xpx, PY0+11*rh-3), fill=LEARN if gate[kk] else CLASSIC, width=2)
            pa.cursor(t, t0)
            dr.text((pa.x0, PY1+22), 'shaded = learned layer in charge · motor: orange learned, grey hover law', font=f_micro, fill=INK2)
            # (2) 姿态与设定点
            x0, x1 = cols[1]
            pb = Panel(dr, x0+40, PY0, x1, PY1, ta, tb, -40, 40, 'attitude (deg): solid = true, dashed = SNN set-point', f_small, f_tiny)
            for v in (-30, -15, 0, 15, 30):
                pb.hline(v, PALE if v else MUTE)
                dr.text((x0, pb.py(v)-7), f'{v:+d}', font=f_tiny, fill=INK2)
            pb.curve(tt, ref_pitch, t, PITCH_C, 2, dash=True)
            pb.curve(tt, ref_roll, t, ROLL_C, 2, dash=True)
            pb.curve(tt, pitch, t, PITCH_C, 2)
            pb.curve(tt, roll, t, ROLL_C, 2)
            pb.cursor(t, t0)
            dr.text((pb.x0, PY1+22), 'pitch', font=f_tiny, fill=PITCH_C)
            dr.text((pb.x0+50, PY1+22), 'roll', font=f_tiny, fill=ROLL_C)
            dr.text((pb.x0+100, PY1+22), f'tilt {math.degrees(tr[k, 9]):4.1f}°  (crash at 60°)', font=f_micro, fill=INK2)
            # (3) 执行器指令
            x0, x1 = cols[2]
            pc = Panel(dr, x0+40, PY0, x1, PY1, ta, tb, -1, 1, 'actuators: gimbal α, β (±1 = ±0.3 rad), motor ω (grey)', f_small, f_tiny)
            for v in (-.5, 0, .5):
                pc.hline(v, PALE if v else MUTE)
                dr.text((x0, pc.py(v)-7), f'{v:+.1f}', font=f_tiny, fill=INK2)
            om = (acts[:, 2]-CENTER[2])/HALF[2]
            pc.curve(times, om, t, (120, 120, 120), 1)
            pc.curve(times, acts[:, 0]/HALF[0], t, ROLL_C, 2)
            pc.curve(times, acts[:, 1]/HALF[1], t, PITCH_C, 2)
            pc.cursor(t, t0)
            ia = min(int(np.searchsorted(times, t)), len(acts)-1)
            dr.text((pc.x0, PY1+22), f'α {math.degrees(acts[ia, 0]):+5.1f}°', font=f_tiny, fill=ROLL_C)
            dr.text((pc.x0+90, PY1+22), f'β {math.degrees(acts[ia, 1]):+5.1f}°', font=f_tiny, fill=PITCH_C)
            dr.text((pc.x0+180, PY1+22), f'ω {acts[ia, 2]:6.0f} rad/s (3115–3807)', font=f_tiny, fill=(110, 110, 110))
            # (4) 任务进度
            x0, x1 = cols[3]
            top_cm = max(GOAL_CM[a.task]+10., float(along[(tt >= ta) & (tt <= tb)].max())+3.)
            pd_ = Panel(dr, x0+40, PY0, x1, PY1, ta, tb, -8, top_cm, 'progress (cm): travel along the command, height', f_small, f_tiny)
            goal_label = f'goal {GOAL_CM[a.task]:g} cm' if a.task != 'redirect' else 'turn at 8 cm'
            pd_.hline(SWITCH_CM if a.task == 'redirect' else GOAL_CM[a.task], (160, 200, 170), goal_label)
            pd_.hline(0, MUTE)
            for v in (0, 5, 10, 15, 20):
                if v <= top_cm:
                    dr.text((x0, pd_.py(v)-7), f'{v}', font=f_tiny, fill=INK2)
            pd_.curve(tt, height, t, (120, 120, 120), 1)
            pd_.curve(tt, along, t, (12, 100, 60), 3)
            if a.task == 'redirect':
                pd_.curve(tt, along2, t, (214, 115, 20), 3)
            pd_.cursor(t, t0)
            dr.text((pd_.x0, PY1+22), 'along command', font=f_tiny, fill=(12, 100, 60))
            if a.task == 'redirect':
                dr.text((pd_.x0+110, PY1+22), 'along command 2', font=f_tiny, fill=(214, 115, 20))
            dr.text((pd_.x0+230, PY1+22), f'height (grey) · speed {speed[k]:4.1f} cm/s', font=f_tiny, fill=(110, 110, 110))
            if res['goal_t'] is not None and t-t0 >= res['goal_t'] and not still_done:
                still_done = True
                img.save(out/f'{stem}_still.png')
            ff.stdin.write(np.asarray(img).tobytes())
        for _ in range(FPS*2):
            ff.stdin.write(np.asarray(img).tobytes())
        img.save(out/f'{stem}_last.png')
        if not still_done:
            img.save(out/f'{stem}_still.png')
    finally:
        ff.stdin.close()
        ff.wait()
    rr.close()
    summary = dict(task=a.task, case=a.case, case_seed=c['seed'], dir_deg=c['dir_deg'], dir2_deg=c.get('dir2_deg'), seed=a.seed,
                   generation=meta['generation'], controller=str(sdir/'selected.json'), video=str(vpath), window_s=[ta, tb], slow=a.slow,
                   score=res['score'], score_classical=res_cls['score'], ratio=ratio, goal_t=res['goal_t'], flown=res['flown'],
                   pulses=res['pulses'], hidden_spikes_in_window=int(hs[(tt >= ta) & (tt <= tb)].sum()),
                   output_spikes_in_window=int(fo[(tt >= ta) & (tt <= tb)].sum()), ticks=int(n), frames=n_frames)
    (out/f'{stem}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


SWITCH_CM = 8.


def _runs(tt, mask):
    """连续 True 段的 (t_start, t_end) 列表。"""
    out, start = [], None
    for k in range(len(mask)):
        if mask[k] and start is None:
            start = tt[k]
        if (not mask[k] or k == len(mask)-1) and start is not None:
            out.append((start, tt[k]))
            start = None
    return out


if __name__ == '__main__':
    main()
