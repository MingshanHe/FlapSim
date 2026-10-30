"""相位锁定电机脉冲的飞行演示视频（第三人称主画面 + 右上机构细节 + 右下实时数据 + 左上阶段提示）。

一次连续的名义悬停：t = 2.90 s 起连续 4 个翅拍在曲柄相位 135° ± 22.5° 打脉冲（抬头），PID 拉回（会过冲、来回摆）；
t = 3.85 s 起连续 4 个翅拍在 315° ± 22.5° 打脉冲（低头），PID 再次拉回。低头组的时刻选在抬头组之后的回摆顶点，
此时打下去效果最干净（单独比较：不打低头组时 4.05 s 的俯仰约 −3°，打了约 −21°）。
为了让效果在画面里看得清，脉冲期间的转速指令是 PEAK（默认 4500 rad/s），**高于论文接口的上限 3807 rad/s**——
这是可视化用的夸大，不是论文实验条件；其余（8.2 ms 平滑、电机模型、PID、物理）不变。
同一案例完全不打脉冲的飞行画在俯仰曲线里作灰色虚线。

    MUJOCO_GL=egl python -W ignore -m experiments.algorithms.spike_steer.render_pulse_flight [--peak 4500] [--beats 4] [--every 2]

输出：artifacts/spike_steer/pulse_video/pulse_flight.mp4、pulse_flight_still_up.png、pulse_flight_still_down.png、pulse_flight.json。
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
import mujoco                                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
from ..robust_control.environment import ResearchEnv                     # noqa: E402
from ..snn_pid_compare.environment import LOW, HIGH, CENTER, HALF        # noqa: E402
from ..snn_pid_compare.common import DT, prepare_source                  # noqa: E402
from ..topology_hover.environment import case as make_case               # noqa: E402
from ..neuro_gates import gates                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/spike_steer/pulse_video'
W, H, FPS = 1920, 1080, 30
HALF_WIN = math.radians(22.5)
BURSTS = [(2.90, 135), (3.85, 315)]      # (开始找窗口的时刻, 窗口中心)，同一次飞行
T0, T1 = 2.86, 4.45                      # 视频覆盖的仿真时间
UPC, DNC = (232, 120, 20), (30, 110, 210)
INK2, GRID, MUTE = (110, 116, 126), (232, 235, 240), (200, 205, 212)
KAPTON, GRIDC, GRIDM, TRAIL = [.86, .55, .14, .6], [.74, .78, .84, 1.], [.58, .64, .74, 1.], [.16, .47, .84, .7]


def font(size):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def add(scn, gtype, size, pos, rgba, mat=None, emission=None):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel() if mat is None else np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    if emission is not None:
        g.emission = emission
    scn.ngeom += 1
    return g


def line(scn, a, b, width, rgba):
    g = add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], [0, 0, 0], rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))


def rotate(v, axis, angle):
    axis = axis/np.linalg.norm(axis)
    return v*math.cos(angle)+np.cross(axis, v)*math.sin(angle)+axis*(axis@v)*(1-math.cos(angle))


# ---------------------------------------------------------------- 仿真
def fly(env, case, peak, beats, chunk, bursts, every=1):
    env.configure(case)
    env.reset(seed=case['seed'])
    m, d = env.sim.model, env.sim.data
    crank, motor = m.joint('carrier_2_joint'), m.joint('motor_joint')
    sim, step0, adv0 = env.sim, env.sim.step, env.air.advance
    st = dict(on=False, which=None, raw=float(CENTER[2]), n=0, acc_on=False, acc_which=None, acc_raw=0.)
    frames = []

    def advance(command, p):              # 脉冲期间把指令换成 PEAK（越过接口上限，仅为可视化）
        command = np.array(command, float)
        if st['on']:
            command[2] = peak
        st['raw'] = float(command[2])
        st['acc_on'] |= st['on']
        st['acc_which'] = st['which'] or st['acc_which']
        st['acc_raw'] = max(st['acc_raw'], st['raw'])
        return adv0(command, p)

    def chunked(steps=1):
        n = 0
        while n < steps:
            k = min(chunk, steps-n)
            sim.backend.advance(k, sim._command, sim.external_joint_forces, sim.external_body_forces, sim._gimbal_command)
            n += k
            if n == steps:
                st['n'] += 1
            if T0 <= float(d.time) <= T1+1e-9 and (k < steps or st['n'] % every == 0):   # 每 every 个 tick 记一帧；期间的脉冲合并进这一帧
                frames.append(dict(t=float(d.time), qpos=d.qpos.copy(), crank=float(d.qpos[crank.qposadr[0]]),
                                   motor=float(d.qvel[motor.dofadr[0]]), raw=st['acc_raw'], smooth=float(env.air.action[2]),
                                   on=st['acc_on'], which=st['acc_which']))
                st['acc_on'], st['acc_which'], st['acc_raw'] = False, None, 0.
        return d
    if chunk:
        sim.step = chunked
    env.air.advance = advance
    oracle = gates.Oracle(env, 'derocked')
    ticks, fired = [], []
    count = {c: 0 for _, c in bursts}
    last = {c: None for _, c in bursts}
    try:
        while env.air.steps*DT < T1:
            t = env.air.steps*DT
            phase = float(d.qpos[crank.qposadr[0]])
            st['on'], st['which'] = False, None
            if peak is not None:
                for tb, c in bursts:
                    if t < tb-1e-9:
                        continue
                    inside = abs(math.remainder(phase-math.radians(c), 2*math.pi)) < HALF_WIN
                    cont = last[c] is not None and last[c] >= t-DT-1e-9
                    if inside and (cont or count[c] < beats):
                        if not cont:
                            count[c] += 1
                        st['on'], st['which'], last[c] = True, c, t
                        fired.append((round(t, 4), c))
            env.step((np.clip(env.teacher_command(), LOW, HIGH)-CENTER)/HALF)
            oracle.update()
            pos, vel, rot = env.air.env._state()
            ticks.append((t+DT, math.degrees(oracle.queue[-1][1]), pos.copy()))
    finally:
        sim.step, env.air.advance = step0, adv0
    return ticks, frames, fired


# ---------------------------------------------------------------- 画面
def style(m):
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_group[g] = 5
    for n in ('left_wing_visual', 'right_wing_visual'):
        m.geom_rgba[m.geom(n).id] = KAPTON
    for n in ('base_link_mount_visual', 'pitch_link_mount_visual'):   # 机体基座（装 IMU、姿态即论文中的俯仰）涂深色
        m.geom_rgba[m.geom(n).id] = [.34, .38, .46, 1.]
    m.light_castshadow[:] = 0
    m.light_diffuse[:] = .75
    m.light_ambient[:] = .25
    m.vis.headlight.ambient[:] = .45
    m.vis.headlight.diffuse[:] = .55
    m.vis.headlight.specular[:] = .1


def backdrop(scn, cam, shade=(.88, .89, .91, 1.), em=.55):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
    u = np.cross(view, [0., 0., 1.])
    u /= np.linalg.norm(u)
    rot = np.column_stack([u, np.cross(u, view), view])
    add(scn, mujoco.mjtGeom.mjGEOM_BOX, [8., 8., .01], np.asarray(cam.lookat)+view*3., shade, rot, emission=em)


def floor(scn, centre):
    for gx in np.arange(-.6, .601, .05):
        major = abs(round(gx/.25)*.25-gx) < 1e-6
        col, w = (GRIDM, .0012) if major else (GRIDC, .0007)
        add(scn, mujoco.mjtGeom.mjGEOM_BOX, [w, .6, .00004], [centre[0]+gx, centre[1], .0003], col)
        add(scn, mujoco.mjtGeom.mjGEOM_BOX, [.6, w, .00004], [centre[0], centre[1]+gx, .0003], col)


def mechanism_overlay(scn, m, d, active, firing):
    """曲柄销（红点）+ 曲柄臂 + 两个相位窗（橙 135°、蓝 315°），正在发脉冲的那个加亮。"""
    j = m.joint('carrier_2_joint').id
    axis, centre = d.xaxis[j].copy(), d.xanchor[j].copy()
    pin = d.site_xpos[m.site('crank_pin').id].copy()
    centre = centre+axis*(axis@(pin-centre))
    up = axis if axis[2] > 0 else -axis
    lift = up*.002
    phi = float(d.qpos[m.joint('carrier_2_joint').qposadr[0]])
    v = pin-centre
    r = np.linalg.norm(v)
    vr = v/r*r*1.6
    for c, col in ((135, UPC), (315, DNC)):
        bright = firing and active == c
        rgba = (*(np.array(col)/255), .98 if bright else .35)
        for k in range(-12, 13):
            a = math.radians(c)+k/12*HALF_WIN-phi
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.00048 if bright else .00036, 0, 0], centre+rotate(vr, axis, a)+lift, rgba)
    line(scn, centre+lift, centre+v+lift, .00018, (1, 1, 1, .95))
    add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.00062, 0, 0], centre+v+lift, (.9, .08, .08, 1.))


def arc_arrow(scn, p0, u, r, rgba):
    """从竖直方向到机体轴 u 的圆弧箭头；弧长 = 倾角。"""
    v = np.array([0., 0., 1.])
    th = math.acos(float(np.clip(u@v, -1, 1)))
    if th < math.radians(.8):
        return
    w = u-(u@v)*v
    w /= np.linalg.norm(w)
    n = max(4, int(math.degrees(th)/2))
    pts = [p0+r*(math.cos(a)*v+math.sin(a)*w) for a in np.linspace(0, th, n+1)]
    for a_, b_ in zip(pts[:-1], pts[1:]):
        line(scn, a_, b_, .0011, rgba)
    tip = pts[-1]
    tang = -math.sin(th)*v+math.cos(th)*w
    rad = math.cos(th)*v+math.sin(th)*w
    L = .0055
    for sgn in (1, -1):
        line(scn, tip, tip-L*(math.cos(math.radians(30))*tang+sgn*math.sin(math.radians(30))*rad), .0011, rgba)


class Strip:
    def __init__(self, dr, box, ta, tb, ya, yb, unit, f):
        self.dr, (self.x0, self.y0, self.x1, self.y1) = dr, box
        self.ta, self.tb, self.ya, self.yb = ta, tb, ya, yb
        for v in np.linspace(ya, yb, 3):
            y = self.py(v)
            dr.line((self.x0, y, self.x1, y), fill=GRID)
            dr.text((self.x0-8, y), f'{v:.0f}', fill=INK2, font=f, anchor='rm')
        dr.text((self.x0-8, self.y0-14), unit, fill=INK2, font=f, anchor='rm')

    def px(self, t):
        return self.x0+(t-self.ta)/(self.tb-self.ta)*(self.x1-self.x0)

    def py(self, v):
        v = min(max(v, self.ya), self.yb)
        return self.y1-(v-self.ya)/(self.yb-self.ya)*(self.y1-self.y0)

    def band(self, t0, t1, col):
        self.dr.rectangle((self.px(t0), self.y0, self.px(t1), self.y1), fill=col)

    def curve(self, tt, vv, col, w=2, dash=False):
        pts = [(self.px(t), self.py(v)) for t, v in zip(tt, vv)]
        if dash:
            for i in range(0, len(pts)-1, 6):
                self.dr.line(pts[i:i+3], fill=col, width=w)
        elif len(pts) > 1:
            self.dr.line(pts, fill=col, width=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peak', type=float, default=4500.)
    ap.add_argument('--beats', type=int, default=4)
    ap.add_argument('--every', type=int, default=2, help='每帧 = every 个 2 ms tick；2 → 每帧 4 ms，慢放约 8 倍')
    ap.add_argument('--seed-offset', type=int, default=7)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    env = ResearchEnv(str(source), mode='direct', task='hover')
    case = make_case(gates.SPLITS['evaluate']+a.seed_offset, T1+.2, 'nominal', 0)
    m = env.sim.model
    chunk = 400
    a.slow = 1/(a.every*DT*FPS)
    ref_ticks, _, _ = fly(env, case, None, 0, 0, BURSTS)
    style(m)
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    MW = 1340                                                    # 主画面宽；右侧 580 像素放细节与曲线
    main_r = mujoco.Renderer(m, height=H, width=MW, max_geom=6000)
    ins_r = mujoco.Renderer(m, height=400, width=520, max_geom=2000)
    opt = mujoco.MjvOption()
    opt.geomgroup[5] = 0
    dv, db = mujoco.MjData(m), mujoco.MjData(m)
    fb = m.joint('floating_base').qposadr[0]
    path = OUT/'pulse_flight.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(path)], stdin=subprocess.PIPE)
    f_ax = font(15)
    pitch_ref_all = np.array([k[1] for k in ref_ticks])
    f_tag = font(26)
    ticks, frames, fired = fly(env, case, a.peak, a.beats, chunk, BURSTS, a.every)
    tt = np.array([k[0] for k in ticks])
    pitch = np.array([k[1] for k in ticks])
    pitch_ref = pitch_ref_all[:len(pitch)]
    win = tt >= T0
    ft = np.array([f['t'] for f in frames])
    pulses = []
    for t, c in fired:
        if pulses and pulses[-1][2] == c and abs(pulses[-1][1]-t) < 1e-6:
            pulses[-1][1] = t+DT
        else:
            pulses.append([t, t+DT, c])
    span = {c: (min(p[0] for p in pulses if p[2] == c), max(p[1] for p in pulses if p[2] == c)) for c in (135, 315)}
    summary = dict(peak_rad_s=a.peak, interface_high_rad_s=float(HIGH[2]), beats=a.beats, slowdown=a.slow, frame_ms=a.every*DT*1e3,
                   pulses=[dict(t0=p[0], t1=p[1], centre=p[2]) for p in pulses],
                   pitch_max_after_up=float(pitch[(tt > span[135][0]) & (tt < span[135][0]+.25)].max()),
                   pitch_min_after_down=float(pitch[(tt > span[315][0]) & (tt < span[315][0]+.25)].min()),
                   pitch_at_down_start=float(pitch[np.searchsorted(tt, span[315][0])]))
    print(json.dumps(summary))

    def phase(t):
        if span[135][0] <= t <= span[135][1]:
            return 'up'
        if span[315][0] <= t <= span[315][1]:
            return 'down'
        return 'recover' if t > span[135][1] else None
    tags = [('up', '\u25b2  UP PULSE', UPC), ('down', '\u25bc  DOWN PULSE', DNC), ('recover', '\u21ba  PID RECOVER', (70, 140, 90))]
    look, stills = None, set()
    for i, f in enumerate(frames):
        dv.qpos[:] = f['qpos']
        mujoco.mj_forward(m, dv)
        top = dv.xpos[m.body('motor_sun').id].copy()
        base = dv.xpos[m.body('thrust_unit').id].copy()
        upv = (top-base)/np.linalg.norm(top-base)
        if look is None:
            rb0 = dv.xmat[m.body('thrust_unit').id].reshape(3, 3)
            yaw0 = math.degrees(math.atan2(rb0[1, 0], rb0[0, 0]))
        target = (top+base)/2+np.array([0, 0, .012])
        look = target if look is None else look+.12*(target-look)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.azimuth, cam.elevation, cam.distance = yaw0+125, -10, .26
        main_r.update_scene(dv, camera=cam, scene_option=opt)
        scn = main_r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        backdrop(scn, cam)
        k = min(int(np.searchsorted(tt, f['t'])), len(pitch)-1)
        tilt_col = np.array(UPC if pitch[k] >= 0 else DNC)/255
        p0 = top+upv*.014
        line(scn, p0, p0+np.array([0, 0, .05]), .0007, (.62, .66, .72, .8))         # 竖直参考
        zb = dv.xmat[m.body('base_link').id].reshape(3, 3)[:, 2].copy()              # 机体基座的轴（云台上方的推力单元几乎保持竖直）
        line(scn, p0, p0+zb*.05, .0013, (*tilt_col, .95))                            # 机体轴
        arc_arrow(scn, p0, zb, .046, (*tilt_col, .95))                               # 倾角圆弧箭头
        on = f['on']
        pc = np.array(UPC if f['which'] == 135 else DNC)/255
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0045 if on else .0018, 0, 0], p0+zb*.05,
            (*pc, 1.) if on else (.45, .48, .54, .8), emission=.6 if on else None)   # 脉冲指示
        img = Image.new('RGB', (W, H), (224, 226, 231))
        img.paste(Image.fromarray(main_r.render()), (0, 0))

        # 右上：机构细节（机体坐标下的俯视，外壳半透明）
        db.qpos[:] = f['qpos']
        db.qpos[fb:fb+3] = (0, 0, .8)
        db.qpos[fb+3:fb+7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, db)
        save = m.geom_rgba.copy()
        for n in ('base_link_mount_visual', 'pitch_link_mount_visual'):
            m.geom_rgba[m.geom(n).id, 3] = 0.
        m.geom_rgba[m.geom('base_link_visual').id, 3] = .16
        j = m.joint('carrier_2_joint').id
        pin = db.site_xpos[m.site('crank_pin').id]
        c2 = db.xanchor[j]+db.xaxis[j]*(db.xaxis[j]@(pin-db.xanchor[j]))
        cam2 = mujoco.MjvCamera()
        cam2.lookat[:] = c2
        cam2.azimuth, cam2.elevation, cam2.distance = 90, -89, .040
        ins_r.update_scene(db, camera=cam2, scene_option=opt)
        mechanism_overlay(ins_r.scene, m, db, f['which'], on)
        backdrop(ins_r.scene, cam2, (.95, .955, .965, 1.), 1.)
        inset = ins_r.render()
        m.geom_rgba[:] = save
        X0, Y0 = MW+30, 30
        img.paste(Image.fromarray(inset), (X0, Y0))
        dr = ImageDraw.Draw(img)
        dr.rectangle((X0-2, Y0-2, X0+521, Y0+401), outline=tuple(int(255*c) for c in pc) if on else MUTE, width=5 if on else 1)

        # 左上：阶段提示
        ph = phase(f['t'])
        for n_, (key, label, col) in enumerate(tags):
            y = 34+n_*54
            if key == ph:
                dr.rounded_rectangle((30, y, 290, y+42), radius=10, fill=col)
                dr.text((46, y+21), label, fill=(255, 255, 255), font=f_tag, anchor='lm')
            else:
                dr.rounded_rectangle((30, y, 290, y+42), radius=10, outline=(196, 200, 207), width=2)
                dr.text((46, y+21), label, fill=(160, 165, 174), font=f_tag, anchor='lm')

        # 右下：实时数据
        dr.rectangle((MW+30, 460, W-30, 1050), fill=(246, 247, 249))
        s1 = Strip(dr, (MW+90, 500, W-50, 700), T0, T1, 1800, a.peak+100, 'rad/s', f_ax)
        s2 = Strip(dr, (MW+90, 770, W-50, 1020), T0, T1, -30, 30, 'deg', f_ax)
        for q0, q1, c in pulses:
            if q0 <= f['t']:
                band = (252, 214, 180) if c == 135 else (196, 218, 246)
                s1.band(q0, min(q1, f['t']), band)
                s2.band(q0, min(q1, f['t']), band)
        sel = ft <= f['t']
        s1.curve(ft[sel], [g['raw'] for g, ok in zip(frames, sel) if ok], (214, 48, 49), 2)
        s1.curve(ft[sel], [g['motor'] for g, ok in zip(frames, sel) if ok], (40, 44, 52), 2)
        s2.curve(tt[win], pitch_ref[win], (170, 175, 184), 2, dash=True)
        idx = np.flatnonzero(win & (tt <= f['t']))
        for q in range(1, len(idx)):
            a_, b_ = idx[q-1], idx[q]
            dr.line((s2.px(tt[a_]), s2.py(pitch[a_]), s2.px(tt[b_]), s2.py(pitch[b_])), fill=UPC if pitch[b_] >= 0 else DNC, width=3)
        xn = s1.px(f['t'])
        dr.line((xn, 490, xn, 1030), fill=(120, 126, 136), width=1)
        ff.stdin.write(np.asarray(img.convert('RGB')).tobytes())
        for key, c in (('up', 135), ('down', 315)):
            if key not in stills and f['t'] >= span[c][0]+.08:
                img.convert('RGB').save(OUT/f'pulse_flight_still_{key}.png')
                stills.add(key)
    for _ in range(int(1.0*FPS)):
        ff.stdin.write(np.asarray(img.convert('RGB')).tobytes())
    ff.stdin.close()
    ff.wait()
    (OUT/'pulse_flight.json').write_text(json.dumps(summary, indent=1))
    print('wrote', path, len(frames), 'frames')


if __name__ == '__main__':
    main()
