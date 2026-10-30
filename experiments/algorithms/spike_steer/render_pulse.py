"""相位锁定电机脉冲的慢动作讲解视频：Scotch yoke 机构 + 论文中的核心脉冲（曲柄相位 135° 抬头 / 315° 低头）。

同一个名义悬停案例飞三次，到脉冲之前三次逐位相同：
  * 参考：只有 PID，不打脉冲；
  * UP：曲柄真实相位进入 135° ± 22.5° 窗口的那几个 2 ms tick，把电机转速指令打到上限 HIGH = 3807 rad/s（只打一个翅拍）；
  * DOWN：同样的脉冲，窗口中心 315°。
指令经过论文中的 8.2 ms 一阶平滑进入电机模型。物理步长 5 µs；把每个 2 ms tick 的 400 个子步拆成 20 子步一段推进
（已验证与一次推进 400 子步逐位相同），每 0.1 ms 记录一次完整状态，所以慢动作的每一帧都是真实仿真状态，不是插值。

    MUJOCO_GL=egl python -W ignore -m experiments.algorithms.spike_steer.render_pulse

输出：artifacts/spike_steer/pulse_video/pulse_slowmo.mp4、pulse_up_still.png、pulse_down_still.png、pulse_summary.json。
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
FPS, CHUNK = 30, 20                      # 每帧 20 个物理子步 = 0.1 ms 仿真时间 → 慢放 1/333
W, H = 1920, 1080
HALF_WIN = math.radians(22.5)            # 经典脉冲律的相位窗半宽
PRE, POST = .012, .075                   # 慢动作片段：脉冲前 12 ms 到脉冲后 75 ms
INK, INK2, MUTE, PALE, BG = (24, 30, 38), (92, 100, 110), (190, 196, 204), (236, 238, 242), (250, 250, 252)
RED, BLUE, GREY, ORANGE, GREEN = (214, 48, 49), (40, 110, 200), (150, 156, 165), (222, 120, 30), (12, 140, 70)


def font(size, bold=False):
    name = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def zh_font(size):
    for name in ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return font(size)


# ---------------------------------------------------------------- 仿真与记录
def fly(env, case, centre_deg, t_from, t_to):
    """飞一次；centre_deg=None 为参考。返回逐 tick 记录、逐 0.1 ms 记录和实际发脉冲的 tick。"""
    env.configure(case)
    env.reset(seed=case['seed'])
    m, d = env.sim.model, env.sim.data
    crank = m.joint('carrier_2_joint')
    motor = m.joint('motor_joint')
    flap = m.joint('left_flapping_joint')
    slider = m.joint('slider_joint')
    sim, original = env.sim, env.sim.step
    frames, raw = [], [float(CENTER[2])]

    def chunked(steps=1):                # 与 sim.step(400) 逐位相同，只是中途停下来记录
        n = 0
        while n < steps:
            k = min(CHUNK, steps-n)
            sim.backend.advance(k, sim._command, sim.external_joint_forces, sim.external_body_forces, sim._gimbal_command)
            n += k
            if t_from <= float(d.time) <= t_to+1e-9:
                frames.append(dict(t=float(d.time), qpos=d.qpos.copy(), crank=float(d.qpos[crank.qposadr[0]]),
                                   motor=float(d.qvel[motor.dofadr[0]]), flap=math.degrees(float(d.qpos[flap.qposadr[0]])),
                                   slider=1e3*float(d.qpos[slider.qposadr[0]]), cmd_raw=raw[0], cmd_smooth=float(env.air.action[2])))
        return d
    sim.step = chunked
    oracle = gates.Oracle(env, 'derocked')
    ticks, fired = [], []
    try:
        while env.air.steps*DT < t_to:
            t = env.air.steps*DT
            cmd = env.teacher_command()
            phase = float(d.qpos[crank.qposadr[0]])
            if centre_deg is not None and t >= t_from+.02-1e-9 and (not fired or fired[-1][0] >= t-DT-1e-9):
                inside = abs(math.remainder(phase-math.radians(centre_deg), 2*math.pi)) < HALF_WIN
                if inside and len(fired) < 3:
                    cmd[2] = HIGH[2]
                    fired.append((round(t, 4), math.degrees(phase) % 360))
            raw[0] = float(np.clip(cmd, LOW, HIGH)[2])
            env.step((np.clip(cmd, LOW, HIGH)-CENTER)/HALF)
            oracle.update()
            pos, vel, rot = env.air.env._state()
            ticks.append((t+DT, oracle.queue[-1][1], oracle.rate[1], rot[:, 2].copy(), rot[:, 0].copy()))
    finally:
        sim.step = original
    return ticks, frames, fired


# ---------------------------------------------------------------- 三维画面
def style(m):
    for n in ('base_link_mount_visual', 'pitch_link_mount_visual'):
        m.geom_rgba[m.geom(n).id, 3] = 0.
    m.geom_rgba[m.geom('base_link_visual').id, 3] = .18
    gi = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    for g in gi:
        m.geom_rgba[g, 3] = 0.
    for n in ('left_wing_visual', 'right_wing_visual'):
        m.geom_rgba[m.geom(n).id] = [.86, .55, .14, .55]
    m.vis.headlight.ambient[:] = .45
    m.vis.headlight.diffuse[:] = .6
    m.light_castshadow[:] = 0


def add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def rotate(v, axis, angle):
    axis = axis/np.linalg.norm(axis)
    return v*math.cos(angle)+np.cross(axis, v)*math.sin(angle)+axis*(axis@v)*(1-math.cos(angle))


def crank_overlay(scn, m, d, centre_deg, firing):
    """曲柄销（红点）、曲柄臂（白线）、脉冲相位窗（红弧，正在发脉冲时加亮）、行程换向点 90°/270°（灰点）。"""
    j = m.joint('carrier_2_joint').id
    axis, centre = d.xaxis[j].copy(), d.xanchor[j].copy()
    up = axis if axis[2] > 0 else -axis
    pin = d.site_xpos[m.site('crank_pin').id].copy()
    centre = centre+axis*(axis@(pin-centre))   # 投影到曲柄销所在平面
    lift = up*.002
    phi = float(d.qpos[m.joint('carrier_2_joint').qposadr[0]])
    v = pin-centre
    v -= axis*(axis@v)
    r = np.linalg.norm(v)
    ring = r*1.55
    vr = v/r*ring
    for k in range(-12, 13):             # 相位窗弧
        a = math.radians(centre_deg)+k/12*HALF_WIN-phi
        add_sphere(scn, centre+rotate(vr, axis, a)+lift, .00042, (1., .15, .1, .95 if firing else .55))
    for ref in (90, 270):                 # 行程换向点
        add_sphere(scn, centre+rotate(vr, axis, math.radians(ref)-phi)+lift, .00035, (.75, .78, .82, .9))
    add_line(scn, centre+lift, centre+v+lift, .00018, (1, 1, 1, .9))
    add_sphere(scn, centre+v+lift, .0006, (1., .1, .1, 1.))


# ---------------------------------------------------------------- 二维面板
class Plot:
    def __init__(self, dr, box, xa, xb, ya, yb, title, ylab, fs, ft):
        self.dr, (self.x0, self.y0, self.x1, self.y1) = dr, box
        self.xa, self.xb, self.ya, self.yb = xa, xb, ya, yb
        dr.rectangle(box, outline=MUTE, width=1)
        dr.text((self.x0, self.y0-26), title, fill=INK, font=fs)
        dr.text((self.x1-4, self.y0-24), ylab, fill=INK2, font=ft, anchor='ra')
        for v in np.linspace(ya, yb, 5):
            y = self.py(v)
            dr.line((self.x0, y, self.x1, y), fill=PALE)
            dr.text((self.x0-6, y), f'{v:.0f}', fill=INK2, font=ft, anchor='rm')

    def px(self, x):
        return self.x0+(x-self.xa)/(self.xb-self.xa)*(self.x1-self.x0)

    def py(self, y):
        y = min(max(y, self.ya), self.yb)
        return self.y1-(y-self.ya)/(self.yb-self.ya)*(self.y1-self.y0)

    def curve(self, xs, ys, col, width=2):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            self.dr.line(pts, fill=col, width=width)

    def band(self, xa, xb, col):
        self.dr.rectangle((self.px(xa), self.y0+1, self.px(xb), self.y1-1), fill=col)

    def xticks(self, ft, step):
        for v in np.arange(math.ceil(self.xa/step)*step, self.xb+1e-9, step):
            self.dr.text((self.px(v), self.y1+4), f'{v:.0f}', fill=INK2, font=ft, anchor='ma')


def dial(dr, cx, cy, R, phi_deg, centre_deg, firing, fs, ft):
    """曲柄相位盘，方向与左边俯视图一致：0° 时曲柄销在正下方，相位增加时顺时针转；90°/270° 是滑块的左右极限（行程换向）。
    PIL 的角度从 3 点钟方向顺时针量，所以图上角度 = 90° + 曲柄相位。"""
    ang = lambda phi: math.radians(90+phi)
    dr.ellipse((cx-R, cy-R, cx+R, cy+R), outline=MUTE, width=3)
    dr.pieslice((cx-R, cy-R, cx+R, cy+R), 90+centre_deg-22.5, 90+centre_deg+22.5, fill=(240, 90, 80) if firing else (248, 200, 196))
    for ref, lab in ((0, '0°'), (90, '90° reversal'), (180, '180°'), (270, '270° reversal')):
        x, y = cx+R*math.cos(ang(ref)), cy+R*math.sin(ang(ref))
        dr.ellipse((x-5, y-5, x+5, y+5), fill=GREY)
        x2, y2 = cx+(R+20)*math.cos(ang(ref)), cy+(R+16)*math.sin(ang(ref))
        dr.text((x2, y2), lab, fill=INK2, font=ft, anchor=('rm' if ref == 90 else 'lm' if ref == 270 else 'mm'))
    x, y = cx+R*.92*math.cos(ang(phi_deg)), cy+R*.92*math.sin(ang(phi_deg))
    dr.line((cx, cy, x, y), fill=INK, width=4)
    dr.ellipse((x-9, y-9, x+9, y+9), fill=RED)
    dr.ellipse((cx-5, cy-5, cx+5, cy+5), fill=INK)
    dr.text((cx, cy+R+52), f'crank phase {phi_deg:5.1f}° (clockwise, as in the top view)', fill=INK, font=ft, anchor='mm')


# ---------------------------------------------------------------- 主程序
SEGMENTS = [dict(key='up', centre=135, title='PITCH-UP pulse: motor speed command at its maximum while the crank is at 135° ± 22.5°',
                 zh='抬头脉冲：曲柄相位 135° ± 22.5° 时电机转速指令打到上限', col=ORANGE),
            dict(key='down', centre=315, title='PITCH-DOWN pulse: the same pulse half a crank turn later, at 315° ± 22.5°',
                 zh='低头脉冲：同样的脉冲，晚半圈，曲柄相位 315° ± 22.5°', col=BLUE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-offset', type=int, default=7)
    ap.add_argument('--start', type=float, default=2.9, help='在此时刻之后找第一个进入窗口的 tick')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT/'source'
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    env = ResearchEnv(str(source), mode='direct', task='hover')
    case = make_case(gates.SPLITS['evaluate']+a.seed_offset, 4.0, 'nominal', 0)
    m = env.sim.model
    style(m)
    dv = mujoco.MjData(m)

    # 先飞参考，找脉冲时刻；再飞 UP / DOWN，每段都和参考在同一时间轴上配对
    t_end = a.start+.25
    ref_ticks, ref_frames, _ = fly(env, case, None, a.start-.02, t_end)
    ref_t = np.array([f['t'] for f in ref_frames])
    summary = dict(case_seed=case['seed'], condition='nominal', chunk_substeps=CHUNK, slowdown=round(1/(CHUNK*m.opt.timestep*FPS)),
                   window_half_width_deg=22.5, high_rad_s=float(HIGH[2]), segments=[])
    segs = []
    for s in SEGMENTS:
        ticks, frames, fired = fly(env, case, s['centre'], a.start-.02, t_end)
        tp = fired[0][0]
        tt = np.array([k[0] for k in ticks])
        rt = np.array([k[0] for k in ref_ticks])
        dq = np.degrees(np.array([k[2] for k in ticks])-np.array([k[2] for k in ref_ticks]))
        dth = np.degrees(np.array([k[1] for k in ticks])-np.array([k[1] for k in ref_ticks]))
        assert np.allclose(tt, rt)
        fr = [f for f in frames if tp-PRE <= f['t'] <= tp+POST]
        idx = np.searchsorted(ref_t, [f['t'] for f in fr])
        rf = [ref_frames[min(i, len(ref_frames)-1)] for i in idx]
        k110 = int(np.argmin(abs(tt-(tp+.11))))
        z_p, z_r, x_r = ticks[k110][3], ref_ticks[k110][3], ref_ticks[k110][4]
        tilt_fwd = float((z_p-z_r)@x_r)      # 推力轴相对参考向机体 +x 方向倾斜的量（无量纲）
        win = [f for f in frames if tp <= f['t'] <= tp+.04]
        rwin = [f for f in ref_frames if tp <= f['t'] <= tp+.04]
        info = dict(key=s['key'], centre_deg=s['centre'], pulse_ticks=[dict(t=t, crank_deg=round(p, 1)) for t, p in fired],
                    peak_dq_deg_s=float(dq[np.argmax(abs(dq))]), t_peak_ms=float((tt[np.argmax(abs(dq))]-tp)*1e3),
                    dpitch_110ms_deg=float(dth[k110]), thrust_axis_shift_toward_body_x=tilt_fwd,
                    motor_peak_pulse=max(f['motor'] for f in win), motor_peak_ref=max(f['motor'] for f in rwin),
                    cmd_smooth_peak=max(f['cmd_smooth'] for f in win))
        dm = np.array([f['motor'] for f in fr])-np.array([g['motor'] for g in rf])
        tf = np.array([f['t'] for f in fr])
        sel = np.flatnonzero((tf >= tp) & (tf <= tp+.03))
        kmax = sel[np.argmax(dm[sel])]
        a_, b_ = kmax, kmax
        while a_ > sel[0] and dm[a_-1] > .3*dm[kmax]:
            a_ -= 1
        while b_ < sel[-1] and dm[b_+1] > .3*dm[kmax]:
            b_ += 1
        info.update(speed_gain_rad_s=float(dm[kmax]), speed_gain_crank_from=math.degrees(fr[a_]['crank']) % 360,
                    speed_gain_crank_to=math.degrees(fr[b_]['crank']) % 360, speed_gain_ms=[float((tf[a_]-tp)*1e3), float((tf[b_]-tp)*1e3)])
        summary['segments'].append(info)
        segs.append(dict(s, tp=tp, fr=fr, rf=rf, tt=tt, dq=dq, dth=dth, fired=fired, info=info))
        print(json.dumps(info))

    # 渲染
    rr_top = mujoco.Renderer(m, height=700, width=900, max_geom=2000)
    rr_side = mujoco.Renderer(m, height=330, width=440, max_geom=2000)
    f_title, f_big, f_mid, f_small, f_tiny = font(30, True), font(24, True), font(20), font(17), font(15)
    f_zh, f_zh_small = zh_font(26), zh_font(19)
    path = OUT/'pulse_slowmo.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(path)], stdin=subprocess.PIPE)
    slow = summary['slowdown']

    def card(lines, seconds):
        img = Image.new('RGB', (W, H), BG)
        dr = ImageDraw.Draw(img)
        y = 300
        for text, f, col in lines:
            dr.text((W//2, y), text, fill=col, font=f, anchor='mm')
            y += 62 if f in (f_title, f_zh) else 46
        for _ in range(int(seconds*FPS)):
            ff.stdin.write(img.tobytes())
        return img

    card([('Phase-locked motor pulse on the IST_robot-V2 flapping-wing vehicle', f_title, INK),
          ('扑翼飞行器上的相位锁定电机脉冲（慢动作）', f_zh, INK),
          (f'MuJoCo, 5 µs physics step; one frame = 0.1 ms of simulated time; played {slow}× slower than real time', f_mid, INK2),
          ('Motor → two-stage planetary gear (16:1) → crank → Scotch yoke slider with racks → wing pinions → flapping', f_mid, INK2),
          ('Pulse: the motor speed command goes to its upper limit 3807 rad/s while the crank phase is inside a ±22.5° window,', f_mid, INK2),
          ('then passes the 8.2 ms actuator lag. Grey curves: the same hover without the pulse (bit-identical until the pulse).', f_mid, INK2),
          ('Window at 135° → pitch up;  window at 315° → pitch down.', f_big, RED)], 6)

    for s in segs:
        tp, fr, rf, col = s['tp'], s['fr'], s['rf'], s['col']
        xs = np.array([(f['t']-tp)*1e3 for f in fr])
        motor_p, motor_r = [f['motor'] for f in fr], [f['motor'] for f in rf]
        flap_p, flap_r = [f['flap'] for f in fr], [f['flap'] for f in rf]
        raw, smooth = [f['cmd_raw'] for f in fr], [f['cmd_smooth'] for f in fr]
        tq = (s['tt']-tp)*1e3
        pulse_ms = [(t-tp)*1e3 for t, _ in s['fired']]
        for i, f in enumerate(fr):
            img = Image.new('RGB', (W, H), BG)
            dr = ImageDraw.Draw(img)
            x_now = xs[i]
            firing = any(p <= x_now < p+2 for p in pulse_ms)
            # 标题
            dr.text((30, 18), s['title'], fill=INK, font=f_big)
            dr.text((30, 52), s['zh'], fill=col, font=f_zh_small)
            dr.text((W-30, 22), f't − t_pulse = {x_now:+6.1f} ms   ({slow}× slow motion)', fill=INK2, font=f_mid, anchor='ra')
            # 三维：机构俯视
            dv.qpos[:] = f['qpos']
            mujoco.mj_forward(m, dv)
            j = m.joint('carrier_2_joint').id
            cam = mujoco.MjvCamera()
            ax_, pin_ = dv.xaxis[j], dv.site_xpos[m.site('crank_pin').id]
            cam.lookat[:] = dv.xanchor[j]+ax_*(ax_@(pin_-dv.xanchor[j]))
            rb = dv.xmat[m.body('thrust_unit').id].reshape(3, 3)
            yaw = math.degrees(math.atan2(rb[1, 0], rb[0, 0]))   # 相机随机身偏航转动，俯视图始终是机体坐标
            cam.azimuth, cam.elevation, cam.distance = 90+yaw, -89, .042
            rr_top.update_scene(dv, camera=cam)
            crank_overlay(rr_top.scene, m, dv, s['centre'], firing)
            img.paste(Image.fromarray(rr_top.render()), (30, 92))
            dr.rectangle((30, 92, 930, 792), outline=MUTE)
            dr.text((44, 102), 'Top view of the gearbox (housing translucent)', fill=(235, 238, 242), font=f_small)
            legend = [('red dot / white arm: crank pin on the 16:1 output gear', (255, 120, 110)),
                      ('red arc: pulse window, bright while the command is at max', (255, 120, 110)),
                      ('grey dots: stroke reversals (crank 90° / 270°)', (215, 220, 226)),
                      ('green: Scotch yoke slider, racks turn the wing pinions', (120, 230, 170))]
            ov = Image.new('RGBA', (470, 108), (0, 0, 0, 170))
            img.paste(ov, (36, 684), ov)
            for k, (txt, c) in enumerate(legend):
                dr.text((44, 690+k*24), txt, fill=c, font=f_tiny)
            if firing:
                dr.rectangle((660, 102, 916, 140), fill=RED)
                dr.text((788, 121), 'COMMAND = 3807 rad/s', fill=(255, 255, 255), font=f_small, anchor='mm')
            # 三维：整机
            cam2 = mujoco.MjvCamera()
            cam2.lookat[:] = dv.xpos[m.body('thrust_unit').id]+np.array([0, 0, .04])
            cam2.azimuth, cam2.elevation, cam2.distance = yaw, -12, .14
            rr_side.update_scene(dv, camera=cam2)
            img.paste(Image.fromarray(rr_side.render()), (960, 92))
            dr.rectangle((960, 92, 1400, 422), outline=MUTE)
            dr.text((972, 100), 'Whole vehicle', fill=(235, 238, 242), font=f_small)
            # 相位盘
            dial(dr, 1655, 232, 112, f['crank'] % (2*math.pi)*180/math.pi, s['centre'], firing, f_mid, f_tiny)
            # 曲线 1：转速
            p1 = Plot(dr, (1030, 485, 1890, 650), xs[0], xs[-1], 1800, 3900, 'Motor speed', 'rad/s', f_small, f_tiny)
            for pm in pulse_ms:
                p1.band(pm, pm+2, (252, 220, 216))
            p1.curve(xs, motor_r, GREY, 2)
            p1.curve(xs[:i+1], raw[:i+1], RED, 2)
            p1.curve(xs[:i+1], smooth[:i+1], (120, 40, 120), 3)
            p1.curve(xs[:i+1], motor_p[:i+1], col, 3)
            lx = 1180
            for txt, c in (('command', RED), ('after 8.2 ms lag', (120, 40, 120)), ('shaft speed', col), ('no pulse', GREY)):
                dr.line((lx, 470, lx+22, 470), fill=c, width=4)
                dr.text((lx+28, 470), txt, fill=INK2, font=f_tiny, anchor='lm')
                lx += 60+int(dr.textlength(txt, font=f_tiny))
            # 曲线 2：扑动角
            p2 = Plot(dr, (1030, 705, 1890, 830), xs[0], xs[-1], -90, 90, 'Wing flapping angle', 'deg', f_small, f_tiny)
            for pm in pulse_ms:
                p2.band(pm, pm+2, (252, 220, 216))
            p2.curve(xs, flap_r, GREY, 2)
            p2.curve(xs[:i+1], flap_p[:i+1], col, 3)
            # 曲线 3：俯仰角速度变化
            lim = 90
            p3 = Plot(dr, (1030, 885, 1890, 1020), xs[0], xs[-1], -lim, lim, 'Body pitch rate: pulse − no pulse', 'deg/s', f_small, f_tiny)
            for pm in pulse_ms:
                p3.band(pm, pm+2, (252, 220, 216))
            dr.line((p3.x0, p3.py(0), p3.x1, p3.py(0)), fill=GREY)
            sel = (tq >= xs[0]) & (tq <= x_now)
            p3.curve(tq[sel], s['dq'][sel], col, 3)
            p3.xticks(f_tiny, 10)
            dr.text((1460, 1062), 'time since the pulse onset (ms)', fill=INK2, font=f_tiny, anchor='mm')
            # 下方说明
            dr.text((30, 812), 'What happens', fill=INK, font=f_big)
            k = s['info']
            steps = [f'1. The crank enters the window at {s["fired"][0][1]:.0f}°: speed command → 3807 rad/s for {len(s["fired"])*2} ms.',
                     f'2. After the 8.2 ms lag the shaft runs up to {k["speed_gain_rad_s"]:.0f} rad/s faster while the crank turns '
                     f'{k["speed_gain_crank_from"]:.0f}° → {k["speed_gain_crank_to"]:.0f}°,',
                     '    so this half-stroke is flapped faster than its mirror half-stroke: lift becomes asymmetric over the beat.',
                     f'3. That asymmetry gives a pitch torque: peak {k["peak_dq_deg_s"]:+.0f} deg/s at {k["t_peak_ms"]:.0f} ms, '
                     f'{k["dpitch_110ms_deg"]:+.1f}° after 110 ms.',
                     ('4. The same pulse half a crank turn later (315°) gives the opposite torque.' if s['key'] == 'up' else
                      '4. Mirror image of the 135° pulse: same command, same speed gain, opposite half-stroke, opposite torque.')]
            for n, txt in enumerate(steps):
                dr.text((34, 852+n*30), txt, fill=INK if n < 4 else INK2, font=f_small)
            ff.stdin.write(img.tobytes())
            if i == len(fr)//2:
                img.save(OUT/f'pulse_{s["key"]}_still.png')
        for _ in range(int(2.5*FPS)):        # 段尾停留
            ff.stdin.write(img.tobytes())

    up, dn = summary['segments']
    card([('Same pulse, opposite phase, opposite torque', f_title, INK),
          ('同样的脉冲，相位相差半圈，力矩方向相反', f_zh, INK),
          (f'135° window: pitch rate {up["peak_dq_deg_s"]:+.0f} deg/s,  pitch {up["dpitch_110ms_deg"]:+.1f}° after 110 ms', f_big, ORANGE),
          (f'315° window: pitch rate {dn["peak_dq_deg_s"]:+.0f} deg/s,  pitch {dn["dpitch_110ms_deg"]:+.1f}° after 110 ms', f_big, BLUE),
          ('The timing inside the 37 ms wingbeat, not the amplitude, sets the sign of the torque.', f_mid, INK2),
          ('This is the channel the spiking layer fires through (3-tick twitches) and the classical base law uses in hover.', f_mid, INK2)], 6)
    ff.stdin.close()
    ff.wait()
    (OUT/'pulse_summary.json').write_text(json.dumps(summary, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()
