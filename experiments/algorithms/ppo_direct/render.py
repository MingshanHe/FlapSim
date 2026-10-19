"""把 PPO 直接控制策略在完整模型上的回合离屏渲染成 mp4。

    MUJOCO_GL=egl python -m experiments.algorithms.ppo_direct.render --model runs/ppo_direct/models/quick.zip --tasks nod,dash,hover --output runs/ppo_direct/media/quick.mp4

每个任务渲染盲测集的第 --case 例（默认第 0 例），每个任务单独一个 mp4（<output 去掉后缀>_<task>.mp4）。
固定第三人称机位（看着起飞点，不跟随），左上角叠加时间、任务/指令状态、高度、倾角、俯仰和 α β ω。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..snn_pid_compare.common import DT
from ..spike_steer import bench as B
from .env import CALIBRATE, DirectManeuverEnv, TICKS_PER_DECISION, prepare_source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tasks", default="nod,dash,hover")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600)
    p.add_argument("--output", type=Path, default=Path("runs/ppo_direct/media/ppo_direct.mp4"))
    p.add_argument("--control", choices=["direct", "residual", "gated"], default="residual")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_source()
    from stable_baselines3 import PPO
    from ..spike_steer import descend_r5 as R5
    policy = PPO.load(a.model, device="cpu")
    env = DirectManeuverEnv(task="hover", control=a.control)
    R5.install()
    levels = json.loads(CALIBRATE.read_text())["levels"]
    model, data = env.env.sim.model, env.env.sim.data
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), a.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), a.height)
    renderer = mujoco.Renderer(model, height=a.height, width=a.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE          # 固定机位，不跟随
    camera.distance = 1.6
    camera.azimuth = 135.
    camera.elevation = -15.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    per_frame = max(1, int(round(1. / (a.fps * TICKS_PER_DECISION * DT))))
    for task in a.tasks.split(","):
        out = a.output.with_name(f"{a.output.stem}_{task}.mp4")
        writer = imageio.get_writer(str(out), fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
        frames = 0
        case = B.cases(task, "test", levels)[a.case]
        obs, info = env.reset(seed=int(case["seed"]) % (2 ** 31), options=dict(task="hover" if task in ("hover", "upset") else task, case=case, forward=True))
        camera.lookat[:] = env.env.air.start + np.array([0., 0., 0.05])   # 看着起飞点
        step, done = 0, False
        while not done:
            if step % per_frame == 0:
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                t = env.env.air.steps * DT
                who = "classical PID (takeoff)" if t < 1.5 else ("PPO direct" if a.control == "direct" else ("PID+law / PPO in charge" if a.control == "gated" and env.in_charge else "PID + PPO set-points/motor" if a.control == "residual" else "classical PID + pulse law"))
                if task in ("nod", "dash"):
                    state = "COMMAND ON" if env.cmd_on else ("goal reached -> hover" if env.goal_done else f"hover, command at {env.t0:.1f} s")
                    what = f"nod {'up' if env.nod_sign > 0 else 'down'}" if task == "nod" else "dash forward 0.15 m"
                else:
                    what, state = ("upset recovery" if "upset" in case else "stress hover"), ""
                cmd = env.env.command
                ref = env.env.teacher.rpy_reference
                lines = [f"{task}: {what}  {state}", f"t = {t:4.2f} s   {who}   ref roll {math.degrees(ref[0]):+5.1f} pitch {math.degrees(ref[1]):+5.1f} deg",
                         f"height {env._truth()['z']:+.3f} m (rel. start)   tilt {math.degrees(env._truth()['tilt']):4.1f} deg   pitch {math.degrees(env._truth()['pitch']):+5.1f} deg",
                         f"alpha {cmd[0]:+.3f}  beta {cmd[1]:+.3f}  omega {cmd[2]:6.0f} rad/s"]
                for i, text in enumerate(lines):
                    draw.text((12, 10 + 26 * i), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
                writer.append_data(np.asarray(frame))
                frames += 1
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        m = info.get("metrics", {})
        print(f"{task}: {m.get('termination')} score {m.get('score', float('nan')):.3f} " + (f"t30 {m.get('t30')} t_recover {m.get('t_recover')}" if task == "nod" else f"t_reach {m.get('t_reach')} t_settle {m.get('t_settle')}" if task == "dash" else ""), flush=True)
        writer.close()
        print(f"wrote {out} ({frames} frames, {frames / a.fps:.1f} s)", flush=True)
    renderer.close()


if __name__ == "__main__":
    main()
    # EGL 上下文在解释器退出时的析构会报 EGL_NOT_INITIALIZED（视频已写完），直接退出避免把链判成失败。
    import sys
    sys.stdout.flush()
    os._exit(0)
