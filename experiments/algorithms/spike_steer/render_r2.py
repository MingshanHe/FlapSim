"""Side-by-side replay video: gated SNN descending layer (plan v4 round 2) vs the same-budget classical opponent.

Both panels replay the SAME blind test case (same seed, same perturbations); every frame is the recorded simulator
state (qpos at control rate), nothing is interpolated or invented. Scores printed on the frame are the benchmark's
own per-episode scores (lower is better), the ratio is learned/opponent.

    MUJOCO_GL=osmesa python -m experiments.algorithms.spike_steer.render_r2 \
        --spec artifacts/spike_steer/plan_v4/r2/finetune_s1/spec_best.json --tag r2_s1 --tasks nod,dash,hover,upset --case 0

Headless rendering needs one of MUJOCO_GL=egl / osmesa (WSL2 usually has egl; CloudPC has osmesa).
"""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco                                                            # noqa: E402
import imageio.v2 as imageio                                             # noqa: E402
from PIL import Image, ImageDraw, ImageFont                              # noqa: E402
from . import bench as B                                                 # noqa: E402
from . import descend_gated as G                                         # noqa: E402

FPS = 30
PANEL = (640, 480)
BG = (17, 31, 46)
TITLES = dict(hover='T1 hover (8 s)', nod='T2 nod: pitch >= 30 deg and recover', dash='T3 dash: forward %.2f m and stop',
              upset='T4 upset: external kick, recover')


def font(size):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/System/Library/Fonts/Helvetica.ttc'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def record(env, task, c, kind, arg):
    """Run one benchmark episode exactly as bench.run_episode does, recording qpos after every control step."""
    d = env.sim.data
    frames = []
    orig = env.step

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


def opponent_arg(baseline, task):
    name = baseline['opponent'][task]
    return name, baseline['arms_full'][task][name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/spike_steer/bench_r2')
    p.add_argument('--spec', required=True)
    p.add_argument('--tag', default='r2')
    p.add_argument('--tasks', default='nod,dash,hover,upset')
    p.add_argument('--case', type=int, default=0, help='index into the blind test set of each task')
    p.add_argument('--slow', type=float, default=1., help='playback slow-down factor (2 = half speed)')
    a = p.parse_args()
    G.install()
    out = Path(a.output)
    baseline = B.load_baseline(out)
    levels = B.upset_levels(out)
    spec = json.loads(Path(a.spec).read_text())
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    B.worker_init(str(source), baseline['prereg']['dash_m'] or .25)
    env = B._ENV
    m, d = env.sim.model, env.sim.data
    body = env.air.env.body
    vdir = out/'videos'
    vdir.mkdir(exist_ok=True)
    big, small = font(20), font(15)
    renderer = mujoco.Renderer(m, height=PANEL[1], width=PANEL[0])
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -10, .4
    manifest = {}

    def scene(q, lookat):
        d.qpos[:] = q
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        cam.lookat[:] = lookat
        renderer.update_scene(d, camera=cam)
        return Image.fromarray(renderer.render())

    for task in a.tasks.split(','):
        c = B.cases(task, 'test', levels)[a.case]
        oname, oarg = opponent_arg(baseline, task)
        runs = [('gated SNN (learned)', 'learned', spec), (f'classical opponent ({oname})', 'classical', oarg)]
        results, traces, frames = [], [], []
        for label, kind, arg in runs:
            res, fr = record(env, task, c, kind, arg)
            results.append(res)
            traces.append(np.array(res.pop('trace')))
            frames.append(fr)
            print(f"{task} case {a.case} seed {c['seed']} {label}: score {res['score']:.3f} flown {res['flown']} term {res['termination']}", flush=True)
        ratio = results[0]['score']/results[1]['score']
        horizon = max(fr[-1][0] for fr in frames)
        t0 = c['upset']['t0'] if task == 'upset' else B.T0
        path = vdir/f'{a.tag}_{task}_case{a.case}.mp4'
        writer = imageio.get_writer(str(path), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
        W, H = 2*PANEL[0], PANEL[1]+190
        lookat = [None, None]
        n_frames = int(round(horizon*FPS*a.slow))
        try:
            for k in range(n_frames):
                t = (k+1)/(FPS*a.slow)
                pic = Image.new('RGB', (W, H), BG)
                draw = ImageDraw.Draw(pic)
                title = TITLES[task] % B.DASH_M['value'] if task == 'dash' else TITLES[task]
                draw.text((16, 10), f'{title} | blind case {a.case}, seed {c["seed"]} | {a.tag} | playback {1/a.slow:g}x', font=big, fill='white')
                draw.text((W-440, 36), f'score ratio learned/opponent = {ratio:.3f}', font=big, fill=(135, 220, 135) if ratio < 1 else (255, 170, 120))
                cmd_on = task in ('nod', 'dash') and t >= t0
                phase = ('COMMAND ON' if cmd_on else 'hover / reflex only') if task in ('nod', 'dash') else \
                        (('KICK' if t0 <= t < t0+.1 else 'recovering') if task == 'upset' and t >= t0 else 'hover')
                draw.text((16, 38), f't = {t:5.2f} s   {phase}', font=small, fill=(255, 210, 90) if cmd_on or (task == 'upset' and t >= t0) else (170, 190, 205))
                for col, ((label, _, _), res, tr, fr) in enumerate(zip(runs, results, traces, frames)):
                    x0 = col*PANEL[0]
                    i = min(int(np.searchsorted([f[0] for f in fr], t)), len(fr)-1)
                    ft, q = fr[i]
                    d.qpos[:] = q
                    mujoco.mj_forward(m, d)
                    pos = d.xpos[body].copy()
                    if lookat[col] is None:
                        lookat[col] = pos.copy()
                    lookat[col] = .85*lookat[col]+.15*pos                      # camera follows the body smoothly
                    pic.paste(scene(q, lookat[col]), (x0, 64))
                    draw = ImageDraw.Draw(pic)
                    ended = t > res['elapsed']+.02 and res['termination'] != 'time_limit'
                    j = min(int(np.searchsorted(tr[:, 0], t)), len(tr)-1)
                    row = tr[j]
                    pitch, tilt, ref, fire = math.degrees(row[1]), math.degrees(row[8]), row[9], row[10] > .5
                    color = (100, 220, 230) if col == 0 else (250, 170, 90)
                    draw.text((x0+14, 70), label, font=big, fill=color)
                    if ended:
                        draw.text((x0+14, 96), f'STOPPED at {res["elapsed"]:.2f} s: {res["termination"]}', font=big, fill=(255, 105, 100))
                    y = PANEL[1]+72
                    draw.text((x0+14, y), f'pitch {pitch:+6.1f} deg   tilt {tilt:5.1f} deg   height {pos[2]:.3f} m', font=small, fill='white')
                    if col == 0:
                        draw.text((x0+14, y+22), f'descending ref {ref:+.2f}   pulse {"FIRE" if fire else "-"}', font=small, fill=(100, 220, 230))
                    else:
                        draw.text((x0+14, y+22), 'PID + tuned pulse law / schedule', font=small, fill=(250, 170, 90))
                    extra = ''
                    r2 = lambda v: 'miss' if v is None else f'{v:.2f}'
                    if task == 'nod':
                        extra = f"t30 {r2(res.get('t30'))} s  recover {r2(res.get('t_recover'))} s  peak {res.get('peak_pitch_deg', 0) or 0:.0f} deg"
                    elif task == 'dash':
                        extra = f"reach {r2(res.get('t_reach'))} s  settle {r2(res.get('t_settle'))} s  max fwd {res.get('max_forward_m', 0) or 0:.2f} m"
                    elif task == 'upset':
                        extra = f"recovery {r2(res.get('recovery_s'))} s  tilt peak {res.get('tilt_peak_deg') or 0:.0f} deg"
                    else:
                        extra = f"pitch rms {res.get('pitch_rms_deg', 0):.1f} deg  height rmse {res.get('height_rmse_cm', 0):.1f} cm"
                    draw.text((x0+14, y+44), extra, font=small, fill=(190, 210, 220))
                    draw.text((x0+14, y+70), f'episode score {res["score"]:.3f} (lower is better)', font=big,
                              fill=(135, 220, 135) if (col == 0 and ratio < 1) or (col == 1 and ratio >= 1) else 'white')
                draw.text((16, H-24), 'Recorded simulator states at control rate; no interpolated motion. Same seed and perturbations in both panels.',
                          font=small, fill=(170, 190, 205))
                writer.append_data(np.asarray(pic))
                if k == min(n_frames-1, int(round((t0+.6)*FPS*a.slow))):
                    pic.save(vdir/f'{a.tag}_{task}_case{a.case}_preview.png')
            for _ in range(FPS):
                writer.append_data(np.asarray(pic))
        finally:
            writer.close()
        manifest[task] = dict(video=str(path), seed=c['seed'], learned=results[0], opponent=dict(results[1], arm=oname), ratio=ratio, fps=FPS, slow=a.slow)
        print(f'wrote {path}', flush=True)
    renderer.close()
    (vdir/f'{a.tag}_case{a.case}_manifest.json').write_text(json.dumps(manifest, indent=1, default=str))


if __name__ == '__main__':
    main()
