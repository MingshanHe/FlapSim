"""Plan v7 comparison video: one grid video, rows = arms (classical, SNN + ES, SNN imitation only, MLP + PPO), columns =
tasks (evade, redirect, dash), every cell the first blind-set case of that task (same case across a column), shared
clock. Run in the PPO venv (stable-baselines3) because the MLP + PPO arm is HomePC's v7ppo policy.

    MUJOCO_GL=osmesa .venv-ppo/bin/python -m experiments.algorithms.spike_steer.render9 --workers 12
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import numpy as np

if not os.environ.get('RENDER9_RECORD'):             # the record stage must not load OSMesa: it crashes next to torch (PPO)
    os.environ.setdefault('MUJOCO_GL', 'osmesa')
    os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
import mujoco                                                            # noqa: E402
import imageio.v2 as imageio                                             # noqa: E402
from PIL import Image, ImageDraw                                         # noqa: E402
from . import bench as B                                                 # noqa: E402
from . import bench8 as E                                                # noqa: E402
from . import bench9 as T                                                # noqa: E402
from .render_v3 import add_geom, font                                    # noqa: E402
from .render_v4 import style_model, GRID, GRID_MAJOR, TRAIL              # noqa: E402

FPS = 30
PW, PH = 520, 340
HEAD, ROWHEAD = 96, 200
INK, INK2, GOOD, BAD = (24, 30, 38), (90, 98, 108), (12, 140, 70), (194, 59, 59)
ARROW1, ARROW2 = [.05, .62, .30, .95], [.84, .45, .08, .95]
NAME = {0: 'forward', 45: 'fwd-left', 90: 'left', 135: 'back-left', 180: 'back', 225: 'back-right', 270: 'right', 315: 'fwd-right'}
TASK_TITLE = dict(evade='Evade 0.10 m', redirect='Redirect (90° turn)', dash='Dash 0.15 m and stop')

_PPO = {}


def _init(source):
    B.worker_init(source, .15)
    E.load_signs()


def _init_ppo(source):
    """PPO jobs only: v7ppo.PpoLayer9Env builds the worker's single simulator itself (a second one in the same process
    crashes the native backend)."""
    E.load_signs()
    from ..ppo_direct import v7ppo as P
    _PPO['env'] = P.PpoLayer9Env()


def record(args):
    arm, task, c, arg = args
    if arm == 'ppo':
        from ..ppo_direct import v7ppo as P
        from stable_baselines3 import PPO
        if 'env' not in _PPO:
            _PPO['env'] = P.PpoLayer9Env()
        env9 = _PPO['env']
        model = _PPO.setdefault(arg, PPO.load(arg, device='cpu'))
        env = B._ENV
    else:
        env = B._ENV
    d = env.sim.data
    frames, orig = [], env.step

    def step(action):
        out = orig(action)
        frames.append((float(d.time), d.qpos.copy()))
        return out
    env.step = step
    try:
        if arm == 'ppo':
            obs, _ = env9.reset(seed=0, options=dict(task=task, case=c))
            done, info = False, {}
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, te, tr, info = env9.step(act)
                done = te or tr
            res = info['metrics']
        elif arm == 'classical':
            res = T.run_episode9(env, task, c, T.Classical9(arg['law'], arg['schedule']))
        else:
            from .descend8 import Learned8
            res = T.run_episode9(env, task, c, Learned8(arg))
    finally:
        del env.step
    res.pop('trace', None)
    return arm, task, res, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='artifacts/spike_steer/plan_v7/videos')
    p.add_argument('--slow', type=float, default=2.)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    v7 = Path('runs/spike_steer/v7')
    base = json.loads((v7/'baseline.json').read_text())
    sel1 = json.loads((v7/'reselect_s1.json').read_text())['selected']
    arms = [('classical', 'Classical (re-tuned)', None),
            ('snn_es', f"SNN + ES (seed 1, gen {sel1['generation']})", json.loads(Path(sel1['spec']).read_text())),
            ('snn_bc', 'SNN, imitation only (seed 1)', json.loads((v7/'finetune_s1/spec_val_g-1.json').read_text())),
            ('ppo', 'MLP + PPO (seed 1, HomePC)', 'artifacts/spike_steer/ppo_direct/v7ppo/s1/centres/ckpt_g1.zip')]
    cases = {t: T.cases9(t, T.BASES['test'], 1)[0] for t in T.TASKS}
    jobs = []
    for arm, _, arg in arms:
        for t in T.TASKS:
            c = cases[t]
            if arm == 'classical':
                arg_ = dict(law=base['law'], schedule=base['opponent'][t][T.klass(c)])
            else:
                arg_ = arg
            jobs.append((arm, t, c, arg_))
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        B.prepare_source(E.SOURCE)
    store = Path(a.out)/'v7_compare_recordings.pkl'
    if not os.environ.get('RENDER9_RECORD'):
        import pickle
        import subprocess
        import sys
        if not store.exists():
            env_ = dict(os.environ, RENDER9_RECORD='1')
            env_.pop('MUJOCO_GL', None)
            env_.pop('PYOPENGL_PLATFORM', None)
            subprocess.run([sys.executable, '-W', 'ignore', '-m', 'experiments.algorithms.spike_steer.render9', '--workers', str(a.workers),
                            '--out', a.out], env=env_, check=True)
        recs = pickle.loads(store.read_bytes())
        render(a, arms, cases, recs)
        return
    jp = [j for j in jobs if j[0] == 'ppo']
    jo = [j for j in jobs if j[0] != 'ppo']
    with ProcessPoolExecutor(len(jp), initializer=_init_ppo, initargs=(str(E.SOURCE),)) as exp, \
            ProcessPoolExecutor(max(1, a.workers-len(jp)), initializer=_init, initargs=(str(E.SOURCE),)) as ex:
        fp = exp.map(record, jp, chunksize=1)
        recs = list(ex.map(record, jo, chunksize=1))+list(fp)
    import pickle
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(pickle.dumps(recs))
    print('recorded', len(recs), 'episodes', flush=True)


def render(a, arms, cases, recs):
    base = None
    rec = {(r[0], r[1]): r for r in recs}

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
    rr = mujoco.Renderer(m, height=PH, width=PW, max_geom=3000)
    f_title, f_col, f_row, f_small = font(28, True), font(19, True), font(17, True), font(14)

    # one camera per column (task), framed on the union of all arms' paths, so the rows are comparable
    panels = {}
    for t in T.TASKS:
        c = cases[t]
        paths = {}
        for arm, _, _ in arms:
            _, _, res, frames = rec[(arm, t)]
            pth = []
            for (_, q) in frames:
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                pth.append(d.xpos[body].copy())
            paths[arm] = np.array(pth)
        _, _, _, frames0 = rec[('classical', t)]
        times0 = np.array([f[0] for f in frames0])
        i0 = min(int(np.searchsorted(times0, c['t0'])), len(frames0)-1)
        d.qpos[:] = frames0[i0][1]
        mujoco.mj_forward(m, d)
        start = d.xpos[body].copy()
        fwd = d.xmat[body].reshape(3, 3)[:, 0].copy()
        fwd[2] = 0.
        fwd /= np.linalg.norm(fwd)+1e-12
        left = np.array([-fwd[1], fwd[0], 0.])
        allp = np.concatenate([p_[int(np.searchsorted(np.array([f[0] for f in rec[(arm, t)][3]]), c['t0']-.5)):] for arm, p_ in paths.items()])
        lo, hi = allp.min(0), allp.max(0)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (lo+hi)/2
        cam.azimuth, cam.elevation = math.degrees(math.atan2(fwd[1], fwd[0]))+128., -32.
        cam.distance = max(.45, .3+1.15*float(np.linalg.norm(hi-lo)))
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        view = np.array([math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)])
        u = np.cross(view, [0., 0., 1.])
        u /= np.linalg.norm(u)
        dirs = [(c['dir_deg'], ARROW1)]+([(c['dir2_deg'], ARROW2)] if t == 'redirect' else [])
        panels[t] = dict(cam=cam, start=start, fwd=fwd, left=left, rot=np.column_stack([fwd, left, [0., 0., 1.]]), paths=paths,
                         bd=(np.asarray(cam.lookat)+view*3., np.column_stack([u, np.cross(u, view), view])), dirs=dirs)

    W, H = ROWHEAD+3*PW, HEAD+len(arms)*PH
    horizon = max(float(rec[k][3][-1][0]) for k in rec)
    n_frames = int(round(horizon*FPS*a.slow))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vpath = out/'v7_compare_blind_case0.mp4'
    writer = imageio.get_writer(str(vpath), fps=FPS, codec='libx264', quality=8, macro_block_size=None)
    try:
        for k in range(n_frames):
            t_ = (k+1)/(FPS*a.slow)
            img = Image.new('RGB', (W, H), (255, 255, 255))
            dr = ImageDraw.Draw(img)
            dr.text((16, 10), 'Plan v7 quick validation  ·  blind case 0 of each task  ·  same case in every column', font=f_title, fill=INK)
            dr.text((16, 46), f'{1/a.slow:g}x speed  ·  t = {t_:4.2f} s  ·  ratio = score ÷ classical score on the same case (lower is better)',
                    font=f_small, fill=INK2)
            for j, tk in enumerate(T.TASKS):
                c = cases[tk]
                lab = f"{TASK_TITLE[tk]}  ·  {NAME[int(c['dir_deg'])]}" + (f" → {NAME[int(c['dir2_deg'])]}" if tk == 'redirect' else '')
                dr.text((ROWHEAD+j*PW+10, HEAD-28), lab, font=f_col, fill=INK)
            for i, (arm, title, _) in enumerate(arms):
                y0 = HEAD+i*PH
                dr.multiline_text((10, y0+PH//2-30), title.replace(' (', '\n('), font=f_row, fill=INK, spacing=4)
                for j, tk in enumerate(T.TASKS):
                    _, _, res, frames = rec[(arm, tk)]
                    P = panels[tk]
                    times = np.array([f[0] for f in frames])
                    ii = min(int(np.searchsorted(times, t_)), len(frames)-1)
                    d.qpos[:] = frames[ii][1]
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
                    for (phi, colr) in P['dirs']:
                        pr = math.radians(phi)
                        dv = math.cos(pr)*fw+math.sin(pr)*lf
                        L = .12
                        mid = st[:2]+dv[:2]*L/2
                        arot = np.column_stack([dv, np.cross([0., 0., 1.], dv), [0., 0., 1.]])
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [L/2, .005, .0004], np.r_[mid, .0008], colr, arot)
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.011, 0, 0], np.r_[st[:2]+dv[:2]*L, .001], colr)
                    pth = P['paths'][arm]
                    for q in range(0, ii, 6):
                        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [.0016, 0, 0], pth[q], TRAIL)
                    x0 = ROWHEAD+j*PW
                    img.paste(Image.fromarray(rr.render()), (x0, y0))
                    dr.rectangle((x0, y0, x0+PW-1, y0+PH-1), outline=(225, 229, 235))
                    if res['goal_t'] is not None and t_-c['t0'] >= res['goal_t']:
                        dr.text((x0+10, y0+8), f"goal at +{res['goal_t']:.2f} s", font=f_small, fill=GOOD)
                    if not res['flown'] and t_ >= res['elapsed']:
                        dr.text((x0+10, y0+28), 'crash', font=f_small, fill=BAD)
                    ref = rec[('classical', tk)][2]['score']
                    s = res['score']
                    dr.text((x0+10, y0+PH-24), f"score {s:.2f}" + ('' if arm == 'classical' else f"  ·  ratio {s/ref:.2f}"), font=f_small,
                            fill=INK2 if arm == 'classical' else (GOOD if s < ref else BAD))
            writer.append_data(np.asarray(img))
        for _ in range(FPS*2):
            writer.append_data(np.asarray(img))
        img.save(vpath.with_suffix('.png'))
    finally:
        writer.close()
    rr.close()
    summary = {f'{arm}/{tk}': dict(score=rec[(arm, tk)][2]['score'], goal_t=rec[(arm, tk)][2]['goal_t'], flown=rec[(arm, tk)][2]['flown'])
               for arm, _, _ in arms for tk in T.TASKS}
    (out/'v7_compare_blind_case0.json').write_text(json.dumps(dict(cases={t: dict(seed=cases[t]['seed'], dir=cases[t]['dir_deg'],
                                                                                    dir2=cases[t].get('dir2_deg')) for t in T.TASKS},
                                                                   episodes=summary), indent=1))
    print('wrote', vpath)
    for k, v in summary.items():
        print(k, v)


if __name__ == '__main__':
    main()
