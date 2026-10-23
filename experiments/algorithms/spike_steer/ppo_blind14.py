"""Plan v14 amendment 2: the MLP + PPO reference arm (HomePC's plan-v7-full policies, frozen) on the plan-v11 nominal blind
set and on the primary kick set, with bench10 scoring, so that the reference arm appears in the same tables as the SNN.
Runs in the PPO venv:   .venv-ppo/bin/python -m experiments.algorithms.spike_steer.ppo_blind14 --workers 12

HomePC 扩展（2026-10-22）：--sets 可以列出任何 blind12 / blind13 的鲁棒性集合（默认 nominal,kick），每个集合跑完立即写出
runs/spike_steer/v14/<set>__mlp_ppo_s<seed>.json，已存在的集合跳过；对手行取 runs/spike_steer/v12|v13/<set>/rows_classical.json。
    MUJOCO_GL= python -W ignore -m experiments.algorithms.spike_steer.ppo_blind14 --sets kick2,kick45,turb15 --workers 18
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .blind12 import cases12, TESTS as T12
from .blind13 import cases13, TESTS as T13
from .paired_report import paired

OUT = Path('runs/spike_steer/v14')
PDIR = Path('artifacts/spike_steer/ppo_direct/v7full')
_S = {}


def _init(source):
    B.worker_init(source, .15)
    E.load_signs()
    from ..ppo_direct import v7ppo as P
    P.T.Episode = T10.Episode                      # bench10 scoring (symmetric altitude term), as for every other arm
    _S['P'] = P
    _S['env9'] = P.PpoLayer9Env()


def job(args):
    seed, ckpt, task, c = args
    from stable_baselines3 import PPO
    P, env9 = _S['P'], _S['env9']
    model = _S.setdefault(ckpt, PPO.load(ckpt, device='cpu'))
    obs, _ = env9.reset(seed=0, options=dict(task=task, case=c))
    done, info = False, {}
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, te, tr, info = env9.step(act)
        done = te or tr
    res = dict(info['metrics'])
    res.pop('trace', None)
    res['controller'] = f'mlp_ppo_s{seed}'
    return res


def build_set(name):
    """(cases per task, path of the classical opponent rows) for a set name: nominal, a blind12 test or a blind13 test."""
    if name == 'nominal':
        return {t: T.cases9(t, 2_099_100_000, 64) for t in D9.TASKS}, Path('runs/spike_steer/v11/test_opponent.json')
    if name in T12:
        K = T12[name]
        return ({t: cases12(t, K['base'], 64, K['condition'], K['kick'], K.get('mag'), K.get('kick_t')) for t in D9.TASKS},
                Path(f'runs/spike_steer/v12/{name}/rows_classical.json'))
    if name in T13:
        return {t: cases13(t, T13[name]) for t in D9.TASKS}, Path(f'runs/spike_steer/v13/{name}/rows_classical.json')
    raise SystemExit(f'unknown set {name}; known: nominal, {", ".join(T12)}, {", ".join(T13)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--sets', default='nominal,kick', help='comma-separated: nominal and any blind12 / blind13 test name')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        B.prepare_source(E.SOURCE)
    ck = {}
    for s in range(1, 6):
        g = json.loads((PDIR/f'reselect_s{s}.json').read_text())['selected']['generation']
        ck[s] = str(PDIR/f's{s}/centres/ckpt_g{g}.zip')
    todo = [n for n in a.sets.split(',') if n]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(E.SOURCE),)) as ex:
        for name in todo:
            if all((OUT/f'{name}__mlp_ppo_s{s}.json').exists() for s in range(1, 6)):
                print(name, 'already done, skipped', flush=True)
                continue
            cs, oppf = build_set(name)
            opp = json.loads(oppf.read_text())
            jobs = [(s, ck[s], t, c) for s in range(1, 6) for t in D9.TASKS for c in cs[t]]
            rows = list(ex.map(job, jobs, chunksize=4))
            n = sum(len(v) for v in cs.values())
            for s in range(1, 6):
                rr = rows[(s-1)*n:s*n]
                (OUT/f'{name}__mlp_ppo_s{s}.json').write_text(json.dumps(rr))
                pr = paired(opp, rr)
                print('%-14s mlp_ppo_s%d  mean %.3f  Δ %+.3f [%+.3f, %+.3f]  wins %d  crashes %d (opp %d)  selected gen %s' % (
                    name, s, pr['mean'], pr['delta'], *pr['delta_ci95'], pr['wins'], pr['crashes'], pr['classical_crashes'], ck[s].split('_g')[-1][:-4]), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
