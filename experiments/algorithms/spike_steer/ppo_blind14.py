"""Plan v14 amendment 2: the MLP + PPO reference arm (HomePC's plan-v7-full policies, frozen) on the plan-v11 nominal blind
set and on the primary kick set, with bench10 scoring, so that the reference arm appears in the same tables as the SNN.
Runs in the PPO venv:   .venv-ppo/bin/python -m experiments.algorithms.spike_steer.ppo_blind14 --workers 12
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        B.prepare_source(E.SOURCE)
    K = T12['kick']
    sets = dict(nominal={t: T.cases9(t, 2_099_100_000, 64) for t in D9.TASKS},
                kick={t: cases12(t, K['base'], 64, K['condition'], True, K['mag']) for t in D9.TASKS})
    ck = {}
    for s in range(1, 6):
        g = json.loads((PDIR/f'reselect_s{s}.json').read_text())['selected']['generation']
        ck[s] = str(PDIR/f's{s}/centres/ckpt_g{g}.zip')
    jobs = [(s, ck[s], t, c) for name, cs in sets.items() for s in range(1, 6) for t in D9.TASKS for c in cs[t]]
    names = [(name, s) for name in sets for s in range(1, 6)]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(E.SOURCE),)) as ex:
        rows = list(ex.map(job, jobs, chunksize=4))
    i = 0
    for name, cs in sets.items():
        n = sum(len(v) for v in cs.values())
        opp = json.loads(Path('runs/spike_steer/v11/test_opponent.json' if name == 'nominal' else 'runs/spike_steer/v12/kick/rows_classical.json').read_text())
        for s in range(1, 6):
            rr = rows[i:i+n]
            i += n
            (OUT/f'{name}__mlp_ppo_s{s}.json').write_text(json.dumps(rr))
            pr = paired(opp, rr)
            print('%-8s mlp_ppo_s%d  mean %.3f  Δ %+.3f [%+.3f, %+.3f]  wins %d  crashes %d (opp %d)  selected gen %s' % (
                name, s, pr['mean'], pr['delta'], *pr['delta_ci95'], pr['wins'], pr['crashes'], pr['classical_crashes'], ck[s].split('_g')[-1][:-4]), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
