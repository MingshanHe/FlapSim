"""Robustness of the frozen learned layers to weight quantisation and synapse loss (hardware fit, result.md 3.4).
Variants of every SNN11 centre and every PPO actor, no retraining:
  q8, q6, q4   weight-only symmetric per-tensor quantisation of every weight and bias tensor to b bits (scale = max|w| / (2^(b-1)-1));
               SNN time constants, thresholds and the spike scale and all activations stay in float
  p10, p20, p30  a random fraction of the synapses (weight entries, not biases) set to zero; one fixed draw per arm and level
Sets: the plan-v15 nominal cases and the plan-v12 primary kick cases (192 each). The intact rows are those already flown
(runs/spike_steer/v15/nominal, v12/kick, v14/*__mlp_ppo_s*), on identical cases.

    python -m experiments.algorithms.spike_steer.damage run --workers 19
    python -m experiments.algorithms.spike_steer.damage report
"""
import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench10 as T10
from . import descend9 as D9
from .blind12 import load_arms, cases12, TESTS as T12
from .blind15 import cases15
from .evaluate_v8 import clopper_pearson
from .paired_report import paired

OUT = Path('runs/spike_steer/damage')
ART = Path('artifacts/spike_steer/damage')
VARIANTS = ('q8', 'q6', 'q4', 'p10', 'p20', 'p30')
SNN_W = ('W_in', 'W_out', 'W_g')
SNN_B = ('b_h', 'b_o', 'b_g')


def quant(w, bits):
    w = np.asarray(w, float)
    m = np.abs(w).max()
    if m == 0:
        return w
    s = m/(2**(bits-1)-1)
    return np.round(w/s)*s


def prune(w, frac, rng):
    w = np.asarray(w, float)
    return w*(rng.random(w.shape) >= frac)


def snn_variant(spec, v, seed):
    s = copy.deepcopy(spec)
    rng = np.random.default_rng([seed, int(v[1:])])
    for k in SNN_W+SNN_B:
        if v[0] == 'q':
            s[k] = quant(s[k], int(v[1:])).tolist()
        elif k in SNN_W:
            s[k] = prune(s[k], int(v[1:])/100., rng).tolist()
    return s


def ppo_variant(src, dst, v, seed):
    import torch
    from stable_baselines3 import PPO
    m = PPO.load(src, device='cpu')
    rng = np.random.default_rng([seed, int(v[1:])])
    with torch.no_grad():
        for n, p in m.policy.named_parameters():
            if not (n.startswith('mlp_extractor.policy_net') or n.startswith('action_net')):
                continue                              # the actor only; the value net is never used when flying
            w = p.detach().numpy()
            if v[0] == 'q':
                w = quant(w, int(v[1:]))
            elif n.endswith('weight'):
                w = prune(w, int(v[1:])/100., rng)
            p.copy_(torch.as_tensor(w, dtype=p.dtype))
    m.save(dst)


def case_sets():
    K = T12['kick']
    return dict(nominal={t: cases15(t, 'nominal') for t in D9.TASKS},
                kick={t: cases12(t, K['base'], 64, K['condition'], K['kick'], K.get('mag'), K.get('kick_t')) for t in D9.TASKS})


def intact(setname, arm):
    if arm.startswith('mlp'):
        f = Path(f'runs/spike_steer/v14/{"v15_nominal" if setname == "nominal" else "kick"}__{arm}.json')
    else:
        f = Path(f'runs/spike_steer/{"v15/nominal" if setname == "nominal" else "v12/kick"}/rows_{arm}.json')
    return json.loads(f.read_text())


def _ppo_init():
    from . import ppo_blind14 as Q
    Q._init(str(E.SOURCE))
    globals()['_Q'] = Q


def _ppo_job(args):
    name, ckpt, task, c = args
    r = _Q.job((0, ckpt, task, c))
    r['controller'] = name
    return r


def stage_run(a):
    from . import cache8 as _C8  # noqa: F401
    from .energy import ppo_ckpts
    OUT.mkdir(parents=True, exist_ok=True)
    arms, _ = load_arms()
    snn = [(n, s) for n, k, s in arms if n.startswith('snn11_')]
    cs = case_sets()
    for setname in ('nominal', 'kick'):
        d = OUT/setname
        d.mkdir(exist_ok=True)
        todo = [(f'{n}_{v}', snn_variant(s, v, int(n[-1]))) for n, s in snn for v in VARIANTS if not (d/f'rows_{n}_{v}.json').exists()]
        if todo:
            print(setname, 'snn variants', len(todo), flush=True)
            with E.pool(a.workers) as ex:
                for i in range(0, len(todo), 5):
                    part = todo[i:i+5]
                    rows = T10.run_grouped(ex, [(n, t, c, s) for n, s in part for t in D9.TASKS for c in cs[setname][t]], chunk=len(part))
                    for n, _ in part:
                        rr = [r for r in rows if r['controller'] == n]
                        (d/f'rows_{n}.json').write_text(json.dumps(rr))
                        print(setname, n, 'crashes', sum(not r['flown'] for r in rr), flush=True)
    ck = ppo_ckpts()
    zdir = OUT/'ppo_ckpts'
    zdir.mkdir(exist_ok=True)
    for n, src in ck.items():
        for v in VARIANTS:
            z = zdir/f'{n}_{v}.zip'
            if not z.exists():
                ppo_variant(src, str(z), v, int(n[-1]))
    for setname in ('nominal', 'kick'):
        d = OUT/setname
        todo = [(f'{n}_{v}', str(zdir/f'{n}_{v}.zip')) for n in ck for v in VARIANTS if not (d/f'rows_{n}_{v}.json').exists()]
        if not todo:
            continue
        print(setname, 'ppo variants', len(todo), flush=True)
        with ProcessPoolExecutor(a.workers, initializer=_ppo_init) as ex:
            rows = list(ex.map(_ppo_job, [(n, z, t, c) for n, z in todo for t in D9.TASKS for c in cs[setname][t]], chunksize=4))
        for n, _ in todo:
            rr = [r for r in rows if r['controller'] == n]
            (d/f'rows_{n}.json').write_text(json.dumps(rr))
            print(setname, n, 'crashes', sum(not r['flown'] for r in rr), flush=True)


def stage_report(a):
    S, lines = {}, []
    for setname in ('nominal', 'kick'):
        opp = json.loads(Path(f'runs/spike_steer/{"v15/nominal" if setname == "nominal" else "v12/kick"}/rows_classical.json').read_text())
        oc = sum(not r['flown'] for r in opp)
        lines.append(f'== {setname} (classical crashes {oc})')
        S[setname] = dict(classical_crashes=oc)
        for arm in [f'snn11_s{s}' for s in range(1, 6)]+[f'mlp_ppo_s{s}' for s in range(1, 6)]:
            base = intact(setname, arm)
            ent = dict(intact=dict(crashes=sum(not r['flown'] for r in base), delta_vs_classical=paired(opp, base)['delta']))
            cells = ['intact %3d %+.3f' % (ent['intact']['crashes'], ent['intact']['delta_vs_classical'])]
            for v in VARIANTS:
                f = OUT/setname/f'rows_{arm}_{v}.json'
                if not f.exists():
                    continue
                rr = json.loads(f.read_text())
                pi, pc = paired(base, rr), paired(opp, rr)
                ent[v] = dict(crashes=pi['crashes'], ci95=clopper_pearson(pi['crashes'], len(rr)), delta_vs_intact=pi['delta'],
                              delta_vs_intact_ci95=pi['delta_ci95'], delta_vs_classical=pc['delta'])
                cells.append('%s %3d %+.3f' % (v, pi['crashes'], pi['delta']))
            S[setname][arm] = ent
            lines.append(f'{arm:11s} ' + ' | '.join(cells))
    lines.append('cells: crashes of 192, paired score delta vs the same arm intact (intact: delta vs classical)')
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'summary.json').write_text(json.dumps(S, indent=1))
    (ART/'report.txt').write_text(txt+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run', 'report'])
    p.add_argument('--workers', type=int, default=19)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    {'run': stage_run, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
