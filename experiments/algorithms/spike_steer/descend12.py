"""Plan v11 (artifacts/spike_steer/plan_v11/preregistration_v11.json): the plan-v9 network and scoring (descend10: 25-input
layer, bench10) with the ES training fixed.

Defect found in v10: with 4 cases per task per generation, bench9.cases9 gives ES the same direction slots every
generation (dash 0/45/90/135 deg only, redirect left turns only) and only the first 4 of the 8 stress strata; selection
and blind sets cover everything. v11:
  * 8 cases per task per generation -> every dash direction, every redirect pair (both turns), every evade direction
    twice, all 8 strata (24 episodes per candidate, common to the population);
  * crash-aware ranking: a candidate with any crashed episode ranks below every crash-free one (ties by objective);
  * step size decays lr_g = lr * LR_DECAY**g;
  * 16 generations, centre saved at odd generations (+ the imitation start), selection on 64 cases per task.

    python -m experiments.algorithms.spike_steer.descend12 finetune --seed 1 --workers 10
    python -m experiments.algorithms.spike_steer.descend12 reselect --seed 1 --n-select 64 --workers 30
    python -m experiments.algorithms.spike_steer.descend12 evaluate --seed 1 --n-test 64 --test-base 2099100000 --workers 30
(default --output runs/spike_steer/v11, tag _v11)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import descend8 as D8
from . import descend9 as D9
from . import descend9b as D9b
from . import descend10 as D10

OUT, TAG = 'runs/spike_steer/v11', '_v11'
CASES, LR, SIGMA, POP, LR_DECAY, GENS, SAVE_EVERY = 8, .01, .02, 16, .85, 16, 2


def stage_finetune(a):
    from . import cache8 as _C8  # noqa: F401
    T = D9.T
    out = Path(a.output)
    norm = D9.load(out)['normalisers']
    spec0 = json.loads((out/f'imitate_spec_s{a.seed}.json').read_text())
    run = out/f'finetune_s{a.seed}{TAG}'
    run.mkdir(parents=True, exist_ok=True)
    (run/'config.json').write_text(json.dumps(dict(lr=LR, lr_decay=LR_DECAY, sigma=SIGMA, pop=POP, save_every=SAVE_EVERY, generations=GENS,
                                                   cases_per_task=CASES, crash_aware=True, seed=a.seed), indent=1))
    rng = np.random.default_rng(1000+a.seed)
    theta = D8.flatten(spec0)
    m_ad, v_ad = np.zeros_like(theta), np.zeros_like(theta)
    gen0 = 0
    if (run/'checkpoint.npz').exists():
        ck = np.load(run/'checkpoint.npz')
        theta, m_ad, v_ad, gen0 = ck['theta'], ck['m'], ck['v'], int(ck['gen'])
        rng = np.random.default_rng([1000+a.seed, gen0])
    with E.pool(a.workers) as ex:
        if gen0 == 0:
            (run/'spec_val_g-1.json').write_text(json.dumps(dict(spec0, generation=-1), indent=1))
        for gen in range(gen0, GENS):
            t_gen = time.monotonic()
            eps = rng.standard_normal((POP//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in D9.TASKS for c in T.cases9(t, T.BASES['train'], CASES, offset=D9.OFF['es']+gen*100)]
            cands = [theta+SIGMA*e for e in eps]+[theta]
            rows = T.run_grouped(ex, [(str(i), t, c, D8.unflatten(spec0, th)) for i, th in enumerate(cands) for t, c in batch], chunk=6)
            per = [[r for r in rows if r['controller'] == str(i)] for i in range(len(cands))]
            objs = [D9.objective(rr, norm)[0] for rr in per]
            crashed = [sum(not r['flown'] for r in rr) for rr in per]
            key = np.array([(1e6 if crashed[i] else 0.)+objs[i] for i in range(POP)])      # lower is better
            ranks = np.empty(POP)
            ranks[np.argsort(key)] = np.arange(POP)
            util = np.maximum(0., math.log(POP/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./POP
            grad = (util[:, None]*eps).sum(0)/SIGMA
            m_ad = .9*m_ad+.1*grad
            v_ad = .999*v_ad+.001*grad*grad
            lr = LR*LR_DECAY**gen
            step = lr*(m_ad/(1-.9**(gen+1)))/(np.sqrt(v_ad/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            row = dict(generation=gen, center=objs[-1], center_ratios=D9.objective(per[-1], norm)[1], best_candidate=float(min(objs[:POP])),
                       mean=float(np.mean(objs[:POP])), flown_center=sum(r['flown'] for r in per[-1]), crashed_candidates=sum(c > 0 for c in crashed[:POP]),
                       episodes_per_candidate=len(batch), lr=lr, step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            if gen % SAVE_EVERY == SAVE_EVERY-1:
                (run/f'spec_val_g{gen}.json').write_text(json.dumps(dict(D8.unflatten(spec0, theta), generation=gen), indent=1))
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_ad, v=v_ad, gen=gen+1)
            print(json.dumps(row), flush=True)


def selftest():
    """Coverage of the ES batches (pure bookkeeping, no simulation)."""
    from . import bench9 as T
    for g in range(GENS):
        for t, want in (('dash', 8), ('redirect', 8), ('evade', 4)):
            cs = T.cases9(t, T.BASES['train'], CASES, offset=D9.OFF['es']+g*100)
            dirs = {(c['dir_deg'], c.get('dir2_deg')) for c in cs}
            assert len(dirs) == want, (g, t, dirs)
            assert sorted(c['stratum'] if 'stratum' in c else i % 8 for i, c in enumerate(cs)) == list(range(8))
    print('descend12 coverage selftest ok: every generation covers all dash directions, all redirect pairs, all evade directions, 8 strata')


def main():
    D10.install()
    if '--output' not in sys.argv:
        sys.argv += ['--output', OUT]
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        selftest()
        return
    if stage == 'finetune':
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--output', default=OUT)
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--workers', type=int, default=10)
        a = p.parse_args()
        E.load_signs()
        B.DASH_M['value'] = .15
        stage_finetune(a)
        return
    if '--tag' not in sys.argv:
        sys.argv += ['--tag', TAG]
    D9b.main()


if __name__ == '__main__':
    main()
