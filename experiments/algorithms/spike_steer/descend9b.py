"""Plan v7 quick validation, ES variant (exploratory): descend9 with configurable step size, perturbation scale and
centre-saving interval, writing to finetune_s<seed><tag>/ so it never touches the baseline run.

Motivation: in the descend9 baseline run (lr 0.02, sigma 0.03, centre saved every 5 generations) both seeds were best at
generations 2-3 (objective 0.97-0.98, not saved) and degraded afterwards (seed 2 crashed from generation 5).

    python -m experiments.algorithms.spike_steer.descend9b finetune --seed 1 --tag _careful --lr 0.01 --sigma 0.02 --save-every 1 --workers 15
    python -m experiments.algorithms.spike_steer.descend9b reselect --seed 1 --tag _careful --workers 30
    python -m experiments.algorithms.spike_steer.descend9b evaluate --seed 1 --tag _careful --workers 30 --machine GoogleCloud
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import descend8 as D8
from . import descend9 as D9


def stage_finetune(a):
    from . import cache8 as _C8  # noqa: F401
    out = Path(a.output)
    norm = D9.load(out)['normalisers']
    spec0 = json.loads((out/f'imitate_spec_s{a.seed}.json').read_text())
    run = out/f'finetune_s{a.seed}{a.tag}'
    run.mkdir(parents=True, exist_ok=True)
    (run/'config.json').write_text(json.dumps(dict(lr=a.lr, sigma=a.sigma, pop=a.pop, save_every=a.save_every, generations=a.generations,
                                                   cases_per_task=D9.ES['cases'], seed=a.seed, soft_tilt_deg=D9.SOFT_TILT_DEG,
                                                   penalty_per_deg=D9.PENALTY_PER_DEG), indent=1))
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
        for gen in range(gen0, a.generations):
            t_gen = time.monotonic()
            eps = rng.standard_normal((a.pop//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in D9.TASKS for c in T.cases9(t, T.BASES['train'], D9.ES['cases'], offset=D9.OFF['es']+gen*100)]
            cands = [theta+a.sigma*e for e in eps]+[theta]
            rows = T.run_grouped(ex, [(str(i), t, c, D8.unflatten(spec0, th)) for i, th in enumerate(cands) for t, c in batch], chunk=6)
            objs = [D9.objective([r for r in rows if r['controller'] == str(i)], norm)[0] for i in range(len(cands))]
            f = -np.array(objs[:a.pop])
            ranks = np.empty(a.pop)
            ranks[np.argsort(-f)] = np.arange(a.pop)
            util = np.maximum(0., math.log(a.pop/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./a.pop
            grad = (util[:, None]*eps).sum(0)/a.sigma
            m_ad = .9*m_ad+.1*grad
            v_ad = .999*v_ad+.001*grad*grad
            step = a.lr*(m_ad/(1-.9**(gen+1)))/(np.sqrt(v_ad/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            cr = [r for r in rows if r['controller'] == str(a.pop)]
            row = dict(generation=gen, center=objs[-1], center_ratios=D9.objective(cr, norm)[1], best_candidate=float(min(objs[:a.pop])),
                       mean=float(np.mean(objs[:a.pop])), flown_center=sum(r['flown'] for r in cr), episodes_per_candidate=len(batch),
                       step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            if gen % a.save_every == a.save_every-1:
                (run/f'spec_val_g{gen}.json').write_text(json.dumps(dict(D8.unflatten(spec0, theta), generation=gen), indent=1))
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_ad, v=v_ad, gen=gen+1)
            print(json.dumps(row), flush=True)


def _with_tag(a, fn):
    """Run a descend9 stage on finetune_s<seed><tag>/ by pointing it at a view directory with the same layout."""
    out = Path(a.output)
    view = out/f'view{a.tag}'
    view.mkdir(exist_ok=True)
    for f in ('baseline.json', f'imitate_spec_s{a.seed}.json', 'test_opponent.json'):
        if (out/f).exists() and not (view/f).exists():
            (view/f).symlink_to((out/f).resolve())
    link = view/f'finetune_s{a.seed}'
    if not link.exists():
        link.symlink_to((out/f'finetune_s{a.seed}{a.tag}').resolve())
    sel = out/f'reselect_s{a.seed}{a.tag}.json'
    if sel.exists() and not (view/f'reselect_s{a.seed}.json').exists():
        (view/f'reselect_s{a.seed}.json').symlink_to(sel.resolve())
    fn(argparse.Namespace(output=str(view), seed=a.seed, workers=a.workers))
    for f in list(view.glob(f'reselect_s{a.seed}.json'))+list(view.glob(f'evaluate_*_s{a.seed}.json')):
        if not f.is_symlink():
            dst = out/f.name.replace(f'_s{a.seed}.json', f'_s{a.seed}{a.tag}.json')
            f.replace(dst)
            if f.name.startswith('reselect_'):
                f.symlink_to(dst.resolve())
    for f in view.glob('test_opponent.json'):
        if not f.is_symlink() and not (out/'test_opponent.json').exists():
            f.replace(out/'test_opponent.json')


def stage_evaluate(a):
    _with_tag(a, D9.stage_evaluate)
    out = Path(a.output)
    for f in out.glob(f'evaluate_*_s{a.seed}{a.tag}.json'):
        d = json.loads(f.read_text())
        d.update(machine=a.machine, variant=a.tag, note=(d.get('note') or '')+f' · ES variant {a.tag}')
        f.write_text(json.dumps(d, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['finetune', 'reselect', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/v7')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tag', default='_careful')
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--sigma', type=float, default=.02)
    p.add_argument('--pop', type=int, default=16)
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--workers', type=int, default=15)
    p.add_argument('--machine', default='CloudPC-2')
    p.add_argument('--n-select', type=int, default=None, help='validation cases per task for selection (descend9 default 16)')
    p.add_argument('--n-test', type=int, default=None, help='blind cases per task (descend9 default 16)')
    p.add_argument('--test-base', type=int, default=None, help='blind-set base seed (bench9 default 2_098_100_000); use a fresh --output dir')
    p.add_argument('--cases-per-task', type=int, default=None, help='ES training cases per task per generation (descend9 default 4)')
    p.add_argument('--soft-tilt', type=float, default=None, help='tilt penalty starts at this true tilt peak, deg (descend9 default 60)')
    p.add_argument('--penalty-per-deg', type=float, default=None, help='objective units per degree beyond --soft-tilt (descend9 default 0.3)')
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    if a.n_select:
        D9.N_SELECT = a.n_select
    if a.n_test:
        D9.N_TEST = a.n_test
    if a.test_base:
        T.BASES['test'] = a.test_base
    if a.cases_per_task:
        D9.ES['cases'] = a.cases_per_task
    if a.soft_tilt is not None:
        D9.SOFT_TILT_DEG = a.soft_tilt
    if a.penalty_per_deg is not None:
        D9.PENALTY_PER_DEG = a.penalty_per_deg
    if a.stage == 'finetune':
        stage_finetune(a)
    elif a.stage == 'reselect':
        _with_tag(a, D9.stage_reselect)
    else:
        stage_evaluate(a)


if __name__ == '__main__':
    main()
