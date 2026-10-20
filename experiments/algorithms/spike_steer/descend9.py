"""Plan v7 quick validation: the plan-v6 SNN pipeline (descend8 network, gate, imitation, OpenAI-ES, selection, blind
test) on the bench9 tasks evade / redirect / dash, at a reduced budget. Exploratory; scale up only if it pays off.

Arms reported on the blind set (bench9 BASES['test'], 16 cases per task): classical opponent, SNN imitation start
(gen -1, 'snn_bc'), SNN + ES (selected centre, 'snn_es').

    python -m experiments.algorithms.spike_steer.descend9 baseline --workers 30
    python -m experiments.algorithms.spike_steer.descend9 collect --workers 30
    python -m experiments.algorithms.spike_steer.descend9 train --seed 1
    python -m experiments.algorithms.spike_steer.descend9 finetune --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend9 reselect --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend9 evaluate --seed 1 --workers 30
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

TASKS = T.TASKS
WEIGHTS = dict(evade=1/3, redirect=1/3, dash=1/3)
N_COLLECT, N_VAL_IMIT = 16, 4
ES = dict(pop=16, sigma=.03, lr=.02, cases=4, validate_every=5)
N_SELECT, N_TEST = 16, 16
OFF = dict(collect=0, collect_val=50_000, select=60_000, es=100_000, norm=99_000)
SOFT_TILT_DEG, PENALTY_PER_DEG, HARD_TILT_DEG = D8.SOFT_TILT_DEG, D8.PENALTY_PER_DEG, D8.HARD_TILT_DEG


def load(out):
    return json.loads((Path(out)/'baseline.json').read_text())


def objective(rows, norm, weights=WEIGHTS):
    rt, pn = {}, {}
    for t in TASKS:
        rs = [r for r in rows if r['task'] == t]
        if not rs:
            continue
        rt[t] = float(np.mean([r['score']/norm[t][r['cls']] for r in rs]))
        pn[t] = float(np.mean([max(0., (r['tilt_peak_deg'] or 0.)-SOFT_TILT_DEG)*PENALTY_PER_DEG for r in rs]))
    tot = sum(weights[t] for t in rt)
    return float(sum(weights[t]*(rt[t]+pn[t]) for t in rt)/tot), rt


def stage_baseline(a):
    out = Path(a.output)
    pilot = json.loads((out/'pilot9.json').read_text())
    law = B.law_tuned()
    opp = pilot['opponent']
    norm = {}
    with E.pool(a.workers) as ex:
        for t in TASKS:
            cs = T.cases9(t, T.BASES['train'], 24, offset=OFF['norm'])
            rows = T.run_jobs(ex, [('opp', t, c, 'classical', dict(law=law, schedule=opp[t][T.klass(c)])) for c in cs])
            norm[t] = {cl: float(np.mean([r['score'] for r in rows if r['cls'] == cl])) for cl in T.CLASSES[t]}
            print(t, json.dumps(norm[t]), 'crashes', sum(not r['flown'] for r in rows), flush=True)
    up = json.loads(Path('artifacts/spike_steer/plan_v4/r5/train/imitate_spec_s1.json').read_text())['upset_law']
    (out/'baseline.json').write_text(json.dumps(dict(opponent=opp, law=law, normalisers=norm, upset_law=up, signs=E.SIGNS), indent=1))


class Traced9(T.Classical9):
    def reset(self, c):
        super().reset(c)
        self.log = []

    def act(self, env, now, t, x, cmd, rel):
        pulse, ref = super().act(env, now, t, x, cmd, rel)
        if t >= 2.:
            u = -(self.law['k_theta']*env.last_reading[1]+self.law['k_q']*self.steer.estimator.slow)
            self.log.append(np.r_[x, float(pulse), float(np.sign(u)), np.asarray(ref, float)])
        return pulse, ref


def collect_job(args):
    t, c, law, sched = args
    pol = Traced9(law, sched)
    res = T.run_episode9(B._ENV, t, c, pol)
    return dict(task=t, score=res['score'], flown=res['flown'], log=np.array(pol.log, np.float32))


def stage_collect(a):
    out = Path(a.output)
    base = load(out)
    with E.pool(a.workers) as ex:
        for split, off, n in (('train', OFF['collect'], N_COLLECT), ('val', OFF['collect_val'], N_VAL_IMIT)):
            jobs = [(t, c, base['law'], base['opponent'][t][T.klass(c)]) for t in TASKS for c in T.cases9(t, T.BASES['train'], n, offset=off)]
            rows = list(ex.map(collect_job, jobs, chunksize=1))
            L = np.array([len(r['log']) for r in rows])
            X = np.zeros((len(rows), int(L.max()), D8.N_IN+4), np.float32)
            for i, r in enumerate(rows):
                X[i, :L[i]] = r['log']
            np.savez_compressed(out/f'collect_{split}.npz', X=X, lengths=L, tasks=np.array([r['task'] for r in rows]))
            print(json.dumps(dict(split=split, episodes=len(rows), flown=sum(r['flown'] for r in rows))), flush=True)


def stage_train(a):
    """descend8.stage_train on the v7 data (same network, loss and schedule)."""
    out = Path(a.output)
    D8.stage_train(argparse.Namespace(output=str(out), seed=a.seed, threads=a.threads, epochs=a.epochs))


def stage_finetune(a):
    from . import cache8 as _C8  # noqa: F401  (registers the ctypes copy reducers)
    out = Path(a.output)
    base = load(out)
    norm = base['normalisers']
    spec0 = json.loads((out/f'imitate_spec_s{a.seed}.json').read_text())
    run = out/f'finetune_s{a.seed}'
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000+a.seed)
    theta = D8.flatten(spec0)
    m_ad, v_ad = np.zeros_like(theta), np.zeros_like(theta)
    pop, sigma, lr = ES['pop'], ES['sigma'], ES['lr']
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
            eps = rng.standard_normal((pop//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in TASKS for c in T.cases9(t, T.BASES['train'], ES['cases'], offset=OFF['es']+gen*100)]
            cands = [theta+sigma*e for e in eps]+[theta]
            rows = T.run_grouped(ex, [(str(i), t, c, D8.unflatten(spec0, th)) for i, th in enumerate(cands) for t, c in batch], chunk=6)
            objs = [objective([r for r in rows if r['controller'] == str(i)], norm)[0] for i in range(len(cands))]
            f = -np.array(objs[:pop])
            ranks = np.empty(pop)
            ranks[np.argsort(-f)] = np.arange(pop)
            util = np.maximum(0., math.log(pop/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./pop
            grad = (util[:, None]*eps).sum(0)/sigma
            m_ad = .9*m_ad+.1*grad
            v_ad = .999*v_ad+.001*grad*grad
            step = lr*(m_ad/(1-.9**(gen+1)))/(np.sqrt(v_ad/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            cr = [r for r in rows if r['controller'] == str(pop)]
            row = dict(generation=gen, center=objs[-1], center_ratios=objective(cr, norm)[1], best_candidate=float(min(objs[:pop])),
                       mean=float(np.mean(objs[:pop])), flown_center=sum(r['flown'] for r in cr), episodes_per_candidate=len(batch),
                       step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            if gen % ES['validate_every'] == ES['validate_every']-1:
                (run/f'spec_val_g{gen}.json').write_text(json.dumps(dict(D8.unflatten(spec0, theta), generation=gen), indent=1))
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_ad, v=v_ad, gen=gen+1)
            print(json.dumps(row), flush=True)


def stage_reselect(a):
    from . import cache8 as _C8  # noqa: F401
    out = Path(a.output)
    norm = load(out)['normalisers']
    run = out/f'finetune_s{a.seed}'
    cs = {t: T.cases9(t, T.BASES['train'], N_SELECT, offset=OFF['select']) for t in TASKS}
    specs = sorted(run.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1]))
    loaded = [json.loads(f.read_text()) for f in specs]
    with E.pool(a.workers) as ex:
        rows = T.run_grouped(ex, [(str(k), t, c, s) for k, s in enumerate(loaded) for t in TASKS for c in cs[t]], chunk=len(loaded))
    res = []
    for k, (f, s) in enumerate(zip(specs, loaded)):
        rr = [r for r in rows if r['controller'] == str(k)]
        obj, rt = objective(rr, norm)
        res.append(dict(generation=s['generation'], objective=obj, ratios=rt, crashes=sum(not r['flown'] for r in rr),
                        over_hard=sum((r['tilt_peak_deg'] or 0) >= HARD_TILT_DEG for r in rr),
                        tilt_peak_max=max(r['tilt_peak_deg'] or 0 for r in rr), spec=str(f)))
        print(json.dumps(res[-1]), flush=True)
    ok = [e for e in res if e['crashes'] == 0 and e['over_hard'] == 0]
    sel = min(ok, key=lambda e: e['objective']) if ok else None
    (out/f'reselect_s{a.seed}.json').write_text(json.dumps(dict(centres=res, selected=sel), indent=1))
    print(json.dumps(dict(selected=sel)), flush=True)


def report(rows, opp_rows):
    rep = {}
    for t in TASKS:
        lr_ = [r for r in rows if r['task'] == t]
        op = [r for r in opp_rows if r['task'] == t]
        med = {cl: float(np.median([r['score'] for r in op if r['cls'] == cl])) for cl in T.CLASSES[t]}
        by = {}
        for cl in T.CLASSES[t]:
            l_ = [r for r in lr_ if r['cls'] == cl]
            o_ = [r for r in op if r['cls'] == cl]
            by[cl] = dict(ratio=float(np.mean([r['score'] for r in l_])/med[cl]), opponent_median=med[cl], n=len(l_),
                          wins=sum(x['score'] < y['score'] for x, y in zip(l_, o_)), crashes=sum(not r['flown'] for r in l_))
        rep[t] = dict(ratio=float(np.mean([r['score']/med[r['cls']] for r in lr_])), n=len(lr_),
                      wins=sum(x['score'] < y['score'] for x, y in zip(lr_, op)), crashes=sum(not r['flown'] for r in lr_),
                      opponent_crashes=sum(not r['flown'] for r in op), by_class=by)
    rep['manoeuvre_mean'] = float(np.mean([rep[t]['ratio'] for t in TASKS]))
    return rep


def stage_evaluate(a):
    """Blind set: selected SNN + ES centre and the SNN imitation start, each vs the classical opponent (paired)."""
    from . import cache8 as _C8  # noqa: F401
    out = Path(a.output)
    base = load(out)
    sel = json.loads((out/f'reselect_s{a.seed}.json').read_text())['selected']
    cs = {t: T.cases9(t, T.BASES['test'], N_TEST) for t in TASKS}
    with E.pool(a.workers) as ex:
        of = out/'test_opponent.json'
        if of.exists():
            opp_rows = json.loads(of.read_text())
        else:
            opp_rows = T.run_jobs(ex, [('opp', t, c, 'classical', dict(law=base['law'], schedule=base['opponent'][t][T.klass(c)]))
                                       for t in TASKS for c in cs[t]])
            of.write_text(json.dumps(opp_rows))
        arms = [('snn_es', json.loads(Path(sel['spec']).read_text()), f"selected gen {sel['generation']}")]
        bc = out/f'finetune_s{a.seed}/spec_val_g-1.json'
        if sel['generation'] != -1 and bc.exists():
            arms.append(('snn_bc', json.loads(bc.read_text()), 'imitation start (gen −1)'))
        rows = T.run_grouped(ex, [(arm, t, c, spec) for arm, spec, _ in arms for t in TASKS for c in cs[t]], chunk=len(arms))
    for arm, _, note in arms:
        rr = [r for r in rows if r['controller'] == arm]
        rep = report(rr, opp_rows)
        doc = dict(arm=arm, seed=a.seed, machine='CloudPC-2', note=note, report=rep, rows=rr)
        (out/f'evaluate_{arm}_s{a.seed}.json').write_text(json.dumps(doc, indent=1))
        print(arm, json.dumps({t: round(rep[t]['ratio'], 3) for t in TASKS}), 'mean', round(rep['manoeuvre_mean'], 3),
              'crashes', sum(rep[t]['crashes'] for t in TASKS), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['baseline', 'collect', 'train', 'finetune', 'reselect', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/v7')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--generations', type=int, default=20)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    {'baseline': stage_baseline, 'collect': stage_collect, 'train': stage_train, 'finetune': stage_finetune,
     'reselect': stage_reselect, 'evaluate': stage_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
