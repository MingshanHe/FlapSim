"""Plan v11 amendment 1: the CLASSICAL UPPER-BOUND ARM. The classical opponent's 28 schedule knobs (7 direction classes x
go_ref, go_beats, brake_beats, brake_ref; evade / redirect / dash) get exactly the plan-v11 ES budget of the learned layer:
population 16, 16 generations, 8 cases per task per generation (full direction / turn / stratum coverage), crash-aware
ranking, lr decay 0.85 per generation, selection on 64 cases per task with 0 crashes, then the v11 blind set (base
2_099_100_000, 64 per task) against the untouched classical opponent. PID, pulse law (same as the learned layer's base
law) and bench10 scoring are unchanged. If this arm cannot reach the 0.95 criterion either, the criterion, not the
learned layer, is what fails.

Knobs are ES-normalised: refs / 0.5 rad, beats / 20; sigma 0.05, lr 0.02 in normalised units (the layer has 2564
parameters at sigma 0.02 / lr 0.01; a 28-knob schedule needs larger relative steps to move at all).

    python -m experiments.algorithms.spike_steer.descend_cls selftest
    python -m experiments.algorithms.spike_steer.descend_cls finetune --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend_cls reselect --seed 1 --workers 30
    python -m experiments.algorithms.spike_steer.descend_cls evaluate --seed 1 --workers 30 --machine CloudPC-2
(default --output runs/spike_steer/v11)
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
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from . import descend12 as D12

OUT = 'runs/spike_steer/v11'
TEST_BASE, N_TEST, N_SELECT = 2_099_100_000, 64, 64
KEYS = ('go_ref', 'go_beats', 'brake_beats', 'brake_ref')
SCALE = dict(go_ref=.5, go_beats=20., brake_beats=20., brake_ref=.5)
SIGMA, LR = .05, .02
CELLS = [(t, cl) for t in D9.TASKS for cl in T.CLASSES[t]]          # 7 cells x 4 knobs = 28


def flatten(opp):
    return np.array([opp[t][cl][k]/SCALE[k] for t, cl in CELLS for k in KEYS])


def unflatten(theta):
    opp, i = {}, 0
    for t, cl in CELLS:
        q = {}
        for k in KEYS:
            v = float(theta[i])*SCALE[k]
            i += 1
            if k.endswith('beats'):
                v = max(0., v)
            elif k == 'go_ref':
                v = float(np.clip(v, 0., B.REF_MAX))
            else:
                v = float(np.clip(v, -B.REF_MAX, B.REF_MAX))
            q[k] = v
        q['gated'] = True
        opp.setdefault(t, {})[cl] = q
    return opp


def jobs_for(name, opp, law, batch):
    return [(name, t, c, 'classical', dict(law=law, schedule=opp[t][T.klass(c)])) for t, c in batch]


def stage_finetune(a):
    out = Path(a.output)
    base = D9.load(out)
    law, norm = base['law'], base['normalisers']
    run = out/f'cls_es_s{a.seed}'
    run.mkdir(parents=True, exist_ok=True)
    (run/'config.json').write_text(json.dumps(dict(lr=LR, lr_decay=D12.LR_DECAY, sigma=SIGMA, pop=D12.POP, generations=D12.GENS,
                                                   cases_per_task=D12.CASES, crash_aware=True, seed=a.seed, scale=SCALE), indent=1))
    rng = np.random.default_rng(5000+a.seed)
    theta = flatten(base['opponent'])
    m_ad, v_ad = np.zeros_like(theta), np.zeros_like(theta)
    gen0 = 0
    if (run/'checkpoint.npz').exists():
        ck = np.load(run/'checkpoint.npz')
        theta, m_ad, v_ad, gen0 = ck['theta'], ck['m'], ck['v'], int(ck['gen'])
        rng = np.random.default_rng([5000+a.seed, gen0])
    POP = D12.POP
    with E.pool(a.workers) as ex:
        if gen0 == 0:
            (run/'opp_g-1.json').write_text(json.dumps(dict(opponent=base['opponent'], generation=-1), indent=1))
        for gen in range(gen0, D12.GENS):
            t_gen = time.monotonic()
            eps = rng.standard_normal((POP//2, theta.size))
            eps = np.concatenate([eps, -eps], 0)
            batch = [(t, c) for t in D9.TASKS for c in T.cases9(t, T.BASES['train'], D12.CASES, offset=D9.OFF['es']+gen*100)]
            cands = [theta+SIGMA*e for e in eps]+[theta]
            rows = T10.run_jobs(ex, [j for i, th in enumerate(cands) for j in jobs_for(str(i), unflatten(th), law, batch)])
            per = [[r for r in rows if r['controller'] == str(i)] for i in range(len(cands))]
            objs = [D9.objective(rr, norm)[0] for rr in per]
            crashed = [sum(not r['flown'] for r in rr) for rr in per]
            key = np.array([(1e6 if crashed[i] else 0.)+objs[i] for i in range(POP)])
            ranks = np.empty(POP)
            ranks[np.argsort(key)] = np.arange(POP)
            util = np.maximum(0., math.log(POP/2+1)-np.log(ranks+1))
            util = util/util.sum()-1./POP
            grad = (util[:, None]*eps).sum(0)/SIGMA
            m_ad = .9*m_ad+.1*grad
            v_ad = .999*v_ad+.001*grad*grad
            lr = LR*D12.LR_DECAY**gen
            step = lr*(m_ad/(1-.9**(gen+1)))/(np.sqrt(v_ad/(1-.999**(gen+1)))+1e-8)
            theta = theta+step
            row = dict(generation=gen, center=objs[-1], center_ratios=D9.objective(per[-1], norm)[1], best_candidate=float(min(objs[:POP])),
                       mean=float(np.mean(objs[:POP])), flown_center=sum(r['flown'] for r in per[-1]), crashed_candidates=sum(c > 0 for c in crashed[:POP]),
                       episodes_per_candidate=len(batch), lr=lr, step_norm=float(np.linalg.norm(step)), wall_s=round(time.monotonic()-t_gen))
            (run/f'opp_g{gen}.json').write_text(json.dumps(dict(opponent=unflatten(theta), generation=gen), indent=1))
            with (run/'history.jsonl').open('a') as h:
                h.write(json.dumps(row)+'\n')
            np.savez(run/'checkpoint.npz', theta=theta, m=m_ad, v=v_ad, gen=gen+1)
            print(json.dumps(row), flush=True)


def stage_reselect(a):
    out = Path(a.output)
    base = D9.load(out)
    law, norm = base['law'], base['normalisers']
    run = out/f'cls_es_s{a.seed}'
    cs = {t: T.cases9(t, T.BASES['train'], N_SELECT, offset=D9.OFF['select']) for t in D9.TASKS}
    files = sorted(run.glob('opp_g*.json'), key=lambda f: int(f.stem.split('g')[-1]))
    files = [f for f in files if int(f.stem.split('g')[-1]) == -1 or int(f.stem.split('g')[-1]) % D12.SAVE_EVERY == D12.SAVE_EVERY-1]
    loaded = [json.loads(f.read_text()) for f in files]
    batch = [(t, c) for t in D9.TASKS for c in cs[t]]
    with E.pool(a.workers) as ex:
        rows = T10.run_jobs(ex, [j for k, d in enumerate(loaded) for j in jobs_for(str(k), d['opponent'], law, batch)])
    res = []
    for k, (f, d) in enumerate(zip(files, loaded)):
        rr = [r for r in rows if r['controller'] == str(k)]
        obj, rt = D9.objective(rr, norm)
        res.append(dict(generation=d['generation'], objective=obj, ratios=rt, crashes=sum(not r['flown'] for r in rr),
                        over_hard=sum((r['tilt_peak_deg'] or 0) >= D9.HARD_TILT_DEG for r in rr),
                        tilt_peak_max=max(r['tilt_peak_deg'] or 0 for r in rr), spec=str(f)))
        print(json.dumps(res[-1]), flush=True)
    ok = [e for e in res if e['crashes'] == 0 and e['over_hard'] == 0]
    sel = min(ok, key=lambda e: e['objective']) if ok else None
    (out/f'reselect_cls_es_s{a.seed}.json').write_text(json.dumps(dict(centres=res, selected=sel), indent=1))
    print(json.dumps(dict(selected=sel)), flush=True)


def stage_evaluate(a):
    out = Path(a.output)
    base = D9.load(out)
    law = base['law']
    sel = json.loads((out/f'reselect_cls_es_s{a.seed}.json').read_text())['selected']
    cs = {t: T.cases9(t, TEST_BASE, N_TEST) for t in D9.TASKS}
    batch = [(t, c) for t in D9.TASKS for c in cs[t]]
    with E.pool(a.workers) as ex:
        of = out/'test_opponent.json'
        if of.exists():
            opp_rows = json.loads(of.read_text())
        else:
            opp_rows = T10.run_jobs(ex, jobs_for('opp', base['opponent'], law, batch))
            of.write_text(json.dumps(opp_rows))
        opp = json.loads(Path(sel['spec']).read_text())['opponent']
        rows = T10.run_jobs(ex, jobs_for('cls_es', opp, law, batch))
    rep = D9.report(rows, opp_rows)
    crashes = sum(not r['flown'] for r in rows)
    doc = dict(arm='cls_es', seed=a.seed, machine=a.machine, plan='v11', note=f"classical upper-bound arm, selected gen {sel['generation']}",
               report=rep, crashes=crashes, opponent=opp, rows=rows)
    (out/f'evaluate_cls_es_s{a.seed}_v11.json').write_text(json.dumps(doc, indent=1))
    print('cls_es', json.dumps({t: round(rep[t]['ratio'], 3) for t in D9.TASKS}), 'mean', round(rep['manoeuvre_mean'], 3), 'crashes', crashes, flush=True)


def selftest():
    base = json.loads(Path('artifacts/spike_steer/plan_v9/baseline.json').read_text())
    th = flatten(base['opponent'])
    assert th.size == 28
    back = unflatten(th)
    for t, cl in CELLS:
        for k in KEYS:
            assert abs(back[t][cl][k]-base['opponent'][t][cl][k]) < 1e-12, (t, cl, k)
    th2 = th+SIGMA*np.random.default_rng(0).standard_normal(28)
    o2 = unflatten(th2)
    assert all(o2[t][cl]['go_beats'] >= 0 and 0 <= o2[t][cl]['go_ref'] <= B.REF_MAX for t, cl in CELLS)
    print('descend_cls selftest ok: 28 knobs round-trip, clipping ok; sigma', SIGMA, '-> ref +-%.3f rad, beats +-%.1f' % (SIGMA*.5, SIGMA*20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'finetune', 'reselect', 'evaluate'])
    p.add_argument('--output', default=OUT)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--machine', default='CloudPC-2')
    a = p.parse_args()
    if a.stage == 'selftest':
        selftest()
        return
    E.load_signs()
    B.DASH_M['value'] = .15
    {'finetune': stage_finetune, 'reselect': stage_reselect, 'evaluate': stage_evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
