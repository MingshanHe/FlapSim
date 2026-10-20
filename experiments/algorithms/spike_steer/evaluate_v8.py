"""Plan v8 (artifacts/spike_steer/plan_v8/preregistration_v8.json): frozen plan-v7-full controllers re-tested on a fresh
blind set (bench9, base 2_098_500_000, 64 cases per task), criterion M4 per SNN + ES seed: average ratio <= 0.95 and
crashed episodes <= the classical opponent's on the same cases; M4 holds if >= 3 of 5 seeds pass.

    python -m experiments.algorithms.spike_steer.evaluate_v8 --workers 30
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import descend9 as D9

BASE, N = 2_098_500_000, 64
OUT = Path('runs/spike_steer/v8')
ART = Path('artifacts/spike_steer/plan_v8')
SNN = Path('artifacts/spike_steer/plan_v7/full/snn')


def clopper_pearson(k, n, a=.05):
    """Exact 95% interval for a binomial rate, by bisection on the binomial tail (no scipy)."""
    def tail_ge(p_):          # P(X >= k)
        return sum(math.comb(n, i)*p_**i*(1-p_)**(n-i) for i in range(k, n+1))

    def tail_le(p_):          # P(X <= k)
        return sum(math.comb(n, i)*p_**i*(1-p_)**(n-i) for i in range(0, k+1))

    def solve(f, target, inc):
        lo, hi = 0., 1.
        for _ in range(60):
            mid = (lo+hi)/2
            if (f(mid) < target) == inc:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    low = 0. if k == 0 else solve(tail_ge, a/2, True)
    high = 1. if k == n else solve(tail_le, a/2, False)
    return low, high


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=30)
    a = p.parse_args()
    from . import cache8 as _C8  # noqa: F401
    E.load_signs()
    B.DASH_M['value'] = .15
    base = json.loads(Path('artifacts/spike_steer/plan_v7/baseline.json').read_text())
    D9.N_TEST = N
    T.BASES['test'] = BASE
    cs = {t: T.cases9(t, BASE, N) for t in T.TASKS}
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for s in range(1, 6):
        specs.append((f'snn_es_s{s}', json.loads((SNN/f's{s}/selected.json').read_text())))
        specs.append((f'snn_bc_s{s}', json.loads((SNN/f's{s}/bc.json').read_text())))
    with E.pool(a.workers) as ex:
        of = OUT/'test_opponent.json'
        if of.exists():
            opp = json.loads(of.read_text())
        else:
            opp = T.run_jobs(ex, [('opp', t, c, 'classical', dict(law=base['law'], schedule=base['opponent'][t][T.klass(c)]))
                                  for t in T.TASKS for c in cs[t]])
            of.write_text(json.dumps(opp))
        print('classical done, crashes', sum(not r['flown'] for r in opp), flush=True)
        rows = T.run_grouped(ex, [(name, t, c, spec) for name, spec in specs for t in T.TASKS for c in cs[t]], chunk=len(specs))
    opp_crash = sum(not r['flown'] for r in opp)
    summary = dict(blind_base=BASE, cases_per_task=N, classical=dict(crashes=opp_crash, n=len(opp), ci95=clopper_pearson(opp_crash, len(opp))))
    passes = 0
    for name, _ in specs:
        rr = [r for r in rows if r['controller'] == name]
        rep = D9.report(rr, opp)
        crashes = sum(not r['flown'] for r in rr)
        ok = bool(rep['manoeuvre_mean'] <= .95 and crashes <= opp_crash)
        arm, seed = name.rsplit('_s', 1)
        doc = dict(arm=arm, seed=int(seed), machine='CloudPC-2', plan='v8', note=f'v8 frozen re-test, blind {BASE} × {N}/task',
                   report=rep, crashes=crashes, M4_pass=ok if arm == 'snn_es' else None, rows=rr)
        (OUT/f'evaluate_{name}.json').write_text(json.dumps(doc, indent=1))
        if arm == 'snn_es':
            passes += ok
        summary[name] = dict(mean=rep['manoeuvre_mean'], **{t: rep[t]['ratio'] for t in T.TASKS}, crashes=crashes,
                             ci95=clopper_pearson(crashes, len(rr)), M4=ok if arm == 'snn_es' else None,
                             crash_cases=[(r['task'], r['seed'], r['dir_deg'], r['termination']) for r in rr if not r['flown']])
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[name].items() if k != 'ci95'}), flush=True)
    summary['M4'] = dict(seeds_passing=passes, holds=passes >= 3)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['M4']), flush=True)


if __name__ == '__main__':
    main()
