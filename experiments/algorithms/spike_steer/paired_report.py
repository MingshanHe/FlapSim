"""Paired analysis of a blind set (plan v11 amendment 1): learned-arm ratios and the classical opponent's own ratio on the
SAME cases, their paired difference with a bootstrap 95 % interval, median-of-ratios, paired win rate and crashes.

ratio_i = score_i / median(classical score of the case's direction class); task ratio = mean_i; manoeuvre = mean over tasks.
The classical self-ratio is > 1 whenever the score distribution is right-skewed, so '<= 0.95' is a stricter demand than
'better than classical'; the paired difference is the comparison the criterion was meant to make.

    python -m experiments.algorithms.spike_steer.paired_report runs/spike_steer/v9/test_opponent.json runs/spike_steer/v9/evaluate_snn_es_s*_careful.json
"""
import json
import sys
from pathlib import Path
import numpy as np
from . import bench9 as T

TASKS = ('evade', 'redirect', 'dash')


def ratios(rows, med):
    return {t: np.array([r['score']/med[t][r['cls']] for r in rows if r['task'] == t]) for t in TASKS}


def paired(opp, rows, n_boot=4000, seed=0):
    key = lambda r: (r['task'], r['seed'])
    O = {key(r): r for r in opp}
    rows = sorted(rows, key=key)
    opp_m = [O[key(r)] for r in rows]
    med = {t: {cl: float(np.median([r['score'] for r in opp if r['task'] == t and r['cls'] == cl])) for cl in T.CLASSES[t]} for t in TASKS}
    a, b = ratios(rows, med), ratios(opp_m, med)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        d = []
        for t in TASKS:
            i = rng.integers(0, len(a[t]), len(a[t]))
            d.append(a[t][i].mean()-b[t][i].mean())
        deltas.append(np.mean(d))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out = dict(mean=float(np.mean([a[t].mean() for t in TASKS])), classical_self=float(np.mean([b[t].mean() for t in TASKS])),
               delta=float(np.mean([a[t].mean()-b[t].mean() for t in TASKS])), delta_ci95=[float(lo), float(hi)],
               median_of_ratios=float(np.mean([np.median(a[t]) for t in TASKS])),
               relative=float(np.mean([a[t].mean() for t in TASKS])/np.mean([b[t].mean() for t in TASKS])),
               wins=int(sum(x['score'] < y['score'] for x, y in zip(rows, opp_m))), n=len(rows),
               crashes=int(sum(not r['flown'] for r in rows)), classical_crashes=int(sum(not r['flown'] for r in opp)),
               per_task={t: dict(ratio=float(a[t].mean()), classical_self=float(b[t].mean()), delta=float(a[t].mean()-b[t].mean()),
                                 median=float(np.median(a[t]))) for t in TASKS})
    return out


def main():
    opp = json.loads(Path(sys.argv[1]).read_text())
    for f in sys.argv[2:]:
        e = json.loads(Path(f).read_text())
        r = paired(opp, e['rows'])
        print(Path(f).name, 'mean %.3f  classical-self %.3f  delta %+.3f [%+.3f, %+.3f]  median-of-ratios %.3f  wins %d/%d  crashes %d (cls %d)' % (
            r['mean'], r['classical_self'], r['delta'], *r['delta_ci95'], r['median_of_ratios'], r['wins'], r['n'], r['crashes'], r['classical_crashes']))


if __name__ == '__main__':
    main()
