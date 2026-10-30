"""Multiplicity correction for the descriptive robustness sets of plan v12 / v13 (result.md 3.2, Fig. R2): for every learned
seed (and the PPO reference), two paired tests against the classical opponent per set, Holm-Bonferroni-corrected across the
sets of the family (the confirmatory 'kick' set and its replication 'kick_rep' are excluded: they had their own criterion).

  crashes   exact McNemar test on the discordant pairs (opponent crashed / arm flew vs the reverse), two-sided
  score     paired bootstrap of the manoeuvre-ratio difference (paired_report.ratios, opponent medians), two-sided
            p = 2 min(P(delta* <= 0), P(delta* >= 0)), 20 000 resamples

    python -m experiments.algorithms.spike_steer.holm
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from . import bench9 as T
from .paired_report import ratios, TASKS

SETS = [('v12', t) for t in ('kick2', 'kick45', 'kick_early', 'kick_late', 'ood')]+\
       [('v13', t) for t in ('kick_pre', 'kick_double', 'kick_rollaxis', 'kick_pitchaxis', 'turb125', 'turb15', 'turb2', 'turb3', 'noise5', 'delay3')]
ARMS = [f'snn11_s{s}' for s in range(1, 6)]+[f'mlp_ppo_s{s}' for s in range(1, 6)]+['cls_es_s1', 'cls_es_s2']
ART = Path('artifacts/spike_steer/holm')
ALPHA = .05


def rows(plan, t, arm):
    f = Path(f'runs/spike_steer/v14/{t}__{arm}.json') if arm.startswith('mlp') else Path(f'runs/spike_steer/{plan}/{t}/rows_{arm}.json')
    return json.loads(f.read_text()) if f.exists() else None


def p_crash(opp, rr):
    O = {(r['task'], r['seed']): r['flown'] for r in opp}
    b = sum(1 for r in rr if not O[(r['task'], r['seed'])] and r['flown'])      # opponent crashed, arm flew
    c = sum(1 for r in rr if O[(r['task'], r['seed'])] and not r['flown'])
    return (1. if b+c == 0 else binomtest(b, b+c, .5).pvalue), b, c


def p_score(opp, rr, n=20000, seed=0):
    key = lambda r: (r['task'], r['seed'])
    O = {key(r): r for r in opp}
    rr = sorted(rr, key=key)
    med = {t: {cl: float(np.median([r['score'] for r in opp if r['task'] == t and r['cls'] == cl])) for cl in T.CLASSES[t]} for t in TASKS}
    a, b = ratios(rr, med), ratios([O[key(r)] for r in rr], med)
    d = {t: a[t]-b[t] for t in TASKS}
    rng = np.random.default_rng(seed)
    boots = np.mean([np.array([d[t][rng.integers(0, len(d[t]), len(d[t]))].mean() for _ in range(n)]) for t in TASKS], axis=0)
    obs = float(np.mean([d[t].mean() for t in TASKS]))
    return min(1., 2*min((boots <= 0).mean(), (boots >= 0).mean())), obs


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.
    for k, i in enumerate(order):
        run = max(run, min(1., (m-k)*ps[i]))
        adj[i] = run
    return adj


def main():
    out, lines = {}, []
    for arm in ARMS:
        recs = []
        for plan, t in SETS:
            opp, rr = rows(plan, t, 'classical'), rows(plan, t, arm)
            if opp is None or rr is None:
                continue
            pc, b, c = p_crash(opp, rr)
            ps, d = p_score(opp, rr)
            recs.append(dict(set=t, crash_p=pc, arm_only_flew=b, opp_only_flew=c, score_p=ps, delta=d))
        if not recs:
            continue
        for k in ('crash', 'score'):
            adj = holm(np.array([r[f'{k}_p'] for r in recs]))
            for r, x in zip(recs, adj):
                r[f'{k}_p_holm'] = float(x)
        out[arm] = recs
        better_c = [r['set'] for r in recs if r['crash_p_holm'] < ALPHA and r['arm_only_flew'] > r['opp_only_flew']]
        worse_c = [r['set'] for r in recs if r['crash_p_holm'] < ALPHA and r['arm_only_flew'] < r['opp_only_flew']]
        better_s = [r['set'] for r in recs if r['score_p_holm'] < ALPHA and r['delta'] < 0]
        worse_s = [r['set'] for r in recs if r['score_p_holm'] < ALPHA and r['delta'] > 0]
        lines.append(f'{arm:11s} of {len(recs)} sets, Holm alpha {ALPHA}: fewer crashes {len(better_c)} {better_c}; more crashes {worse_c}; '
                     f'better score {len(better_s)}; worse score {worse_s}')
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'holm.json').write_text(json.dumps(out, indent=1))
    (ART/'holm.txt').write_text(txt+'\n')


if __name__ == '__main__':
    main()
