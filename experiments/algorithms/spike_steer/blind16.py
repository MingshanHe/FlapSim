"""Plan v16 (artifacts/spike_steer/plan_v16/preregistration.json): confirmatory replication of the sustained-disturbance
observation (SNN vs the classical opponent with its phase pulse law kept on) on fresh seed bases at three turbulence levels.
Frozen controllers, no training. The PPO arm runs separately: ppo_blind14 --sets v16_<test>.

    python -m experiments.algorithms.spike_steer.blind16 run --test turbc15 --workers 19
    python -m experiments.algorithms.spike_steer.blind16 report
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import bench8 as E
from . import bench as B
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .ablate14 import job as job14
from .blind12 import load_arms
from .blind13 import cases13
from .evaluate_v8 import clopper_pearson
from .paired_report import paired, ratios

OUT = Path('runs/spike_steer/v16')
ART = Path('artifacts/spike_steer/plan_v16')
TESTS = dict(turbc125=dict(base=2_103_200_000, scale=dict(pulse_force=1.25, pulse_torque=1.25)),
             turbc15=dict(base=2_103_400_000, scale=dict(pulse_force=1.5, pulse_torque=1.5)),
             turbc175=dict(base=2_103_600_000, scale=dict(pulse_force=1.75, pulse_torque=1.75)))
SNN = [f'snn11_s{s}' for s in range(1, 6)]
PPO = [f'mlp_ppo_s{s}' for s in range(1, 6)]


def cases16(task, test):
    return cases13(task, TESTS[test])


def stage_run(a):
    from . import cache8 as _C8  # noqa: F401
    d = OUT/a.test
    d.mkdir(parents=True, exist_ok=True)
    arms, law = load_arms()
    opp = arms[0][2]
    cs = {t: cases16(t, a.test) for t in D9.TASKS}
    (d/'cases.json').write_text(json.dumps(cs))
    with E.pool(a.workers) as ex:
        cls = [('classical', 'classical', opp), ('cls_unsup', 'classical_unsup', opp)]+[(n, 'classical', p) for n, k, p in arms if n == 'cls_es_s1']
        for name, kind, o in cls:
            if (d/f'rows_{name}.json').exists():
                continue
            rows = list(ex.map(job14, [(name, t, c, kind, dict(law=law, schedule=o[t][T.klass(c)])) for t in D9.TASKS for c in cs[t]], chunksize=1))
            _save(d, name, rows)
        learned = [(n, s) for n, k, s in arms if n in SNN and not (d/f'rows_{n}.json').exists()]
        if learned:
            rows = T10.run_grouped(ex, [(n, t, c, s) for n, s in learned for t in D9.TASKS for c in cs[t]], chunk=len(learned))
            for n, _ in learned:
                _save(d, n, [r for r in rows if r['controller'] == n])


def _save(d, name, rows):
    (d/f'rows_{name}.json').write_text(json.dumps(rows))
    print(name, 'crashes', sum(not r['flown'] for r in rows), flush=True)


def _load(test, arm):
    f = Path(f'runs/spike_steer/v14/v16_{test}__{arm}.json') if arm.startswith('mlp') else OUT/test/f'rows_{arm}.json'
    return json.loads(f.read_text()) if f.exists() else None


def family_delta(rule, A, Bf, n_boot=4000, seed=0):
    """Per case: mean ratio over family A minus mean ratio over family B (rule medians); bootstrap over cases per task."""
    med = {t: {cl: float(np.median([r['score'] for r in rule if r['task'] == t and r['cls'] == cl])) for cl in T.CLASSES[t]} for t in D9.TASKS}
    key = lambda r: (r['task'], r['seed'])

    def fam(rowsets):
        per = [{key(r): r['score']/med[r['task']][r['cls']] for r in rs} for rs in rowsets]
        ks = sorted(per[0])
        return ks, np.array([[p[k] for k in ks] for p in per]).mean(0)
    ks, a = fam(A)
    _, b = fam(Bf)
    d = a-b
    tasks = np.array([k[0] for k in ks])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(np.mean([d[tasks == t][rng.integers(0, (tasks == t).sum(), (tasks == t).sum())].mean() for t in D9.TASKS]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean([d[tasks == t].mean() for t in D9.TASKS])), [float(lo), float(hi)]


def stage_report(a):
    S = dict(plan='v16')
    lines = []
    for test in TESTS:
        rule = _load(test, 'cls_unsup')
        if rule is None:
            continue
        rc = sum(not r['flown'] for r in rule)
        T_ = dict(cls_unsup=dict(crashes=rc, ci95=clopper_pearson(rc, len(rule))))
        lines.append(f'{test}: cls_unsup crashes {rc}')
        for arm in ['classical', 'cls_es_s1']+SNN+PPO:
            rows = _load(test, arm)
            if rows is None:
                continue
            p = paired(rule, rows)
            ok = bool(p['delta_ci95'][1] < 0 and p['crashes'] <= rc)
            T_[arm] = dict(crashes=p['crashes'], delta_vs_rule=p['delta'], ci95=p['delta_ci95'], beats_rule=ok)
            lines.append('  %-11s crashes %3d  Δ vs rule %+.3f [%+.3f, %+.3f]%s' % (arm, p['crashes'], p['delta'], *p['delta_ci95'], '  beats rule' if ok else ''))
        sn = [_load(test, x) for x in SNN]
        pp = [_load(test, x) for x in PPO]
        if all(sn) and all(pp):
            dl, ci = family_delta(rule, sn, pp)
            T_['snn_minus_ppo'] = dict(delta=dl, ci95=ci)
            lines.append('  SNN family - PPO family: %+.3f [%+.3f, %+.3f]' % (dl, *ci))
        T_['snn_beating_rule'] = sum(T_[x]['beats_rule'] for x in SNN if x in T_)
        S[test] = T_
    c = S.get('turbc15', {})
    S['R1'] = dict(seeds=c.get('snn_beating_rule'), holds=bool(c.get('snn_beating_rule', 0) >= 3))
    if 'snn_minus_ppo' in c:
        S['R2'] = dict(**c['snn_minus_ppo'], holds=bool(c['snn_minus_ppo']['ci95'][1] < 0))
    lines.append('R1 (replication at x1.5): %s' % json.dumps(S['R1']))
    lines.append('R2 (SNN family better than PPO family at x1.5): %s' % json.dumps(S.get('R2')))
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'summary.json').write_text(json.dumps(S, indent=1))
    (ART/'report.txt').write_text(txt+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run', 'report'])
    p.add_argument('--test', choices=list(TESTS), default='turbc15')
    p.add_argument('--workers', type=int, default=19)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    {'run': stage_run, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
