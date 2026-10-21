"""Plan v12 (artifacts/spike_steer/plan_v12/preregistration_v12.json): blind-only robustness tests of FROZEN controllers.
No training. The bench10 tasks (evade / redirect / dash) and scoring are unchanged; the cases get one new element each:

  kick  a single 20 ms torque hit (KICK_MAG N m, upset.install's mechanism) about a random horizontal axis at
        t0 + U(KICK_T) s, i.e. during the manoeuvre; stress condition otherwise as bench9;
  ood   the environment's built-in 'ood' condition: lower bodies +12.5 % mass (stress disturbances on).

Arms (all frozen): the classical opponent; the classical upper-bound arm (plan v11 cls_es, seeds 1-2); the plan-v11
selected SNN + ES centres (primary, seeds 1-5); the plan-v9 selected centres (secondary); the v11 imitation start.
Analysis: paired difference vs the classical opponent on identical cases (paired_report), crashes with exact intervals,
and the preregistered settle-free dash score variant (dash score minus its settle term, which no controller meets in
the pitch / diag classes).

    python -m experiments.algorithms.spike_steer.blind12 selftest
    python -m experiments.algorithms.spike_steer.blind12 run --test kick --workers 30
    python -m experiments.algorithms.spike_steer.blind12 run --test kick --arms snn11_s4,snn11_s5 --workers 30
    python -m experiments.algorithms.spike_steer.blind12 report --test kick
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .evaluate_v8 import clopper_pearson
from .paired_report import paired

OUT = Path('runs/spike_steer/v12')
ART = Path('artifacts/spike_steer/plan_v12')
N = 64
TESTS = dict(kick=dict(base=2_099_300_000, condition='stress', kick=True, mag=3e-3),
             ood=dict(base=2_099_500_000, condition='ood', kick=False),
             kick2=dict(base=2_099_700_000, condition='stress', kick=True, mag=2e-3),      # amendment 3: dose-response
             kick45=dict(base=2_099_900_000, condition='stress', kick=True, mag=4.5e-3))
KICK_MAG, KICK_T = 3e-3, (.15, .45)
DASH_MISS = B.WINDOW['dash']                      # the settle miss constant (s)
A9, A11 = Path('artifacts/spike_steer/plan_v9'), Path('artifacts/spike_steer/plan_v11')


def cases12(task, base, n, condition, kick, mag=None):
    """bench9.cases9 with a selectable condition and an optional mid-manoeuvre kick."""
    out = []
    for i in range(n):
        c = T.make_case(base+i, T.SECONDS[task], condition, i % 8)
        k = (i+i//8)
        c['task'] = task
        if task == 'redirect':
            phi1 = T.DIRS['redirect'][k % 4]
            turn = T.TURNS[(k//4) % 2]
            c['dir_deg'], c['dir2_deg'] = float(phi1), float((phi1+turn) % 360)
        else:
            c['dir_deg'] = float(T.DIRS[task][k % len(T.DIRS[task])])
        if task == 'evade':
            c['t0'] = 2.2+.6*float(np.random.default_rng(np.random.SeedSequence([c['seed'], 917])).random())
        else:
            c['t0'] = B.T0
        if kick:
            r = np.random.default_rng(np.random.SeedSequence([c['seed'], 1201]))
            t_hit = c['t0']+KICK_T[0]+(KICK_T[1]-KICK_T[0])*float(r.random())
            c['upset'] = dict(t0=c['t0'], hits=[(float(t_hit), float(360.*r.random()), float(mag if mag is not None else KICK_MAG))])
        out.append(c)
    return out


def load_arms():
    """(name, kind, payload): kind 'classical' -> opponent dict; 'learned' -> spec."""
    base = json.loads((A9/'baseline.json').read_text())
    arms = [('classical', 'classical', base['opponent'])]
    for s, f in ((1, A11/'cls_es/s1/opp_g15.json'), (2, A11/'homepc/cls_es_s2/opp_g3.json')):
        if f.exists():
            arms.append((f'cls_es_s{s}', 'classical', json.loads(f.read_text())['opponent']))
    for s in range(1, 6):
        for tag, d in (('snn11', A11/'snn'), ('snn11', A11/'homepc_snn')):
            f = d/f's{s}/selected.json'
            if f.exists() and not any(n == f'snn11_s{s}' for n, _, _ in arms):
                arms.append((f'snn11_s{s}', 'learned', json.loads(f.read_text())))
        f = A9/f'snn/s{s}/selected.json'
        if f.exists():
            arms.append((f'snn9_s{s}', 'learned', json.loads(f.read_text())))
    f = A11/'snn/s1/bc.json'
    if f.exists():
        arms.append(('snn_bc_s1', 'learned', json.loads(f.read_text())))
    return arms, base['law']


def stage_run(a):
    from . import cache8 as _C8  # noqa: F401
    t = TESTS[a.test]
    d = OUT/a.test
    d.mkdir(parents=True, exist_ok=True)
    arms, law = load_arms()
    if a.arms:
        want = set(a.arms.split(','))
        arms = [x for x in arms if x[0] in want]
    arms = [x for x in arms if not (d/f'rows_{x[0]}.json').exists()]
    cs = {task: cases12(task, t['base'], N, t['condition'], t['kick'], t.get('mag')) for task in D9.TASKS}
    (d/'cases.json').write_text(json.dumps(cs))
    print(a.test, 'arms', [x[0] for x in arms], flush=True)
    with E.pool(a.workers) as ex:
        for name, kind, payload in arms:
            if kind == 'classical':
                rows = T10.run_jobs(ex, [(name, task, c, 'classical', dict(law=law, schedule=payload[task][T.klass(c)]))
                                         for task in D9.TASKS for c in cs[task]])
                (d/f'rows_{name}.json').write_text(json.dumps(rows))
                print(name, 'crashes', sum(not r['flown'] for r in rows), flush=True)
        learned = [(name, spec) for name, kind, spec in arms if kind == 'learned']
        if learned:
            rows = T10.run_grouped(ex, [(name, task, c, spec) for name, spec in learned for task in D9.TASKS for c in cs[task]], chunk=len(learned))
            for name, _ in learned:
                rr = [r for r in rows if r['controller'] == name]
                (d/f'rows_{name}.json').write_text(json.dumps(rr))
                print(name, 'crashes', sum(not r['flown'] for r in rr), flush=True)


def settle_free(rows):
    out = []
    for r in rows:
        r = dict(r)
        if r['task'] == 'dash':
            r['score'] = r['score']-(r['t_settle'] if r.get('t_settle') is not None else DASH_MISS)
        out.append(r)
    return out


def stage_report(a):
    d = OUT/a.test
    opp = json.loads((d/'rows_classical.json').read_text())
    oc = sum(not r['flown'] for r in opp)
    summary = dict(plan='v12', test=a.test, **TESTS[a.test], cases_per_task=N, classical=dict(crashes=oc, n=len(opp), ci95=clopper_pearson(oc, len(opp))))
    passes, n11 = 0, 0
    for f in sorted(d.glob('rows_*.json')):
        name = f.stem[len('rows_'):]
        if name == 'classical':
            continue
        rows = json.loads(f.read_text())
        p = paired(opp, rows)
        pv = paired(settle_free(opp), settle_free(rows))
        ok = bool(p['delta_ci95'][1] < 0 and p['crashes'] <= oc)
        arm, seed = (name.rsplit('_s', 1)+[None])[:2]
        rep = D9.report(rows, opp)
        doc = dict(arm=arm, seed=int(seed) if seed else None, machine=a.machine, plan='v12', test=a.test,
                   note=f"v12 {a.test}: Δ {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}], {p['crashes']} crashed of {len(rows)} (classical: {oc})",
                   report=rep, crashes=p['crashes'], paired=p, paired_settle_free=pv, H1_pass=ok if arm == 'snn11' else None, rows=rows)
        (d/f'evaluate_{name}.json').write_text(json.dumps(doc, indent=1))
        summary[name] = dict(mean=p['mean'], classical_self=p['classical_self'], delta=p['delta'], delta_ci95=p['delta_ci95'],
                             median_of_ratios=p['median_of_ratios'], wins=p['wins'], crashes=p['crashes'], ci95=clopper_pearson(p['crashes'], len(rows)),
                             settle_free=dict(mean=pv['mean'], classical_self=pv['classical_self'], delta=pv['delta'], delta_ci95=pv['delta_ci95']),
                             per_task={t: p['per_task'][t]['delta'] for t in D9.TASKS}, H1=ok if arm == 'snn11' else None,
                             crash_cases=[(r['task'], r['cls'], r['termination']) for r in rows if not r['flown']])
        if arm == 'snn11':
            n11 += 1
            passes += ok
        print('%-12s mean %.3f  self %.3f  Δ %+.3f [%+.3f, %+.3f]  settle-free Δ %+.3f  wins %3d  crashes %d%s' % (
            name, p['mean'], p['classical_self'], p['delta'], *p['delta_ci95'], pv['delta'], p['wins'], p['crashes'], '  H1 ' + ('PASS' if ok else 'fail') if arm == 'snn11' else ''), flush=True)
    summary['H1'] = dict(seeds_passing=passes, seeds=n11, holds=bool(passes >= 3))
    (d/'summary.json').write_text(json.dumps(summary, indent=1))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/f'summary_{a.test}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['H1']), flush=True)


def selftest():
    for test, t in TESTS.items():
        for task in D9.TASKS:
            cs = cases12(task, t['base'], N, t['condition'], t['kick'], t.get('mag'))
            assert len({c['seed'] for c in cs}) == N and (('upset' in cs[0]) == t['kick'])
            if t['kick']:
                assert all(c['t0']+KICK_T[0] <= c['upset']['hits'][0][0] <= c['t0']+KICK_T[1] and c['upset']['hits'][0][2] == t['mag'] for c in cs)
            if t['condition'] == 'ood':
                assert cs[0]['lower_mass_scale'] == 1.125
        print(test, 'cases ok')
    arms, law = load_arms()
    print('arms', [n for n, _, _ in arms])
    E.load_signs()
    B.DASH_M['value'] = .15
    B.worker_init(str(E.SOURCE), .15)
    c = cases12('dash', TESTS['kick']['base'], 4, 'stress', True)[1]
    c0 = dict(c)
    del c0['upset']
    pol = lambda: T.Classical9(law, arms[0][2]['dash'][T.klass(c)])
    r0 = T10.run_episode10(B._ENV, 'dash', c0, pol(), trace=True)
    r1 = T10.run_episode10(B._ENV, 'dash', c, pol(), trace=True)
    a0, a1 = np.array(r0['trace']), np.array(r1['trace'])
    t_hit = c['upset']['hits'][0][0]
    n = min(len(a0), len(a1))
    before = a0[:n, 0] < t_hit
    assert np.array_equal(a0[:n][before], a1[:n][before], equal_nan=True), 'kick changed the pre-hit trajectory'
    assert not np.array_equal(a0[:n][~before], a1[:n][~before], equal_nan=True), 'kick had no effect'
    print('kick: identical before t_hit=%.3f, differs after; score %.3f -> %.3f, tilt peak %.1f -> %.1f deg' % (
        t_hit, r0['score'], r1['score'], r0['tilt_peak_deg'], r1['tilt_peak_deg']))
    print('blind12 selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'run', 'report'])
    p.add_argument('--test', choices=list(TESTS), default='kick')
    p.add_argument('--arms', default='')
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--machine', default='CloudPC-2')
    a = p.parse_args()
    if a.stage == 'selftest':
        selftest()
        return
    E.load_signs()
    B.DASH_M['value'] = .15
    {'run': stage_run, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
