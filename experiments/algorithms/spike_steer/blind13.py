"""Plan v13 tests 3-9 (artifacts/spike_steer/plan_v13/amendment_2_overnight.json): more blind-only robustness tests of the
frozen controllers, each a bench10 case set with one change (no training; the arms of blind12.load_arms):

  turb2 / turb3   the stress disturbance pulses (pulse_force, pulse_torque) x2 / x3, no hit       -> sustained turbulence
  noise5          IMU attitude noise 0.02 -> 0.10 deg, gyro noise 0.002 -> 0.010                  -> sensor degradation
  delay3          sensor delay 1 -> 3 steps (6 ms)                                                -> sensor degradation
  kick_pre        3 mN m hit at t0 - U(0.3, 0.6) s: identical flight until t0 (gate closed), the command arrives during
                  the recovery; the pre-t0 rows are checked for bit identity between arms          -> disturbed start
  kick_double     3 mN m hit at t0 + U(0.15, 0.45) s and a second one 80-200 ms later, random axis -> double hit
  kick_rollaxis   3 mN m hit about the heading-frame roll axis only (phi in {0, 180})               -> axis of the hit
  kick_pitchaxis  3 mN m hit about the pitch axis only (phi in {90, 270})                          -> axis of the hit

    python -m experiments.algorithms.spike_steer.blind13 selftest
    python -m experiments.algorithms.spike_steer.blind13 run --test turb2 --workers 30
    python -m experiments.algorithms.spike_steer.blind13 report --test turb2
    python -m experiments.algorithms.spike_steer.blind13 table            # one crash/delta table over every finished test
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .blind12 import load_arms, settle_free, N, KICK_T
from .evaluate_v8 import clopper_pearson
from .paired_report import paired

OUT = Path('runs/spike_steer/v13')
ART = Path('artifacts/spike_steer/plan_v13')
MAG = 3e-3
TESTS = dict(
    turb2=dict(base=2_100_500_000, scale=dict(pulse_force=2., pulse_torque=2.)),
    turb3=dict(base=2_100_700_000, scale=dict(pulse_force=3., pulse_torque=3.)),
    noise5=dict(base=2_100_900_000, set=dict(attitude_noise_deg=.1, gyro_noise=.01)),
    delay3=dict(base=2_101_100_000, set=dict(delay_steps=3)),
    kick_pre=dict(base=2_101_300_000, kick_t=(-.6, -.3)),
    kick_double=dict(base=2_101_500_000, kick_t=KICK_T, double=True),
    kick_rollaxis=dict(base=2_101_700_000, kick_t=KICK_T, phis=(0., 180.)),
    kick_pitchaxis=dict(base=2_101_900_000, kick_t=KICK_T, phis=(90., 270.)),
)


def cases13(task, t, n=N):
    out = []
    for i in range(n):
        c = T.make_case(t['base']+i, T.SECONDS[task], 'stress', i % 8)
        for k, v in t.get('scale', {}).items():
            c[k] = c[k]*v
        for k, v in t.get('set', {}).items():
            c[k] = v
        k = (i+i//8)
        c['task'] = task
        if task == 'redirect':
            phi1 = T.DIRS['redirect'][k % 4]
            turn = T.TURNS[(k//4) % 2]
            c['dir_deg'], c['dir2_deg'] = float(phi1), float((phi1+turn) % 360)
        else:
            c['dir_deg'] = float(T.DIRS[task][k % len(T.DIRS[task])])
        c['t0'] = 2.2+.6*float(np.random.default_rng(np.random.SeedSequence([c['seed'], 917])).random()) if task == 'evade' else B.T0
        if 'kick_t' in t:
            r = np.random.default_rng(np.random.SeedSequence([c['seed'], 1201]))
            kt = t['kick_t']
            t_hit = c['t0']+kt[0]+(kt[1]-kt[0])*float(r.random())
            u = float(r.random())
            phi = float(t['phis'][int(u*len(t['phis']))]) if 'phis' in t else 360.*u
            hits = [(float(t_hit), phi, MAG)]
            if t.get('double'):
                hits.append((float(t_hit+.08+.12*float(r.random())), float(360.*r.random()), MAG))
            c['upset'] = dict(t0=c['t0'], hits=hits)
        out.append(c)
    return out


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
    cs = {task: cases13(task, t) for task in D9.TASKS}
    (d/'cases.json').write_text(json.dumps(cs))
    print(a.test, 'arms', [x[0] for x in arms], flush=True)
    trace = a.test == 'kick_pre'
    with E.pool(a.workers) as ex:
        for name, kind, payload in arms:
            if kind == 'classical':
                rows = T10.run_jobs(ex, [(name, task, c, 'classical', dict(law=law, schedule=payload[task][T.klass(c)]), trace)
                                         for task in D9.TASKS for c in cs[task]])
                _save(d, name, rows, trace)
        learned = [(name, spec) for name, kind, spec in arms if kind == 'learned']
        if learned:
            if trace:
                for name, spec in learned:
                    rows = T10.run_jobs(ex, [(name, task, c, 'learned', spec, True) for task in D9.TASKS for c in cs[task]])
                    _save(d, name, rows, trace)
            else:
                rows = T10.run_grouped(ex, [(name, task, c, spec) for name, spec in learned for task in D9.TASKS for c in cs[task]], chunk=len(learned))
                for name, _ in learned:
                    _save(d, name, [r for r in rows if r['controller'] == name], False)


def _save(d, name, rows, trace):
    if trace:                                       # keep only the pre-t0 fingerprint, not the full traces
        for r in rows:
            a = np.array(r.pop('trace'))
            pre = a[a[:, 0] < r['t0']-1e-9] if 't0' in r else a[:0]
            r['pre_t0_fingerprint'] = [len(pre), float(pre[:, 1:10].sum()) if len(pre) else 0., float(np.abs(pre[:, 1:10]).sum()) if len(pre) else 0.]
    (d/f'rows_{name}.json').write_text(json.dumps(rows))
    print(name, 'crashes', sum(not r['flown'] for r in rows), flush=True)


def stage_report(a):
    d = OUT/a.test
    opp = json.loads((d/'rows_classical.json').read_text())
    oc = sum(not r['flown'] for r in opp)
    summary = dict(plan='v13', test=a.test, **{k: v for k, v in TESTS[a.test].items()}, cases_per_task=N,
                   classical=dict(crashes=oc, n=len(opp), ci95=clopper_pearson(oc, len(opp))))
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
        doc = dict(arm=arm, seed=int(seed) if seed else None, machine=a.machine, plan='v13', test=a.test,
                   note=f"v13 {a.test}: Δ {p['delta']:+.3f} [{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}], {p['crashes']} crashed of {len(rows)} (classical: {oc})",
                   report=rep, crashes=p['crashes'], paired=p, paired_settle_free=pv, H1_pass=ok if arm == 'snn11' else None, rows=rows)
        if a.test == 'kick_pre' and 'pre_t0_fingerprint' in rows[0]:
            same = sum(r['pre_t0_fingerprint'] == o['pre_t0_fingerprint'] for r, o in zip(rows, opp))
            doc['pre_t0_identical_to_classical'] = [same, len(rows)]
        (d/f'evaluate_{name}.json').write_text(json.dumps(doc, indent=1))
        summary[name] = dict(mean=p['mean'], classical_self=p['classical_self'], delta=p['delta'], delta_ci95=p['delta_ci95'], median_of_ratios=p['median_of_ratios'],
                             wins=p['wins'], crashes=p['crashes'], ci95=clopper_pearson(p['crashes'], len(rows)), per_task={t_: p['per_task'][t_]['delta'] for t_ in D9.TASKS},
                             settle_free_delta=pv['delta'], H1=ok if arm == 'snn11' else None, crash_cases=[(r['task'], r['cls'], r['termination']) for r in rows if not r['flown']],
                             **({'pre_t0_identical_to_classical': doc['pre_t0_identical_to_classical']} if 'pre_t0_identical_to_classical' in doc else {}))
        if arm == 'snn11':
            n11 += 1
            passes += ok
        print('%-14s mean %.3f  self %.3f  Δ %+.3f [%+.3f, %+.3f]  wins %3d  crashes %2d%s%s' % (
            name, p['mean'], p['classical_self'], p['delta'], *p['delta_ci95'], p['wins'], p['crashes'], '  H1 ' + ('PASS' if ok else 'fail') if arm == 'snn11' else '',
            f"  pre-t0 identical {doc['pre_t0_identical_to_classical'][0]}/{doc['pre_t0_identical_to_classical'][1]}" if 'pre_t0_identical_to_classical' in doc else ''), flush=True)
    summary['H1'] = dict(seeds_passing=passes, seeds=n11, holds=bool(passes >= 3))
    (d/'summary.json').write_text(json.dumps(summary, indent=1))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/f'summary_{a.test}.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['H1']), flush=True)


def stage_table(a):
    """One table over every finished v12 / v13 test: crashes per arm and the v11 H1 count."""
    tests = [('v12', t) for t in ('kick2', 'kick', 'kick45', 'kick_early', 'kick_late', 'ood')]+[('v13', t) for t in TESTS]
    arms = ['classical', 'cls_es_s1', 'cls_es_s2', 'snn11_s1', 'snn11_s2', 'snn11_s3', 'snn11_s4', 'snn11_s5', 'snn_bc_s1']
    lines = ['| test | ' + ' | '.join(arms) + ' | H1 (v11) |', '|' + '---|'*(len(arms)+2)]
    for plan, t in tests:
        f = Path(f'runs/spike_steer/{plan}/{t}/summary.json')
        if not f.exists():
            continue
        S = json.loads(f.read_text())
        cells = [str(S['classical']['crashes'])]+[(str(S[x]['crashes']) if x in S else '–') for x in arms[1:]]
        lines.append(f'| {t} | ' + ' | '.join(cells) + f" | {S['H1']['seeds_passing']}/{S['H1']['seeds']} |")
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'crash_table.md').write_text(txt+'\n')


def selftest():
    for name, t in TESTS.items():
        for task in D9.TASKS:
            cs = cases13(task, t, 16)
            assert len({c['seed'] for c in cs}) == 16
            if 'kick_t' in t:
                for c in cs:
                    h = c['upset']['hits']
                    assert c['t0']+t['kick_t'][0]-1e-9 <= h[0][0] <= c['t0']+t['kick_t'][1]+1e-9 and h[0][2] == MAG
                    if 'phis' in t:
                        assert h[0][1] in t['phis']
                    if t.get('double'):
                        assert len(h) == 2 and .08-1e-9 <= h[1][0]-h[0][0] <= .2+1e-9
            else:
                assert 'upset' not in cs[0]
            for k, v in t.get('scale', {}).items():
                assert abs(cs[0][k]-T.make_case(t['base'], T.SECONDS[task], 'stress', 0)[k]*v) < 1e-12
            for k, v in t.get('set', {}).items():
                assert cs[0][k] == v
        print(name, 'cases ok')
    arms, _ = load_arms()
    print('arms', [n for n, _, _ in arms])
    print('blind13 selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'run', 'report', 'table'])
    p.add_argument('--test', choices=list(TESTS), default='turb2')
    p.add_argument('--arms', default='')
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--machine', default='CloudPC-2')
    a = p.parse_args()
    if a.stage == 'selftest':
        selftest()
        return
    if a.stage == 'table':
        stage_table(a)
        return
    E.load_signs()
    B.DASH_M['value'] = .15
    {'run': stage_run, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
