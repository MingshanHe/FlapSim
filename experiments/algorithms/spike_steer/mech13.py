"""Plan v13, test 1 (artifacts/spike_steer/plan_v13/preregistration_v13.json): WHY does the learned layer crash less under
mid-manoeuvre kicks? Two hypotheses, frozen controllers, no training:

  conservatism  the SNN simply flies at a smaller tilt when the hit lands (less margin used), so the same hit tips it less;
  reaction      the SNN changes its set-points / pulses within ~150 ms after the hit, against the induced tilt, while the
                classical schedule (timer-based) cannot.

Stages
  strata  offline, from the v12 kick rows (2 / 3 / 4.5 mN m): crash rate per arm by hit direction relative to the manoeuvre
          direction (4 bins of 90 deg) and by hit time tertile;
  refly   the 3 mN m kick cases re-flown with traces for a few arms; per episode: tilt at the hit, tilt peak within 0.5 s,
          induced tilt vector d = (roll, pitch)(t_hit+60 ms) - (t_hit), set-point change dref = ref(t_hit+150 ms) - ref(t_hit),
          its projection on d (negative = corrective), pulses 150 ms before / after, recovery time below 25 deg;
  report  paired comparisons vs the classical on identical cases, and crash rate at matched tilt-at-hit bins.

    python -m experiments.algorithms.spike_steer.mech13 strata
    python -m experiments.algorithms.spike_steer.mech13 refly --workers 30
    python -m experiments.algorithms.spike_steer.mech13 report
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
from .blind12 import cases12, TESTS, load_arms
from .evaluate_v8 import clopper_pearson

OUT = Path('runs/spike_steer/v13')
ART = Path('artifacts/spike_steer/plan_v13')
REFLY_ARMS = ('classical', 'cls_es_s1', 'snn11_s4', 'snn11_s2', 'snn11_s3', 'snn_bc_s1')
PRE, POST_D, POST_R, PEAK = .15, .06, .15, .5


def rel_bin(c):
    """Hit axis angle relative to the manoeuvre direction (both in the heading frame), 4 bins of 90 deg centred on 0/90/180/270."""
    phi = c['upset']['hits'][0][1]
    d = c['dir2_deg'] if (c['task'] == 'redirect' and c['upset']['hits'][0][0]-c['t0'] > 0.) and False else c['dir_deg']
    return int(((phi-d+45.) % 360)//90)


def stage_strata(a):
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for test in ('kick2', 'kick', 'kick45'):
        d = Path('runs/spike_steer/v12')/test
        if not (d/'cases.json').exists():
            continue
        cases = {(t, c['seed']): c for t, cs in json.loads((d/'cases.json').read_text()).items() for c in cs}
        res[test] = {}
        for f in sorted(d.glob('rows_*.json')):
            arm = f.stem[len('rows_'):]
            rows = json.loads(f.read_text())
            by_dir, by_time = {}, {}
            for r in rows:
                c = cases[(r['task'], r['seed'])]
                rb = rel_bin(c)
                tb = int((c['upset']['hits'][0][0]-c['t0']-.15)/.1)      # 0: 0.15-0.25, 1: 0.25-0.35, 2: 0.35-0.45 s
                by_dir.setdefault(rb, []).append(not r['flown'])
                by_time.setdefault(tb, []).append(not r['flown'])
            res[test][arm] = dict(by_rel_dir={k: [int(sum(v)), len(v)] for k, v in sorted(by_dir.items())},
                                  by_hit_time={k: [int(sum(v)), len(v)] for k, v in sorted(by_time.items())})
    (OUT/'strata.json').write_text(json.dumps(res, indent=1))
    for test, arms in res.items():
        print(f'== {test}: crashes/n by hit axis relative to the manoeuvre (0=axis along motion, 90, 180, 270) | by hit time (0.15-0.25, 0.25-0.35, 0.35-0.45 s)')
        for arm in ('classical', 'cls_es_s1', 'snn11_s1', 'snn11_s2', 'snn11_s3', 'snn11_s4', 'snn11_s5', 'snn_bc_s1'):
            if arm in arms:
                v = arms[arm]
                print('  %-10s dir %s | time %s' % (arm, ' '.join('%2d/%d' % tuple(v['by_rel_dir'][k]) for k in sorted(v['by_rel_dir'])),
                                                   ' '.join('%2d/%d' % tuple(v['by_hit_time'][k]) for k in sorted(v['by_hit_time']))))


def features(res, c):
    a = np.array(res['trace'])
    t = a[:, 0]
    th = c['upset']['hits'][0][0]
    i0 = int(np.searchsorted(t, th))
    if i0 >= len(t):
        return dict(flown=res['flown'], termination=res['termination'], hit_before_end=False)
    def at(dt):
        return a[min(int(np.searchsorted(t, th+dt)), len(t)-1)]
    r0, rd, rr = a[i0], at(POST_D), at(POST_R)
    pre = a[(t >= th-PRE) & (t < th)]
    post = a[(t >= th) & (t < th+POST_R)]
    peak = a[(t >= th) & (t < th+PEAK)]
    d = np.array([rd[2]-r0[2], rd[1]-r0[1]])                      # (roll, pitch) induced over 60 ms
    dref = np.array([rr[10]-r0[10], rr[11]-r0[11]])              # (roll, pitch) set-point change over 150 ms
    nd = np.linalg.norm(d)
    proj = float(dref@d/nd) if nd > 1e-9 else 0.
    tilt = np.degrees(a[:, 9])
    above = np.flatnonzero((t >= th) & (tilt > 25.))
    rec = None
    if len(above):
        below = np.flatnonzero((t > t[above[0]]) & (tilt <= 25.))
        rec = float(t[below[0]]-th) if len(below) else None
    return dict(flown=res['flown'], termination=res['termination'], hit_before_end=True, task=c['task'], seed=c['seed'], cls=T.klass(c),
                t_hit_rel=float(th-c['t0']), tilt_at_hit=float(np.degrees(r0[9])), tilt_peak_post=float(tilt[(t >= th) & (t < th+PEAK)].max()) if len(peak) else None,
                d_roll=float(d[0]), d_pitch=float(d[1]), induced_deg=float(np.degrees(nd)), dref_roll=float(dref[0]), dref_pitch=float(dref[1]),
                dref_norm=float(np.linalg.norm(dref)), proj_on_induced=proj, corrective=bool(proj < -1e-4),
                ref_norm_pre=float(np.hypot(pre[:, 10], pre[:, 11]).mean()) if len(pre) else None,
                ref_norm_post=float(np.hypot(post[:, 10], post[:, 11]).mean()) if len(post) else None,
                pulses_pre=int(pre[:, 12].sum()) if len(pre) else 0, pulses_post=int(post[:, 12].sum()) if len(post) else 0,
                t_recover25=rec, crash_after_hit=(float(res['elapsed']-th) if not res['flown'] else None))


def stage_refly(a):
    OUT.mkdir(parents=True, exist_ok=True)
    K = TESTS['kick']
    arms, law = load_arms()
    arms = [x for x in arms if x[0] in REFLY_ARMS]
    cs = {task: cases12(task, K['base'], 64, K['condition'], True, K['mag']) for task in D9.TASKS}
    with E.pool(a.workers) as ex:
        for name, kind, payload in arms:
            f = OUT/f'refly_{name}.json'
            if f.exists():
                continue
            if kind == 'classical':
                jobs = [(name, task, c, 'classical', dict(law=law, schedule=payload[task][T.klass(c)]), True) for task in D9.TASKS for c in cs[task]]
            else:
                jobs = [(name, task, c, 'learned', payload, True) for task in D9.TASKS for c in cs[task]]
            rows = T10.run_jobs(ex, jobs)
            feats = [features(r, c) for r, c in zip(rows, [c for task in D9.TASKS for c in cs[task]])]
            f.write_text(json.dumps(feats))
            print(name, 'episodes', len(feats), 'crashes', sum(not x['flown'] for x in feats), flush=True)


def boot_mean(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), [float(v) for v in np.percentile(m, [2.5, 97.5])]


def stage_report(a):
    F = {f.stem[len('refly_'):]: json.loads(f.read_text()) for f in sorted(OUT.glob('refly_*.json'))}
    base = F['classical']
    key = lambda x: (x['task'], x['seed'])
    bi = {key(x): x for x in base if x.get('hit_before_end')}
    rep = {}
    print('== per-arm (paired with the classical on identical cases; mean [95% CI])')
    for arm, rows in F.items():
        rows = [x for x in rows if x.get('hit_before_end') and key(x) in bi]
        pr = [bi[key(x)] for x in rows]
        d_tilt = [x['tilt_at_hit']-y['tilt_at_hit'] for x, y in zip(rows, pr)]
        d_peak = [x['tilt_peak_post']-y['tilt_peak_post'] for x, y in zip(rows, pr) if x['tilt_peak_post'] is not None and y['tilt_peak_post'] is not None]
        corr = np.mean([x['corrective'] for x in rows])
        proj = [x['proj_on_induced'] for x in rows]
        drefn = [x['dref_norm'] for x in rows]
        pp = [x['pulses_post']-x['pulses_pre'] for x in rows]
        bins = {}
        for x in rows:
            b = '<20' if x['tilt_at_hit'] < 20 else ('20-35' if x['tilt_at_hit'] < 35 else '>35')
            bins.setdefault(b, []).append(not x['flown'])
        rep[arm] = dict(n=len(rows), crashes=int(sum(not x['flown'] for x in rows)), tilt_at_hit=boot_mean([x['tilt_at_hit'] for x in rows]),
                        d_tilt_at_hit_vs_classical=boot_mean(d_tilt), d_tilt_peak_vs_classical=boot_mean(d_peak),
                        corrective_fraction=float(corr), proj_on_induced=boot_mean(proj), dref_norm=boot_mean(drefn), pulses_post_minus_pre=boot_mean(pp),
                        crash_rate_by_tilt_at_hit={b: [int(sum(v)), len(v), [round(q, 3) for q in clopper_pearson(int(sum(v)), len(v))]] for b, v in sorted(bins.items())},
                        induced_deg=boot_mean([x['induced_deg'] for x in rows]))
        r = rep[arm]
        print('  %-10s n %d crashes %2d | tilt@hit %.1f [%.1f, %.1f] (Δ vs cls %+.1f [%+.1f, %+.1f]) | peak Δ %+.1f [%+.1f, %+.1f] | corrective %.0f%% proj %+.3f | |Δref| %.3f | pulses post−pre %+.2f | crash by tilt@hit %s' % (
            arm, r['n'], r['crashes'], r['tilt_at_hit'][0], *r['tilt_at_hit'][1], r['d_tilt_at_hit_vs_classical'][0], *r['d_tilt_at_hit_vs_classical'][1],
            r['d_tilt_peak_vs_classical'][0], *r['d_tilt_peak_vs_classical'][1], 100*r['corrective_fraction'], r['proj_on_induced'][0], r['dref_norm'][0], r['pulses_post_minus_pre'][0],
            {b: '%d/%d' % (v[0], v[1]) for b, v in r['crash_rate_by_tilt_at_hit'].items()}))
    strata = json.loads((OUT/'strata.json').read_text()) if (OUT/'strata.json').exists() else {}
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'mechanism_summary.json').write_text(json.dumps(dict(refly=rep, strata=strata), indent=1))
    (OUT/'mechanism_summary.json').write_text(json.dumps(dict(refly=rep, strata=strata), indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['strata', 'refly', 'report'])
    p.add_argument('--workers', type=int, default=30)
    a = p.parse_args()
    if a.stage == 'refly':
        E.load_signs()
        B.DASH_M['value'] = .15
    {'strata': stage_strata, 'refly': stage_refly, 'report': stage_report}[a.stage](a)


if __name__ == '__main__':
    main()
