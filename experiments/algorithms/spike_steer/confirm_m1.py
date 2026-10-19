"""Confirmatory re-test M1 (pre-registered in artifacts/spike_steer/confirm_m1/preregistration_m1.json before it is run):
the five FROZEN plan-v4 round-5 selected controllers on a FRESH nod/dash blind set, judged by a manoeuvre-focused rule.

Rationale: the learned layer acts only while a manoeuvre command is on (gate); hover and upset are flown by the classical
layer and are 1.000 / 1.050 by construction, so the four-task composite mixes two constants into the learned layer's
score. M1 makes the manoeuvres the primary test and hover / upset non-inferiority constraints.

Per seed:  M1 = (nod + dash)/2 <= 0.95  AND  hover <= 1.06  AND  upset <= 1.06  AND  0 crashes in the 32 nod+dash episodes.
Verdict:   pass if >= 3 of 5 seeds pass.   No retraining, no re-selection, run once.

    python -W ignore -m experiments.algorithms.spike_steer._local.fresh_baseline --base 2_096_900_000 --round 0 --output runs/spike_steer/bench_m1 --workers 30
    python -W ignore -m experiments.algorithms.spike_steer.confirm_m1 selftest
    python -W ignore -m experiments.algorithms.spike_steer.confirm_m1 run --workers 30
"""
import json
import sys
from pathlib import Path
from . import bench as B
from . import descend_r3 as R
from . import descend_r4 as R4
from . import descend_r5 as R5
from . import maneuver_v2 as M

TEST_BASE_M1 = 2_096_900_000
OUT = Path('runs/spike_steer/bench_m1')
ART = Path('artifacts/spike_steer/confirm_m1')
R5_SELECTED = Path('artifacts/spike_steer/plan_v4/r5/claude_ea')
MANOEUVRE_MAX, NONINF_MAX = .95, 1.06
SEEDS = (1, 2, 3, 4, 5)

_install_r5 = R5.install


def cases_m1(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_M1, n or 16, task)
    return R._cases0(task, split, levels, n, offset)


def install(fresh_test=True):
    _install_r5(fresh_test=fresh_test)
    B.cases = cases_m1 if fresh_test else R._cases0


R.install = install
R4.install = install
R5.install = install


def specs():
    out = {}
    for s in SEEDS:
        sel = json.loads((R5_SELECTED/f'reselect_r5_s{s}.json').read_text())['selected']
        g = sel['generation']
        f = Path(f'artifacts/spike_steer/plan_v4/r5/finetune_s{s}/spec_val_g{g}.json')
        out[s] = (g, f)
    return out


def verdict_of(r):
    q = r['ratios']
    m1 = (q['nod']+q['dash'])/2
    ok = dict(m1=m1 <= MANOEUVRE_MAX, hover=q['hover'] <= NONINF_MAX, upset=q['upset'] <= NONINF_MAX, no_crash=r['crashes'] == 0)
    return m1, ok, all(ok.values())


def run(workers):
    install()
    baseline = B.load_baseline(OUT)
    assert baseline['normaliser_notes']['test_base_nod_dash'] == TEST_BASE_M1
    ART.mkdir(parents=True, exist_ok=True)
    res = {}
    with B.pool(workers, baseline['prereg']['dash_m'] or .25) as ex:
        for s, (g, f) in specs().items():
            spec = json.loads(f.read_text())
            rep = B.evaluate_spec(ex, OUT, spec, f'm1_s{s}', baseline, ('intact',))
            i = rep['variants']['intact']
            crashes = sum(not r['flown'] for t in ('nod', 'dash') for r in i['rows'][t])
            r = dict(seed=s, generation=g, spec=str(f), ratios=i['ratios'], composite4=i['composite'], crashes=crashes)
            m1, ok, passed = verdict_of(r)
            r.update(m1=m1, checks=ok, passed=passed)
            res[s] = r
            (ART/f'evaluate_m1_s{s}.json').write_text(json.dumps(dict(rep, spec=str(f)), indent=1))
            print(f"seed {s} (r5 g{g}): nod {i['ratios']['nod']:.3f} dash {i['ratios']['dash']:.3f} M1 {m1:.3f} hover {i['ratios']['hover']:.3f} "
                  f"upset {i['ratios']['upset']:.4f} crashes {crashes} -> {'PASS' if passed else 'fail'} {ok}", flush=True)
    n = sum(r['passed'] for r in res.values())
    summary = dict(seeds=res, n_pass=n, verdict='PASS' if n >= 3 else 'FAIL', rule='>= 3 of 5 seeds pass M1 + non-inferiority + no crash')
    (ART/'result_m1.json').write_text(json.dumps(summary, indent=1))
    print(f'M1 verdict: {summary["verdict"]} ({n}/5)', flush=True)


def selftest():
    install()
    assert B.cases('nod', 'test')[0]['seed'] == TEST_BASE_M1 and len(B.cases('dash', 'test')) == 16
    sp = specs()
    assert len(sp) == 5 and all(f.exists() for g, f in sp.values()), sp
    r = dict(ratios=dict(nod=.9, dash=.99, hover=1., upset=1.0502), crashes=0)
    m1, ok, p = verdict_of(r)
    assert abs(m1-.945) < 1e-9 and p
    print('selftest ok (fresh blind set 2_096_900_000; frozen r5 specs', {s: g for s, (g, f) in sp.items()}, ')')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        selftest()
    elif stage == 'run':
        run(int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 30)
