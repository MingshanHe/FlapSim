"""Plan v5 round 1: SAFE ES -- round-5 configuration (gated layer + refractory 37 ms, nod:dash 2:1 objective with tilt
penalty from 60 deg, 8 nod + 4 dash cases per ES candidate, 128-case selection) plus a per-generation SAFETY CHECK:

  after every ES update the new centre is flown on a fixed safety set of 64 nod + 64 dash training cases (offset
  SAFETY_OFFSET, disjoint from the ES batches, the selection cases and every blind set); if it crashes on any case or
  reaches the hard tilt (70 deg) on any case, the update is REVERTED (theta and the Adam moments restored) and the next
  generation draws new perturbations from the reverted point.

Why (plan v4 rounds 5-6): the ES objective sees a 1-3 % crash rate only rarely (12-32 cases per candidate, rank-based
utilities), so the centre drifts toward faster-but-riskier policies; on the 128 selection cases most late centres crash
and the zero-crash rule falls back to early, under-trained generations. The safety check pins every accepted centre to
zero crashes on 128 cases, so late centres stay eligible.

Implementation: descend.finetune is run one generation at a time through its own checkpoint/resume (each resume
re-seeds the perturbation stream with [1000+seed, gen], as documented in descend.py); between generations the checkpoint
is checked and possibly rolled back. No existing module is edited. Validated centres written by finetune for a reverted
generation are rewritten with the reverted (safe) parameters.

    python -W ignore -m experiments.algorithms.spike_steer.descend_safe selftest
    python -W ignore -m experiments.algorithms.spike_steer.descend_safe prepare --output runs/spike_steer/bench_v5
    python -W ignore -m experiments.algorithms.spike_steer.descend_safe finetune --output runs/spike_steer/bench_v5 --seed 1 --generations 40 --workers 10
    python -W ignore -m experiments.algorithms.spike_steer.descend_safe reselect --output runs/spike_steer/bench_v5 --seed 1 --workers 30
    python -W ignore -m experiments.algorithms.spike_steer.descend_safe evaluate --output runs/spike_steer/bench_v5 --seed 1 --spec <selected> --tag v5_s1 --variants intact
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_r3 as R
from . import descend_r4 as R4
from . import descend_r5 as R5
from . import maneuver_v2 as M

SAFETY_OFFSET = 150_000            # training generator; ES uses 100_000 + gen*100 (< 104_000), selection 60_000, diagnostics 70-90_000
N_SAFETY = 64                      # per task
TEST_BASE_V5 = 2_096_800_000       # fresh nod/dash blind set (used before: .1, .4, .5, .6, .7 x 1e6 above 2_096_000_000)
SEEDS = (1, 2, 3)

_install_r5 = R5.install


def cases_v5(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_V5, n or 16, task)
    if split == 'train' and offset >= SAFETY_OFFSET:
        return R._cases0(task, split, levels, n, offset)               # safety set: exactly n per task (no ES nod doubling)
    return R5.cases_r5(task, 'train', levels, n, offset) if split == 'train' else R._cases0(task, split, levels, n, offset)


def install(fresh_test=True):
    _install_r5(fresh_test=fresh_test)
    B.cases = cases_v5 if fresh_test else R._cases0


R.install = install
R4.install = install
R5.install = install


def safety_check(ex, spec, hard):
    tasks = ('nod', 'dash')
    rows = B.run_jobs(ex, [('safety', t, c, 'learned', spec) for t in tasks for c in B.cases(t, 'train', None, n=N_SAFETY, offset=SAFETY_OFFSET)])
    crashes = sum(not r['flown'] for r in rows)
    over = sum((r['tilt_peak_deg'] or 0) >= hard for r in rows)
    return dict(crashes=crashes, over_hard=over, tilt_peak_max=float(max(r['tilt_peak_deg'] or 0 for r in rows)),
                ok=crashes == 0 and over == 0)


def finetune(a):
    install()
    out = Path(a.output)
    tr = json.loads((out/'train'/'baseline.json').read_text())
    hard = float(tr['prereg']['hard_tilt_deg'])
    run = out/'train'/f'finetune_s{a.seed}'
    spec0 = json.loads((out/'train'/f'imitate_spec_s{a.seed}.json').read_text())
    log = run/'safety.jsonl'
    argv0 = [sys.argv[0], 'finetune', '--output', a.output, '--seed', str(a.seed), '--workers', str(a.workers), '--hours', str(a.hours)]
    if a.quick:                                                         # smoke test only: tiny ES and safety sets
        argv0.append('--quick')
        globals()['N_SAFETY'] = 2
    start = int(np.load(run/'checkpoint.npz')['gen']) if (run/'checkpoint.npz').exists() else 0
    t0 = time.monotonic()
    with B.pool(a.workers, tr['prereg']['dash_m'] or .25) as ex:
        for gen in range(start, a.generations):
            prev = dict(np.load(run/'checkpoint.npz')) if (run/'checkpoint.npz').exists() else None
            sys.argv = argv0 + ['--generations', str(gen+1)]
            R.main()                                                    # exactly one ES generation (resumes from the checkpoint)
            ck = dict(np.load(run/'checkpoint.npz'))
            res = safety_check(ex, Dn.unflatten(spec0, ck['theta']), hard)
            if a.force_fail_gen == gen:                                 # smoke test only: exercise the revert path
                res.update(ok=False, forced=True)
            res.update(generation=gen, accepted=res['ok'], wall_s=round(time.monotonic()-t0))
            if not res['ok'] and prev is not None:                       # revert: keep the generation count, restore the safe point
                np.savez(run/'checkpoint.npz', theta=prev['theta'], m=prev['m'], v=prev['v'], gen=gen+1)
                safe = dict(Dn.unflatten(spec0, prev['theta']), trained_by='descend_safe (reverted)', seed=a.seed, generation=gen)
                (run/'spec_last.json').write_text(json.dumps(safe, indent=1))
                vf = run/f'spec_val_g{gen}.json'
                if vf.exists():
                    vf.write_text(json.dumps(safe, indent=1))
            elif not res['ok']:
                res['note'] = 'generation 0 failed the check; no earlier safe point -- kept (start spec is the fallback centre g-1)'
            with log.open('a') as f:
                f.write(json.dumps(res)+'\n')
            print(json.dumps(dict(safety=res)), flush=True)


def prepare(a):
    install()
    out, src = Path(a.output), Path(a.source)
    tr = json.loads((out/'train'/'baseline.json').read_text())
    ev = json.loads((out/'baseline.json').read_text())
    assert ev['normaliser_notes']['test_base_nod_dash'] == TEST_BASE_V5
    tr['prereg'] = dict(tr['prereg'], plan='v5', round=1, envelope_deg=None, soft_tilt_deg=R4.SOFT_TILT_DEG, hard_tilt_deg=R4.HARD_TILT_DEG,
                        penalty_per_deg=R4.PENALTY_PER_DEG, refractory_s=R4.REFRACTORY_S, weights=R5.WEIGHTS,
                        nod_case_factor=R5.NOD_CASE_FACTOR, n_val_select=R5.N_VAL_SELECT,
                        safety=dict(offset=SAFETY_OFFSET, n_per_task=N_SAFETY, rule='revert on any crash or tilt >= hard_tilt'))
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    for s in SEEDS:
        spec = json.loads((src/'train'/f'imitate_spec_s{s}.json').read_text())
        assert spec.get('refractory_s') == R4.REFRACTORY_S
        (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
    print(json.dumps(dict(prepared=str(out), seeds=list(SEEDS), normalisers=tr['normalisers'])), flush=True)


def reselect(a):
    from . import reselect_r3 as S
    S.N_VAL2 = R5.N_VAL_SELECT
    sys.argv = [sys.argv[0], '--seed', str(a.seed), '--output', a.output, '--workers', str(a.workers),
                '--tag', a.tag or f'v5_s{a.seed}'] + (['--centres', a.spec] if a.spec else [])
    S.main()


def selftest():
    install()
    assert B.cases('nod', 'test')[0]['seed'] == TEST_BASE_V5
    assert len(B.cases('nod', 'train', None, n=4, offset=100_000)) == 8 and len(B.cases('dash', 'train', None, n=4, offset=100_000)) == 4
    safety = {c['seed'] for t in ('nod', 'dash') for c in B.cases(t, 'train', None, n=N_SAFETY, offset=SAFETY_OFFSET)}
    es = {c['seed'] for g in range(60) for t in ('nod', 'dash') for c in B.cases(t, 'train', None, n=4, offset=100_000+g*100)}
    val = {c['seed'] for t in ('nod', 'dash') for c in B.cases(t, 'train', None, n=64, offset=Dn.COLLECT_OFFSET['val']+10_000)}
    assert not (safety & es) and not (safety & val) and len(safety) >= 64, (len(safety), len(safety & es), len(safety & val))
    print('selftest ok (round-5 configuration, safety set disjoint from ES batches and selection cases, fresh blind set 2_096_800_000)')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        return selftest()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('stage')
    p.add_argument('--output', default='runs/spike_steer/bench_v5')
    p.add_argument('--source', default='runs/spike_steer/bench_r5')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--hours', type=float, default=40.)
    p.add_argument('--tag', default='')
    p.add_argument('--spec', default=None)
    p.add_argument('--variants', default='intact')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--force-fail-gen', type=int, default=-99)
    a, _ = p.parse_known_args()
    if stage == 'prepare':
        return prepare(a)
    if stage == 'reselect':
        return reselect(a)
    if stage == 'finetune':
        return finetune(a)
    R.main()                                                            # evaluate


if __name__ == '__main__':
    main()
