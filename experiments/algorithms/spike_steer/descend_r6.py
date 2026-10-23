"""Plan v4 round 6 -- the FINAL, confirmatory round: round 5 with the ES search itself seeing many more cases, so that
late generations stay crash-free and the 128-case selection does not have to fall back to early, under-trained centres.

Change vs round 5 (one change, plus the generation count that pays for it):
  * 16 nod + 16 dash training cases per ES candidate per generation (round 5: 8 + 4); same generator, ES offsets
    100_000 + gen*100 (16 cases fit the 100 spacing).
  * 30 generations (round 5: 60); every round-5 selection fell in generations 4-44, four of five in 4-19.
Why (plan_v4/r5/claude_ea/reselect_r5_s*.json): with 12 cases per candidate the ES cannot see a 1-5 % crash rate; on the
128 selection cases 8 of 13 seed-4 centres and 10 of 13 seed-5 centres crash, crashes grow with the generation, and the
zero-crash rule then picks generation 4-9 in three seeds (blind nod/dash 0.88-1.01).

Everything else is round 5 (nod:dash 2:1 objective and selection, 128-case selection, refractory 37 ms, tilt penalty
from 60 deg x0.3, gate, start specs) on a fresh nod/dash blind set. This module only rebinds hooks at run time.

    python -W ignore -m experiments.algorithms.spike_steer.descend_r6 selftest
    python -W ignore -m experiments.algorithms.spike_steer.descend_r6 prepare  --output runs/spike_steer/bench_r6
    python -W ignore -m experiments.algorithms.spike_steer.descend_r6 finetune --output runs/spike_steer/bench_r6 --seed 1 --generations 30 --hours 40 --workers 10
    python -W ignore -m experiments.algorithms.spike_steer.descend_r6 reselect --output runs/spike_steer/bench_r6 --seed 1 --workers 30
    python -W ignore -m experiments.algorithms.spike_steer.descend_r6 evaluate --output runs/spike_steer/bench_r6 --seed 1 --spec <selected> --tag r6_s1 --variants intact
"""
import json
import sys
from pathlib import Path
from . import bench as B
from . import descend_r3 as R
from . import descend_r4 as R4
from . import descend_r5 as R5
from . import maneuver_v2 as M

ES_CASES = dict(nod=16, dash=16)   # per ES candidate per generation (round 5: nod 8, dash 4)
GENERATIONS = 30
TEST_BASE_R6 = 2_096_700_000       # fresh nod/dash blind set (r1-2 .1, r3 .4, r4 .5, r5 .6 x 1e6 above 2_096_000_000)
SEEDS = (1, 2, 3, 4, 5)

_install_r5 = R5.install


def cases_r6(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_R6, n or 16, task)
    if task in ES_CASES and split == 'train' and n is not None and offset >= R5.ES_OFFSET_MIN:
        n = ES_CASES[task]                                              # ES batch only (gen*100 spacing leaves room)
    return R._cases0(task, split, levels, n, offset)


def install(fresh_test=True):
    """Round-5 hooks (round-4 dispatch, penalty slope), then the round-6 case generator. R.install is rebound so every
    entry point and pool worker runs the round-6 configuration."""
    _install_r5(fresh_test=fresh_test)
    B.cases = cases_r6 if fresh_test else R._cases0


R.install = install
R4.install = install
R5.install = install


def prepare(a):
    install()
    out, src = Path(a.output), Path(a.source)
    tr = json.loads((out/'train'/'baseline.json').read_text())          # written by _local/fresh_baseline.py (claude-ea)
    ev = json.loads((out/'baseline.json').read_text())
    assert ev['normaliser_notes']['test_base_nod_dash'] == TEST_BASE_R6
    tr['prereg'] = dict(tr['prereg'], round=6, envelope_deg=None, soft_tilt_deg=R4.SOFT_TILT_DEG, hard_tilt_deg=R4.HARD_TILT_DEG,
                        penalty_per_deg=R4.PENALTY_PER_DEG, refractory_s=R4.REFRACTORY_S, weights=R5.WEIGHTS,
                        es_cases=ES_CASES, n_val_select=R5.N_VAL_SELECT, generations=GENERATIONS)
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    for s in SEEDS:
        spec = json.loads((src/'train'/f'imitate_spec_s{s}.json').read_text())
        assert spec.get('refractory_s') == R4.REFRACTORY_S
        (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
    print(json.dumps(dict(prepared=str(out), seeds=list(SEEDS), normalisers=tr['normalisers'], weights=R5.WEIGHTS, es_cases=ES_CASES)), flush=True)


def reselect(a):
    from . import reselect_r3 as S
    S.N_VAL2 = R5.N_VAL_SELECT
    sys.argv = [sys.argv[0], '--seed', str(a.seed), '--output', a.output, '--workers', str(a.workers),
                '--tag', a.tag or f'r6_s{a.seed}'] + (['--centres', a.spec] if a.spec else [])
    S.main()


def selftest():
    install()
    assert B.cases('nod', 'test')[0]['seed'] == TEST_BASE_R6 and len(B.cases('dash', 'test')) == 16
    assert B.cases('hover', 'test')[0]['seed'] == R._cases0('hover', 'test')[0]['seed']
    for t in ('nod', 'dash'):
        assert len(B.cases(t, 'train', None, n=4, offset=100_000+29*100)) == 16
        assert len(B.cases(t, 'train', None, n=4, offset=60_000)) == 4                 # validation untouched
        assert B.cases(t, 'train', None, n=4, offset=100_000)[:4] == R._cases0(t, 'train', None, 4, 100_000)
    seeds = [c['seed'] for g in (0, 1) for c in B.cases('dash', 'train', None, n=4, offset=100_000+g*100)]
    assert len(set(seeds)) == 32                                                      # consecutive generations do not overlap
    spec = dict(R.Dn.DescendingSNN(hidden=2, seed=0).export(), gate=R.G.GATE, base_law=None, tail_s=.5, refractory_s=R4.REFRACTORY_S)
    assert isinstance(B.Learned(spec), R4.RefractoryLearned)
    print('selftest ok (round-5 dispatch and weights, 16 nod + 16 dash ES cases, fresh blind set 2_096_700_000)')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        return selftest()
    if stage in ('prepare', 'reselect'):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--output', default='runs/spike_steer/bench_r6')
        p.add_argument('--source', default='runs/spike_steer/bench_r5')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--workers', type=int, default=30)
        p.add_argument('--tag', default='')
        p.add_argument('--spec', default=None, help='reselect: centres directory')
        a = p.parse_args()
        return (prepare if stage == 'prepare' else reselect)(a)
    R.main()


if __name__ == '__main__':
    main()
