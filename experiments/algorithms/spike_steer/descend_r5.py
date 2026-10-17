"""Plan v4 round 5: round 4 (gated layer + wingbeat refractory period, tilt penalty from 60 deg x0.3) with three changes
aimed at the nod, the task that decided every failure of rounds 2-4, and a fresh nod/dash blind set:

  1. ES / selection objective weights nod : dash = 2 : 1 (round 4: 1 : 1). Blind composite unchanged (25/25/25/25).
     Why: H1 needs (nod + dash)/2 <= 0.848 because hover = 1.000 and upset = 1.050 by construction; dash is already
     0.82-0.87 with 16/16 wins in every round-4 seed, nod was 0.915-1.748 (plan_v4/r4/claude_ea/evaluate_r4_s*.json).
  2. 8 nod cases per ES candidate per generation (dash stays 4; round 4: 4 + 4). Why: nod gradient noise; seeds 3-5
     of round 4 had sudden nod regressions mid-search.
  3. Selection on 64 nod + 64 dash validation cases (round 4: 8 + 8). Why: re-validating every round-4 centre on 128
     cases (plan_v4/r4/reselect_wide/) found that 4 of the 5 selected centres crash 1-3 times per 128 episodes, and that
     crash counts grow with the generation; 16 cases cannot see a 1-3 % crash rate.

Everything else (network, inputs, gate, tail, refractory 37 ms, righting reflex, laws, ES settings, start specs = round-2
gated imitation specs, zero-crash + tilt < 70 deg selection rule) is round 4. This module only rebinds hooks at run time.

    python -W ignore -m experiments.algorithms.spike_steer.descend_r5 selftest
    python -W ignore -m experiments.algorithms.spike_steer.descend_r5 prepare  --output runs/spike_steer/bench_r5
    python -W ignore -m experiments.algorithms.spike_steer.descend_r5 finetune --output runs/spike_steer/bench_r5 --seed 1 --generations 60 --hours 20 --workers 10
    python -W ignore -m experiments.algorithms.spike_steer.descend_r5 reselect --output runs/spike_steer/bench_r5 --seed 1 --workers 30
    python -W ignore -m experiments.algorithms.spike_steer.descend_r5 evaluate --output runs/spike_steer/bench_r5 --seed 1 --spec <selected> --tag r5_s1 --variants intact
"""
import json
import sys
from pathlib import Path
from . import bench as B
from . import descend_r3 as R
from . import descend_r4 as R4
from . import maneuver_v2 as M

WEIGHTS = dict(nod=2/3, dash=1/3)  # ES and selection objective (blind composite stays 25/25/25/25)
NOD_CASE_FACTOR = 2                # ES: nod cases per candidate per generation = 2 x cases_per_task (4) = 8
N_VAL_SELECT = 64                  # selection: validation cases per task (round 4: 8)
ES_OFFSET_MIN = 100_000            # ES batches use offset 100_000 + gen*100 (descend.finetune); validation uses 60_000
TEST_BASE_R5 = 2_096_600_000       # fresh nod/dash blind set (r1-2 2_096_100_000, r3 2_096_400_000, r4 2_096_500_000)
SEEDS = (1, 2, 3, 4, 5)

_install_r4 = R4.install


def cases_r5(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_R5, n or 16, task)
    if task == 'nod' and split == 'train' and n is not None and offset >= ES_OFFSET_MIN:
        n = n*NOD_CASE_FACTOR                                           # ES batch only (gen*100 spacing leaves room)
    return R._cases0(task, split, levels, n, offset)


def install(fresh_test=True):
    """Round-4 hooks (dispatch with refractory period, penalty slope), then the round-5 case generator. R.install is
    rebound so every round-3 entry point and pool worker runs the round-5 configuration."""
    _install_r4(fresh_test=fresh_test)
    B.cases = cases_r5 if fresh_test else R._cases0


R.install = install
R4.install = install


def prepare(a):
    install()
    out, src = Path(a.output), Path(a.source)
    tr = json.loads((out/'train'/'baseline.json').read_text())          # written by _local/fresh_baseline.py (claude-ea)
    ev = json.loads((out/'baseline.json').read_text())
    assert ev['normaliser_notes']['test_base_nod_dash'] == TEST_BASE_R5
    tr['prereg'] = dict(tr['prereg'], round=5, envelope_deg=None, soft_tilt_deg=R4.SOFT_TILT_DEG, hard_tilt_deg=R4.HARD_TILT_DEG,
                        penalty_per_deg=R4.PENALTY_PER_DEG, refractory_s=R4.REFRACTORY_S, weights=WEIGHTS,
                        nod_case_factor=NOD_CASE_FACTOR, n_val_select=N_VAL_SELECT)
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    for s in SEEDS:
        spec = json.loads((src/'train'/f'imitate_spec_s{s}.json').read_text())
        assert spec.get('refractory_s') == R4.REFRACTORY_S
        (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
    print(json.dumps(dict(prepared=str(out), seeds=list(SEEDS), normalisers=tr['normalisers'], weights=WEIGHTS)), flush=True)


def reselect(a):
    from . import reselect_r3 as S
    S.N_VAL2 = N_VAL_SELECT
    sys.argv = [sys.argv[0], '--seed', str(a.seed), '--output', a.output, '--workers', str(a.workers),
                '--tag', a.tag or f'r5_s{a.seed}'] + (['--centres', a.spec] if a.spec else [])
    S.main()


def selftest():
    install()
    assert B.cases('nod', 'test')[0]['seed'] == TEST_BASE_R5 and len(B.cases('dash', 'test')) == 16
    assert B.cases('hover', 'test')[0]['seed'] == R._cases0('hover', 'test')[0]['seed']
    assert len(B.cases('nod', 'train', None, n=4, offset=100_000+59*100)) == 8
    assert len(B.cases('dash', 'train', None, n=4, offset=100_000)) == 4
    assert len(B.cases('nod', 'train', None, n=4, offset=60_000)) == 4                 # validation untouched
    assert B.cases('nod', 'train', None, n=8, offset=100_000)[:4] == R._cases0('nod', 'train', None, 4, 100_000)
    spec = dict(R.Dn.DescendingSNN(hidden=2, seed=0).export(), gate=R.G.GATE, base_law=None, tail_s=.5, refractory_s=R4.REFRACTORY_S)
    assert isinstance(B.Learned(spec), R4.RefractoryLearned)
    R.SOFT['value'] = R4.SOFT_TILT_DEG
    rows = [dict(task='nod', score=1., tilt_peak_deg=50.), dict(task='dash', score=2., tilt_peak_deg=70.)]
    comp, _ = R.composite_r3(rows, ['nod', 'dash'], dict(nod=1., dash=1.), WEIGHTS)
    assert abs(comp-(2/3*1.+1/3*(2.+3.))) < 1e-9, comp
    print('selftest ok (round-4 dispatch, 8 nod ES cases, nod:dash 2:1, fresh blind set 2_096_600_000)')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        return selftest()
    if stage in ('prepare', 'reselect'):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--output', default='runs/spike_steer/bench_r5')
        p.add_argument('--source', default='runs/spike_steer/bench_r4')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--workers', type=int, default=30)
        p.add_argument('--tag', default='')
        p.add_argument('--spec', default=None, help='reselect: centres directory')
        a = p.parse_args()
        return (prepare if stage == 'prepare' else reselect)(a)
    R.main()


if __name__ == '__main__':
    main()
