"""Exploratory pilot (NOT a pre-registered round; training-split cases only, never a blind set): does a CLIMB penalty in
the ES objective remove the upward height escapes, with or without height inputs?

  arm A: 24 inputs (height + vertical speed, height_inputs_pilot) + climb penalty
  arm B: 20 inputs (round 4)                                      + climb penalty
Both warm-start from the round-4 seed-2 selected centre (spec_val_g59; arm A with zero weights on the 4 height channels)
and run 10 ES generations in the round-4 configuration (gate, refractory 37 ms, tilt penalty from 60 deg x0.3), then
fly the same 128 nod + 128 dash training cases as plan_v4/pilot_hin (offset 90_000). Earlier reference arms:
height inputs without penalty (pilot_hin/compare_rounds.json, hin_g9) and the round-4 centre (pilot_hin/compare_r4ref.json).

Climb penalty (training signal only, like the tilt penalty; the controller never sees the simulator state): per task,
mean over cases of max(0, climb_m - 0.35) * 1.0 ratio units per metre, climb_m = max true height above the start height.
Normal flights climb <= 0.2-0.3 m (HomePC diagnostic); escapes terminate at +1 m.

    python -W ignore -m experiments.algorithms.spike_steer._local.climb_pilot setup --arm A
    python -W ignore -m experiments.algorithms.spike_steer._local.climb_pilot finetune --arm A --generations 10 --hours 8 --workers 14
    python -W ignore -m experiments.algorithms.spike_steer._local.climb_pilot compare --arm A --gen 9 --workers 14 --n 128
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np

CLIMB_SOFT_M = .35
CLIMB_PER_M = 1.
ART = Path('artifacts/spike_steer/plan_v4')
WARM_FROM = ART/'r4/finetune_s2/spec_val_g59.json'
SEED = 2


def arm_of(argv):
    a = argv[argv.index('--arm')+1] if '--arm' in argv else None
    assert a in ('A', 'B'), 'use --arm A (24 inputs) or --arm B (20 inputs)'
    return a


ARM = arm_of(sys.argv) if __name__ == '__main__' else 'B'
if ARM == 'A':
    from . import height_inputs_pilot as H        # importing it rebinds the install hook to the 24-input version
from .. import bench as B
from .. import descend as Dn
from .. import descend_r3 as R
from .. import descend_r4 as R4

OUT = Path(f'runs/spike_steer/pilot_climb_{ARM}')
_run0 = B.run_episode
_install_prev = R.install                        # 24-input install (arm A) or round-4 install (arm B)


def run_episode_climb(env, task, c, policy, trace=False):
    res = _run0(env, task, c, policy, True)
    a = np.asarray(res['trace'])
    res['climb_m'] = float(a[:, 4].max()-env.air.start[2])
    if not trace:
        res.pop('trace')
    return res


def composite_climb(rows, tasks, normalisers, weights):
    comp, scores = R.composite_r3(rows, tasks, normalisers, weights)
    tot = sum(weights.values())
    pen = {t: float(np.mean([max(0., r.get('climb_m', 0.)-CLIMB_SOFT_M)*CLIMB_PER_M for r in rows if r['task'] == t]) or 0.)
           for t in tasks if any(r['task'] == t for r in rows)}
    return comp+sum(weights[t]*pen.get(t, 0.) for t in weights)/tot, scores


def install(fresh_test=False):
    _install_prev(fresh_test=fresh_test)
    B.run_episode = run_episode_climb
    Dn.composite_of = composite_climb


R.install = install
R4.install = install


def setup():
    for d in (OUT, OUT/'train'):
        (d/'upset_v2').mkdir(parents=True, exist_ok=True)
        shutil.copy(ART/'r3/upset_v2/calibrate.json', d/'upset_v2'/'calibrate.json')
    shutil.copy(ART/'r4/claude_ea/baseline.json', OUT/'baseline.json')
    shutil.copy(ART/'r4/train/baseline.json', OUT/'train'/'baseline.json')
    spec = json.loads(WARM_FROM.read_text())
    if ARM == 'A':
        W = np.asarray(spec['W_in'])
        spec['W_in'] = np.concatenate([W, np.zeros((W.shape[0], 4, W.shape[2]))], 1).tolist()
        spec['inputs'] = 24
    spec.pop('generation', None)
    (OUT/'train'/f'imitate_spec_s{SEED}.json').write_text(json.dumps(spec, indent=1))
    print('setup ok', OUT, 'arm', ARM, 'W_in', np.asarray(spec['W_in']).shape)


def compare(gen, workers, n):
    from . import compare_rounds as C
    run = OUT/'train'/f'finetune_s{SEED}'
    lab = f'climb{ARM}_g{gen}'
    sys.argv = [sys.argv[0], '--workers', str(workers), '--n', str(n), '--only', lab, '--extra', f'{lab}={run}/spec_val_g{gen}.json',
                '--out', str(OUT/'compare')]
    C.main()


def selftest():
    install()
    rows = [dict(task='dash', score=1., tilt_peak_deg=10., climb_m=.2), dict(task='dash', score=1., tilt_peak_deg=10., climb_m=1.)]
    R.SOFT['value'] = 60.
    comp, _ = composite_climb(rows, ['dash'], dict(dash=1.), dict(dash=1.))
    assert abs(comp-(1.+(0.+.65)/2)) < 1e-9, comp
    assert B.N_IN == (24 if ARM == 'A' else 20), B.N_IN
    print(f'selftest ok (arm {ARM}: {B.N_IN} inputs, climb penalty from {CLIMB_SOFT_M} m x{CLIMB_PER_M})')


def main():
    stage = sys.argv[1]
    if stage == 'selftest':
        return selftest()
    if stage == 'setup':
        return setup()
    install()
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
    if stage == 'compare':
        return compare(arg('--gen', 9), arg('--workers', 14), arg('--n', 128))
    if stage == 'finetune':
        rest = [x for i, x in enumerate(sys.argv[2:]) if x != '--arm' and (i == 0 or sys.argv[2:][i-1] != '--arm')]
        sys.argv = [sys.argv[0], 'finetune', '--output', str(OUT), '--seed', str(SEED)] + rest
        return R.main()
    raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()
