"""Exploratory pilot (NOT a pre-registered round; training-split cases only, never a blind set): "structural direction 1"
-- give the learned descending layer the height channels the robot already senses (rangefinder height relative to the
height reference, vertical speed), so it can see its own climb. Round-4 configuration otherwise (gate, refractory
37 ms, tilt penalty from 60 deg x0.3), seed 1, compared with round-4 seed 1 on the same 64 nod + 64 dash training cases.

Inputs 20 -> 24: bench.features + [max(dz,0), max(-dz,0), max(vz,0), max(-vz,0)] with dz = (z_meas - z_reference)/0.2 m,
vz = vz_meas/0.2 m/s (rectified halves like every other channel). Sensors, actuators, rates and limits are unchanged.
Everything is rebound at run time (bench.features, bench.N_IN, the torch twin's default width); no module is edited.

Stages (repo root, OMP_NUM_THREADS=1):
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot selftest
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot setup
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot collect  --workers 14
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot train    --seed 1 --epochs 40
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot startspec --seed 1
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot finetune --seed 1 --generations 20 --hours 12 --workers 14
    python -W ignore -m experiments.algorithms.spike_steer._local.height_inputs_pilot compare  --gen 19 --workers 14

Faster warm-start path (used): setup -> warmstart -> finetune --seed 2 --generations 10 -> compare --gen 9 --n 128, plus the
plain-process round-4 reference (see compare()). Checked on claude-ea: the zero-weight warm start reproduces the round-4
centre case for case.
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from .. import bench as B
from .. import descend as Dn
from .. import descend_r3 as R
from .. import descend_r4 as R4

OUT = Path('runs/spike_steer/pilot_hin')
ART = Path('artifacts/spike_steer/plan_v4')
Z_SCALE, VZ_SCALE = .2, .2
N_IN_H = 24
_features0 = B.features


def features_h(reading, cmd):
    r = np.asarray(reading, dtype=float)
    env = getattr(B, '_ENV', None)
    zref = float(env.teacher.z_reference) if env is not None else 0.
    dz, vz = (r[9]-zref)/Z_SCALE, r[10]/VZ_SCALE
    return np.r_[_features0(reading, cmd), max(dz, 0.), max(-dz, 0.), max(vz, 0.), max(-vz, 0.)]


_install_r4 = R4.install


def install(fresh_test=False):
    _install_r4(fresh_test=fresh_test)
    B.features = features_h
    B.N_IN = N_IN_H
    d = list(Dn.DescendingSNN.__init__.__defaults__)
    d[0] = N_IN_H                                                       # torch twin default width (n_in, hidden, seed, spec)
    Dn.DescendingSNN.__init__.__defaults__ = tuple(d)


R.install = install
R4.install = install


def setup():
    """Round-4 claude-ea baseline (opponents, normalisers, tilt rule) and upset calibration under OUT."""
    for d in (OUT, OUT/'train'):
        (d/'upset_v2').mkdir(parents=True, exist_ok=True)
        shutil.copy(ART/'r3/upset_v2/calibrate.json', d/'upset_v2'/'calibrate.json')
    shutil.copy(ART/'r4/claude_ea/baseline.json', OUT/'baseline.json')
    shutil.copy(ART/'r4/train/baseline.json', OUT/'train'/'baseline.json')
    print('setup ok', OUT)


def startspec(seed):
    """Round-4 start spec of the same seed (gate, laws, refractory) with the network replaced by the 24-input imitation."""
    base = json.loads((ART/'r4/train'/f'imitate_spec_s{seed}.json').read_text())
    net = json.loads((OUT/f'imitate_spec_s{seed}.json').read_text())
    spec = dict(base, **{k: v for k, v in net.items() if k in ('W_in', 'b_h', 'W_out', 'b_o', 'W_g', 'b_g', 'tau_m', 'tau_s', 'inputs', 'hidden', 'kind', 'spike_scale', 'tau_g')})
    assert np.asarray(spec['W_in']).shape[1] == N_IN_H and spec.get('refractory_s') == R4.REFRACTORY_S
    (OUT/'train'/f'imitate_spec_s{seed}.json').write_text(json.dumps(spec, indent=1))
    print('start spec', OUT/'train'/f'imitate_spec_s{seed}.json', 'W_in', np.asarray(spec['W_in']).shape)


WARM_FROM = ART/'r4/finetune_s2/spec_val_g59.json'   # round-4 seed-2 selected centre: most height escapes (3/128 cases)
WARM_SEED = 2


def warmstart():
    """Faster variant: the round-4 selected centre with the 4 height input channels appended at ZERO weight, so the
    start policy is tick-for-tick the round-4 one; a short ES then decides whether to use the height channels."""
    spec = json.loads(WARM_FROM.read_text())
    W = np.asarray(spec['W_in'])
    assert W.shape[1] == B.N_IN-4 == 20, W.shape
    spec['W_in'] = np.concatenate([W, np.zeros((W.shape[0], 4, W.shape[2]))], 1).tolist()
    spec['inputs'] = N_IN_H
    spec.pop('generation', None)
    (OUT/'train').mkdir(parents=True, exist_ok=True)
    (OUT/'train'/f'imitate_spec_s{WARM_SEED}.json').write_text(json.dumps(spec, indent=1))
    print('warm start spec', OUT/'train'/f'imitate_spec_s{WARM_SEED}.json', 'from', WARM_FROM, 'W_in', np.asarray(spec['W_in']).shape)


def compare(gen, workers, n=128):
    """Warm-start centres on the same training cases as the round-4 start (offset 90_000, n per task). Run in a 24-input
    process; the round-4 20-input reference must run in a plain process:
        python -W ignore -m experiments.algorithms.spike_steer._local.compare_rounds --workers W --n N --only r4_s2_g59 \
               --extra r4_s2_g59=artifacts/spike_steer/plan_v4/r4/finetune_s2/spec_val_g59.json --out runs/spike_steer/pilot_hin/compare_r4ref
    hin_g-1 (zero height weights) must reproduce it case for case."""
    from . import compare_rounds as C
    run = OUT/'train'/f'finetune_s{WARM_SEED}'
    gens = [gen]                                                        # g-1 is checked equal to the round-4 start
    extra = [f'hin_g{g}={run}/spec_val_g{g}.json' for g in gens]
    sys.argv = [sys.argv[0], '--workers', str(workers), '--n', str(n), '--only', ','.join(e.split('=')[0] for e in extra),
                '--extra', ','.join(extra), '--out', str(OUT/'compare_warm')]
    C.main()


def selftest():
    install()

    class E:
        class teacher:
            z_reference = .5
    B._ENV = E
    x = B.features(np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0, .7, -.1], np.array([1., 0.]))
    assert x.shape == (N_IN_H,) and abs(x[20]-1.) < 1e-9 and x[21] == 0. and x[22] == 0. and abs(x[23]-.5) < 1e-9, x[20:]
    B._ENV = None
    m = Dn.DescendingSNN(hidden=8, seed=0)
    assert m.n_in == N_IN_H
    spec = dict(B.random_spec(hidden=2, n_in=N_IN_H), gate=R.G.GATE, base_law=None, tail_s=.5, refractory_s=R4.REFRACTORY_S)
    pol = B.Learned(spec)
    assert isinstance(pol, R4.RefractoryLearned) and pol.net.n_in == N_IN_H if hasattr(pol, 'net') else True
    print('selftest ok (24 inputs: height and vertical speed, rectified; torch twin width 24; round-4 dispatch)')


def main():
    stage = sys.argv[1]
    if stage == 'selftest':
        return selftest()
    if stage == 'setup':
        return setup()
    install()
    if stage == 'startspec':
        return startspec(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 1)
    if stage == 'warmstart':
        return warmstart()
    if stage == 'compare':
        arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
        return compare(arg('--gen', 9), arg('--workers', 14), arg('--n', 128))
    if stage in ('collect', 'train'):
        sys.argv = [sys.argv[0], stage, '--output', str(OUT)] + sys.argv[2:]
        return Dn.main()
    if stage == 'finetune':
        sys.argv = [sys.argv[0], 'finetune', '--output', str(OUT)] + sys.argv[2:]
        return R.main()
    raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()
