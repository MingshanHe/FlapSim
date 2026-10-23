"""Round-7 PILOT (exploratory; training and validation cases only, never a blind set). Ideas from MonoRace (Bahnam et al.,
Sci. Robot. 2026): cap the speed reward, penalise the aggressive action, train from perturbed states. Arms on top of the
plan-v4 round-5 configuration (refractory 37 ms, nod:dash 2:1, tilt penalty from 60 deg x0.3, 8 nod + 4 dash cases per ES
candidate, same start specs and ES seeds):

  A (control) = round 5 unchanged -- its seed-1/seed-4 searches and 128-case centre evaluations already exist
                (plan_v4/r5/finetune_s*, claude_ea/reselect_r5_s*.json), generations 0-29 are reused.
  B = A + nod reward cap (per-case nod score counted no lower than NOD_CAP x the nod normaliser: no reward for being
        faster than 15 % better than the opponent) + pulse penalty (PULSE_PEN ratio units per motor pulse per episode)
  C = B + mid-manoeuvre perturbation in a quarter of the ES training cases: one 20 ms pitch torque impulse at
        T0 + U(0.2, 0.6) s, magnitude PERTURB_FRAC x the weakest calibrated pitch upset level, random sign

All arms are compared on the same 64 nod + 64 dash validation cases with the plain round-5 score (no cap, no pulse
penalty, no perturbation): fraction of crash-free centres, and the best eligible centre (zero crashes, tilt < 70 deg).

    python -W ignore -m experiments.algorithms.spike_steer.pilot7 selftest
    python -W ignore -m experiments.algorithms.spike_steer.pilot7 setup --arm B
    python -W ignore -m experiments.algorithms.spike_steer.pilot7 finetune --arm B --seed 1 --generations 30 --workers 15
    python -W ignore -m experiments.algorithms.spike_steer.pilot7 evaluate --arm B --seed 1 --workers 30
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_r3 as R
from . import descend_r4 as R4
from . import descend_r5 as R5
from . import upset as U

NOD_CAP = .85
PULSE_PEN = .001
PERTURB_FRAC = .5
PERTURB_EVERY = 4                     # every 4th ES training case of each task is perturbed
ART = Path('artifacts/spike_steer/plan_v4/r5')


def arm_of(argv):
    a = argv[argv.index('--arm')+1] if '--arm' in argv else 'B'
    assert a in ('B', 'C'), a
    return a


ARM = arm_of(sys.argv) if __name__ == '__main__' else 'C'
OUT = Path(f'runs/spike_steer/pilot7_{ARM}')
_install_r5 = R5.install
_levels = {}


def perturb(c, i, task):
    if ARM != 'C' or i % PERTURB_EVERY != PERTURB_EVERY-1:
        return c
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], 7_777]))
    mag = PERTURB_FRAC*float(_levels['pitch']['levels'][0])
    t0 = B.T0+float(rng.uniform(.2, .6))
    phi = float(rng.choice([90., 270.]))
    c = dict(c)
    c['upset'] = dict(t0=t0, hits=[(t0, phi, mag)], phi=phi, mag=mag, double=False, mid_manoeuvre=True)
    return c


def cases_p7(task, split, levels=None, n=None, offset=0):
    out = R5.cases_r5(task, split, levels, n, offset)
    if split == 'train' and task in ('nod', 'dash') and offset >= R5.ES_OFFSET_MIN:
        out = [perturb(c, i, task) for i, c in enumerate(out)]
    return out


def composite_p7(rows, tasks, normalisers, weights):
    capped = [dict(r, score=max(r['score'], NOD_CAP*normalisers['nod'])) if r['task'] == 'nod' else r for r in rows]
    comp, scores = R.composite_r3(capped, tasks, normalisers, weights)
    tot = sum(weights.values())
    pen = {t: PULSE_PEN*float(np.mean([r['pulses'] for r in rows if r['task'] == t])) for t in tasks if any(r['task'] == t for r in rows)}
    return comp+sum(weights[t]*pen.get(t, 0.) for t in weights)/tot, scores


def install(fresh_test=False):
    _install_r5(fresh_test=fresh_test)
    if not _levels:
        _levels.update(json.loads(Path('artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json').read_text())['levels'])
    B.cases = cases_p7
    Dn.composite_of = composite_p7


R.install = install
R4.install = install
R5.install = install


def setup():
    for d in (OUT, OUT/'train'):
        (d/'upset_v2').mkdir(parents=True, exist_ok=True)
        shutil.copy('artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json', d/'upset_v2'/'calibrate.json')
    shutil.copy(ART/'claude_ea'/'baseline.json', OUT/'baseline.json')
    shutil.copy(ART/'train'/'baseline.json', OUT/'train'/'baseline.json')
    for s in (1, 4):
        shutil.copy(ART/'train'/f'imitate_spec_s{s}.json', OUT/'train'/f'imitate_spec_s{s}.json')
    print('setup ok', OUT)


def evaluate(seed, workers):
    """Every saved centre on the 128 round-5 selection cases with the PLAIN round-5 score (same metric for all arms)."""
    _install_r5(fresh_test=False)                                       # plain cases and objective
    tr = json.loads((OUT/'train'/'baseline.json').read_text())
    pr = tr['prereg']
    R.SOFT['value'] = float(pr['soft_tilt_deg'])
    hard, norm, weights = float(pr['hard_tilt_deg']), tr['normalisers'], pr['weights']
    tasks = ('nod', 'dash')
    cases = {t: R._cases0(t, 'train', None, R5.N_VAL_SELECT, Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    run = OUT/'train'/f'finetune_s{seed}'
    out = []
    with B.pool(workers, pr['dash_m'] or .25) as ex:
        for f in sorted(run.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1])):
            g = int(f.stem.split('g')[-1])
            if g % 5 != 4 and g != -1:
                continue
            rows = B.run_jobs(ex, [(f'g{g}', t, c, 'learned', json.loads(f.read_text())) for t in tasks for c in cases[t]])
            pen, scores = R.composite_r3(rows, tasks, norm, weights)
            crashed = [r for r in rows if not r['flown']]
            e = dict(generation=g, penalised16=pen, ratios={t: scores[t]/norm[t] for t in tasks}, crashes=len(crashed),
                     over_hard=sum((r['tilt_peak_deg'] or 0) >= hard for r in rows), pulses_mean=float(np.mean([r['pulses'] for r in rows])))
            out.append(e)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}), flush=True)
    (OUT/f'eval_s{seed}.json').write_text(json.dumps(out, indent=1))


def selftest():
    install()
    norm = dict(nod=1., dash=1.)
    rows = [dict(task='nod', score=.7, tilt_peak_deg=10., pulses=50), dict(task='dash', score=1., tilt_peak_deg=10., pulses=100)]
    R.SOFT['value'] = 60.
    comp, _ = composite_p7(rows, ['nod', 'dash'], norm, dict(nod=2/3, dash=1/3))
    assert abs(comp-(2/3*(.85+.05)+1/3*(1.+.1))) < 1e-9, comp
    es = B.cases('nod', 'train', None, n=4, offset=100_000)
    assert len(es) == 8
    pert = [c for c in es if 'upset' in c]
    if ARM == 'C':
        assert len(pert) == 2 and all(B.T0+.2 <= c['upset']['t0'] <= B.T0+.6 for c in pert), pert
        assert 0 < pert[0]['upset']['mag'] < _levels['pitch']['levels'][0]
    else:
        assert not pert
    val = R._cases0('nod', 'train', None, 64, Dn.COLLECT_OFFSET['val']+10_000)
    assert not any('upset' in c for c in val)
    print(f'selftest ok (arm {ARM}: nod cap {NOD_CAP}, pulse penalty {PULSE_PEN}/pulse, perturbed ES cases {len(pert)}/8 nod)')


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'selftest':
        return selftest()
    if stage == 'setup':
        return setup()
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k)+1]) if k in sys.argv else d
    if stage == 'evaluate':
        return evaluate(arg('--seed', 1), arg('--workers', 30))
    if stage == 'finetune':
        install()
        rest = [x for i, x in enumerate(sys.argv[2:]) if x != '--arm' and (i == 0 or sys.argv[2:][i-1] != '--arm')]
        sys.argv = [sys.argv[0], 'finetune', '--output', str(OUT)] + rest
        return R.main()
    raise SystemExit(stage)


if __name__ == '__main__':
    main()
