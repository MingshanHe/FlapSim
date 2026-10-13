"""T4 large-upset recovery, v2 calibration (research plan v4, stage S0). Reuses upset.py (scenario, controllers, score).

What changed from Stage A (artifacts/spike_steer/upset_a/report.md, "two problems that must be reported"):
  * calibration is done per direction class on a log-spaced magnitude grid wide enough to bracket the survival
    curve (1-40 mN m), with the classical HOVER controller of the composite benchmark (retuned PID + phase-locked
    pulse law), not the bare PID; levels are read off a monotone (isotonic) survival curve and are CLAMPED to the
    sampled range and separated by at least 20 % in magnitude; a class whose curve does not reach the target
    survival inside the grid is flagged;
  * recovery window 2.5 s (was 1.5 s), episodes 5 s;
  * the blind test set is 8 directions x 3 levels x 2 seeds = 48 cases (half with a second hit).
Classical arms for S0: pid (gain grid) and pulse (36-point law grid), both tuned on the 8 training cases.
The per-case score is upset.py's J (tilt integral before and after the hit + 0.2 x |dz| cm; crash 60 + 40 x unflown).

    python -m experiments.algorithms.spike_steer.upset_v2 selftest
    python -m experiments.algorithms.spike_steer.upset_v2 all --output runs/spike_steer/upset_v2 --workers 30 [--quick]
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from ..neuro_gates import gates
from . import upset as U
from . import pilot_v3 as V

WINDOW, SECONDS = 2.5, 5.
CALIB_MAGS = tuple(float(x) for x in np.geomspace(1e-3, 4e-2, 9))
TARGET_SURVIVAL = (.8, .5, .2)
MIN_SEP = 1.2
BASES = dict(calib=2_070_100_000, train=2_070_300_000, test=2_070_400_000)
HOVER_GAINS = (1.5, .85)
HOVER_LAW = None
QUICK = False


def patch():
    """upset.py's scenario constants: longer recovery window, longer episode (module globals read at call time)."""
    U.WINDOW, U.SECONDS = WINDOW, SECONDS


def n(full, quick):
    return quick if QUICK else full


def law():
    global HOVER_LAW
    if HOVER_LAW is None:
        b = json.loads(V.V2_LAW.read_text())['best']
        HOVER_LAW = dict(k_theta=b['k_theta'], k_q=b['k_q'], threshold=b['threshold'], width=b['width'])
    return HOVER_LAW


def isotonic_decreasing(y, w=None):
    """Pool-adjacent-violators for a non-increasing fit."""
    y = [float(v) for v in y]
    w = [1.]*len(y) if w is None else [float(v) for v in w]
    blocks = [[v, wt, 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks)-1:
        if blocks[i][0] < blocks[i+1][0]:
            a, b = blocks[i], blocks[i+1]
            m = (a[0]*a[1]+b[0]*b[1])/(a[1]+b[1])
            blocks[i:i+2] = [[m, a[1]+b[1], a[2]+b[2]]]
            i = max(i-1, 0)
        else:
            i += 1
    out = []
    for m, _, k in blocks:
        out += [m]*k
    return out


def levels_from_curve(mags, survival):
    """Invert a monotone survival curve at TARGET_SURVIVAL (log-magnitude interpolation); clamp to the sampled range
    and keep successive levels >= MIN_SEP apart. Returns (levels, flags)."""
    x = np.log(np.asarray(mags, dtype=float))
    s = np.asarray(isotonic_decreasing(survival))
    levels, flags = [], []
    for target in TARGET_SURVIVAL:
        if s[0] <= target:
            levels.append(float(mags[0]))
            flags.append('below_grid')
            continue
        if s[-1] >= target:
            levels.append(float(mags[-1]))
            flags.append('above_grid')
            continue
        j = int(np.flatnonzero(s > target)[-1])     # last point above the target; s[j+1] <= target
        if s[j] == s[j+1]:
            lv = x[j+1]
        else:
            lv = x[j]+(s[j]-target)/(s[j]-s[j+1])*(x[j+1]-x[j])
        levels.append(float(math.exp(lv)))
        flags.append('ok')
    for k in range(1, 3):
        if levels[k] < levels[k-1]*MIN_SEP:
            levels[k] = min(levels[k-1]*MIN_SEP, float(mags[-1]))
            flags[k] = flags[k] if flags[k] != 'ok' else 'separated'
    return levels, flags


def calibrate_levels(rows):
    out = {}
    for cls in U.CLASSES:
        items = [r for r in rows if r['cls'] == cls]
        mags = sorted({r['mag'] for r in items})
        surv = [float(np.mean([r['flown'] for r in items if r['mag'] == m])) for m in mags]
        levels, flags = levels_from_curve(mags, surv)
        out[cls] = dict(levels=levels, flags=flags, survival_by_mag={f'{m:.4g}': s for m, s in zip(mags, surv)})
    return out


def stage_calibrate(ex, out):
    if (r := U.load(out, 'calibrate.json')):
        return r
    mags, seeds = CALIB_MAGS if not QUICK else CALIB_MAGS[::4], n(4, 1)
    jobs = [('calib', U.upset_case(BASES['calib']+1000*j+100*d+s, phi, m, False, s), 'pulse', law(), HOVER_GAINS)
            for j, m in enumerate(mags) for d, phi in enumerate(U.DIRECTIONS[:n(8, 3)]) for s in range(seeds)]
    rows = U.run_jobs(ex, jobs)
    if QUICK:
        rows += [dict(r, cls=c) for r in rows for c in U.CLASSES if c != r['cls']]
    levels = calibrate_levels(rows)
    result = dict(stage='calibrate', controller='pid+pulse', gains=HOVER_GAINS, law=law(), window=WINDOW, seconds=SECONDS,
                  mags=list(mags), levels=levels, rows=rows)
    U.save(out, 'calibrate.json', result)
    return result


def train_cases(levels):
    lv = (1, 2, 0, 1, 2, 0, 1, 1)
    return [U.upset_case(BASES['train']+k, phi, levels[U.klass(phi)]['levels'][lv[k]], k % 2 == 1, k)
            for k, phi in enumerate(U.DIRECTIONS)]


def test_cases(levels):
    out, i = [], 0
    for phi in U.DIRECTIONS:
        for level in range(3):
            for s in range(n(2, 1)):
                out.append(U.upset_case(BASES['test']+i, phi, levels[U.klass(phi)]['levels'][level], s % 2 == 1, i))
                i += 1
    return out[:n(48, 4)]


def stage_tune(ex, out, cases):
    """Classical arms on the 8 training cases: PID gain grid (bare PID) and the pulse-law grid on top of the retuned PID."""
    if (r := U.load(out, 'tune.json')):
        return r
    grid = gates.GRID if not QUICK else gates.GRID[::8]
    rows = U.run_jobs(ex, [(f'pid|{g[0]:g},{g[1]:g}', c, 'pid', {}, g) for g in grid for c in cases])
    pid = {}
    for g in grid:
        items = [r for r in rows if r['controller'] == f'pid|{g[0]:g},{g[1]:g}']
        pid[f'{g[0]:g},{g[1]:g}'] = dict(flown=sum(r['flown'] for r in items), J=U.mean(items, 'J'))
    pid_best = max(pid, key=lambda k: (pid[k]['flown'], -pid[k]['J']))
    pulse_grid = U.PULSE_GRID if not QUICK else U.PULSE_GRID[:2]
    rows2 = U.run_jobs(ex, [(f'pulse|{i}', c, 'pulse', q, HOVER_GAINS) for i, q in enumerate(pulse_grid) for c in cases])
    pulse = []
    for i, q in enumerate(pulse_grid):
        items = [r for r in rows2 if r['controller'] == f'pulse|{i}']
        pulse.append(dict(q=q, flown=sum(r['flown'] for r in items), J=U.mean(items, 'J')))
    pulse_best = max(range(len(pulse)), key=lambda i: (pulse[i]['flown'], -pulse[i]['J']))
    result = dict(stage='tune', pid=dict(grid=pid, best=pid_best, gains=[float(v) for v in pid_best.split(',')]),
                  pulse=dict(candidates=pulse, best=pulse[pulse_best], gains=HOVER_GAINS))
    U.save(out, 'tune.json', result)
    print(json.dumps(dict(pid_best=pid_best, pid=pid[pid_best], pulse_best=pulse[pulse_best])), flush=True)
    return result


def stage_test(ex, out, cases, tune):
    if (r := U.load(out, 'test.json')):
        return r
    U.save(out, 'preregistration.json', dict(
        version='upset_v2 (research plan v4, stage S0, task T4)', window=WINDOW, seconds=SECONDS,
        score='J = tilt integral [2 s, t0) + [t0, t0+2.5 s] (deg s) + 0.2 |dz| cm; crash = 60 + 40 x unflown fraction',
        calibration='levels at 80/50/20 % survival of the classical hover controller (PID x1.5/0.85 + pulse law), per class, '
                    'isotonic curve on a 1-40 mN m grid, clamped to the grid, >= 20 % apart',
        arms=['pid (gain grid on 8 training cases)', 'pulse (36-point law grid on 8 training cases)'],
        classical_opponent='the better of pid/pulse on the TRAINING cases is the T4 opponent in the composite benchmark',
        cases=[c['seed'] for c in cases], time=time.strftime('%Y-%m-%d %H:%M:%S')))
    arms = [('pid', 'pid', {}, tuple(tune['pid']['gains'])), ('pulse', 'pulse', tune['pulse']['best']['q'], tuple(tune['pulse']['gains']))]
    rows = U.run_jobs(ex, [(name, c, kind, q, g) for name, kind, q, g in arms for c in cases])
    by = {name: sorted([r for r in rows if r['controller'] == name], key=lambda r: r['seed']) for name, *_ in arms}
    table = {name: U.summary(v) for name, v in by.items()}
    by_class = {f'{name}|{cls}': U.summary([r for r in v if r['cls'] == cls]) for name, v in by.items() for cls in U.CLASSES}
    by_level = {}
    for name, v in by.items():
        for k in range(3):
            lv = {cls: tune_levels(out)[cls]['levels'][k] for cls in U.CLASSES}
            by_level[f'{name}|L{k}'] = U.summary([r for r in v if abs(r['mag']-lv[r['cls']]) < 1e-12])
    result = dict(stage='test', table=table, by_class=by_class, by_level=by_level,
                  pulse_vs_pid=U.compare(by['pulse'], by['pid']), rows=rows)
    U.save(out, 'test.json', result)
    for name, s in table.items():
        print(f"{name:6s} flown {s['flown']}/{s['episodes']} J {s['J']:.2f} peak {s['tilt_peak_deg']:.1f} "
              f"rec {s['recovery_censored_s']:.2f}s recovered {s['recovered']}", flush=True)
    print(json.dumps(result['pulse_vs_pid']), flush=True)
    return result


def tune_levels(out):
    return U.load(out, 'calibrate.json')['levels']


def worker_init(source):
    patch()
    U.worker_init(source)


def pool(workers):
    from concurrent.futures import ProcessPoolExecutor
    from ..snn_pid_compare.common import prepare_source
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(source),))


def selftest():
    patch()
    assert U.WINDOW == 2.5 and U.SECONDS == 5.
    assert isotonic_decreasing([1., .9, 1., .5, .6, 0.]) == [1., .95, .95, .55, .55, 0.]
    mags = list(CALIB_MAGS)
    lv, fl = levels_from_curve(mags, [1., 1., 1., .9, .6, .3, .1, 0., 0.])
    assert fl == ['ok', 'ok', 'ok'] and lv[0] < lv[1] < lv[2] and mags[3] < lv[0] < mags[4] and mags[5] < lv[2] < mags[6], (lv, fl)
    lv, fl = levels_from_curve(mags, [1.]*9)
    assert fl == ['above_grid']*3 and lv[0] == mags[-1]
    lv, fl = levels_from_curve(mags, [0.]*9)
    assert fl == ['below_grid']*3
    lv, fl = levels_from_curve(mags, [1., 1., 1., 1., 0., 0., 0., 0., 0.])
    assert lv[1] >= lv[0]*MIN_SEP-1e-12 and lv[2] >= lv[1]*MIN_SEP-1e-12, lv
    c = U.upset_case(BASES['test'], 90., 9e-3, True)
    assert c['seconds'] == 5. and len(c['upset']['hits']) == 2
    assert len(CALIB_MAGS) == 9 and abs(CALIB_MAGS[0]-1e-3) < 1e-12 and abs(CALIB_MAGS[-1]-4e-2) < 1e-12
    fake = {cls: dict(levels=[5e-3, 8e-3, 1.2e-2]) for cls in U.CLASSES}
    assert len(train_cases(fake)) == 8 and len(test_cases(fake)) == 48 and len({x['seed'] for x in test_cases(fake)}) == 48
    assert set(law()) == {'k_theta', 'k_q', 'threshold', 'width'}
    print('selftest ok')


def main():
    global QUICK
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all', 'calibrate'])
    p.add_argument('--output', type=Path, default=Path('runs/spike_steer/upset_v2'))
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.stage == 'selftest':
        return selftest()
    QUICK = a.quick
    U.QUICK = a.quick
    patch()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    with pool(a.workers) as ex:
        cal = stage_calibrate(ex, out)
        print(json.dumps({k: dict(levels=[round(x*1e3, 2) for x in v['levels']], flags=v['flags']) for k, v in cal['levels'].items()}), flush=True)
        if a.stage == 'calibrate':
            return
        cases = train_cases(cal['levels'])
        tune = stage_tune(ex, out, cases)
        stage_test(ex, out, test_cases(cal['levels']), tune)


if __name__ == '__main__':
    main()
