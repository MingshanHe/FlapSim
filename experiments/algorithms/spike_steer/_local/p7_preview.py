"""Round-7 pilot PREVIEW (any machine; validation cases only). Interim centres pushed to
artifacts/spike_steer/pilot7/interim/<arm>/s<seed>/spec_val_g<G>.json are flown on the 64 nod + 64 dash round-5
selection cases with the PLAIN round-5 score (no nod cap, no pulse penalty, no perturbation) -- the same metric as
pilot7.evaluate -- and printed next to arm A (round 5, same seed and generation, from plan_v4/r5/claude_ea/reselect_r5_s*.json).
Results are cached in runs/spike_steer/p7_preview/preview.json.

    python -W ignore -m experiments.algorithms.spike_steer._local.p7_preview --workers 14
"""
import argparse
import json
from pathlib import Path
from .. import bench as B
from .. import descend as Dn
from .. import descend_r3 as R
from .. import descend_r5 as R5

INTERIM = Path('artifacts/spike_steer/pilot7/interim')
OUT = Path('runs/spike_steer/p7_preview')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=14)
    a = p.parse_args()
    R5.install(fresh_test=False)
    tr = json.loads(Path('artifacts/spike_steer/plan_v4/r5/train/baseline.json').read_text())
    pr = tr['prereg']
    R.SOFT['value'] = float(pr['soft_tilt_deg'])
    hard, norm, weights = float(pr['hard_tilt_deg']), tr['normalisers'], pr['weights']
    tasks = ('nod', 'dash')
    cases = {t: R._cases0(t, 'train', None, R5.N_VAL_SELECT, Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT/'preview.json'
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    todo = sorted(INTERIM.glob('*/s*/spec_val_g*.json'), key=lambda f: (f.parent.parent.name, f.parent.name, int(f.stem.split('g')[-1])))
    with B.pool(a.workers, pr['dash_m'] or .25) as ex:
        for f in todo:
            key = f'{f.parent.parent.name}/{f.parent.name}/{f.stem}'
            if key in cache:
                continue
            rows = B.run_jobs(ex, [(key, t, c, 'learned', json.loads(f.read_text())) for t in tasks for c in cases[t]])
            pen, scores = R.composite_r3(rows, tasks, norm, weights)
            crashed = [r for r in rows if not r['flown']]
            cache[key] = dict(penalised=pen, nod=scores['nod']/norm['nod'], dash=scores['dash']/norm['dash'], crashes=len(crashed),
                              over_hard=sum((r['tilt_peak_deg'] or 0) >= hard for r in rows), tilt_max=max(r['tilt_peak_deg'] or 0 for r in rows))
            cf.write_text(json.dumps(cache, indent=1))
    print('arm/seed/gen        | pilot: score  nod   dash  crashes | arm A (round 5): score crashes')
    for key in sorted(cache, key=lambda k: (k.split('/')[0], k.split('/')[1], int(k.split('g')[-1]))):
        arm, s, g = key.split('/')[0], int(key.split('/')[1][1:]), int(key.split('g')[-1])
        e = cache[key]
        ref = json.loads(Path(f'artifacts/spike_steer/plan_v4/r5/claude_ea/reselect_r5_s{s}.json').read_text())
        A = next((c for c in ref['centres'] if c['generation'] == g), None)
        ok = e['crashes'] == 0 and e['over_hard'] == 0
        print(f"{arm} s{s} g{g:<3}         | {e['penalised']:.3f} {e['nod']:.3f} {e['dash']:.3f} {e['crashes']:2d}{' ok' if ok else '   '}"
              f" | {A['penalised16']:.3f} {A['crashes']:2d}" if A else '', flush=True)


if __name__ == '__main__':
    main()
