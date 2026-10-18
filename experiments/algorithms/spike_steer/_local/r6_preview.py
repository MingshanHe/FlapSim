"""Round-6 selection PREVIEW (any machine; validation cases only, never a blind set). Interim ES centres pushed during the
round-6 searches (artifacts/spike_steer/plan_v4/r6/interim/s<S>/spec_val_g<G>.json) are flown on exactly the 64 nod +
64 dash validation cases the pre-registered selection uses (reselect_r3 with N_VAL2 = 64 via descend_r6), and scored
with the same rule (2:1 penalised composite, zero crashes, tilt < 70 deg). It only shows early what the claude-ea
selection will see (evaluation is reproducible across machines to 2e-6, r5 amendment 1); it decides nothing.
Already-evaluated centres are cached in runs/spike_steer/r6_preview/preview.json and skipped.

    python -W ignore -m experiments.algorithms.spike_steer._local.r6_preview --workers 14
"""
import argparse
import json
from pathlib import Path
from .. import bench as B
from .. import descend as Dn
from .. import descend_r6 as R6
from .. import descend_r3 as R
from .. import descend_r5 as R5

INTERIM = Path('artifacts/spike_steer/plan_v4/r6/interim')
OUT = Path('runs/spike_steer/r6_preview')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=14)
    a = p.parse_args()
    R6.install()
    base_tr = json.loads(Path('artifacts/spike_steer/plan_v4/r6/train/baseline.json').read_text())
    pr = base_tr['prereg']
    R.SOFT['value'] = float(pr['soft_tilt_deg'])
    hard, norm, weights = float(pr['hard_tilt_deg']), base_tr['normalisers'], pr['weights']
    tasks = list(base_tr['opponent'])
    cases = {t: B.cases(t, 'train', None, n=R5.N_VAL_SELECT, offset=Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT/'preview.json'
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    todo = sorted(INTERIM.glob('s*/spec_val_g*.json'), key=lambda f: (f.parent.name, int(f.stem.split('g')[-1])))
    with B.pool(a.workers, pr['dash_m'] or .25) as ex:
        for f in todo:
            key = f'{f.parent.name}/{f.stem}'
            if key in cache:
                continue
            spec = json.loads(f.read_text())
            rows = B.run_jobs(ex, [(key, t, c, 'learned', spec) for t in tasks for c in cases[t]])
            pen, scores = R.composite_r3(rows, tasks, norm, weights)
            crashed = [r for r in rows if not r['flown']]
            cache[key] = dict(penalised=pen, ratios={t: scores[t]/norm[t] for t in tasks}, crashes=len(crashed),
                              tilt=sum(r['termination'] == 'tilt' for r in crashed), height=sum(r['termination'] == 'height_escape' for r in crashed),
                              over_hard=sum((r['tilt_peak_deg'] or 0) >= hard for r in rows), tilt_peak_max=max(r['tilt_peak_deg'] or 0 for r in rows))
            cache_f.write_text(json.dumps(cache, indent=1))
            e = cache[key]
            print(f"{key:22s} pen {e['penalised']:.3f} nod {e['ratios']['nod']:.3f} dash {e['ratios']['dash']:.3f} "
                  f"crashes {e['crashes']} (tilt {e['tilt']}, height {e['height']}) tilt_max {e['tilt_peak_max']:.0f}", flush=True)
    print('\nper seed: best eligible so far (zero crashes, tilt < %.0f deg)' % hard)
    for s in sorted({k.split('/')[0] for k in cache}):
        ent = {k: v for k, v in cache.items() if k.startswith(s+'/')}
        ok = {k: v for k, v in ent.items() if v['crashes'] == 0 and v['over_hard'] == 0}
        best = min(ok.items(), key=lambda kv: kv[1]['penalised']) if ok else None
        print(f"{s}: {len(ent)} centres, {len(ok)} eligible; best {best[0].split('/')[1] if best else '-'} "
              f"{best[1]['penalised']:.3f} (nod {best[1]['ratios']['nod']:.3f}, dash {best[1]['ratios']['dash']:.3f})" if best else
              f"{s}: {len(ent)} centres, 0 eligible", flush=True)


if __name__ == '__main__':
    main()
