"""Head-to-head of the round-2, round-3 and round-4 controllers on ONE machine and ONE set of training-split cases
(64 nod + 64 dash at offset 90_000, the round-4 pre-check cases; never a blind set), plus the classical opponent
(schedule) on the same cases for paired wins. Round-4 specs (refractory_s) dispatch to RefractoryLearned via descend_r4;
round-2/3 specs (no envelope) to GatedLearned. Ratios use the round-4 normalisers.

    python -W ignore -m experiments.algorithms.spike_steer._local.compare_rounds --workers 14 [--extra label=path,...] [--n 64]
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path
from .. import bench as B
from .. import descend_r4 as R4
from .. import descend_r3 as R

OFFSET = 90_000
P = 'artifacts/spike_steer/plan_v4'
SPECS = {'r2_s1_g44': f'{P}/r2/finetune_s1/spec_val_g44.json', 'r2_s2_g29': f'{P}/r2/finetune_s2/spec_val_g29.json',
         'r2_s3c_g59': f'{P}/r2/finetune_s3_cloudpc/spec_val_g59.json', 'r2_s5_g14': f'{P}/r2/finetune_s5/spec_val_g14.json',
         'r3_s1_g59': f'{P}/r3/finetune_s1/spec_val_g59.json', 'r3_s2_g9': f'{P}/r3/finetune_s2/spec_val_g9.json',
         'r3_s5_g9': f'{P}/r3/finetune_s5/spec_val_g9.json',
         'r4i_s1_g29': f'{P}/r4/interim_g29/s1_spec_val_g29.json', 'r4i_s2_g29': f'{P}/r4/interim_g29/s2_spec_val_g29.json',
         'r4i_s3_g29': f'{P}/r4/interim_g29/s3_spec_val_g29.json'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=14)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--extra', default='', help='label=path,... more specs (e.g. final round-4 selections)')
    p.add_argument('--only', default='', help='comma list of labels to run (default: all)')
    p.add_argument('--out', default='runs/spike_steer/compare_rounds')
    a = p.parse_args()
    R4.install(fresh_test=False)
    specs = dict(SPECS)
    for kv in filter(None, a.extra.split(',')):
        k, v = kv.split('=', 1)
        specs[k] = v
    if a.only:
        specs = {k: v for k, v in specs.items() if k in a.only.split(',')}
    ev = json.loads(Path(f'{P}/r4/claude_ea/baseline.json').read_text())
    norm = ev['normalisers']
    cases = {t: B.cases(t, 'train', None, n=a.n, offset=OFFSET) for t in ('nod', 'dash')}
    jobs = [('schedule', t, c, 'classical', ev['arms_full'][t]['schedule']) for t in cases for c in cases[t]]
    jobs += [(lab, t, c, 'learned', json.loads(Path(f).read_text())) for lab, f in specs.items() for t in cases for c in cases[t]]
    t0 = time.time()
    with B.pool(a.workers, ev['prereg']['dash_m'] or .25) as ex:
        rows = B.run_jobs(ex, jobs)
    opp = {(r['task'], r['seed']): r for r in rows if r['controller'] == 'schedule'}
    summ = {}
    for lab in ['schedule'] + list(specs):
        for t in cases:
            rs = [r for r in rows if r['controller'] == lab and r['task'] == t]
            crashed = [r for r in rs if not r['flown']]
            flown = [r for r in rs if r['flown']]
            s = dict(n=len(rs), ratio=st.mean(r['score'] for r in rs)/norm[t],
                     ratio_flown=(st.mean(r['score'] for r in flown)/norm[t]) if flown else None,
                     crashes=len(crashed), tilt=sum(r['termination'] == 'tilt' for r in crashed),
                     height=sum(r['termination'] == 'height_escape' for r in crashed),
                     wins=sum(r['score'] < opp[(t, r['seed'])]['score'] for r in rs) if lab != 'schedule' else None,
                     median_paired=st.median(r['score']/opp[(t, r['seed'])]['score'] for r in rs) if lab != 'schedule' else None,
                     tilt_peak_max=max(r['tilt_peak_deg'] or 0 for r in rs), over60=sum((r['tilt_peak_deg'] or 0) > 60 for r in rs))
            summ[f'{lab}|{t}'] = s
            print(f"{lab:12s} {t:4s} ratio {s['ratio']:.3f} flown-only {s['ratio_flown'] or float('nan'):.3f} crashes {s['crashes']:2d} "
                  f"(tilt {s['tilt']}, height {s['height']}) wins {s['wins']} med {s['median_paired'] or float('nan'):.3f} >60deg {s['over60']}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'compare_rounds.json').write_text(json.dumps(dict(offset=OFFSET, n=a.n, normalisers=norm, specs=specs, summary=summ,
                                                             wall_s=time.time()-t0, rows=rows)))
    print('wrote', out/'compare_rounds.json', f'{time.time()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
