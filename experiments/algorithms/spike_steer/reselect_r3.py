"""Round-3 centre selection (pre-registered rule, plan v4 round 3): among the validated ES centres saved during a seed's
search (spec_val_g*.json), re-validate every centre on 16 validation cases (8 per task; the first 4 per task are the
in-run ones) ON THE SELECTION MACHINE (CloudPC) and pick the lowest penalised composite among centres with zero crashes
and no envelope trigger (true tilt peak < cmd_envelope_deg on all 16 cases). Fallbacks (reported as such): zero-crash
centres only; then all centres. The round-2 rules (8-case lowest composite; 16-case zero-crash) are reported as well.

    python -m experiments.algorithms.spike_steer.reselect_r3 --seed 1 --centres runs/spike_steer/bench_r3/train/finetune_s1 --tag r3_s1 --workers 30
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_r3 as R

N_VAL2 = 8      # per task (nod, dash)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--output', default='runs/spike_steer/bench_r3')
    p.add_argument('--centres', default=None, help='directory with spec_val_g*.json (default: <output>/train/finetune_s<seed>)')
    p.add_argument('--tag', default=None)
    a = p.parse_args()
    tag = a.tag or f'r3_s{a.seed}'
    R.install()
    out = Path(a.output)
    base_tr = json.loads((out/'train'/'baseline.json').read_text())
    env_deg = float(base_tr['prereg']['envelope_deg'])
    R.ENVELOPE['value'] = env_deg
    tasks = list(base_tr['opponent'])
    norm, weights = base_tr['normalisers'], base_tr['prereg']['weights']
    cdir = Path(a.centres or out/'train'/f'finetune_s{a.seed}')
    centres = sorted(cdir.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1]))
    val_cases = {t: B.cases(t, 'train', None, n=N_VAL2, offset=Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    first4 = {c['seed'] for t in tasks for c in val_cases[t][:4]}
    rows_out = []
    with B.pool(a.workers, base_tr['prereg']['dash_m'] or .25) as ex:
        for f in centres:
            spec = json.loads(f.read_text())
            gen = int(f.stem.split('g')[-1])
            rows = B.run_jobs(ex, [(f'g{gen}', t, c, 'learned', spec) for t in tasks for c in val_cases[t]])
            pen16, scores = R.composite_r3(rows, tasks, norm, weights)
            comp16, _ = R._composite0(rows, tasks, norm, weights)
            comp8, _ = R._composite0([r for r in rows if r['seed'] in first4], tasks, norm, weights)
            tilts = [r['tilt_peak_deg'] or 0. for r in rows]
            entry = dict(generation=gen, penalised16=pen16, composite16=comp16, composite8=comp8, crashes=sum(not r['flown'] for r in rows),
                         triggers=sum(tp > env_deg for tp in tilts), tilt_peak_max=float(max(tilts)),
                         ratios={t: scores[t]/norm[t] for t in tasks}, spec=str(f))
            rows_out.append(entry)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items() if k != 'ratios'}), flush=True)
    clean = [e for e in rows_out if e['crashes'] == 0 and e['triggers'] == 0]
    nocrash = [e for e in rows_out if e['crashes'] == 0]
    if clean:
        pick, fallback = min(clean, key=lambda e: e['penalised16']), None
    elif nocrash:
        pick, fallback = min(nocrash, key=lambda e: e['penalised16']), 'no centre without an envelope trigger: zero-crash centres only'
    else:
        pick, fallback = min(rows_out, key=lambda e: e['penalised16']), 'every centre crashed at least once: all centres'
    report = dict(seed=a.seed, envelope_deg=env_deg, n_val_per_task=N_VAL2,
                  rule='lowest penalised composite on 16 validation cases among centres with zero crashes and zero envelope triggers',
                  fallback=fallback, centres=rows_out, selected=pick,
                  selected_by_r2_rule_16=min(nocrash or rows_out, key=lambda e: e['composite16']),
                  selected_by_r2_rule_8=min(rows_out, key=lambda e: e['composite8']))
    print(f"r3 rule picks gen {pick['generation']}{' (fallback: '+fallback+')' if fallback else ''} | r2 16-case rule gen "
          f"{report['selected_by_r2_rule_16']['generation']} | 8-case rule gen {report['selected_by_r2_rule_8']['generation']}", flush=True)
    (out/f'reselect_{tag}.json').write_text(json.dumps(report, indent=1))
    print('wrote', out/f'reselect_{tag}.json', flush=True)


if __name__ == '__main__':
    main()
