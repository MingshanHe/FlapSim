"""Exploratory check for plan v4 amendment 2 (NOT a pre-registered result): re-select a round-2 seed's ES centre with a
larger validation set (8 cases per task instead of 4; the first 4 are the original ones) and a zero-crash constraint,
then blind-test the re-selected centre once. Only the validated centres saved during the original run are considered
(spec_val_g*.json), so no new search happens; the blind set is untouched by the selection.

    python -m experiments.algorithms.spike_steer.reselect_r2 --seed 3 --workers 8
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_gated as G

N_VAL2 = 8      # per task (nod, dash)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output', default='runs/spike_steer/bench_r2')
    p.add_argument('--centres', default=None, help='directory with spec_val_g*.json (default: artifacts/spike_steer/plan_v4/r2/finetune_s<seed>)')
    p.add_argument('--blind', action='store_true', help='also blind-test the re-selected centre (intact only)')
    p.add_argument('--tag', default=None, help='output tag (default r2_s<seed>); blind test tag = <tag>_resel')
    a = p.parse_args()
    tag = a.tag or f'r2_s{a.seed}'
    G.install()
    out = Path(a.output)
    base_ev = B.load_baseline(out)
    base_tr = json.loads((out/'train'/'baseline.json').read_text())
    tasks = list(base_tr['opponent'])
    norm, weights = base_tr['normalisers'], base_tr['prereg']['weights']
    cdir = Path(a.centres or f'artifacts/spike_steer/plan_v4/r2/finetune_s{a.seed}')
    centres = sorted(cdir.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1]))
    val_cases = {t: B.cases(t, 'train', None, n=N_VAL2, offset=Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    rows_out = []
    with B.pool(a.workers, base_tr['prereg']['dash_m'] or .25) as ex:
        for f in centres:
            spec = json.loads(f.read_text())
            gen = int(f.stem.split('g')[-1])
            rows = B.run_jobs(ex, [(f'g{gen}', t, c, 'learned', spec) for t in tasks for c in val_cases[t]])
            comp, scores = Dn.composite_of(rows, tasks, norm, weights)
            old = [r for r in rows if r['seed'] in {c['seed'] for t in tasks for c in val_cases[t][:4]}]
            comp_old, _ = Dn.composite_of(old, tasks, norm, weights)
            crashes = sum(not r['flown'] for r in rows)
            tilt = float(np.max([r['tilt_peak_deg'] or 0. for r in rows if r['task'] == 'dash']))
            entry = dict(generation=gen, composite16=comp, composite8=comp_old, crashes=crashes, dash_tilt_peak_max=tilt,
                         ratios={t: scores[t]/norm[t] for t in tasks}, spec=str(f))
            rows_out.append(entry)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items() if k != 'ratios'}), flush=True)
        ok = [e for e in rows_out if e['crashes'] == 0]
        pick = min(ok, key=lambda e: e['composite16']) if ok else None
        pick8 = min(rows_out, key=lambda e: e['composite8'])
        report = dict(seed=a.seed, rule='lowest composite on 16 validation cases among zero-crash centres', n_val_per_task=N_VAL2,
                      centres=rows_out, selected=pick, selected_by_old_rule=pick8)
        print('old rule (8 cases) picks gen', pick8['generation'], '| new rule picks gen', pick and pick['generation'], flush=True)
        if a.blind and pick:
            spec = json.loads(Path(pick['spec']).read_text())
            rep = B.evaluate_spec(ex, out, spec, f'{tag}_resel', base_ev, ('intact',))
            report['blind'] = dict(composite=rep['variants']['intact']['composite'], ratios=rep['variants']['intact']['ratios'],
                                   table={t: {k: v for k, v in rep['variants']['intact']['table'][t].items() if k in ('flown', 'episodes', 'ratio', 'paired_vs_opponent')} for t in tasks+['hover', 'upset'] if t in rep['variants']['intact']['table']})
    (out/f'reselect_{tag}.json').write_text(json.dumps(report, indent=1))
    print('wrote', out/f'reselect_{tag}.json', flush=True)


if __name__ == '__main__':
    main()
