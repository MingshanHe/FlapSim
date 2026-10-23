"""Plan v4 round 3, amendment (CloudPC lost): rebuild the round-3 baseline on THIS machine (claude-ea), the new selection
and blind-test machine. Every classical arm of every task is re-run here on the unchanged blind sets (hover/upset: rounds
1-2 sets; nod/dash: the fresh round-3 set) and the normalisers are recomputed with the pre-registered rule (nod/dash:
opponent per-case median; hover/upset: opponent mean). Everything else (prereg, arms, tuning, start specs) is copied
unchanged from runs/spike_steer/bench_r3 (= artifacts/spike_steer/plan_v4/r3). The CloudPC numbers stay as the secondary
normalisation.

    python -m experiments.algorithms.spike_steer._local.amend5_baseline --workers 2
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .. import bench as B
from .. import descend_r3 as R


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='runs/spike_steer/bench_r3')
    p.add_argument('--output', default='runs/spike_steer/bench_r3_local')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    R.install()
    src, out = Path(a.source), Path(a.output)
    ev = json.loads((src/'baseline.json').read_text())
    tr = json.loads((src/'train'/'baseline.json').read_text())
    for d in (out, out/'train'):
        (d/'upset_v2').mkdir(parents=True, exist_ok=True)
        shutil.copy(src/'upset_v2'/'calibrate.json', d/'upset_v2'/'calibrate.json')
    for f in (src/'train').glob('imitate_spec_s*.json'):
        shutil.copy(f, out/'train'/f.name)
    levels = B.upset_levels(out)
    tasks = list(ev['opponent'])
    test_rows, table, diff = {}, {}, {}
    with B.pool(a.workers, ev['prereg']['dash_m'] or .25) as ex:
        for t in tasks:
            tc = B.cases(t, 'test', levels)
            rows = B.run_jobs(ex, [(name, t, c, 'classical', arg) for name, arg in ev['arms_full'][t].items() for c in tc])
            test_rows[t] = rows
            table[t] = {name: B.summary([r for r in rows if r['controller'] == name]) for name in ev['arms_full'][t]}
            ref = {(r['controller'], r['seed']): r for r in ev['test_rows'][t]}
            d = [abs(r['score']-ref[(r['controller'], r['seed'])]['score'])/max(abs(ref[(r['controller'], r['seed'])]['score']), 1e-9)
                 for r in rows if (r['controller'], r['seed']) in ref]
            same = sum(r['flown'] == ref[(r['controller'], r['seed'])]['flown'] for r in rows if (r['controller'], r['seed']) in ref)
            diff[t] = dict(matched=len(d), same_outcome=same, mean_rel=float(np.mean(d)), max_rel=float(np.max(d)))
            for name, s in table[t].items():
                print(f"{t:6s} {name:13s} flown {s['flown']}/{s['episodes']} score {s['score']:.4f} (CloudPC {ev['table'][t][name]['score']:.4f})", flush=True)
            print(t, json.dumps(diff[t]), flush=True)
    opp = {t: [r for r in test_rows[t] if r['controller'] == ev['opponent'][t]] for t in tasks}
    norm = {t: (float(np.median([r['score'] for r in opp[t]])) if t in ('nod', 'dash') else table[t][ev['opponent'][t]]['score']) for t in tasks}
    notes = dict(ev['normaliser_notes'], machine='claude-ea (replaces CloudPC, lost 2026-10-16)',
                 opponent_mean_all={t: float(np.mean([r['score'] for r in opp[t]])) for t in tasks},
                 opponent_mean_flown={t: float(np.mean([r['score'] for r in opp[t] if r['flown']])) for t in tasks},
                 opponent_flown={t: sum(r['flown'] for r in opp[t]) for t in tasks},
                 opponent_median={t: float(np.median([r['score'] for r in opp[t]])) for t in tasks},
                 cloudpc_normalisers=ev['normalisers'], cloudpc_vs_local=diff)
    ev2 = dict(ev, test_rows=test_rows, table=table, normalisers=norm, normaliser_notes=notes)
    ev2['prereg'] = dict(ev['prereg'], normalisers=norm, normaliser_rule=notes, machine='claude-ea')
    (out/'baseline.json').write_text(json.dumps(ev2))
    tr2 = dict(tr, normalisers={t: norm[t] for t in tr['opponent']})
    (out/'train'/'baseline.json').write_text(json.dumps(tr2))
    print('normalisers local', json.dumps(norm), '| CloudPC', json.dumps(ev['normalisers']), flush=True)


if __name__ == '__main__':
    main()
