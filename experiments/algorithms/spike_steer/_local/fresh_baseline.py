"""Fresh nod/dash blind set + classical baseline on claude-ea for a new round (generalises r4_baseline.py, which stays as
archived for round 4). Every classical nod/dash arm is re-run on maneuver_v2 cases starting at --base; hover/upset rows
(unchanged blind sets) are copied from the claude-ea round-3 baseline. Normalisers by the round-3/4 rule (nod/dash:
opponent per-case median; hover/upset: opponent mean). No learned controller is run on the fresh set.

Used blind bases: 2_096_100_000 (rounds 1-2), 2_096_400_000 (round 3), 2_096_500_000 (round 4); training cases < 2_096_310_000.

    python -m experiments.algorithms.spike_steer._local.fresh_baseline --base 2_096_600_000 --round 5 --output runs/spike_steer/bench_r5 --workers 20
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .. import bench as B
from .. import descend_r3 as R
from .. import maneuver_v2 as M

USED = (2_096_100_000, 2_096_400_000, 2_096_500_000, 2_096_600_000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=lambda s: int(s.replace('_', '')), required=True)
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--source', default='runs/spike_steer/bench_r3_local')
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=20)
    a = p.parse_args()
    assert all(abs(a.base-u) >= 1000 for u in USED) and a.base >= 2_096_310_000, a.base
    R.install()
    B.cases = lambda task, split, levels=None, n=None, offset=0: (M.cases(a.base, n or 16, task) if task in ('nod', 'dash') and split == 'test'
                                                                  else R._cases0(task, split, levels, n, offset))
    src, out = Path(a.source), Path(a.output)
    ev = json.loads((src/'baseline.json').read_text())
    tr = json.loads((src/'train'/'baseline.json').read_text())
    for d in (out, out/'train'):
        (d/'upset_v2').mkdir(parents=True, exist_ok=True)
        shutil.copy(src/'upset_v2'/'calibrate.json', d/'upset_v2'/'calibrate.json')
    levels = B.upset_levels(out)
    tasks = list(ev['opponent'])
    test_rows, table = dict(ev['test_rows']), {t: dict(v) for t, v in ev['table'].items()}
    seeds = {}
    with B.pool(a.workers, ev['prereg']['dash_m'] or .25) as ex:
        for t in ('nod', 'dash'):
            tc = B.cases(t, 'test', levels)
            seeds[t] = [c['seed'] for c in tc]
            assert min(seeds[t]) == a.base
            rows = B.run_jobs(ex, [(name, t, c, 'classical', arg) for name, arg in ev['arms_full'][t].items() for c in tc])
            test_rows[t] = rows
            table[t] = {name: B.summary([r for r in rows if r['controller'] == name]) for name in ev['arms_full'][t]}
            for name, s in table[t].items():
                print(f"{t:6s} {name:13s} flown {s['flown']}/{s['episodes']} score {s['score']:.4f}", flush=True)
    opp = {t: [r for r in test_rows[t] if r['controller'] == ev['opponent'][t]] for t in tasks}
    norm = {t: (float(np.median([r['score'] for r in opp[t]])) if t in ('nod', 'dash') else table[t][ev['opponent'][t]]['score']) for t in tasks}
    notes = dict(rule='nod/dash opponent per-case median on the fresh blind set (learned side: mean incl. crashes); hover/upset opponent mean (unchanged sets, claude-ea rows)',
                 machine='claude-ea', test_base_nod_dash=a.base,
                 opponent_mean_all={t: float(np.mean([r['score'] for r in opp[t]])) for t in tasks},
                 opponent_flown={t: sum(r['flown'] for r in opp[t]) for t in tasks},
                 opponent_median={t: float(np.median([r['score'] for r in opp[t]])) for t in tasks},
                 opponent_misses={t: sum(r.get('goal_t') is None for r in opp[t]) for t in ('nod', 'dash')})
    ev2 = dict(ev, test_rows=test_rows, table=table, normalisers=norm, normaliser_notes=notes)
    ev2['prereg'] = dict(ev['prereg'], round=a.round, normalisers=norm, normaliser_rule=notes, machine='claude-ea',
                         test_sets=dict(ev['prereg'].get('test_sets', {}), nod=seeds['nod'], dash=seeds['dash']))
    (out/'baseline.json').write_text(json.dumps(ev2))
    (out/'train'/'baseline.json').write_text(json.dumps(dict(tr, normalisers={t: norm[t] for t in tr['opponent']})))
    print(f'round-{a.round} normalisers', json.dumps(norm), flush=True)
    print(json.dumps(notes, indent=1), flush=True)


if __name__ == '__main__':
    main()
