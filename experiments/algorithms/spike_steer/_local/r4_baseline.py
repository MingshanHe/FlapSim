"""Plan v4 round 4 baseline on claude-ea: fresh nod/dash blind set (maneuver_v2 base TEST_BASE_R4, never used before:
rounds 1-2 used 2_096_100_000, round 3 2_096_400_000; ES/validation/diagnostic train cases are < 2_096_310_000).
Every classical nod/dash arm is re-run on the fresh set; hover/upset rows (unchanged blind sets) are copied from the
claude-ea round-3 baseline (same machine, same code). Normalisers by the round-3 rule (nod/dash: opponent per-case
median; hover/upset: opponent mean). No learned controller is run on the fresh set here.

    python -m experiments.algorithms.spike_steer._local.r4_baseline --workers 30
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .. import bench as B
from .. import descend_r3 as R
from .. import maneuver_v2 as M

TEST_BASE_R4 = 2_096_500_000


def cases_r4(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_R4, n or 16, task)
    return R._cases0(task, split, levels, n, offset)


def install():
    R.install()
    B.cases = cases_r4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='runs/spike_steer/bench_r3_local')
    p.add_argument('--output', default='runs/spike_steer/bench_r4')
    p.add_argument('--workers', type=int, default=30)
    a = p.parse_args()
    install()
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
            assert min(seeds[t]) == TEST_BASE_R4, seeds[t][:3]
            rows = B.run_jobs(ex, [(name, t, c, 'classical', arg) for name, arg in ev['arms_full'][t].items() for c in tc])
            test_rows[t] = rows
            table[t] = {name: B.summary([r for r in rows if r['controller'] == name]) for name in ev['arms_full'][t]}
            for name, s in table[t].items():
                print(f"{t:6s} {name:13s} flown {s['flown']}/{s['episodes']} score {s['score']:.4f}", flush=True)
    opp = {t: [r for r in test_rows[t] if r['controller'] == ev['opponent'][t]] for t in tasks}
    norm = {t: (float(np.median([r['score'] for r in opp[t]])) if t in ('nod', 'dash') else table[t][ev['opponent'][t]]['score']) for t in tasks}
    notes = dict(rule='round-3 rule: nod/dash opponent per-case median on the fresh blind set (learned side: mean incl. crashes); hover/upset opponent mean (unchanged sets, claude-ea rows)',
                 machine='claude-ea', test_base_nod_dash=TEST_BASE_R4,
                 opponent_mean_all={t: float(np.mean([r['score'] for r in opp[t]])) for t in tasks},
                 opponent_flown={t: sum(r['flown'] for r in opp[t]) for t in tasks},
                 opponent_median={t: float(np.median([r['score'] for r in opp[t]])) for t in tasks},
                 opponent_misses={t: sum(r.get('goal_t') is None for r in opp[t]) for t in ('nod', 'dash')},
                 round3_normalisers_claude_ea=ev['normalisers'])
    ev2 = dict(ev, test_rows=test_rows, table=table, normalisers=norm, normaliser_notes=notes)
    ev2['prereg'] = dict(ev['prereg'], round=4, normalisers=norm, normaliser_rule=notes, machine='claude-ea',
                         test_sets=dict(ev['prereg'].get('test_sets', {}), nod=seeds['nod'], dash=seeds['dash']))
    (out/'baseline.json').write_text(json.dumps(ev2))
    (out/'train').mkdir(parents=True, exist_ok=True)
    (out/'train'/'baseline.json').write_text(json.dumps(dict(tr, normalisers={t: norm[t] for t in tr['opponent']})))
    print('round-4 normalisers', json.dumps(norm), '| round-3 claude-ea', json.dumps(ev['normalisers']), flush=True)
    print(json.dumps(notes, indent=1), flush=True)


if __name__ == '__main__':
    main()
