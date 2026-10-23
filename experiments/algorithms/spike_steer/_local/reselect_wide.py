"""Round-5 design diagnostic (after round 4 was concluded): would a wider validation set have picked better round-4 centres?
Every saved centre of every round-4 seed is re-validated on 64 nod + 64 dash TRAINING cases (validation generator,
offset COLLECT_OFFSET['val']+10_000 = the round-4 selection cases extended from 8 to 64 per task; never a blind set),
then the round-4 rule (zero crashes, tilt < 70 deg, lowest penalised composite) is applied on 128 cases.

    python -W ignore -m experiments.algorithms.spike_steer._local.reselect_wide --workers 30
"""
import argparse
import json
from pathlib import Path
from .. import bench as B
from .. import descend as Dn
from .. import descend_r4 as R4
from .. import descend_r3 as R

N_TASK = 64


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seeds', default='1,2,3,4,5')
    p.add_argument('--out', default='runs/spike_steer/reselect_wide')
    a = p.parse_args()
    R4.install()
    base_tr = json.loads(Path('runs/spike_steer/bench_r4/train/baseline.json').read_text())
    pr = base_tr['prereg']
    R.SOFT['value'] = float(pr['soft_tilt_deg'])
    hard = float(pr['hard_tilt_deg'])
    norm, weights = base_tr['normalisers'], pr['weights']
    tasks = list(base_tr['opponent'])
    cases = {t: B.cases(t, 'train', None, n=N_TASK, offset=Dn.COLLECT_OFFSET['val']+10_000) for t in tasks}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    with B.pool(a.workers, pr['dash_m'] or .25) as ex:
        for s in map(int, a.seeds.split(',')):
            cdir = Path(f'artifacts/spike_steer/plan_v4/r4/finetune_s{s}')
            sel16 = json.loads(Path(f'artifacts/spike_steer/plan_v4/r4/claude_ea/reselect_r4_s{s}.json').read_text())['selected']['generation']
            entries = []
            for f in sorted(cdir.glob('spec_val_g*.json'), key=lambda f: int(f.stem.split('g')[-1])):
                gen = int(f.stem.split('g')[-1])
                rows = B.run_jobs(ex, [(f'g{gen}', t, c, 'learned', json.loads(f.read_text())) for t in tasks for c in cases[t]])
                pen, scores = R.composite_r3(rows, tasks, norm, weights)
                crashes = [r for r in rows if not r['flown']]
                e = dict(generation=gen, penalised=pen, ratios={t: scores[t]/norm[t] for t in tasks},
                         crashes=len(crashes), tilt=sum(r['termination'] == 'tilt' for r in crashes),
                         height=sum(r['termination'] == 'height_escape' for r in crashes),
                         over_hard=sum((r['tilt_peak_deg'] or 0) >= hard for r in rows),
                         nod_misses=sum(r['task'] == 'nod' and r['goal_t'] is None for r in rows), spec=str(f))
                entries.append(e)
                print(f"s{s} g{gen:2d} pen {pen:.3f} nod {e['ratios']['nod']:.3f} dash {e['ratios']['dash']:.3f} crashes {e['crashes']} "
                      f"(tilt {e['tilt']}, height {e['height']}) misses {e['nod_misses']}{'  <- 16-case pick' if gen == sel16 else ''}", flush=True)
            clean = [e for e in entries if e['crashes'] == 0 and e['over_hard'] == 0]
            pick = min(clean or entries, key=lambda e: e['penalised'])
            report[s] = dict(selected_wide=pick, selected_16=sel16, fallback=None if clean else 'no clean centre', centres=entries)
            print(f"s{s}: wide pick g{pick['generation']} ({'clean' if clean else 'FALLBACK'}) vs 16-case pick g{sel16}", flush=True)
            (out/'reselect_wide.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
