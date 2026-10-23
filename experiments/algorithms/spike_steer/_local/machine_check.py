"""Re-run the classical opponents on the round-3 fresh nod/dash blind set on THIS machine and compare per case with the
CloudPC rows stored in bench_r3/baseline.json (decides whether this machine can replace CloudPC as the selection machine)."""
import json
from pathlib import Path
from .. import bench as B
from .. import descend_r3 as R

R.install()
out = Path('runs/spike_steer/bench_r3')
ev = json.loads((out/'baseline.json').read_text())
levels = B.upset_levels(out)
res = {}
with B.pool(2, ev['prereg']['dash_m'] or .25) as ex:
    for t in ('nod', 'dash'):
        opp = ev['opponent'][t]
        tc = B.cases(t, 'test', levels)
        rows = B.run_jobs(ex, [(opp, t, c, 'classical', ev['arms_full'][t][opp]) for c in tc])
        ref = {r['seed']: r for r in ev['test_rows'][t] if r['controller'] == opp}
        diffs = [(r['seed'], ref[r['seed']]['score'], r['score'], ref[r['seed']]['termination'], r['termination']) for r in rows]
        res[t] = diffs
        for d in diffs:
            print(t, d, flush=True)
Path('logs/machine_check.json').write_text(json.dumps(res, indent=1))
