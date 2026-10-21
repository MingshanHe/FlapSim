"""Copy SNN results of a plan (selected centre, imitation start, selection, blind, history) into artifacts/ for every seed
that has finished selection.   python docs/dashboard/pack_plan.py v9|v10
HomePC seeds (artifacts/spike_steer/plan_<v>/homepc/) are packed by HomePC itself."""
import json
import shutil
import sys
from pathlib import Path
v = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else '_careful'
R = Path(f'runs/spike_steer/{v}')
A = Path(f'artifacts/spike_steer/plan_{v}/snn')
for rs in sorted(R.glob(f'reselect_s*{tag}.json')):
    S = int(rs.stem.split('_s')[1].split('_')[0])
    sel = json.loads(rs.read_text())['selected']
    if sel is None:
        print('seed', S, 'no admissible centre')
        continue
    d = A/f's{S}'
    d.mkdir(parents=True, exist_ok=True)
    gen = sel['generation']
    run = R/f'finetune_s{S}{tag}'
    shutil.copy(run/f'spec_val_g{gen}.json', d/'selected.json')
    shutil.copy(run/'spec_val_g-1.json', d/'bc.json')
    for f in ('history.jsonl', 'config.json'):
        if (run/f).exists():
            shutil.copy(run/f, d/f)
    shutil.copy(rs, d/'reselect.json')
    for f in R.glob(f'evaluate_*_s{S}{tag}.json'):
        shutil.copy(f, d/f.name.replace(tag, ''))
    (d/'meta.json').write_text(json.dumps(dict(seed=S, generation=gen, objective=sel['objective'], machine='CloudPC-2')))
    print('packed seed', S, 'gen', gen)
for f in [R/'test_opponent.json']+list(R.glob('evaluate_snn9_s*.json')):
    if f.exists():
        shutil.copy(f, A.parent/f.name)
