"""Copy plan v7 full SNN results (selected centre, imitation start, selection, blind, history) into artifacts/ for every
seed that has finished selection.   python docs/dashboard/pack_v7full.py"""
import json
import shutil
from pathlib import Path
R = Path('runs/spike_steer/v7full')
A = Path('artifacts/spike_steer/plan_v7/full/snn')
for rs in sorted(R.glob('reselect_s*_careful.json')):
    S = int(rs.stem.split('_s')[1].split('_')[0])
    d = A/f's{S}'
    d.mkdir(parents=True, exist_ok=True)
    sel = json.loads(rs.read_text())['selected']
    gen = sel['generation']
    run = R/f'finetune_s{S}_careful'
    shutil.copy(run/f'spec_val_g{gen}.json', d/'selected.json')
    shutil.copy(run/'spec_val_g-1.json', d/'bc.json')
    shutil.copy(run/'history.jsonl', d/'history.jsonl')
    shutil.copy(run/'config.json', d/'config.json')
    shutil.copy(rs, d/'reselect.json')
    for f in R.glob(f'evaluate_*_s{S}_careful.json'):
        shutil.copy(f, d/f.name.replace('_careful', ''))
    (d/'meta.json').write_text(json.dumps(dict(seed=S, generation=gen, objective=sel['objective'])))
    print('packed seed', S, 'gen', gen)
if (R/'test_opponent.json').exists():
    shutil.copy(R/'test_opponent.json', A.parent/'test_opponent_v7full_cloudpc2.json')
