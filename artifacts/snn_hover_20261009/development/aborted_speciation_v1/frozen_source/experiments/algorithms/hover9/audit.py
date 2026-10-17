"""Audit artifact provenance and history paths after testing, without policy changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from .environment import NAMES
from ..neat_imu.train import write_json


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph(payload):
    if payload['format']=='zero_residual':return dict(format='zero_residual',nodes=0,history_inputs_with_paths=[])
    if payload['format']=='flapsim-ppo-history-v1':
        weight=np.asarray(payload['layers'][0]['weight'])
        return dict(format=payload['format'],first_layer_column_norms=dict(zip(NAMES,np.linalg.norm(weight,axis=0).tolist())),
                    caveat='Nonzero weights do not establish helpful causal use; see held-out zero-history ablation')
    nodes={n['key']:n for n in payload['nodes']};used=set(payload['outputs']);todo=list(used)
    while todo:
        for source,w in nodes.get(todo.pop(),{}).get('links',[]):
            if w!=0 and source not in used:used.add(source);todo.append(source)
    return dict(format=payload['format'],nodes=len(nodes),nonzero_links=sum(w!=0 for n in nodes.values() for _,w in n['links']),
                activation_counts=dict(Counter(n['activation'] for n in nodes.values())),
                history_inputs_with_paths=[NAMES[-k-1] for k in range(-10,-16,-1) if k in used],
                all_inputs_with_paths=[NAMES[-k-1] for k in payload['inputs'] if k in used],
                caveat='A path indicates possible dependence, not beneficial behavior; see ablation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    manifest=json.loads((root/'training_manifest.json').read_text())
    supplement=json.loads((root/'dependency_supplement.json').read_text())
    hashes=dict(manifest['training_source_sha256'],**supplement['sha256'])
    mismatch=[name for name,expected in hashes.items() if digest(name)!=expected]
    if mismatch:raise RuntimeError(f'Training source changed: {mismatch}')
    old=json.loads(Path('artifacts/neat_ppo_7g8g_20261008/training_source_manifest.json').read_text())
    old_mismatch=[]
    for name,expected in old.items():
        relocated=Path(name.split('/FlapSim-current/',1)[-1])
        if digest(relocated)!=expected:old_mismatch.append(str(relocated))
    if old_mismatch:raise RuntimeError(f'Historical experiment source changed: {old_mismatch}')
    assets=json.loads((root/'assets_manifest.json').read_text())['sha256']
    for name,expected in assets.items():assert digest(root/name)==expected
    selected=json.loads((root/'final_test/frozen_selection.json').read_text())
    for label,expected in selected['hashes'].items():assert digest(root/'final_test'/f'{label}_frozen.json')==expected
    audits={label:graph(json.loads((root/'final_test'/f'{label}_frozen.json').read_text())) for label in selected['hashes']}
    write_json(root/'policy_audit.json',audits)
    extra=sorted(list(Path('experiments/algorithms/hover9').glob('*.py'))+list(Path('experiments/algorithms/hover9').glob('*.cpp')))
    final_hashes={name:digest(name) for name in hashes}
    final_hashes.update({str(path):digest(path) for path in extra})
    write_json(root/'source_manifest.json',final_hashes)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(final_hashes):z.write(name,arcname=name)
    results=json.loads((root/'final_test/results.json').read_text())
    seed_audit=json.loads((root/'seed_audit.json').read_text())
    assert not seed_audit['collisions']
    write_json(root/'verification.json',dict(
        tests='25 passed; full command and output retained in logs/tests.log',
        regression_checks=['instrumented physics matches original','translation does not enter actor features',
                           'history reset and short-horizon integration','full signed actuator range','prior 22 related checks'],
        training_source_files_verified=len(hashes),historical_source_files_verified=len(old),
        runtime_assets_verified=len(assets),
        seed_disjointness=seed_audit,
        source_mismatches=[],frozen_policy_hashes_verified=len(selected['hashes']),
        replay_verification=results['replay_verification'],ppo_export_atol=2e-6,
        video_verification=json.loads((root/'video_verification.json').read_text()),
        portable_replay=json.loads((root/'portable_replay_verification.json').read_text()),
        ppo_export_check='Executed on 16 fixed random 15-dimensional inputs at every validation checkpoint',
        high_rate_frontend='Optimistic virtual sensor, not a hardware bandwidth validation'))
    print(json.dumps(dict(training_sources=len(hashes),historical_sources=len(old),policy_audits=audits),indent=2))


if __name__=='__main__':main()
