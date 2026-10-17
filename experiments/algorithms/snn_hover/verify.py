"""Audit all generations, frozen sources, split membership, paired tests and media."""
import argparse,gzip,hashlib,json,pickle,re,subprocess,zipfile
from pathlib import Path
from .design import read_protocol
import numpy as np
from .common import write
from .evaluate import entries,core_cases,extra_cases,variants
from .environment import case

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evolution(folder):
    history=[r for r in read(folder/'history.json') if r['generation']>=0];known={};births={};steps=0;allocated=0;count=0
    args=read(folder/'arguments.json')
    previous_ids=None
    for entry in history:
        g=entry['generation']
        with gzip.open(folder/f'generation_{g:04d}.json.gz','rt') as f:archive=json.load(f)
        assert archive['generation']==g and len(archive['individuals'])==args['population']
        assert len({r['id'] for r in archive['individuals']})==args['population']
        ids={r['id'] for r in archive['individuals']}
        if previous_ids is not None:assert len(ids-previous_ids)>=32
        previous_ids=ids
        assert len({r['species'] for r in archive['individuals']})<=16
        for r in archive['individuals']:
            k=r['id'];s=r['policy'];steps+=r['episode']['steps'];allocated+=round(r['episode']['case']['seconds']*100);count+=1
            if k in known:assert known[k]==s,'Elite changed without a new genome ID';continue
            assert all(parent in known and births[parent]<g for parent in r['parents'])
            nodes={n['id']:n for n in s['nodes']};edges={(e['source'],e['target']):e for e in s['edges']}
            assert len(nodes)==len(s['nodes'])<=35 and len(edges)==len(s['edges']) and {0,1,2}<=nodes.keys()
            for e in s['edges']:
                assert e['target'] in nodes and (e['source'] in nodes or -11<=e['source']<0)
                assert e['source'] not in [0,1,2] and not(e['source']<0 and e['target'] in [0,1,2])
            for n in nodes.values():
                assert .005-1e-12<=n['tau_m']<=.2+1e-12 and .002-1e-12<=n['tau_s']<=.1+1e-12
            event=r['mutation']
            for c in event.get('changed_weights',[]):assert edges[tuple(c['edge'])]['weight']==c['after']
            for c in event.get('toggled_edges',[]):assert edges[tuple(c['edge'])]['enabled']==c['enabled']
            for c in event.get('changed_nodes',[]):
                n=nodes[c['id']];np.testing.assert_allclose([n['bias'],np.log(n['tau_m']),np.log(n['tau_s'])],c['after'],atol=1e-12,rtol=0)
            assert all(tuple(e) in edges for e in event.get('added_edges',[]))
            assert all(tuple(e) not in edges for e in event.get('removed_edges',[]))
            assert all(n in nodes for n in event.get('added_nodes',[]))
            assert all(n not in nodes for n in event.get('removed_nodes',[]))
            known[k]=s;births[k]=g
        assert entry['training_steps']==steps and entry['allocated_steps']==allocated
    r=read(folder/'results.json');selection=read(folder/'selection.json');s=read(folder/'best_policy.json')
    assert list(range(40))==[h['generation'] for h in history]
    assert steps==r['training_steps'] and allocated==r['allocated_steps']==1024000
    assert known[selection['genome_id']]==s
    # Trusted local checkpoint only; never load arbitrary downloaded pickle files.
    with gzip.open(folder/'checkpoint.pkl.gz','rb') as f:population,rng,accounting=pickle.load(f)
    assert population.generation==40 and len(population.population)==64 and accounting['training_steps']==steps
    manifest=read(folder/'visuals/visualization_manifest.json');assert manifest['archived_individual_evaluations']==count
    for filename in ['explorer.html','selected_network.png','evolution_summary.png','selected_family_tree.png','lineage_weights.png','lineage_biases.png','topology_evolution.gif','population.csv.gz','nodes.csv.gz','weights.csv.gz']:
        assert (folder/'visuals'/filename).stat().st_size>0
    return dict(generations=40,individual_evaluations=count,unique_genomes=len(known),actual_training_steps=steps,checkpoint_load=True)

def audit(root):
    p=read_protocol(root);assert read(root/'campaign_status.json')['status']=='training_complete'
    assert read(root/'evaluation_status.json')['status']=='completed'
    for path,expected in p['source_hashes'].items():
        assert sha(path)==sha(root/'frozen_source'/path)==expected,path
    frozen_evaluation=read(root/'evaluation_code_hashes.json')
    assert frozen_evaluation['status']=='frozen_before_independent_tests'
    for path,expected in frozen_evaluation['files'].items():
        assert sha(path)==expected,f'Evaluation or statistics code changed: {path}'
    test_manifest=read(root/'test_manifest.json')
    assert test_manifest['protocol_sha256']==sha(root/'protocol.json')
    assert test_manifest['evaluation_addendum_sha256']==sha(root/'evaluation_addendum.json')
    assert frozen_evaluation['created_unix']<test_manifest['timestamp']
    evidence=read('docs/research/snn_neat_20261009/evidence/manifest.json')
    for r in evidence['files']:assert sha(r['snapshot'])==r['sha256']
    historical=['experiments/algorithms/topology_hover','artifacts/topology_hover_20261008','experiments/algorithms/pid_airborne','artifacts/pid_airborne_20261008']
    olddiff=subprocess.check_output(['git','diff',p['base_commit'],'--',*historical],text=True);assert not olddiff,'Historical files changed'
    index=entries(root,p);assert index==test_manifest['policies']
    rows=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows)==len({r['key'] for r in rows})==p['evaluation_addendum']['total_evaluation_episodes']
    expected={}
    for e in index:
        for cid,c in core_cases(p):expected[f'{e["name"]}|core|{cid}|original']=c
        if e['stage']=='baseline' or e['stage']=='new' and e['budget']==p['final_budget']:
            for group,cid,c in extra_cases(p):expected[f'{e["name"]}|{group}|{cid}|original']=c
        if e['method']=='snn_neat' and e['budget']==p['final_budget']:
            spec=read(e['path'])
            for variant,_ in variants(spec,e['training_seed']):
                for i,seed in enumerate(p['interventions']['seeds']):
                    expected[f'{e["name"]}|intervention|{seed}|{variant}']=case(seed,20.,'stress',i%8)
    assert len(expected)==p['evaluation_addendum']['total_evaluation_episodes']
    for r in rows:
        assert r['case']==expected.pop(r['key'])
        assert r['steps']>0 and r['elapsed']<=r['case']['seconds']+1e-9
        assert all(np.isfinite(r[k]) for k in ['tilt_rms_deg','height_rmse_m','minimum_clearance_m','return_sum'])
    assert not expected
    heldout={r['seed'] for r in rows};runs=[]
    for method in p['methods']:
        for seed in p['seeds']:
            folder=root/f'training/{method}_seed{seed}';result=read(folder/'results.json');selection=read(folder/'selection.json')
            assert result['status']=='completed' and result['budget']==p['final_budget']
            validation_rows=[r for f in folder.glob('validation_*.json') for r in read(f)]
            assert sum(r['steps'] for r in validation_rows)==result['validation_steps']
            assert not {r['seed'] for r in validation_rows}&heldout
            vals=[read(f) for f in folder.glob('validation_*.json')]
            assert len(vals)==21
            info=dict(method=method,seed=seed,validation_steps=result['validation_steps'])
            if method.endswith('_ppo'):
                logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
                actual={r['seed'] for line in logs for r in line['episodes']}
                assert result['training_steps']==1024000 and logs[-1]['training_steps']==1024000
                assert all(360000000<=s<400000000 for s in actual)
                assert not actual&heldout
                info.update(actual_training_steps=1024000,completed_training_case_seeds=len(actual))
            else:
                info.update(evolution(folder));actual={h['case']['seed'] for h in read(folder/'history.json') if 'case' in h};assert not actual&heldout
            for budget in [256000,512000,1024000]:
                b=folder/f'budget_{budget}';assert (b/'best_policy.json').exists() and (b/'selection.json').exists()
                accounting=read(b/'accounting.json');assert accounting.get('allocated_steps',accounting['training_steps'])==budget
            runs.append(info)
    # Replay checks are independent of rendering and must reproduce test outcomes.
    media=[]
    for path in sorted((root/'traces').glob('*.json')):
        meta=read(path);assert meta['verification']['maximum_recomputed_error']<2e-6
        archive=np.load(path.with_suffix('.npz'));assert len(archive['trace'])>0 and archive['trace'].shape[1]==49
        media.append(dict(label=path.stem,steps=len(archive['activations']),actor_recompute_error=meta['verification']['maximum_recomputed_error']))
    assert len(media)==24
    for path in (root/'videos').glob('verification_*.json'):
        meta=read(path);assert meta['frame_position_check_atol']==1e-9
        assert all(v['peak_to_peak']>10 for v in meta['wing_joint_ranges_deg'].values())
    assert len(list((root/'videos').glob('verification_*.json')))==4
    links=[]
    for path in [root/'report.md',root/'report_final.md',Path('report_final.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (path.parent/target.split('#')[0]).exists(),f'Missing report link: {target}'
            links.append(target)
    sources=sorted(x for x in Path('experiments/algorithms/snn_hover').iterdir() if x.is_file() and x.suffix in ['.py','.md','.txt'])
    sources += [Path('tests/test_snn_hover.py'),Path('tests/test_snn_analysis.py'),Path('tests/test_snn_publication.py')]
    write(root/'final_implementation_hashes.json',{str(f):sha(f) for f in sources})
    with zipfile.ZipFile(root/'final_implementation_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sources:z.write(f,str(f))
    result=dict(status='passed',frozen_training_source_hashes=len(p['source_hashes']),
        frozen_evaluation_source_hashes=len(frozen_evaluation['files']),test_protocol_hashes_verified=True,
        physical_evidence_files=len(evidence['files']),
        historical_files_unchanged=True,training_runs=runs,test_episodes=len(rows),test_keys_unique=True,test_case_design_verified=True,
        heldout_seed_count=len(heldout),test_seed_overlap_with_new_training_or_validation=0,media=media,report_links_checked=len(links),
        final_report_sha256=sha('report_final.md'),artifact_final_report_sha256=sha(root/'report_final.md'),
        limitation='Five training seeds, simulation only, height-assisted11D, no horizontal stationkeeping or energy claim.')
    write(root/'verification.json',result)
    manifest={str(f.relative_to(root)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ['manifest.json','finish_status.json'] and '.tmp' not in f.name}
    write(root/'manifest.json',dict(files=manifest,total_bytes=sum(x['bytes'] for x in manifest.values())))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=audit(a.root);print(json.dumps(dict(status=r['status'],runs=len(r['training_runs']),episodes=r['test_episodes'])))
if __name__=='__main__':main()
