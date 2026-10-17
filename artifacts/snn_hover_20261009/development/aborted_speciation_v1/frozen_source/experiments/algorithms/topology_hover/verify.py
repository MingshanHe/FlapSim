"""Audit saved evidence, including ancestry and budget accounting, before delivery."""
import argparse
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path
import numpy as np

from .common import write
from .checkpoint import load_evolution_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def snapshot_implementation(root):
    paths=sorted(path for path in Path('experiments/algorithms/topology_hover').iterdir()
                 if path.is_file() and path.suffix in {'.py','.cpp','.md'})
    paths.append(Path('tests/test_topology_hover.py'))
    hashes={path.as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write(root/'final_implementation_hashes.json',hashes)
    with zipfile.ZipFile(root/'final_implementation_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,path.as_posix())
    return hashes


def check_transfer_controls(parent,transferred,randomized):
    def signature(spec):
        return {(edge['source'],edge['target']) for edge in spec['edges'] if edge['enabled']}
    def degrees(spec):
        active=signature(spec)
        return {node['id']:(sum(source<0 and target==node['id'] for source,target in active),
                            sum(source>=0 and target==node['id'] for source,target in active))
                for node in spec['nodes']}
    assert parent['inputs']==transferred['inputs']==randomized['inputs']==11
    assert parent['outputs']==transferred['outputs']==randomized['outputs']
    assert {n['id'] for n in parent['nodes']}=={n['id'] for n in transferred['nodes']}=={n['id'] for n in randomized['nodes']}
    assert signature(parent)==signature(transferred), 'Transferred actor changed the selected NEAT connectivity'
    assert degrees(parent)==degrees(randomized), 'Randomized control violates per-target indegree matching'
    assert signature(parent)!=signature(randomized), 'Randomized control retained the exact selected connectivity'
    return dict(nodes=len(parent['nodes']),enabled_connections=len(signature(parent)),
                selected_connectivity_preserved=True,random_control_input_and_recurrent_indegrees_matched=True,
                shared_connections_with_random_control=len(signature(parent)&signature(randomized)))


def check_split(root,index,protocol):
    """Check actual completed cases and a conservative prefix of active PPO cases."""
    heldout=set(protocol['test']['seeds'])|set(protocol['test']['long_seeds'])
    heldout.update(range(340200000,340200004));heldout.update(range(340300000,340300004))
    records=[]
    for entry in index:
        folder=(root/entry['path']).parent;training=set();validation=set();demonstration=set()
        for path in folder.glob('validation_*.json'):validation.update(row['seed'] for row in read(path))
        if (folder/'neat.ini').exists():
            training.update(row['case']['seed'] for row in read(folder/'history.json'))
        else:
            completed=set()
            with (folder/'training.jsonl').open() as stream:
                for line in stream:completed.update(row['seed'] for row in json.loads(line)['episodes'])
            args=read(folder/'arguments.json');training.update(completed)
            # Include the first uncompleted/reset case per worker as well. This
            # over-counts an unused reset at exact episode boundaries, safely.
            for rank in range(args['workers']):
                for episode_index in range(entry['training_steps']//args['workers']+2):
                    seed=int(np.random.default_rng(np.random.SeedSequence([entry['seed'],rank,episode_index,8163])).integers(100000,900000000))
                    training.add(seed)
                    if seed not in completed:break
            if entry['method']=='direct_ppo':demonstration.update(range(322000000,322000016))
        overlap=heldout&(training|validation|demonstration)
        assert not overlap, f'Test seeds overlap training/selection for {folder}: {sorted(overlap)}'
        records.append(dict(policy=entry['name'],training_case_seed_superset=len(training),validation_case_seeds=len(validation),demonstration_case_seeds=len(demonstration),test_overlap=[]))
    result=dict(status='passed',test_seed_count=len(heldout),policies=records,
                note='PPO check conservatively includes reset/partially consumed cases, not only completed episodes. Same training cases across methods are allowed; no test seed appears in training, validation or demonstrations.')
    write(root/'split_verification.json',result)
    return result


def audit_evolution(folder):
    history=read(folder/'history.json');known={};births={};steps=0;count=0
    generations=[]
    for entry in history:
        g=entry['generation']
        with gzip.open(folder/f'generation_{g:04d}.json.gz','rt') as f:archive=json.load(f)
        assert archive['generation']==g
        ids={row['id'] for row in archive['individuals']}
        assert len(ids)==len(archive['individuals'])
        assert ids=={i for members in archive['species'].values() for i in members}
        for row in archive['individuals']:
            key=row['id'];spec=row['policy'];steps+=row['episode']['steps'];count+=1
            if key in known:
                assert known[key]==spec, f'Elite genome changed without a new ID: {folder} #{key}'
                continue
            assert all(parent in known for parent in row['parents']), f'Missing parent of #{key}'
            assert all(births[parent]<g for parent in row['parents'])
            nodes={n['id'] for n in spec['nodes']};edges={(e['source'],e['target']):e for e in spec['edges']}
            assert len(nodes)==len(spec['nodes']) and len(edges)==len(spec['edges'])
            assert {0,1,2}<=nodes
            assert all(e['target'] in nodes and (e['source'] in nodes or -spec['inputs']<=e['source']<0) for e in spec['edges'])
            event=row['mutation']
            for change in event.get('changed_weights',[]):
                assert edges[tuple(change['edge'])]['weight']==change['after']
            for change in event.get('toggled_edges',[]):
                assert edges[tuple(change['edge'])]['enabled']==change['enabled']
            assert all(tuple(edge) in edges for edge in event.get('added_edges',[]))
            assert all(tuple(edge) not in edges for edge in event.get('removed_edges',[]))
            assert all(node in nodes for node in event.get('added_nodes',[]))
            assert all(node not in nodes for node in event.get('removed_nodes',[]))
            known[key]=spec;births[key]=g
        assert steps==entry['training_steps']
        generations.append(g)
    assert generations==list(range(len(generations)))
    result=read(folder/'results.json');chosen=read(folder/'selection.json')
    assert steps==result['training_steps']
    population,random_state,accounting=load_evolution_checkpoint(folder/'checkpoint.pkl.gz')
    assert accounting['training_steps']==steps
    assert population.generation==len(generations)
    assert len(population.population)==read(folder/'arguments.json')['population']
    assert known[chosen['genome_id']]==read(folder/'best_policy.json')
    visual=read(folder/'visuals/visualization_manifest.json')
    assert visual['archived_individual_evaluations']==count
    for name in ['explorer.html','selected_family_tree.png','lineage_weight_curves.png','lineage_biases.png',
                 'population.csv.gz','weights.csv.gz','biases.csv.gz','topology_evolution.gif']:
        assert (folder/'visuals'/name).stat().st_size>0, name
    return dict(generations=len(generations),individual_evaluations=count,unique_genomes=len(known),training_steps=steps,checkpoint_load_verified=True)


def audit(root):
    status=read(root/'campaign_status.json')
    assert status['status']=='training_complete'
    assert read(root/'evaluation_status.json')['stage']=='completed'
    seeds=status['seeds'];protocol=read(root/'protocol.json');runs={};evolution={}
    for folder in sorted(root.glob('*/*')):
        if not (folder/'results.json').exists():continue
        result=read(folder/'results.json')
        assert result['status']=='completed'
        assert read(folder/'status.json')==result
        runs[result['method'],result['seed']]=folder
        if (folder/'neat.ini').exists():evolution[str(folder.relative_to(root))]=audit_evolution(folder)
    methods=protocol['methods']+['graph_ppo','random_graph_ppo']
    assert set(runs)=={(m,s) for m in methods for s in seeds}
    transfer_checks={str(seed):check_transfer_controls(
        read(runs['neat',seed]/'best_policy.json'),read(runs['graph_ppo',seed]/'best_policy.json'),
        read(runs['random_graph_ppo',seed]/'best_policy.json')) for seed in seeds}
    hashes=read(root/'source_hashes.json')
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in hashes.items()), 'Training source changed'
    manifest=read(root/'heldout_manifest.json')
    assert len(manifest['policies'])==len(runs)
    check_split(root,manifest['policies'],protocol)
    for policy in manifest['policies']:
        assert hashlib.sha256((root/policy['path']).read_bytes()).hexdigest()==policy['sha256']
    heldout=read(root/'heldout_episodes.json');nscenarios=len(protocol['test']['seeds'])*len(protocol['test']['conditions'])
    assert len(heldout)==(len(runs)+1)*nscenarios
    assert len({(x['policy'],x['seed'],x['case']['condition']) for x in heldout})==len(heldout)
    assert len(read(root/'long_episodes.json'))==(len(methods)+1)*8
    assert len(read(root/'intervention_episodes.json'))==len(seeds)*9*4
    assert len(read(root/'sensor_ablation.json')['episodes'])==12
    all_episodes=heldout+read(root/'long_episodes.json')+read(root/'intervention_episodes.json')+read(root/'sensor_ablation.json')['episodes']
    for episode in all_episodes:
        assert len(episode['criterion_pass'])==6
        assert all(episode['criterion_pass'].values())==episode['success']
        assert episode['criterion_pass']['height_late_range']==(episode['height_late_peak_to_peak_m']<.2)
    for condition in ['nominal','stress']:
        flight=read(root/'flight'/f'neat_{condition}_verification.json')
        assert flight['status']=='passed' and flight['maximum_recomputed_state_or_action_error']<2e-6
        assert flight['weights_fixed_during_flight']
        assert (root/'flight'/f'neat_{condition}.html').stat().st_size>10000
        replay=read(root/'videos'/f'verification_{condition}.json')
        assert replay['frame_position_check_atol']==1e-9
        for filename in [f'comparison_{condition}.mp4',f'neat_wings_{condition}_10x.mp4']:
            assert (root/'videos'/filename).stat().st_size>10000
    report=root/'report.md';missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if '://' in target or target.startswith('#'):continue
        if not (report.parent/target.split('#')[0]).is_file():missing.append(target)
    assert not missing, missing
    large=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.stat().st_size>=95_000_000]
    assert not large, f'Files too large for ordinary GitHub delivery: {large}'
    result=dict(status='passed',training_runs=len(runs),training_seeds=seeds,evolution=evolution,
                transfer_controls=transfer_checks,
                heldout_episodes=len(heldout),training_core_unchanged=True,policy_hashes_verified=True,
                report_links_verified=True,media_records_present=True,
                scope='Artifact integrity audit; complements controller tests and visual/video inspection, not evidence of real-world flight.')
    snapshot_implementation(root)
    write(root/'artifact_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
