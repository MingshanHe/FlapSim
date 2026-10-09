"""Frozen, resumable final/bridge tests and mechanistic interventions."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import write,evaluate,selection
from .environment import case
from .design import read_protocol


def entries(root,protocol):
    index=[]
    for seed in protocol['seeds']:
        for method in protocol['methods']:
            run=root/'training'/f'{method}_seed{seed}';result=json.loads((run/'results.json').read_text())
            for budget in [protocol['bridge_budget'],protocol['final_budget']]:
                folder=run/f'budget_{budget}';path=folder/'best_policy.json';s=json.loads((folder/'selection.json').read_text())
                accounting=json.loads((folder/'accounting.json').read_text())
                index.append(dict(name=f'{method}_seed{seed}_b{budget}',method=method,training_seed=seed,budget=budget,stage='new',mode='residual',
                    path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validation=s['validation'],selection=s,
                    actual_training_steps=accounting['training_steps'],validation_steps=accounting['validation_steps']))
    old=Path('artifacts/topology_hover_20261008/pilot')
    for method in ['neat','fixed_evo','rnn_ppo','ff_ppo']:
        for seed in range(1201,1206):
            folder=old/f'{method}_seed{seed}';path=folder/'best_policy.json';s=json.loads((folder/'selection.json').read_text())
            r=json.loads((folder/'results.json').read_text())
            index.append(dict(name=f'old_{method}_seed{seed}',method='old_'+method,training_seed=seed,budget=262144,stage='historical',mode='residual',
                path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validation=s['validation'],selection=s,
                actual_training_steps=r['training_steps'],validation_steps=r['validation_steps']))
    index.append(dict(name='pid',method='pid',training_seed=0,budget=0,stage='baseline',mode='pid',path=None))
    return index


def core_cases(protocol):
    for condition in protocol['test']['conditions']:
        for i,seed in enumerate(protocol['test']['seeds']):
            c=case(seed,protocol['test']['seconds'],condition,i%8)
            if condition=='ood':c.update(degrees=8.,delay_steps=5,motor_uncertainty=.03,height_delay=.04,pulse_force=.009,pulse_torque=.000012)
            yield f'{condition}_{seed}',c


def extra_cases(protocol):
    p=protocol['long_test']
    for condition in p['conditions']:
        for i,seed in enumerate(p['seeds']):yield 'long',f'{condition}_{seed}',case(seed,p['seconds'],condition,i%8)
    p=protocol['recovery']
    for angle in p['angles']:
        for i,seed in enumerate(p['angle_seeds']):
            c=case(seed,p['seconds'],'nominal',i%8);c.update(tilt_deg=angle,direction_deg=p['directions'][i%8])
            yield 'angle',f'{angle}_{seed}',c
    for strength in p['pulse_torques']:
        for i,seed in enumerate(p['pulse_seeds']):
            c=case(seed,p['seconds'],'nominal',i%8);c.update(impulse_torque=strength,impulse_start=p['pulse_start'],impulse_duration=p['pulse_duration'],direction_deg=p['directions'][i%8])
            yield 'pulse',f'{strength}_{seed}',c
    p=protocol['factors']
    for name in p['names']:
        for i,seed in enumerate(p['seeds']):
            c=case(seed,p['seconds'],'stress',i%8)
            c.update({'stress_baseline':{},'delay_10ms':dict(delay_steps=5),'motor_3pct':dict(motor_uncertainty=.03),
                'lower9g':dict(lower_mass_scale=1.125),'pulse_1p5x':dict(pulse_force=.009,pulse_torque=.000012)}[name])
            c['factor']=name;yield 'factor',f'{name}_{seed}',c


def variants(spec,seed):
    yield 'original',copy.deepcopy(spec)
    s=copy.deepcopy(spec)
    for n in s['nodes']:n.update(tau_m=.02,tau_s=.01)
    yield 'tau_initial',s
    s=copy.deepcopy(spec);rng=np.random.default_rng(seed+729)
    for output in [False,True]:
        nodes=[n for n in s['nodes'] if (n['id'] in s['outputs'])==output]
        values=[(n['tau_m'],n['tau_s']) for n in nodes]
        for n,i in zip(nodes,rng.permutation(len(nodes))):n['tau_m'],n['tau_s']=values[i]
    yield 'tau_shuffle',s
    s=copy.deepcopy(spec);s['reset_each_tick']=True;yield 'reset_each_tick',s
    enabled=[i for i,e in enumerate(spec['edges']) if e['enabled']]
    recurrent=[i for i in enabled if spec['edges'][i]['source']>=0 and spec['edges'][i]['target'] not in spec['outputs']]
    s=copy.deepcopy(spec)
    for i in recurrent:s['edges'][i]['enabled']=False
    yield 'remove_recurrent',s
    for repeat in range(3):
        s=copy.deepcopy(spec);rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,771]))
        for i in rng.choice(enabled,len(recurrent),replace=False):s['edges'][int(i)]['enabled']=False
        yield f'random_remove_{repeat}',s


def intervention_task(task):
    # Observation-only test instrumentation; frozen training code stays unchanged.
    spec,c,mode,sensor,trace=task
    from .common import ENVS,record_criteria
    from .environment import HoverEnv,rollout,Zero
    from .networks import policy
    key=(mode,sensor)
    if key not in ENVS:ENVS[key]=HoverEnv(mode=mode,sensor_mode=sensor)
    env=ENVS[key];p=policy(spec) if spec else Zero();old=p.act
    reset=bool(spec and spec.get('reset_each_tick'));previous=None;action_diff=[];spikes=None;neural_ticks=0
    def act(x):
        nonlocal previous,spikes,neural_ticks
        if reset:p.reset()
        before=p.spike_count.copy() if hasattr(p,'spike_count') else None
        y=old(x)
        if previous is not None:action_diff.append(y-previous)
        previous=np.asarray(y).copy()
        if before is not None:
            delta=p.spike_count-before;spikes=delta.copy() if spikes is None else spikes+delta;neural_ticks+=10
        return y
    p.act=act
    commands=[];advance=env.air.advance
    def instrumented(command,params):
        # Plant gain uncertainty is applied after controller clipping. Measure
        # saturation on the requested command, not the gain-scaled plant input.
        commands.append(env.command.copy());return advance(command,params)
    env.air.advance=instrumented
    try:r=record_criteria(rollout(env,p,c,trace),env)
    finally:env.air.advance=advance
    delta=np.asarray(action_diff) if action_diff else np.zeros((1,3));physical=np.diff(commands,axis=0)
    if not len(physical):physical=np.zeros((1,3))
    requested=np.asarray(commands)
    r['saturation_fraction']=np.mean(np.column_stack([np.abs(requested[:,:2])>=.3-1e-9,(requested[:,2]<=3115.224079+1e-6)|(requested[:,2]>=3807.496097-1e-6)]),axis=0).tolist()
    r['saturation_measurement']='Controller command after clipping, before plant motor-gain uncertainty; observation-only test correction of training instrumentation.'
    r.update(actor_step_delta_rms=np.sqrt(np.mean(delta**2,axis=0)).tolist(),command_step_delta_rms=np.sqrt(np.mean(physical**2,axis=0)).tolist(),
        command_step_delta_units=['rad/2ms','rad/2ms','rad_per_second/2ms'],actor_step_delta_interval_s=.01)
    if spikes is not None:
        r.update(spikes=int(spikes.sum()),hidden_spike_hz=(spikes[p.hidden]/(neural_ticks*.001)).tolist(),
            silent_hidden_fraction=float(np.mean(spikes[p.hidden]==0)) if len(p.hidden) else 1.)
    return r


def run(root,workers=16):
    protocol=read_protocol(root)
    if json.loads((root/'campaign_status.json').read_text())['status']!='training_complete':raise RuntimeError('Finish all training before independent tests')
    for path,expected in protocol['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Frozen source changed: {path}')
    index=entries(root,protocol);manifest=root/'test_manifest.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text());assert prior['policies']==index,'Policies changed after tests began'
        assert prior['evaluation_addendum_sha256']==hashlib.sha256((root/'evaluation_addendum.json').read_bytes()).hexdigest(),'Test addendum changed'
    else:write(manifest,dict(status='frozen_before_tests',policies=index,protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        evaluation_addendum_sha256=hashlib.sha256((root/'evaluation_addendum.json').read_bytes()).hexdigest(),timestamp=time.time()))
    selected={}
    for e in index:
        if e['stage']=='new' and e['budget']==protocol['final_budget']:
            if e['method'] not in selected or selection(e['validation'])>selection(selected[e['method']]['validation']):selected[e['method']]=e
    write(root/'representatives.json',selected)
    tasks=[]
    for e in index:
        spec=json.loads(Path(e['path']).read_text()) if e['path'] else None
        for cid,c in core_cases(protocol):tasks.append((e,'core',cid,'original',spec,c))
        if e['stage']=='baseline' or e['stage']=='new' and e['budget']==protocol['final_budget']:
            for group,cid,c in extra_cases(protocol):tasks.append((e,group,cid,'original',spec,c))
        if e['method']=='snn_neat' and e['budget']==protocol['final_budget']:
            for name,s in variants(spec,e['training_seed']):
                for i,seed in enumerate(protocol['interventions']['seeds']):
                    c=case(seed,20.,'stress',i%8);tasks.append((e,'intervention',str(seed),name,s,c))
    def key(e,g,c,v):return f'{e["name"]}|{g}|{c}|{v}'
    output=root/'evaluation';output.mkdir(exist_ok=True);done=set();rows=[]
    journal=output/'episodes.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():r=json.loads(line);rows.append(r);done.add(r['key'])
    total=len(tasks);tasks=[t for t in tasks if key(t[0],t[1],t[2],t[3]) not in done]
    with journal.open('a',buffering=1) as log,ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(intervention_task,(s,c,e['mode'],'height',False)):(e,g,cid,v) for e,g,cid,v,s,c in tasks}
        last_print=0.
        for future in as_completed(futures):
            e,g,cid,v=futures[future];result=future.result()
            r=dict(key=key(e,g,cid,v),policy=e['name'],method=e['method'],training_seed=e['training_seed'],budget=e['budget'],stage=e['stage'],group=g,variant=v,**result)
            log.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r)
            if time.monotonic()-last_print>30:
                progress=dict(status='evaluating',completed=len(rows),total=total,last_group=g,last_method=e['method'])
                write(root/'evaluation_status.json',progress);print(json.dumps(progress),flush=True);last_print=time.monotonic()
    for group in sorted({r['group'] for r in rows}):write(output/f'{group}.json',[r for r in rows if r['group']==group])
    write(root/'evaluation_status.json',dict(status='completed',completed=len(rows),total=total))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args();run(a.root,a.workers)
if __name__=='__main__':main()
