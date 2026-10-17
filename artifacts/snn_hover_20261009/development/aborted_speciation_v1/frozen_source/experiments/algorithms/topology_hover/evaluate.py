"""Frozen held-out comparison, topology interventions and sensor ablations."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import write,summary,evaluate as base_evaluate,selection,ENVS
from .environment import HoverEnv,case,rollout,Zero
from .networks import NumpyPolicy,complexity


def record_criteria(result,env):
    """Retain each existing success criterion, without changing the rollout."""
    samples=np.asarray(env.samples)
    late=samples[samples[:,0]>=max(0,samples[-1,0]-5)]
    result['height_late_peak_to_peak_m']=float(np.ptp(late[:,2]))
    passed=dict(
        full_horizon=result['termination']=='time_limit' and result['elapsed']>=result['case']['seconds']-1e-6,
        clearance=result['minimum_clearance_m']>.03,
        tilt=result['tilt_rms_deg']<10,
        height_rms=result['height_rmse_m']<.15,
        height_late_range=result['height_late_peak_to_peak_m']<.2,
        height_late_slope=abs(result['height_slope_m_s'])<.03)
    result['criterion_pass']={key:bool(value) for key,value in passed.items()}
    assert all(result['criterion_pass'].values())==result['success']
    return result


def evaluate(task):
    result=base_evaluate(task)
    return record_criteria(result,ENVS[(task[2],task[3])])


def policy_index(root):
    entries=[]
    for stage in ['pilot','formal','transfer']:
        for path in sorted((root/stage).glob('*/results.json')):
            meta=json.loads(path.read_text());policy=path.parent/'best_policy.json'
            entries.append(dict(name=path.parent.name,stage=stage,method=meta['method'],seed=meta['seed'],
                mode='direct' if meta['method']=='direct_ppo' else 'residual',sensor='height',
                path=str(policy.relative_to(root)),sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                validation=meta['best_validation'],training_steps=meta['training_steps'],
                validation_steps=meta['validation_steps'],demonstration_steps=meta.get('demonstration_steps',0),
                inference=meta['inference']))
    return entries


def interventions(spec,seed):
    s=copy.deepcopy(spec);memory=[i for i,e in enumerate(s['edges']) if e['enabled'] and e['source']>=0]
    cross=[i for i,e in enumerate(s['edges']) if e['enabled'] and (e['source'],e['target']) in [(-1,0),(-4,0),(-2,1),(-5,1)]]
    for label,indices in [('remove_past_state_paths',memory),('remove_direct_cross_axis',cross)]:
        variant=copy.deepcopy(spec)
        for i in indices:variant['edges'][i]['enabled']=False
        yield label,variant
        enabled=[i for i,e in enumerate(spec['edges']) if e['enabled']]
        for repeat in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([seed,repeat,len(indices),412]))
            variant=copy.deepcopy(spec)
            for i in rng.choice(enabled,len(indices),replace=False):variant['edges'][i]['enabled']=False
            yield f'{label}_random_control_{repeat}',variant


def remove_height_inputs(spec):
    spec=copy.deepcopy(spec);spec['inputs']=9
    if spec['kind']=='graph':spec['edges']=[e for e in spec['edges'] if e['source']>=-9]
    else:spec['layers'][0]['weight']=np.array(spec['layers'][0]['weight'])[:,:9].tolist()
    return spec


def trace_task(task):
    entry,spec,c=task;env=HoverEnv(mode=entry['mode'],sensor_mode=entry['sensor'])
    p=Zero() if spec is None else NumpyPolicy(spec);activations=[];old=p.act
    def act(obs):
        action=old(obs)
        activations.append([env.air.steps*.002,*obs,*action,*getattr(p,'state',[])])
        return action
    p.act=act;r=rollout(env,p,c,True);r['activation_trace']=activations;r['reference_position']=env.air.start.tolist()
    return record_criteria(r,env)


def save_trace(root,label,result):
    trace=np.asarray(result.pop('trace'),dtype=np.float64)
    activations=np.asarray(result.pop('activation_trace'),dtype=np.float64)
    target=root/'traces';target.mkdir(exist_ok=True)
    np.savez_compressed(target/f'{label}.npz',trace=trace,activations=activations)
    write(target/f'{label}.json',dict(episode=result,
        trace_columns=['time','x','y','z','vx','vy','vz','tilt','alpha_command','beta_command','motor_command',
            'roll_measured','pitch_measured','yaw_measured','gyro_x','gyro_y','gyro_z','accel_x','accel_y','accel_z','height_estimate','vz_estimate',
            'residual_alpha','residual_beta','residual_motor','qpos...'],
        activation_columns='time, actor observation, three actions, recurrent node states (if any)',
        physical_state_recording_hz=500,activation_recording_hz=100))


def run(root,workers=16):
    protocol=json.loads((root/'protocol.json').read_text());index=policy_index(root)
    if not index:raise RuntimeError('No completed policies')
    if json.loads((root/'campaign_status.json').read_text())['status']!='training_complete':raise RuntimeError('Training queue must complete before held-out tests')
    from .verify import check_split
    check_split(root,index,protocol)
    manifest_path=root/'heldout_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['policies']!=index:raise RuntimeError('Frozen policies changed after testing started')
    else:write(manifest_path,dict(status='frozen_before_test',policies=index,protocol=protocol['test'],timestamp=time.time()))
    selected={}
    for e in index:
        if e['method'] not in selected or selection(e['validation'])>selection(selected[e['method']]['validation']):selected[e['method']]=e
    write(root/'representatives.json',selected)
    baseline=dict(name='pid',method='pid',seed=0,mode='pid',sensor='height')
    entries=[baseline]+index;tasks=[]
    for e in entries:
        spec=json.loads((root/e['path']).read_text()) if e.get('path') else None
        for condition in protocol['test']['conditions']:
            for i,seed in enumerate(protocol['test']['seeds']):
                c=case(seed,protocol['test']['seconds'],condition,i)
                if condition=='ood':
                    c.update(degrees=8.,delay_steps=5,motor_uncertainty=.03,height_delay=.04,pulse_force=.009,pulse_torque=.000012)
                tasks.append((e,c,spec))
    result_path=root/'heldout_episodes.json';results=json.loads(result_path.read_text()) if result_path.exists() else []
    done={(r['policy'],r['case']['condition'],r['seed']) for r in results}
    tasks=[t for t in tasks if (t[0]['name'],t[1]['condition'],t[1]['seed']) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(evaluate,(s,c,e['mode'],e['sensor'],False)):(e,c) for e,c,s in tasks}
        for f in as_completed(futures):
            e,c=futures[f];r=f.result();results.append(dict(policy=e['name'],method=e['method'],training_seed=e['seed'],**r))
            write(result_path,results);write(root/'evaluation_status.json',dict(stage='heldout',completed=len(results),total=len(entries)*24))
            if len(results)%16==0:print(json.dumps(dict(stage='heldout',completed=len(results))),flush=True)
        # Representative selection used validation only, never these held-out scores.
        long_path=root/'long_episodes.json';long=json.loads(long_path.read_text()) if long_path.exists() else []
        completed={(r['policy'],r['case']['condition'],r['seed']) for r in long};tasks=[]
        for e in [baseline]+list(selected.values()):
            spec=json.loads((root/e['path']).read_text()) if e.get('path') else None
            for condition in protocol['test']['long_conditions']:
                for i,seed in enumerate(protocol['test']['long_seeds']):
                    if (e['name'],condition,seed) not in completed:tasks.append((e,case(seed,60.,condition,i),spec))
        futures={pool.submit(evaluate,(s,c,e['mode'],e['sensor'],False)):e for e,c,s in tasks}
        for f in as_completed(futures):
            e=futures[f];long.append(dict(policy=e['name'],method=e['method'],training_seed=e['seed'],**f.result()));write(long_path,long)
            write(root/'evaluation_status.json',dict(stage='long',completed=len(long),total=(len(selected)+1)*8))
        # Distinguish destructive interventions from retrained graph-transfer controls.
        intervention_path=root/'intervention_episodes.json';rows=json.loads(intervention_path.read_text()) if intervention_path.exists() else []
        completed={(r['policy'],r['intervention'],r['seed']) for r in rows};tasks=[]
        for e in [x for x in index if x['method']=='neat']:
            spec=json.loads((root/e['path']).read_text())
            for label,s in [('unchanged',spec),*interventions(spec,e['seed'])]:
                for i,seed in enumerate(range(340200000,340200004)):
                    if (e['name'],label,seed) not in completed:tasks.append((e,label,s,case(seed,20.,'stress',i)))
        futures={pool.submit(evaluate,(s,c,'residual','height',False)):(e,label) for e,label,s,c in tasks}
        for f in as_completed(futures):
            e,label=futures[f];rows.append(dict(policy=e['name'],method='neat',training_seed=e['seed'],intervention=label,**f.result()));write(intervention_path,rows)
            write(root/'evaluation_status.json',dict(stage='interventions',completed=len(rows),total=sum(e['method']=='neat' for e in index)*36))
        # Height removal is explicitly a sensor ablation of trained 11-D policies, not retraining in 9-D.
        sensor_path=root/'sensor_ablation.json'
        if not sensor_path.exists():
            sensor=[]
            for e in [baseline,selected['neat'],selected['rnn_ppo']]:
                spec=remove_height_inputs(json.loads((root/e['path']).read_text())) if e.get('path') else None
                tasks=[(spec,case(340300000+i,20.,'nominal' if i<2 else 'stress',i),e['mode'],'imu',False) for i in range(4)]
                sensor.extend(dict(policy=e['name'],**r) for r in pool.map(evaluate,tasks))
            write(sensor_path,dict(scope='Remove height sensor and its two policy inputs without retraining. All compared controllers use the same inertial height observer. Not a 9-D training benchmark.',episodes=sensor))
        # Save one predeclared nominal and stressed trajectory for each representative.
        trace_futures={}
        for e in [baseline]+list(selected.values()):
            spec=json.loads((root/e['path']).read_text()) if e.get('path') else None
            for condition in ['nominal','stress']:
                label=f'{e["method"]}_{condition}'
                if (root/'traces'/f'{label}.npz').exists() and (root/'traces'/f'{label}.json').exists():continue
                c=case(protocol['test']['seeds'][0],20.,condition,0)
                trace_futures[pool.submit(trace_task,(e,spec,c))]=(label,e,c)
        total_traces=(len(selected)+1)*2;completed_traces=total_traces-len(trace_futures)
        for f in as_completed(trace_futures):
            label,e,c=trace_futures[f];r=f.result()
            expected=next(x for x in results if x['policy']==e['name'] and x['case']['condition']==c['condition'] and x['seed']==c['seed'])
            for key in ['elapsed','return_sum','tilt_rms_deg','height_rmse_m','minimum_clearance_m']:
                np.testing.assert_allclose(r[key],expected[key],rtol=0,atol=1e-12)
            save_trace(root,label,r);completed_traces+=1
            write(root/'evaluation_status.json',dict(stage='traces',completed=completed_traces,total=total_traces))
    write(root/'evaluation_status.json',dict(stage='completed',heldout_episodes=len(results),long_episodes=len(long),intervention_episodes=len(rows),replay_metric_check_atol=1e-12))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args();run(a.root,a.workers)
