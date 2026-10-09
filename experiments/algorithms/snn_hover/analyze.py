"""All-seed statistics, explicit hypothesis limits, and scientific figures."""
import argparse,itertools,json
from pathlib import Path
from .design import read_protocol
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write,summary,benchmark
from .networks import complexity

def interval(x,level=.95):
    x=np.asarray(x,dtype=float)
    if len(x)<2:return [float(x.mean())]*2
    rng=np.random.default_rng(4817);means=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    return np.quantile(means,[(1-level)/2,1-(1-level)/2]).tolist()

def signflip(x):
    x=np.asarray(x,dtype=float);observed=abs(x.mean())
    values=[abs(np.mean(x*np.array(s))) for s in itertools.product([-1,1],repeat=len(x))]
    return float(np.mean(np.array(values)>=observed-1e-12))

def holm(values):
    order=np.argsort(values);result=np.zeros(len(values));previous=0.
    for rank,i in enumerate(order):previous=max(previous,min(1.,values[i]*(len(values)-rank)));result[i]=previous
    return result.tolist()

def aggregate(rows):
    r=summary(rows);seeds=sorted({x['training_seed'] for x in rows})
    r.update(success_count=sum(x['success'] for x in rows),recovery_count=sum(x['recovery_success'] for x in rows),
        full_count=sum(x['criterion_pass']['full_horizon'] for x in rows),seed_count=len(seeds),
        maximum_height_loss_m=float(np.mean([x['maximum_height_loss_m'] for x in rows])),
        xy_end_mean_m=float(np.mean([x['xy_offset_m'] for x in rows])),
        saturation_fraction=np.mean([x['saturation_fraction'] for x in rows],axis=0).tolist(),
        range_angle_unavailable_fraction=float(np.mean([x['height_sensor_out_of_angle_fraction'] for x in rows])),
        maximum_tilt_mean_deg=float(np.mean([x['maximum_tilt_deg'] for x in rows])),
        actor_step_delta_rms=np.mean([x['actor_step_delta_rms'] for x in rows],axis=0).tolist(),
        command_step_delta_rms=np.mean([x['command_step_delta_rms'] for x in rows],axis=0).tolist(),
        recovered_times_s=[x['recovery_time_s'] for x in rows if x['recovery_time_s'] is not None],
        terminations={k:sum(x['termination']==k for x in rows) for k in sorted({x['termination'] for x in rows})},
        failed_criteria={k:sum(not x['criterion_pass'][k] for x in rows) for k in rows[0]['criterion_pass']},
        per_seed=[dict(seed=s,success_rate=float(np.mean([x['success'] for x in rows if x['training_seed']==s])),
            recovery_rate=float(np.mean([x['recovery_success'] for x in rows if x['training_seed']==s])),
            tilt_rms_deg=float(np.mean([x['tilt_rms_deg'] for x in rows if x['training_seed']==s]))) for s in seeds])
    r['success_ci']=interval([x['success_rate'] for x in r['per_seed']]) if seeds!=[0] else None
    return r

def analyze(root):
    status=json.loads((root/'evaluation_status.json').read_text());assert status['status']=='completed'
    protocol=read_protocol(root);index=json.loads((root/'test_manifest.json').read_text())['policies']
    rows=[json.loads(line) for line in (root/'evaluation/episodes.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows)==len({r['key'] for r in rows})==status['total']
    groups={}
    for r in rows:
        condition=r['case']['condition']
        if r['group']=='angle':condition=str(r['case']['tilt_deg'])
        if r['group']=='pulse':condition=str(r['case']['impulse_torque'])
        if r['group']=='factor':condition=r['case']['factor']
        if r['group']=='intervention':condition=r['variant']
        key=f'{r["method"]}/{r["budget"]}/{r["group"]}/{condition}'
        groups.setdefault(key,[]).append(r)
    data={k:aggregate(v) for k,v in groups.items()}
    main=protocol['methods'];cap=protocol['final_budget'];contrasts=[]
    for other in ['snn_fixed_tau','snn_fixed_topology','leaky_ann_neat','ann_neat','rnn_ppo','ff_ppo']:
        a=data[f'snn_neat/{cap}/core/stress'];b=data[f'{other}/{cap}/core/stress']
        A={v['seed']:v for v in a['per_seed']};B={v['seed']:v for v in b['per_seed']};seeds=sorted(A.keys()&B.keys())
        ds=[A[s]['success_rate']-B[s]['success_rate'] for s in seeds];dt=[A[s]['tilt_rms_deg']-B[s]['tilt_rms_deg'] for s in seeds]
        primary=other in ('snn_fixed_tau','snn_fixed_topology');level=.975 if primary else .95
        contrasts.append(dict(a='snn_neat',b=other,primary=primary,seeds=seeds,success_differences=ds,
            success_difference=float(np.mean(ds)),success_interval=interval(ds,level),interval_level=level,
            tilt_difference_deg=float(np.mean(dt)),tilt_interval_deg=interval(dt,level),signflip_p=signflip(ds)))
    adjusted=holm([r['signflip_p'] for r in contrasts[:2]])
    for r,p in zip(contrasts[:2],adjusted):r['holm_adjusted_p']=p
    pairwise=[]
    for a,b in itertools.combinations(main,2):
        A={x['seed']:x['success_rate'] for x in data[f'{a}/{cap}/core/stress']['per_seed']}
        B={x['seed']:x['success_rate'] for x in data[f'{b}/{cap}/core/stress']['per_seed']}
        d=[A[s]-B[s] for s in sorted(A)];pairwise.append(dict(a=a,b=b,success_difference=float(np.mean(d)),success_interval=interval(d),scope='Exploratory; unadjusted95% interval; all21 pairs disclosed'))
    factor_effects=[]
    for m in ['pid']+main:
        b=0 if m=='pid' else cap
        baseline={x['seed']:x['success_rate'] for x in data[f'{m}/{b}/factor/stress_baseline']['per_seed']}
        for name in protocol['factors']['names'][1:]:
            altered={x['seed']:x['success_rate'] for x in data[f'{m}/{b}/factor/{name}']['per_seed']}
            delta=[altered[s]-baseline[s] for s in sorted(baseline)]
            factor_effects.append(dict(method=m,factor=name,success_difference=float(np.mean(delta)),success_interval=interval(delta) if m!='pid' else None,
                per_seed_difference=delta,scope='Same policy, same8 scenario seeds/noise/phases; change listed factor only; exploratory95% seed interval'))
    costs=[]
    for e in index:
        if e['stage']!='new' or e['budget']!=cap:continue
        spec=json.loads(Path(e['path']).read_text());result=json.loads((Path(e['path']).parents[1]/'results.json').read_text())
        cost=dict(method=e['method'],seed=e['training_seed'],**benchmark(spec),actual_training_steps=e['actual_training_steps'],
            validation_steps=e['validation_steps'],validation_plateau=result['validation_plateau'],wall_seconds=result['wall_seconds'],
            selected_generation=e['selection'].get('generation'),selected_training_steps=e['selection']['training_steps'])
        if 'nodes' in spec:
            cost['hidden_recurrent_connections']=sum(e['enabled'] and e['source']>=0 and e['target'] not in spec['outputs'] for e in spec['edges'])
        costs.append(cost)
    temporal=[]
    for e in index:
        if e['stage']!='new' or e['budget']!=cap or e['method'] not in ['snn_neat','snn_fixed_tau','snn_fixed_topology','leaky_ann_neat']:continue
        spec=json.loads(Path(e['path']).read_text());selected_rows=[r for r in rows if r['policy']==e['name'] and r['group']=='core' and r['case']['condition']=='stress']
        constants=[dict(node=n['id'],readout=n['id'] in spec['outputs'],tau_m_ms=1000*n['tau_m'],tau_s_ms=1000*n['tau_s']) for n in spec['nodes']]
        boundary=sum(n['tau_m']<=.005*1.01 or n['tau_m']>=.2/1.01 or n['tau_s']<=.002*1.01 or n['tau_s']>=.1/1.01 for n in spec['nodes'])
        temporal.append(dict(method=e['method'],seed=e['training_seed'],nodes=constants,near_boundary_nodes=boundary,
            mean_hidden_spike_hz=float(np.mean([np.mean(r['hidden_spike_hz']) if r['hidden_spike_hz'] else 0 for r in selected_rows])),
            mean_silent_hidden_fraction=float(np.mean([r['silent_hidden_fraction'] for r in selected_rows])),
            note='Leaky ANN has no spikes by construction; boundary tolerance1% multiplicative.'))
    findings=[]
    for r in contrasts[:2]:
        findings.append(dict(hypothesis='H1_time_constants' if r['b']=='snn_fixed_tau' else 'H2_topology',
            exploratory_large_positive_effect=r['success_interval'][0]>0 and r['success_difference']>=.1,
            primary_exact_test_rejects=r['holm_adjusted_p']<.05,
            caveat='Only five independent training seeds. Bootstrap interval and discrete exact test need not agree; neither is a stability proof.'))
    maincost=[r for r in costs if r['method']=='snn_neat'];fixedcost=[r for r in costs if r['method']=='snn_fixed_topology']
    compression=1-np.mean([r['effective_connections'] for r in maincost])/np.mean([r['effective_connections'] for r in fixedcost])
    topology=contrasts[1]
    findings.append(dict(hypothesis='H2_parsimony',effective_connection_reduction=float(compression),
        exploratory_noninferiority=topology['success_interval'][0]>=-.05 and compression>=.2))
    engineering={}
    representatives=json.loads((root/'representatives.json').read_text())
    for method in main:
        nominal=data[f'{method}/{cap}/core/nominal'];stress=data[f'{method}/{cap}/core/stress']
        N={x['seed']:x['success_rate'] for x in nominal['per_seed']};S={x['seed']:x['success_rate'] for x in stress['per_seed']}
        good=sum(N[s]>=.95 and S[s]>=.8 for s in N);rep=representatives[method]['training_seed']
        lr=[r for r in rows if r['method']==method and r['group']=='long' and r['training_seed']==rep]
        rep_pass={c:sum(r['success'] for r in lr if r['case']['condition']==c) for c in ['nominal','stress']}
        engineering[method]=dict(qualifying_training_seeds=good,total_training_seeds=5,representative_seed=rep,
            representative_long_success=rep_pass,meets_proposed_gate=good>=4 and min(rep_pass.values())>=7)
    out=dict(groups=data,contrasts=contrasts,all_pairwise_exploratory=pairwise,factor_effects=factor_effects,costs=costs,temporal=temporal,hypothesis_checks=findings,engineering=engineering,
        total_evaluation_episodes=len(rows),actual_evaluation_steps=sum(r['steps'] for r in rows),
        statistical_scope='Training seeds are the resampling units, conditional on this fixed common test set. Two primary success-rate contrasts use97.5% bootstrap intervals and Holm-adjusted exact sign-flip tests. Secondary contrasts exploratory; no generalization-distribution or stability guarantee.')
    write(root/'analysis.json',out)
    figures(root,out,protocol)
    return out

def figures(root,data,protocol):
    target=root/'figures';target.mkdir(exist_ok=True);methods=protocol['methods'];cap=protocol['final_budget'];colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i,m in enumerate(['pid']+methods):
        budget=0 if m=='pid' else cap
        y=[data['groups'][f'{m}/{budget}/core/{c}']['success_rate']*100 for c in ['nominal','stress','ood']]
        axes[0].plot(range(3),y,'o-',color=colors(i),label=m)
        if m!='pid':
            cs=[r for r in data['costs'] if r['method']==m];params=np.mean([r['parameters'] for r in cs])
            axes[1].scatter(params,y[1],color=colors(i),label=m,s=65)
    axes[0].set(xticks=range(3),xticklabels=['Nominal','Stress','Combined OOD'],ylabel='Strict success (%)',ylim=(-3,103));axes[0].legend(fontsize=7)
    axes[1].set(xscale='log',xlabel='Actor parameters (including evolved time constants)',ylabel='Stress strict success (%)',ylim=(-3,103));axes[1].legend(fontsize=7)
    fig.savefig(target/'performance_complexity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i,m in enumerate(['pid']+methods):
        b=0 if m=='pid' else cap;angles=protocol['recovery']['angles']
        groups=[data['groups'][f'{m}/{b}/angle/{a}'] for a in angles]
        axes[0].plot(angles,[100*g['recovery_count']/g['episodes'] for g in groups],'o-',label=m,color=colors(i))
        axes[1].plot(angles,[g['maximum_height_loss_m'] for g in groups],'o-',label=m,color=colors(i))
    axes[0].set(xlabel='Initial true tilt (deg)',ylabel='Recovery success (%)',ylim=(-3,103));axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Initial true tilt (deg)',ylabel='Mean maximum height loss (m)');axes[1].legend(fontsize=7)
    fig.savefig(target/'large_angle_recovery.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for i,m in enumerate(methods):
        for seed in protocol['seeds']:
            h=json.loads((root/f'training/{m}_seed{seed}/history.json').read_text())
            h=[r for r in h if 'best_validation' in r]
            xs=[r.get('allocated_steps',r['training_steps']) for r in h]
            axes[0].plot(xs,[r['best_validation']['success_rate'] for r in h],color=colors(i),alpha=.35,label=m if seed==protocol['seeds'][0] else None)
            axes[1].plot(xs,[r['best_validation']['tilt_rms_deg'] for r in h],color=colors(i),alpha=.35)
    axes[0].set(xlabel='Allocated training transitions',ylabel='Best validation strict success');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Allocated training transitions',ylabel='Selected validation tilt RMS (deg)')
    fig.suptitle('Validation selection trajectories; not independent test performance')
    fig.savefig(target/'validation_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,r in zip(axes,data['contrasts'][:2]):
        a=data['groups'][f'snn_neat/{cap}/core/stress']['per_seed'];b=data['groups'][f'{r["b"]}/{cap}/core/stress']['per_seed']
        for x,y in zip(a,b):ax.plot([0,1],[100*x['success_rate'],100*y['success_rate']],'o-',label=str(x['seed']))
        ax.set(xticks=[0,1],xticklabels=['Joint SNN',r['b']],ylim=(-3,103),ylabel='Stress strict success (%)',title=f'Difference {100*r["success_difference"]:+.1f} pp; exact p={r["holm_adjusted_p"]:.3f}')
        ax.legend(title='Training seed',fontsize=8);ax.grid(alpha=.2)
    fig.savefig(target/'primary_seed_pairs.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    for i,m in enumerate(['snn_neat','snn_fixed_tau','snn_fixed_topology','leaky_ann_neat']):
        nodes=[n for r in data['temporal'] if r['method']==m for n in r['nodes'] if not n['readout']]
        ax.scatter([n['tau_m_ms'] for n in nodes],[n['tau_s_ms'] for n in nodes],label=m,color=colors(i),alpha=.65,s=32)
    ax.set(xscale='log',yscale='log',xlim=(4.5,220),ylim=(1.8,110),xlabel='Membrane time constant (ms)',ylabel='Synaptic time constant (ms)',title='Selected hidden-node time scales across all5 training seeds')
    ax.axvline(20,ls=':',color='gray');ax.axhline(10,ls=':',color='gray');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(target/'time_constants.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();analyze(a.root)
if __name__=='__main__':main()
