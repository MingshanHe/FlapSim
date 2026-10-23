"""Post-hoc diagnostics on saved data and frozen policies; never train or select."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from .common import read, write, sha
from .networks import policy, Actor

DATA = {}


def initialize(root):
    global DATA
    root = Path(root)
    for split in ('demo', 'offline', 'dagger1', 'dagger2'):
        examples = []
        for row in read(root/'data'/split/'manifest.json')['episodes']:
            p = root/'data'/split/row['file']
            if sha(p) != row['sha256']:
                raise ValueError(f'Data changed: {p}')
            with np.load(p) as z:
                examples.append({k:z[k].astype(np.float64) for k in ('obs', 'action', 'teacher_raw', 'time')})
        DATA[split] = examples


def metrics(prediction, target):
    error = prediction-target
    variance = target.var(0)
    return dict(samples=len(target), rmse=np.sqrt((error**2).mean(0)).tolist(),
                bias=error.mean(0).tolist(), target_std=np.sqrt(variance).tolist(),
                prediction_std=prediction.std(0).tolist(),
                r2=(1-(error**2).mean(0)/np.maximum(variance, 1e-15)).tolist(),
                correlation=[float(np.corrcoef(prediction[:,k],target[:,k])[0,1]) for k in range(3)])


def model_task(task):
    root, round_index, name = task
    path = Path(root)/'development'/f'round{round_index}'/name/'best_policy.json'
    spec = read(path)
    result = dict(policy=str(path), sha256=sha(path), splits={})
    for split, examples in DATA.items():
        # Teacher training errors use four fixed episodes; validation uses all eight.
        selected = examples[:4] if split == 'demo' else examples
        predicted, targets, times, smooth_pred, smooth_target = [], [], [], [], []
        for episode in selected:
            model = policy(spec)
            y = np.array([model.act(x) for x in episode['obs']])
            predicted.append(y); targets.append(episode['action']); times.append(episode['time'])
            # Identical causal 20 ms averages, computed separately in each episode.
            smooth_pred.append(lfilter(np.ones(10)/10, [1], y, axis=0))
            smooth_target.append(lfilter(np.ones(10)/10, [1], episode['action'], axis=0))
        p, y, t = np.concatenate(predicted), np.concatenate(targets), np.concatenate(times)
        rows = dict(all=metrics(p,y), first_second=metrics(p[t<1],y[t<1]))
        if np.any(t>=2):
            rows['after_two_seconds'] = metrics(p[t>=2],y[t>=2])
        rows['twenty_ms_average'] = metrics(np.concatenate(smooth_pred),np.concatenate(smooth_target))
        if split == 'offline':
            shifts = []
            for delay in range(-10,11):
                pp, yy = [], []
                for a,b in zip(predicted,targets):
                    a,b = a[1000:], b[1000:]
                    if delay > 0:
                        a,b = a[delay:], b[:-delay]
                    elif delay < 0:
                        a,b = a[:delay], b[-delay:]
                    pp.append(a); yy.append(b)
                shifts.append(dict(delay_steps=delay, rmse=np.sqrt(np.mean((np.concatenate(pp)-np.concatenate(yy))**2,axis=0)).tolist()))
            rows['lag_diagnostic'] = shifts
        result['splits'][split] = rows
    if spec['kind'] == 'ann':
        result['local_step_response'] = ann_response(spec, DATA['offline'])
    return f'round{round_index}/{name}', result


def ann_response(spec, examples):
    """Exact ANN tangent response along saved teacher trajectories, no intervention on plant."""
    actor = Actor(spec)
    W = (actor.weight*actor.mask).detach().numpy()
    Win, Wrec = W[:,:11], W[:,11:]
    rows = {k:[] for k in (1,5,25,50,250)}
    for episode in examples:
        model = policy(spec)
        states = np.array([step_state(model,obs) for obs in episode['obs']])
        for start in (500,1500,3000,5000,7000):
            J = np.zeros((len(spec['nodes']),11))
            for step in range(250):
                J = (1-states[start+step]**2)[:,None]*(Win+Wrec@J)
                if step+1 in rows:
                    rows[step+1].append(J[actor.out])
    return {str(k*2)+'ms':dict(mean=np.mean(v,axis=0).tolist(),
                              min=np.min(v,axis=0).tolist(), max=np.max(v,axis=0).tolist()) for k,v in rows.items()}


def step_state(model, obs):
    model.act(obs)
    return model.state.copy()


def dataset_summary(root):
    initialize(root)
    result = {}
    for split, examples in DATA.items():
        x = np.concatenate([r['obs'] for r in examples]); y=np.concatenate([r['action'] for r in examples])
        result[split] = dict(episodes=len(examples), samples=len(x), lengths=[len(r['obs']) for r in examples],
                            first_second_fraction=sum(min(500,len(r['obs'])) for r in examples)/len(x),
                            observed_rpy_abs_degrees_quantiles=np.quantile(abs(x[:,:2])*.35*180/np.pi,[.5,.9,.99,1],axis=0).tolist(),
                            action_std=y.std(0).tolist(), action_mean=y.mean(0).tolist(),
                            action_saturation_fraction=np.mean(abs(y)>=.999,axis=0).tolist())
    lengths = [len(e['obs']) for split in ('demo','dagger1','dagger2') for e in DATA[split]]
    groups = np.array([0]*len(DATA['demo'])+[1]*len(DATA['dagger1'])+[2]*len(DATA['dagger2']))
    weights = {}
    for seed in (5101,5102,5103):
        rng = np.random.default_rng(seed)
        totals = np.zeros(3); updates = 0
        for _ in range(30):
            order = rng.permutation(len(lengths))
            for i in range(0,len(order),8):
                batch = order[i:i+8]
                for begin in range(0,max(lengths[j] for j in batch),500):
                    count = np.array([max(0,min(500,lengths[j]-begin)) for j in batch])
                    for g in range(3):totals[g]+=count[groups[batch]==g].sum()/count.sum()
                    updates+=1
        weights[str(seed)] = dict(mean_window_loss_share=(totals/updates).tolist(), updates=updates)
    result['round2_mse_window_weighting'] = weights
    return result


def failure_summary(root):
    result = {}
    for directory in sorted((root/'development').glob('round*/*')):
        if not (directory/'gate.json').exists():continue
        g=read(directory/'gate.json');name=str(directory.relative_to(root/'development'))
        result[name] = {}
        for label, path in [('nominal',g['selected_checkpoint'].replace('policy_epoch','closed_loop_epoch')),
                            ('long','long_validation.json'),('stress','stress_validation.json')]:
            episodes=read(directory/path)['episodes']
            result[name][label]=dict(successes=sum(r['success'] for r in episodes), episodes=len(episodes),
                terminations=dict(Counter(r['termination'] for r in episodes)),
                mean_duration=float(np.mean([r['elapsed'] for r in episodes])),
                successes_by_phase={str(p):sum(r['success'] for r in episodes if r['phase']==p)
                                    for p in sorted({r['phase'] for r in episodes})})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'data_distribution.json',dataset_summary(args.root))
    write(args.output/'failure_modes.json',failure_summary(args.root))
    tasks=[(str(args.root),r,f'{kind}_seed{s}') for r in (0,2) for kind in ('snn','ann') for s in (5101,5102,5103)]
    result={}
    with ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(str(args.root),)) as pool:
        for name,row in pool.map(model_task,tasks):
            result[name]=row
            print('analyzed',name,flush=True)
            write(args.output/'offline_model_diagnostics.json',result)
    write(args.output/'scope.json',dict(type='posthoc_diagnostic',training_performed=False,
          final_test_accessed=False,policy_parameters_modified=False,
          teacher_training_episodes='first four',offline_validation_episodes='all eight',
          student_state_episodes='all recorded dagger1 and dagger2; these are training data, not independent tests'))


if __name__=='__main__':main()
