"""Evaluate an immutable checkpoint on a seed matrix using separate processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(1000,1010)))
    p.add_argument('--trajectories',nargs='+',default=['hover'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shard-trajectories',action='store_true',
                   help='parallelize individual seed/trajectory pairs for larger validation matrices')
    p.add_argument('--seconds',type=float)
    p.add_argument('--period',type=float)
    p.add_argument('--reference-physics',action='store_true')
    p.add_argument('--collect-only',action='store_true',help='aggregate already completed seed evaluations')
    p.add_argument('--resume',action='store_true',help='keep completed seeds and run only missing evaluations')
    a=p.parse_args()
    if (a.workers < 1 or len(set(a.seeds)) != len(a.seeds)
            or len(set(a.trajectories)) != len(a.trajectories)):
        p.error('workers must be positive; seeds and trajectory names must be unique')
    if a.output.exists() and any(a.output.iterdir()) and not (a.collect_only or a.resume):
        p.error('output must be empty')
    a.output.mkdir(parents=True,exist_ok=True)
    policy_hash = hashlib.sha256((a.run/'policy.zip').read_bytes()).hexdigest()
    normalizer_hash = hashlib.sha256((a.run/'vecnormalize.pkl').read_bytes()).hexdigest()
    metadata = json.loads((a.run/'run.json').read_text())
    seconds = a.seconds if a.seconds is not None else metadata['tracking']['episode_seconds']
    period = a.period if a.period is not None else metadata['tracking']['period']
    engine = ('split' if metadata['arguments'].get('split_step') and not a.reference_physics
              else 'original')

    def verify(seed, rows, names):
        if [r['trajectory'] for r in rows] != names:
            raise ValueError(f'seed {seed}: cached trajectories differ from this request')
        for row in rows:
            if (row['policy_sha256'] != policy_hash or row['seed'] != seed
                    or row['episode_seconds'] != seconds or not row['randomized_resets']
                    or row['physics_engine'] != engine or row['controller'] != 'ppo'
                    or len(row['episodes']) != 1
                    or row.get('nominal_period_s', metadata['tracking']['period']) != period
                    or row.get('normalizer_sha256', normalizer_hash) != normalizer_hash):
                raise ValueError(f'seed {seed}: checkpoint or evaluation settings changed')
        return seed, rows

    def evaluate(task):
        seed, names = task
        out=a.output/f'seed_{seed}'
        if a.shard_trajectories:
            out = out/names[0]
        if a.collect_only or (a.resume and (out/'summary.json').exists()):
            return verify(seed, json.loads((out/'summary.json').read_text()), names)
        command=[sys.executable,'-m','flapsim.rl.evaluate','--run',str(a.run),
                 '--output',str(out),'--episodes','1','--seed',str(seed),'--randomize',
                 '--trajectories',*names]
        if a.seconds is not None:
            command.extend(['--seconds',str(a.seconds)])
        if a.period is not None:
            command.extend(['--period',str(a.period)])
        if a.reference_physics:
            command.append('--reference-physics')
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log_name = f'seed_{seed}_{names[0]}' if a.shard_trajectories else f'seed_{seed}'
        with (a.output/f'{log_name}.log').open('w') as log:
            subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        return verify(seed, json.loads((out/'summary.json').read_text()), names)
    results=[]
    tasks = ([(seed, [name]) for seed in a.seeds for name in a.trajectories]
             if a.shard_trajectories else [(seed, a.trajectories) for seed in a.seeds])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for seed,rows in pool.map(evaluate,tasks):
            results.extend({**row,'seed':seed} for row in rows)
            print(f'Completed seed {seed}: {", ".join(r["trajectory"] for r in rows)}',flush=True)
    if a.shard_trajectories:
        for seed in a.seeds:
            directory = a.output/f'seed_{seed}'
            rows = [r for r in results if r['seed'] == seed]
            (directory/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
            for name in a.trajectories:
                shutil.copyfile(directory/name/f'{name}.csv', directory/f'{name}.csv')
    summary={}
    for name in a.trajectories:
        rows=[r for r in results if r['trajectory']==name]
        episodes=[dict(seed=r['seed'],**e) for r in rows for e in r['episodes']]
        summary[name]=dict(success_rate=mean(e['success'] for e in episodes),
                           completion_rate=mean(e['termination_reason']=='time_limit' for e in episodes),
                           mean_seconds=mean(e['seconds'] for e in episodes),
                           mean_rmse_m=mean(e['rmse_m'] for e in episodes),
                           worst_rmse_m=max(e['rmse_m'] for e in episodes),episodes=episodes)
    result=dict(run=str(a.run.resolve()),seeds=a.seeds,reference_physics=a.reference_physics,
                policy_sha256=policy_hash, normalizer_sha256=normalizer_hash,
                episode_seconds=seconds, nominal_period_s=period, physics_engine=engine, trajectories=summary)
    if len({r['policy_sha256'] for r in results})!=1:
        raise RuntimeError('policy changed during evaluation; export an immutable checkpoint first')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{m:v for m,v in r.items() if m!='episodes'} for k,r in summary.items()}),flush=True)


if __name__=='__main__':
    main()
