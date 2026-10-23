"""Immediate finite experiment queue, with accounting and resumable completed jobs."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .common import write,validate,summary
from .report import generate
from .visualize import build


def launch(root,folder,method,seed,steps,workers,topology=None,population=128):
    path=root/folder/f'{method}_seed{seed}'
    if (path/'results.json').exists():return json.loads((path/'results.json').read_text())
    if path.exists():raise RuntimeError(f'Incomplete output retained for investigation: {path}')
    module='evolve' if method in ('neat','fixed_evo') else 'ppo'
    command=[sys.executable,'-u','-m',f'experiments.algorithms.topology_hover.{module}',
        '--output',str(path),'--method',method,'--seed',str(seed),'--steps',str(steps),'--workers',str(workers)]
    if module=='evolve':command+=['--population',str(population)]
    if topology:command+=['--topology',str(topology)]
    logs=root/'logs';logs.mkdir(exist_ok=True);log=logs/f'{folder}_{method}_{seed}.log'
    write(root/'campaign_status.json',dict(status='running',current=str(path),command=command,started=time.time()))
    with log.open('w') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic();last_report=0.
        while proc.poll() is None:
            if time.monotonic()-started>7200:
                os.killpg(proc.pid,signal.SIGTERM);raise TimeoutError(f'Two-hour job timeout: {path}')
            if time.monotonic()-last_report>60:
                generate(root);last_report=time.monotonic()
            time.sleep(5)
    if proc.returncode:raise RuntimeError(f'{path} exited {proc.returncode}; see {log}')
    if module=='evolve':build(path)
    generate(root)
    return json.loads((path/'results.json').read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--wait-first',action='store_true');a=p.parse_args();root=a.root
    protocol=json.loads((root/'protocol.json').read_text());seeds=list(protocol['seeds'])
    if a.wait_first:
        target=root/'pilot/neat_seed1201/results.json';start=time.monotonic()
        while not target.exists():
            if time.monotonic()-start>7200:raise TimeoutError('Initial externally launched training did not finish')
            time.sleep(5)
        build(target.parent);generate(root)
    try:
        for seed in seeds:
            result=launch(root,'pilot','neat',seed,protocol['target_actor_steps_per_run'],a.workers)
            for method in ['rnn_ppo','ff_ppo','direct_ppo','fixed_evo']:
                launch(root,'pilot',method,seed,result['training_steps'],a.workers)
        # This gate uses validation only. Test seeds remain untouched until all selection is frozen.
        baselines={}
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for seed in seeds:
                s,rows=validate(pool,None,'pid','height',seed)
                baselines[str(seed)]=dict(summary=s,episodes=rows)
        write(root/'pid_validation.json',baselines)
        benefits=[]
        for seed in seeds:
            s=json.loads((root/f'pilot/neat_seed{seed}/results.json').read_text())['best_validation'];b=baselines[str(seed)]['summary']
            benefits.append(s['mean_return']>b['mean_return']*1.01 and s['tilt_rms_deg']<b['tilt_rms_deg']*.95)
        expand=sum(benefits)>=3
        write(root/'expansion_decision.json',dict(validation_only=True,criterion='At least 3/5 NEAT seeds improve validation return >1% AND tilt RMS >5% over paired PID',benefits=benefits,expand_to_ten=expand))
        if expand:
            for seed in range(1206,1211):
                seeds.append(seed)
                result=launch(root,'formal','neat',seed,protocol['target_actor_steps_per_run'],a.workers)
                for method in ['rnn_ppo','ff_ppo','direct_ppo','fixed_evo']:
                    launch(root,'formal',method,seed,result['training_steps'],a.workers)
        # All graph-transfer fits reset weights and use independent initialization streams.
        for seed in seeds:
            stage='pilot' if seed<=1205 else 'formal';parent=root/stage/f'neat_seed{seed}'
            steps=json.loads((parent/'results.json').read_text())['training_steps']
            for method in ['graph_ppo','random_graph_ppo']:
                launch(root,'transfer',method,seed,steps,a.workers,parent/'best_policy.json')
        write(root/'campaign_status.json',dict(status='training_complete',seeds=seeds,next='held-out evaluation and causal interventions'))
        generate(root)
    except BaseException as error:
        write(root/'campaign_failure.json',dict(type=type(error).__name__,message=str(error),time=time.time()));raise


if __name__=='__main__':main()
