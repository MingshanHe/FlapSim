"""Finish matched PPO runs when NEAT completes, then freeze and evaluate once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ..neat_imu.train import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--neat-pids',type=int,nargs=2,required=True);p.add_argument('--timeout',type=float,default=10800)
    a=p.parse_args();root=a.root.resolve();protocol=json.loads((root/'protocol.json').read_text())
    started=time.monotonic();processes={};logs={};completed=set()
    logdir=root/'logs';logdir.mkdir(exist_ok=True)
    def status(stage,**extra):
        state=dict(status=stage,completed=sorted(completed),wall_seconds=time.monotonic()-started,**extra)
        write_json(root/'campaign_status.json',state);print(json.dumps(state),flush=True)
    status('waiting_for_neat')
    try:
        while len(completed)<4:
            if time.monotonic()-started>a.timeout:raise TimeoutError('Campaign exceeded declared wall-clock budget')
            for seed,pid in zip(protocol['seeds'],a.neat_pids):
                label=f'neat_seed{seed}';result=root/label/'results.json'
                if label not in completed:
                    if (root/label/'failure.json').exists():raise RuntimeError(f'{label} reported failure')
                    if result.exists():
                        assert json.loads(result.read_text())['status']=='completed';completed.add(label)
                        command=[sys.executable,'-m','experiments.algorithms.hover9.train','--root',str(root),
                                 '--method','ppo','--seed',str(seed),'--workers','8']
                        logs[seed]=(logdir/f'ppo_seed{seed}.log').open('w')
                        processes[seed]=subprocess.Popen(command,stdout=logs[seed],stderr=subprocess.STDOUT)
                        status('training_ppo',started_seed=seed)
                    else:
                        try:os.kill(pid,0)
                        except ProcessLookupError:raise RuntimeError(f'{label} process exited without completion')
                if seed in processes and f'ppo_seed{seed}' not in completed:
                    code=processes[seed].poll()
                    if code is not None:
                        logs[seed].close()
                        if code!=0:raise RuntimeError(f'PPO {seed} exited with code {code}')
                        result=json.loads((root/f'ppo_seed{seed}/results.json').read_text())
                        assert result['status']=='completed';completed.add(f'ppo_seed{seed}');status('training_ppo')
            time.sleep(5)
        status('final_testing')
        from .results import final_test,video
        final_test(root)
        status('rendering_videos')
        for label in [f'{m}_seed{s}' for m in ('neat','ppo') for s in protocol['seeds']]+['shared_prior']:
            video(root,label);print(json.dumps(dict(video_completed=label)),flush=True)
        status('completed')
    except BaseException as e:
        status('failed',error_type=type(e).__name__,error=str(e))
        for child in processes.values():
            if child.poll() is None:child.terminate()
        raise
    finally:
        for f in logs.values():f.close()


if __name__=='__main__':main()
