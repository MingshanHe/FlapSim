"""Bounded local completion queue: training -> tests -> analysis -> media -> audit.

This is an immediate finite job, not a recurring automation. It never pushes Git
or edits frozen policies. Exceptions stop the queue and preserve all evidence.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .common import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    start=time.monotonic();logs=root/'logs';logs.mkdir(exist_ok=True)
    try:
        while True:
            status=json.loads((root/'campaign_status.json').read_text())
            if (root/'pause_request.json').exists() or status['status'] in ('paused','pausing_after_current_run'):
                write(root/'finish_status.json',dict(stage='paused_by_user',status='paused',completed=status.get('completed'),total=35))
                return
            if status['status']=='training_complete':break
            if (root/'campaign_failure.json').exists():raise RuntimeError('Training failed; inspect campaign_failure.json')
            if time.monotonic()-start>72*3600:raise TimeoutError('72-hour training wait bound reached')
            write(root/'finish_status.json',dict(stage='waiting_for_training',completed=status.get('completed'),total=35))
            time.sleep(30)
        commands=[
            ('evaluate',['--workers','16']),('analyze',[]),('visualize',[]),('traces',['--workers','8']),('render',[]),('report',[]),('verify',[])]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')
        for stage,extra in commands:
            write(root/'finish_status.json',dict(stage=stage,status='running',time=time.time()))
            with (logs/f'finish_{stage}.log').open('a') as f:
                subprocess.run([sys.executable,'-u','-m',f'experiments.algorithms.snn_hover.{stage}','--root',str(root),*extra],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=48*3600)
            print(json.dumps(dict(stage=stage,status='completed')),flush=True)
        write(root/'finish_status.json',dict(stage='complete',status='completed',wall_seconds=time.monotonic()-start))
    except BaseException as error:
        write(root/'finish_failure.json',dict(error=repr(error),time=time.time()));raise
if __name__=='__main__':main()
