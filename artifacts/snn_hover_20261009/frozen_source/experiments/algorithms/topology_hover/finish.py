"""Continue this finite campaign through evaluation and report/media generation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .common import write,benchmark
from .report import generate
from .visualize import build

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args();root=a.root
    start=time.monotonic()
    while True:
        if (root/'campaign_failure.json').exists():raise RuntimeError('Training queue failed; inspect retained failure record')
        s=root/'campaign_status.json'
        if s.exists() and json.loads(s.read_text())['status']=='training_complete':break
        if time.monotonic()-start>24*3600:raise TimeoutError('24-hour campaign wait exceeded')
        time.sleep(10)
    try:
        for folder in sorted(root.glob('*/*')):
            if (folder/'results.json').exists() and (folder/'neat.ini').exists():build(folder)
        # Re-measure all actors after training processes exit, with one common idle-host benchmark.
        measurements={}
        for path in sorted(root.glob('*/*/best_policy.json')):
            if not (path.parent/'results.json').exists():continue
            measurements[str(path.parent.relative_to(root))]=benchmark(json.loads(path.read_text()),10000)
        write(root/'idle_inference_benchmark.json',measurements)
        generate(root)
        for module in ['evaluate','analyze','render','dashboard','verify']:
            command=[sys.executable,'-u','-m',f'experiments.algorithms.topology_hover.{module}','--root',str(root)]
            if module=='evaluate':command+=['--workers',str(a.workers)]
            env=dict(os.environ,MUJOCO_GL='egl')
            with (root/'logs'/f'final_{module}.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
            if module=='render':generate(root)
        generate(root)
        write(root/'delivery_status.json',dict(status='ready_for_final_review',report=str(root/'report.md'),github_pushed=False))
    except BaseException as error:
        write(root/'finalization_failure.json',dict(type=type(error).__name__,message=str(error),time=time.time()));raise

if __name__=='__main__':main()
