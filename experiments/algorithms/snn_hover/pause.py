"""Observe the explicitly requested run boundary without starting more work."""
import argparse
import json
from pathlib import Path
import time
from .common import write


def watch(root):
    request=json.loads((root/'pause_request.json').read_text())
    target=root/'training'/request['after_run'];pid=request['training_pid']
    started=time.monotonic()
    while True:
        done=target/'results.json'
        alive=Path(f'/proc/{pid}').exists()
        if done.exists() and json.loads(done.read_text()).get('status')=='completed' and not alive:
            break
        if not alive:
            write(root/'pause_failure.json',dict(status='stopped_without_completed_result',run=target.name,time=time.time()))
            raise RuntimeError('Requested run stopped before a completed result; no next job was launched')
        if time.monotonic()-started>5*3600:
            raise TimeoutError('Pause watcher bound reached; inspect the preserved training process')
        time.sleep(5)
    completed=sorted(p.parent.name for p in (root/'training').glob('*/results.json') if json.loads(p.read_text()).get('status')=='completed')
    result=dict(status='paused',after_run=target.name,completed=len(completed),total=35,completed_runs=completed,
                next_run='snn_fixed_tau_seed2202',remaining=35-len(completed),paused_unix=time.time(),
                reason='User requested pause; training, full testing and final publication remain stopped.')
    write(root/'pause_completed.json',result)
    write(root/'campaign_status.json',result)
    write(root/'finish_status.json',dict(stage='paused_by_user',status='paused',completed=len(completed),total=35))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    watch(p.parse_args().root)

if __name__=='__main__':main()
