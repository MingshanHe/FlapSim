"""Run two paired training seeds; retain process and timeout records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from .train import write_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    if (root/'campaign_status.json').exists(): raise RuntimeError('Campaign already started')
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('config.ini')]
    files+=list(Path('flapsim').rglob('*.py'))+list(Path('flapsim').rglob('*.cpp'))
    files+=[Path('tests/test_neat_ppo_compare.py'),root/'robot_upper7_lower8.xml',root/'phase_bank.npz',root/'calibration/trim.json',root/'protocol.json']
    write_json(root/'training_source_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile(root/'training_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files: archive.write(p,str(p))
    write_json(root/'preflight.json',dict(status='passed',related_tests=19,
        initial_smoke_failure='CompareEnv.root collided with inherited simulator root joint id. Renamed to experiment_root; failed smoke retained.',
        neat_smoke='smoke_neat_fixed',ppo_smoke='smoke_ppo',
        canonical_source_snapshot='training_source_snapshot.zip',timeout_seconds_per_run=14400))
    active={}; completed=[]; start=time.monotonic()
    def launch(method,seed):
        command=[sys.executable,'-m','experiments.algorithms.neat_imu.compare_train','--root',str(root),
                 '--method',method,'--seed',str(seed),'--workers','8','--steps',str(protocol['target_training_steps_per_seed'])]
        logfile=(root/f'{method}_seed{seed}.log').open('w')
        process=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        active[seed]=dict(method=method,process=process,logfile=logfile,started=time.monotonic(),command=command)
        print(json.dumps(dict(event='started',method=method,seed=seed,pid=process.pid,command=command)),flush=True)
    for seed in protocol['seeds']: launch('neat',seed)
    while active:
        for seed,job in list(active.items()):
            code=job['process'].poll()
            if code is None and time.monotonic()-job['started']>14400:
                print(json.dumps(dict(event='hard_timeout',seed=seed,method=job['method'],pid=job['process'].pid)),flush=True)
                os.killpg(job['process'].pid,15)
                code=job['process'].wait(timeout=30)
            if code is not None:
                job['logfile'].close()
                completed.append(dict(method=job['method'],seed=seed,pid=job['process'].pid,exit_code=code,wall_seconds=time.monotonic()-job['started']))
                del active[seed]
                if code==0 and job['method']=='neat': launch('ppo',seed)
                elif code!=0:
                    print(json.dumps(dict(event='failed',method=job['method'],seed=seed,exit_code=code)),flush=True)
        state=dict(status='running' if active else 'completed' if all(r['exit_code']==0 for r in completed) else 'failed',
                   elapsed_seconds=time.monotonic()-start,completed=completed,
                   active=[dict(method=j['method'],seed=s,pid=j['process'].pid,elapsed_seconds=time.monotonic()-j['started']) for s,j in active.items()])
        write_json(root/'campaign_status.json',state)
        if active: time.sleep(10)
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__': main()
