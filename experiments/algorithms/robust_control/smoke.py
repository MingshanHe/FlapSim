"""Exercise every new trainer/task interface before expensive training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from .networks import initial_spec
from ..snn_pid_compare.common import ROOT, write, read


def run(root, source):
    def one(key):
        task, mode, kind, method = key
        out = root/f'{task}_{mode}_{kind}_{method}'
        out.mkdir(parents=True, exist_ok=True)
        initial = out/'initial.json'
        write(initial, initial_spec(kind, 6180, inputs=11 if task == 'hover' else 16, zero_readout=True))
        args = [sys.executable, '-u', '-m', 'experiments.algorithms.robust_control.train_'+method,
                '--source', str(source), '--initial', str(initial), '--output', str(out),
                '--mode', mode, '--task', task, '--seed', '6180', '--workers', '2', '--smoke']
        args += (['--steps','40','--rollout','10','--window','10','--epochs','1'] if method == 'ppo'
                 else ['--population','4','--generations','1','--validate-every','1'])
        with (out/'train.log').open('w') as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = read(out/'results.json')
        return dict(task=task, mode=mode, kind=kind, method=method, completed=result['status']=='completed')
    keys = [(t,mode,k,m) for t,mode in [('hover','residual'),('hover','direct'),('velocity','residual')]
            for k in ('snn','ann') for m in ('ppo','neat')]
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(one, keys))
    from . import environment,train_ppo,train_neat,networks,evolution
    from ..snn_pid_compare.common import sha
    write(root/'results.json', dict(passed=all(r['completed'] for r in rows), runs=rows,
                                  source_hashes={m.__name__:sha(m.__file__) for m in
                                      (environment,train_ppo,train_neat,networks,evolution)},
                                  scientific_performance_evidence=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    run(a.root.resolve(), a.source.resolve())
