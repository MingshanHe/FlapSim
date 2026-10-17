"""Run transfers of already frozen pilot graphs alongside the primary queue.

This is scheduling only: command arguments and training budgets match campaign.py.
Use only while the primary queue is well before its final transfer stage. The main
queue reuses completed results and deliberately rejects any incomplete directory.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', choices=['graph_ppo','random_graph_ppo'],
                        default=['graph_ppo','random_graph_ppo'])
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root = args.root
    completed = []
    status_path = root / 'parallel_transfer_status.json'
    try:
        for seed in args.seeds:
            parent = root / 'pilot' / f'neat_seed{seed}'
            result = json.loads((parent / 'results.json').read_text())
            assert result['status'] == 'completed'
            for method in args.methods:
                path = root / 'transfer' / f'{method}_seed{seed}'
                if (path / 'results.json').exists():
                    completed.append(str(path))
                    continue
                if path.exists():
                    raise RuntimeError(f'Incomplete output retained: {path}')
                command = [sys.executable, '-u', '-m',
                    'experiments.algorithms.topology_hover.ppo', '--output', str(path),
                    '--method', method, '--seed', str(seed),
                    '--steps', str(result['training_steps']), '--workers', str(args.workers),
                    '--topology', str(parent / 'best_policy.json')]
                write(status_path, dict(status='running', current=str(path),
                    command=command, completed=completed, started=time.time(),
                    note='Concurrent scheduling; do not compare per-run wall time.'))
                with (root / 'logs' / f'transfer_{method}_{seed}.log').open('w') as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    started = time.monotonic()
                    while proc.poll() is None:
                        if time.monotonic() - started > 7200:
                            os.killpg(proc.pid, signal.SIGTERM)
                            raise TimeoutError(f'Two-hour job timeout: {path}')
                        time.sleep(5)
                    if proc.returncode:
                        raise RuntimeError(f'{path} exited {proc.returncode}')
                completed.append(str(path))
        write(status_path, dict(status='completed', completed=completed, finished=time.time()))
    except BaseException as error:
        write(root / 'parallel_transfer_failure.json',
              dict(type=type(error).__name__, message=str(error), time=time.time()))
        raise


if __name__ == '__main__':
    main()
