"""Evaluate a sealed interim cohort using the original frozen simulation code.

Run this file by path, not with -m: sys.path is redirected before importing any
experiment package. No training or checkpoint selection occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--interim', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    plan_path = a.interim/'plan.json'
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != a.plan_sha256:
        raise ValueError('Interim plan changed after sealing')
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((a.root/'protocol.json').read_text())
    frozen = Path(protocol['frozen_source']).resolve()
    sys.path.insert(0, str(frozen))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from experiments.algorithms.snn_pid_compare.common import read, write, sha
    from experiments.algorithms.snn_pid_compare.evaluate import summarize
    from experiments.algorithms.robust_control import evaluate as module
    for name in ('experiments.algorithms.robust_control.evaluate',
                 'experiments.algorithms.robust_control.environment',
                 'experiments.algorithms.robust_control.networks',
                 'experiments.algorithms.snn_pid_compare.environment'):
        path = Path(sys.modules[name].__file__).resolve()
        relative = str(path.relative_to(frozen))
        if sha(path) != protocol['source_hashes'][relative]:
            raise ValueError(f'Changed computation: {name}')
    sealed = read(a.interim/'models'/f'{a.model}.json')
    spec = None
    if sealed['policy_path']:
        if sha(sealed['policy_path']) != sealed['policy_sha256']:
            raise ValueError('Sealed interim policy changed')
        spec = read(sealed['policy_path'])
    target = a.interim/'evaluation'/f'{a.model}.json'
    partial = a.interim/'partial'/f'{a.model}.json'
    identity = dict(plan_sha256=a.plan_sha256, model=a.model,
                    policy_sha256=sealed['policy_sha256'])
    if target.exists():
        if read(target)['identity'] != identity:
            raise ValueError('Mismatching completed interim result')
        return
    if partial.exists():
        previous = read(partial)
        if previous['identity'] != identity:
            raise ValueError('Mismatching partial interim result')
        by_index = {r['index']: r['episode'] for r in previous['rows']}
    else:
        by_index = {}
    cases = plan['cases']
    remaining = [i for i in range(len(cases)) if i not in by_index]
    # Long cases first reduces the idle tail without changing any scenario.
    remaining.sort(key=lambda i: -cases[i]['seconds'])
    started = time.time()
    with ProcessPoolExecutor(a.workers, initializer=module.worker_init,
                             initargs=(str(a.root/'source'), 'hover')) as pool:
        futures = {pool.submit(module.evaluate_task, (spec, sealed['mode'], cases[i])): i
                   for i in remaining}
        for future in as_completed(futures):
            index = futures[future]
            by_index[index] = future.result()
            write(partial, dict(identity=identity, updated_unix=time.time(),
                               rows=[dict(index=i, episode=by_index[i]) for i in sorted(by_index)]))
            write(a.interim/'evaluation_progress.json', dict(model=a.model,
                  completed_cases=len(by_index), total_cases=len(cases), unix=time.time()))
    rows = [by_index[i] for i in range(len(cases))]
    summary = summarize(rows)
    summary['minimum_clearance_m'] = min(r['minimum_clearance_m'] for r in rows)
    write(target, dict(identity=identity, summary=summary, episodes=rows,
                       invocation_started_unix=started, finished_unix=time.time(),
                       frozen_evaluate_path=module.__file__))


if __name__ == '__main__':
    main()
