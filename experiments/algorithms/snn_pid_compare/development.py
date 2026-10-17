"""Finite development campaign with a hard closed-loop gate and two DAgger rounds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from .common import ROOT, read, write, sha, event, heartbeat
from .environment import ControlEnv, scenarios, CENTER, HALF
from .networks import policy
from .imitation import dataset, offline
from .evaluate import worker_init, evaluate, selection, gate


def command(module, arguments, log):
    with Path(log).open('a') as handle:
        subprocess.run([sys.executable, '-u', '-m', 'experiments.algorithms.snn_pid_compare.'+module,
                        *map(str, arguments)], stdout=handle, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def fit_task(task):
    root, kind, seed, round_index, data, initial, epochs = task
    root = Path(root)
    directory = root/'development'/f'round{round_index}'/f'{kind}_seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'completed.json').exists():
        return str(directory)
    args = ['--kind', kind, '--seed', seed, '--output', directory,
            '--data', *data, '--validation', root/'data/offline', '--epochs', epochs]
    if initial:
        args += ['--initial', initial]
    command('imitation', args, directory/'train.log')
    return str(directory)


def select_model(pool, directory, source, teacher, validation_examples, candidate_epochs):
    directory = Path(directory)
    if (directory/'gate.json').exists():
        return read(directory/'gate.json')
    best = None
    for epoch in candidate_epochs:
        path = directory/f'policy_epoch{epoch:03d}.json'
        result_path = directory/f'closed_loop_epoch{epoch:03d}.json'
        result = read(result_path) if result_path.exists() else evaluate(pool, read(path), 'direct', scenarios('validation', 16))
        write(result_path, result)
        if best is None or selection(result['summary']) > selection(best[1]['summary']):
            best = (path, result)
    path, nominal = best
    spec = read(path)
    write(directory/'best_policy.json', spec)
    rmse = offline(spec, validation_examples)
    long = evaluate(pool, spec, 'direct', scenarios('long_validation', 4, 60.))
    stress = evaluate(pool, spec, 'direct', scenarios('stress_validation', 16, 20., 'stress'))
    write(directory/'long_validation.json', long)
    write(directory/'stress_validation.json', stress)
    result = gate(nominal, long, teacher, rmse)
    result.update(selected_checkpoint=path.name, policy_sha256=sha(directory/'best_policy.json'),
                  stress=stress['summary'])
    write(directory/'gate.json', result)
    return result


def dagger_task(task):
    source, specification, c, output, index = task
    env = ControlEnv(source, mode='direct')
    env.configure(c)
    obs, _ = env.reset(seed=c['seed'])
    controller = policy(specification)
    x, y, raw, applied = [], [], [], []
    while True:
        target = env.teacher_command()
        x.append(obs.copy()); y.append((target-CENTER)/HALF); raw.append(env.teacher.raw.copy())
        obs, _, done, truncated, info = env.step(controller.act(obs))
        applied.append(info['applied_command'])
        if done or truncated:
            break
    path = Path(output)/f'episode_{index:03d}.npz'
    np.savez_compressed(path, obs=np.array(x), action=np.array(y), teacher_raw=np.array(raw),
                        applied=np.array(applied), time=np.arange(len(x))*.002)
    return dict(file=path.name, sha256=sha(path), samples=len(x), accepted=True, case=c,
                metrics=info['episode'], student_sha256=sha_spec(specification),
                acceptance_reason='causal teacher labels on student states, including failed student episodes')


def sha_spec(spec):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def aggregate(root, models, round_index, workers):
    output = root/'data'/f'dagger{round_index}'
    if (output/'manifest.json').exists():
        return output
    output.mkdir(parents=True, exist_ok=True)
    cases = scenarios('dagger', 12, 20.)
    tasks = []
    for index, c in enumerate(cases):
        c['seed'] += round_index*100_000
        directory = Path(models[index//2])
        tasks.append((str(root/'source'), read(directory/'best_policy.json'), c, str(output), index))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(dagger_task, tasks))
    write(output/'manifest.json', dict(split='dagger_training', round=round_index,
          episodes=rows, samples=sum(r['samples'] for r in rows), shared_by=['snn', 'ann']))
    return output


def wait_dataset(root, name):
    path = root/'data'/name/'manifest.json'
    start = time.monotonic()
    while not path.exists():
        if time.monotonic()-start > 3600:
            raise TimeoutError(f'Data collection has not completed: {path}')
        time.sleep(5)
    result = read(path)
    if result['accepted_samples'] < (160_000 if name == 'demo' else 40_000):
        raise ValueError(f'Insufficient successful teacher coverage: {path}')


def run(root, workers, run_formal):
    event(root, 'waiting_for_demonstrations')
    wait_dataset(root, 'demo')
    wait_dataset(root, 'offline')
    protocol = read(root/'development_protocol.json')
    source = str(root/'source')
    data = [str(root/'data/demo')]
    validation_examples = dataset([root/'data/offline'])
    initial = {}
    models = []
    all_gates = []
    for round_index in range(protocol['dagger_max_rounds']+1):
        event(root, 'imitation_training', round=round_index, seeds=protocol['development_seeds'])
        models = []
        tasks = [(str(root), kind, seed, round_index, data, initial.get((kind, seed)), protocol['bc_epochs'])
                 for kind in ('snn', 'ann') for seed in protocol['development_seeds']]
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            models.extend(pool.map(fit_task, tasks))
        event(root, 'closed_loop_validation', round=round_index)
        with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(source,)) as pool:
            teacher_path = root/'teacher_validation.json'
            teacher = read(teacher_path) if teacher_path.exists() else evaluate(pool, None, 'pid', scenarios('validation', 16))
            write(teacher_path, teacher)
            gates = []
            for directory in models:
                result = select_model(pool, directory, source, teacher, validation_examples,
                                      protocol['closed_loop_candidate_epochs'])
                gates.append(result)
                event(root, 'closed_loop_validation', round=round_index, model=Path(directory).name,
                      completed=len(gates), total=len(models), result=result)
        all_gates.append(dict(round=round_index, models=models, gates=gates))
        write(root/'development_results.json', all_gates)
        if all(g['passed'] for g in gates):
            write(root/'gate_passed.json', dict(round=round_index, models=models, gates=gates))
            event(root, 'development_gate_passed', round=round_index)
            if run_formal:
                command('campaign', ['--root', root, '--workers', workers], root/'formal_campaign.log')
            return
        if round_index < protocol['dagger_max_rounds']:
            event(root, 'collecting_dagger', round=round_index+1)
            data.append(str(aggregate(root, models, round_index+1, workers)))
            for kind_index, kind in enumerate(('snn', 'ann')):
                for seed_index, seed in enumerate(protocol['development_seeds']):
                    initial[(kind, seed)] = str(Path(models[kind_index*3+seed_index])/'best_policy.json')
    event(root, 'development_gate_not_met', formal_training_started=False,
          reason='Closed-loop imitation thresholds remain unmet after the two predeclared aggregation rounds',
          report=str(root/'development_results.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--run-formal', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    with (args.root/'development.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with heartbeat(args.root):
                run(args.root, args.workers, args.run_formal)
        except KeyboardInterrupt:
            event(args.root, 'interrupted', reason='KeyboardInterrupt; completed checkpoints preserved')
            raise
        except Exception:
            event(args.root, 'failed', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
