"""Gated, immutable-protocol eight-arm campaign. Final tests run only at the end."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from .common import ROOT, read, write, sha, event, heartbeat
from .development import command, select_model
from .imitation import dataset
from .networks import initial_spec, complexity
from .evaluate import worker_init, evaluate
from .environment import scenarios
from .train_neat import allocated_budget


def freeze(root):
    gate = read(root/'gate_passed.json')
    if not all(row['passed'] for row in gate['gates']):
        raise ValueError('Development gate is not satisfied')
    if not read(root/'training_smoke.json')['passed']:
        raise ValueError('All eight trainer smoke checks must pass')
    sources = sorted(p for p in (ROOT/'experiments/algorithms/snn_pid_compare').iterdir()
                     if p.suffix in ('.py', '.cpp', '.sh', '.txt'))
    sources += [ROOT/p for p in read(ROOT/'artifacts/snn_hover_20261009/protocol.json')['source_hashes']]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    path = root/'formal_protocol.json'
    if path.exists():
        protocol = read(path)
        if hashes != protocol['source_hashes']:
            raise ValueError('Implementation changed after the formal protocol was frozen')
        return protocol
    development = read(root/'development_protocol.json')
    # Budget is sized after the host's development runs, before any formal seed.
    population, generations = 16, 24
    protocol = dict(version='v2.1-formal-1', frozen_unix=time.time(),
                    seeds=development['formal_seeds'], population=population, generations=generations,
                    steps=allocated_budget(population, generations), ppo_workers=8,
                    concurrent_training_runs=2, simulation_workers_per_run=8,
                    sequence_backend='original_torch_eager_float64',
                    ppo_rollout=500, ppo_window=500, ppo_epochs=4,
                    curriculum_seconds=[2., 4., 8.], cases_per_generation=4,
                    validation_after_generations=[0, 8, 16, 24],
                    source_hashes=hashes, accepted_dagger_rounds=gate['round'],
                    control_hz=500, bc_epochs=development['bc_epochs'],
                    factor_settings=dict(delay_steps=2, motor_uncertainty=.03, lower_mass_scale=1.125),
                    final_tests_accessed=False,
                    allocation_note='16 individuals x 24 generations x four cases; PPO receives the same 3584000 allocated action steps. Actual consumed steps and all pretraining/validation costs are separate.')
    write(path, protocol)
    snapshot = root/'frozen_source'
    for relative in hashes:
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
    return protocol


def pretrain(root, protocol, workers):
    models = {}
    validation = dataset([root/'data/offline'])
    teacher = read(root/'teacher_validation.json')
    previous = {}
    data = [str(root/'data/demo')]
    for round_index in range(protocol['accepted_dagger_rounds']+1):
        if round_index:
            data.append(str(root/'data'/f'dagger{round_index}'))
        def fit_one(key):
            kind, seed = key
            directory = root/'initialization'/f'{kind}_seed{seed}'/f'round{round_index}'
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory/'completed.json').exists():
                args = ['--kind', kind, '--seed', seed, '--output', directory, '--data', *data,
                        '--validation', root/'data/offline', '--epochs', protocol['bc_epochs']]
                if key in previous:
                    args += ['--initial', previous[key]]
                command('imitation', args, directory/'train.log')
            return key, directory
        keys = [(kind, seed) for seed in protocol['seeds'] for kind in ('snn', 'ann')]
        event(root, 'formal_imitation', round=round_index, models=len(keys))
        with ThreadPoolExecutor(min(workers, len(keys))) as pool:
            directories = list(pool.map(fit_one, keys))
        with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(root/'source'),)) as pool:
            for key, directory in directories:
                event(root, 'formal_imitation_validation', model=directory.parent.name, round=round_index)
                select_model(pool, directory, str(root/'source'), teacher, validation, [0, 10, 20, 30])
                previous[key] = directory/'best_policy.json'
    for (kind, seed), path in previous.items():
        models[f'direct_{kind}_{seed}'] = str(path)
        residual = root/'initialization'/f'{kind}_seed{seed}'/'zero_residual.json'
        write(residual, initial_spec(kind, seed, zero_readout=True))
        models[f'residual_{kind}_{seed}'] = str(residual)
    write(root/'initializations.json', {key:dict(path=value, sha256=sha(value)) for key, value in models.items()})
    return models


def final_cases(protocol):
    result = scenarios('final', 32, 20.)+scenarios('final', 32, 20., 'stress')
    result += scenarios('final', 8, 60.)+scenarios('final', 8, 60., 'stress')
    for c in result:
        c['test_family'] = ('long_' if c['seconds'] == 60. else '')+c['condition']
    for key, value in protocol['factor_settings'].items():
        for c in scenarios('final', 8, 20.):
            c[key] = value
            c['test_family'] = 'factor_'+key
            result.append(c)
    return result


def finish(root, protocol, initializations, workers):
    cases = final_cases(protocol)
    write(root/'final_cases.json', cases)
    models = [('pid', None, 'pid')]
    models += [(f'bc_{kind}_{seed}', path, 'direct') for key, path in initializations.items()
               for mode, kind, seed in [key.split('_')] if mode == 'direct']
    for mode in ('direct', 'residual'):
        for seed in protocol['seeds']:
            for kind in ('snn', 'ann'):
                for method in ('ppo', 'neat'):
                    name = f'{mode}_{kind}_{method}_seed{seed}'
                    directory = root/'training'/name
                    if not (directory/'results.json').exists():
                        raise ValueError(f'Formal training is incomplete: {name}')
                    models.append((name, str(directory/'best_policy.json'), mode))
    output = root/'evaluation'
    output.mkdir(exist_ok=True)
    results = {}
    with ProcessPoolExecutor(workers, initializer=worker_init, initargs=(str(root/'source'),)) as pool:
        for index, (name, path, mode) in enumerate(models):
            event(root, 'final_evaluation', model=name, completed=index, total=len(models))
            target = output/f'{name}.json'
            result = read(target) if target.exists() else evaluate(pool, read(path) if path else None, mode, cases)
            write(target, result)
            results[name] = result
    summary = {}
    for name, result in results.items():
        families = {}
        for family in sorted({row['case']['test_family'] for row in result['episodes']}):
            rows = [r for r in result['episodes'] if r['case']['test_family'] == family]
            families[family] = dict(success_rate=float(np.mean([r['success'] for r in rows])),
                                    tilt_rms_deg=float(np.mean([r['tilt_rms_deg'] for r in rows])),
                                    height_rmse_m=float(np.mean([r['height_rmse_m'] for r in rows])))
        summary[name] = families
    write(root/'analysis.json', summary)
    from .analysis import generate
    generate(root, protocol, summary, results)
    lines = ['# SNN/ANN 直接与残差控制结果', '',
             '本报告保留五个训练种子的独立结果。旧 2201 为不同协议的阶段证据，不加入本表。', '',
             '| 方法 | 常规成功率（五种子均值） | 扰动成功率（五种子均值） |', '|---|---:|---:|']
    for mode in ('direct', 'residual'):
        for kind in ('snn', 'ann'):
            for method in ('ppo', 'neat'):
                name = f'{mode}_{kind}_{method}'
                rows = [summary[f'{name}_seed{seed}'] for seed in protocol['seeds']]
                lines.append(f'| {name} | {np.mean([r["nominal"]["success_rate"] for r in rows]):.3f} | {np.mean([r["stress"]["success_rate"] for r in rows]):.3f} |')
    lines += ['', '完整场景、长期控制、单因素变化与逐种子结果见 analysis.json 和 evaluation/。',
              '配对效应见 paired_contrasts.json；控制器规模与总推理成本见 costs.json；学习曲线见 learning_curves.png。',
              '数据为仿真结果；五种子按探索性证据解释，不以测试片段数量代替独立训练样本。', '']
    (root/'report.md').write_text('\n'.join(lines))
    event(root, 'completed', training_runs=40, final_models=len(models), report=str(root/'report.md'))


def run(root, workers):
    protocol = freeze(root)
    initializations = pretrain(root, protocol, workers)
    def fit_one(task):
        mode, seed, kind, method = task
        name = f'{mode}_{kind}_{method}_seed{seed}'
        output = root/'training'/name
        output.mkdir(parents=True, exist_ok=True)
        if (output/'results.json').exists():
            return name
        args = ['--source', root/'source', '--initial', initializations[f'{mode}_{kind}_{seed}'],
                '--output', output, '--mode', mode, '--seed', seed]
        if method == 'ppo':
            args += ['--workers', protocol['ppo_workers'], '--steps', protocol['steps']]
        else:
            args += ['--workers', protocol['simulation_workers_per_run'], '--population', protocol['population'], '--generations', protocol['generations']]
        command('train_'+method, args, output/'train.log')
        return name
    tasks = [(mode, seed, kind, method) for mode in ('direct', 'residual')
             for seed in protocol['seeds'] for kind in ('snn', 'ann') for method in ('ppo', 'neat')]
    concurrency = protocol['concurrent_training_runs']
    with ThreadPoolExecutor(concurrency) as pool:
        pending = {}
        next_task = 0
        completed = 0
        while next_task < len(tasks) or pending:
            pausing = (root/'pause_request.json').exists()
            while not pausing and len(pending) < concurrency and next_task < len(tasks):
                task = tasks[next_task]
                pending[pool.submit(fit_one, task)] = task
                next_task += 1
            if not pending:
                event(root, 'paused_at_run_boundary', completed=completed)
                return
            event(root, 'formal_training', runs=[f'{m}_{k}_{a}_seed{s}' for m, s, k, a in pending.values()],
                  completed=completed, total=len(tasks), budget=protocol['steps'], pause_requested=pausing)
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                del pending[future]
                completed += 1
            if pausing and not pending:
                event(root, 'paused_at_run_boundary', completed=completed)
                return
    finish(root, protocol, initializations, workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    args.root = args.root.resolve()
    with (args.root/'formal.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with heartbeat(args.root):
                run(args.root, args.workers)
        except KeyboardInterrupt:
            event(args.root, 'interrupted', reason='KeyboardInterrupt; completed checkpoints preserved')
            raise
        except Exception:
            event(args.root, 'failed', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
