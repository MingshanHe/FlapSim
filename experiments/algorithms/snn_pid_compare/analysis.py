"""Seed-level contrasts and transparent learning/cost summaries."""
import itertools
from pathlib import Path
import numpy as np
from .common import read, write
from .networks import complexity


def paired_effect(left, right):
    difference = np.asarray(left)-np.asarray(right)
    # Descriptive interval over observed seed effects; no episode pseudoreplication.
    null = [abs(np.mean(difference*np.asarray(signs)))
            for signs in itertools.product((-1, 1), repeat=len(difference))]
    return dict(seed_differences=difference.tolist(), mean=float(difference.mean()),
                minimum=float(difference.min()), maximum=float(difference.max()),
                sign_flip_p=float(np.mean(np.asarray(null) >= abs(difference.mean())-1e-12)),
                interpretation='exploratory paired training-seed comparison; unadjusted p reported descriptively')


def generate(root, protocol, summary, evaluations):
    root = Path(root)
    contrasts = {}
    seeds = protocol['seeds']
    def values(mode, kind, method):
        return [summary[f'{mode}_{kind}_{method}_seed{seed}']['stress']['success_rate'] for seed in seeds]
    for mode in ('direct', 'residual'):
        for kind in ('snn', 'ann'):
            contrasts[f'{mode}_{kind}_ppo_minus_neat'] = paired_effect(values(mode, kind, 'ppo'), values(mode, kind, 'neat'))
        for method in ('ppo', 'neat'):
            contrasts[f'{mode}_{method}_snn_minus_ann'] = paired_effect(values(mode, 'snn', method), values(mode, 'ann', method))
    for kind in ('snn', 'ann'):
        for method in ('ppo', 'neat'):
            contrasts[f'{kind}_{method}_residual_minus_direct'] = paired_effect(values('residual', kind, method), values('direct', kind, method))
            contrasts[f'direct_{kind}_{method}_minus_imitation'] = paired_effect(
                values('direct', kind, method), [summary[f'bc_{kind}_{s}']['stress']['success_rate'] for s in seeds])
            contrasts[f'residual_{kind}_{method}_minus_pid'] = paired_effect(
                values('residual', kind, method), [summary['pid']['stress']['success_rate']]*len(seeds))
    write(root/'paired_contrasts.json', contrasts)
    costs = {}
    curves = {}
    for directory in sorted((root/'training').iterdir()):
        spec = read(directory/'best_policy.json')
        result = read(directory/'results.json')
        episodes = evaluations[directory.name]['episodes']
        control_seconds = sum(r['complete_controller_seconds'] for r in episodes)
        steps = sum(r['steps'] for r in episodes)
        costs[directory.name] = dict(**complexity(spec), allocated_steps=result['allocated_steps'],
                                    actual_training_steps=result['training_steps'],
                                    validation_steps=result['validation_steps'], wall_seconds=result['wall_seconds'],
                                    controller_us_per_action=1e6*control_seconds/steps,
                                    controller_has_pid=directory.name.startswith('residual'),
                                    additional_pid_state_floats=8 if directory.name.startswith('residual') else 0,
                                    initial_checkpoint_retained=(read(directory/'selection.json').get('update',
                                                                 read(directory/'selection.json').get('generation')) == 0))
        history = [__import__('json').loads(line) for line in (directory/'history.jsonl').read_text().splitlines()]
        curves[directory.name] = history
    write(root/'costs.json', costs)
    write(root/'learning_curves.json', curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, mode in enumerate(('direct', 'residual')):
        for column, kind in enumerate(('snn', 'ann')):
            ax = axes[row, column]
            for method, color in [('ppo', 'tab:blue'), ('neat', 'tab:orange')]:
                for index, seed in enumerate(seeds):
                    rows = curves[f'{mode}_{kind}_{method}_seed{seed}']
                    ax.plot([r['training_steps'] for r in rows],
                            [r['best_validation']['successes']/32 for r in rows],
                            color=color, alpha=.45, label=method if index == 0 else None)
            ax.set(title=f'{mode} / {kind}', xlabel='Actual training action steps', ylabel='Best validation success rate', ylim=(-.02, 1.02))
            ax.legend()
    figure.savefig(root/'learning_curves.png', dpi=160)
    plt.close(figure)
    return costs, contrasts
