"""Plot one declared seed plus aggregate metrics from a frozen policy evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--trajectories', nargs='+', help='optional named subset for the displayed task group')
    args = parser.parse_args()
    summary = json.loads((args.evaluation/'summary.json').read_text())
    names = args.trajectories or list(summary['trajectories'])
    if any(name not in summary['trajectories'] for name in names):
        parser.error('each displayed trajectory must have evaluated episodes')
    columns = 3 if len(names) <= 5 else 4
    fig = plt.figure(figsize=(4*columns, 8.5), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.15, h_pad=.08, wspace=.12, hspace=.08)
    rows = int(np.ceil((len(names)+1)/columns))
    for index, name in enumerate(names):
        data = np.genfromtxt(args.evaluation/f'seed_{args.seed}'/f'{name}.csv',
                             delimiter=',', names=True)
        spatial = name in ('helix', 'lissajous')
        axis = fig.add_subplot(rows, columns, index+1, projection='3d' if spatial else None)
        target = [data['target_'+xyz] for xyz in 'xyz']
        actual = [data[xyz] for xyz in 'xyz']
        dimension = 3 if spatial else 2
        axis.plot(*target[:dimension], '--', color='#697580', linewidth=1.5, label='Reference')
        axis.plot(*actual[:dimension], color='#087e8b', linewidth=1.3, label='PPO')
        if name == 'hover':
            axis.scatter(target[0][0], target[1][0], marker='+', color='#697580', s=70)
        axis.set(xlabel='x (m)', ylabel='y (m)')
        if spatial:
            axis.set_zlabel('z (m)', labelpad=0)
            axis.tick_params(labelsize=8)
        else:
            axis.set_aspect('equal', adjustable='datalim')
        axis.grid(alpha=.2)
        result = summary['trajectories'][name]
        axis.set_title(f'{name} | {result["success_rate"]:.0%} success\n'
                       f'mean RMSE {100*result["mean_rmse_m"]:.1f} cm', fontsize=11)
        if index == 0:
            axis.legend(fontsize=8)
    axis = fig.add_subplot(rows, columns, len(names)+1)
    means = [100*summary['trajectories'][n]['mean_rmse_m'] for n in names]
    worst = [100*summary['trajectories'][n]['worst_rmse_m'] for n in names]
    axis.barh(names, means, color='#087e8b', label='Mean')
    axis.scatter(worst, names, marker='|', color='#bd4e2f', s=100, label='Worst')
    axis.axvline(10, color='#697580', linestyle='--', label='Success threshold')
    axis.invert_yaxis()
    axis.set(xlabel='3D position RMSE (cm)', title='Displayed tasks, all held-out seeds')
    axis.legend(fontsize=8)
    axis.grid(axis='x', alpha=.2)
    fig.suptitle(f'IST_robot-V2 PPO | 15 g, normal gravity, full articulated physics\n'
                 f'Traces: seed {args.seed}; metrics: {len(summary["seeds"])} seeds, '
                 f'{summary["episode_seconds"]:g} s each; randomized phase, size, period and initial state',
                 fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
