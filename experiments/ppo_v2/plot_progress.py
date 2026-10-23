"""Plot training duration and whole-episode success, without hiding early failures."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('runs/report/progress.png'))
    args = p.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for directory in args.runs:
        path = directory/'progress.csv'
        if not path.exists():
            continue
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        def values(x, y):
            return np.array([(float(r[x]), float(r[y])) for r in rows if r.get(x) and r.get(y)]).reshape(-1, 2)
        data = values('time/total_timesteps', 'tracking/seconds')
        if len(data):
            ax[0].plot(data[:, 0], data[:, 1], label=directory.name)
        data = values('time/total_timesteps', 'tracking/success')
        if len(data):
            ax[1].plot(data[:, 0], 100*data[:, 1], label=directory.name)
        path = directory/'evaluation_history/evaluations.npz'
        if path.exists():
            history = np.load(path)
            ax[0].scatter(history['timesteps'], history['ep_lengths'].mean(axis=1)*.01,
                          marker='x', s=35, color='black')
    ax[0].axhline(8, color='#218b64', linestyle='--', label='8 s target')
    ax[0].set(xlabel='PPO environment steps', ylabel='Flight duration (s)', ylim=(0, 8.5))
    ax[1].set(xlabel='PPO environment steps', ylabel='Complete episode success (%)', ylim=(-2, 102))
    for axis in ax:
        axis.grid(alpha=.2)
        axis.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
    ax[0].legend(fontsize=8)
    fig.suptitle('IST_robot-V2 | full mechanism, 15 g, normal gravity, 5 us physics\n'
                 'Success requires an entire flight and position RMSE < 0.1 m; x = evaluation')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
