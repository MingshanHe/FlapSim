"""Plan v9: the plan-v7-full pipeline (descend9 objective, descend9b careful ES / selection / blind) with bench10
scoring (symmetric altitude term). Implemented by pointing descend9 / descend9b at a bench9 view whose episode runners are
bench10's, so the training, selection and blind code paths are literally the v7-full ones.

    python -m experiments.algorithms.spike_steer.descend10 baseline --workers 30
    python -m experiments.algorithms.spike_steer.descend10 finetune --seed 1 --tag _careful --lr 0.01 --sigma 0.02 --save-every 1 --generations 20 --workers 10
    python -m experiments.algorithms.spike_steer.descend10 reselect --seed 1 --tag _careful --n-select 32 --workers 30
    python -m experiments.algorithms.spike_steer.descend10 evaluate --seed 1 --tag _careful --n-test 64 --test-base 2098700000 --workers 30
(default --output runs/spike_steer/v9)
"""
import argparse
import sys
import types
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from . import descend9b as D9b


def install():
    view = types.SimpleNamespace(**{k: getattr(T, k) for k in dir(T) if not k.startswith('__')})
    view.run_grouped, view.run_jobs, view.run_episode9, view.Episode = T10.run_grouped, T10.run_jobs, T10.run_episode10, T10.Episode
    D9.T = view
    D9b.T = view
    return view


def main():
    install()
    if '--output' not in sys.argv:
        sys.argv += ['--output', 'runs/spike_steer/v9']
    if len(sys.argv) > 1 and sys.argv[1] == 'baseline':
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--workers', type=int, default=30)
        p.add_argument('--output', default='runs/spike_steer/v9')
        a = p.parse_args()
        from . import bench as B
        from . import bench8 as E
        E.load_signs()
        B.DASH_M['value'] = .15
        D9.stage_baseline(a)
        return
    D9b.main()


if __name__ == '__main__':
    main()
