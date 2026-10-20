"""Plan v10 (artifacts/spike_steer/plan_v10/preregistration_v10.json): the plan-v9 pipeline (bench10 tasks and scoring,
classical opponents, imitation, careful ES, selection, blind test) with the 37-input learned layer of bench11 (phasic
command copies + height channels). Only the controller inputs change; the imitation data are re-collected because the
input vector changed.

    python -m experiments.algorithms.spike_steer.descend11 collect --workers 8
    python -m experiments.algorithms.spike_steer.descend11 train --seed 1
    python -m experiments.algorithms.spike_steer.descend11 finetune --seed 1 --tag _careful --lr 0.01 --sigma 0.02 --save-every 1 --generations 20 --workers 10
    python -m experiments.algorithms.spike_steer.descend11 reselect --seed 1 --tag _careful --n-select 32 --workers 30
    python -m experiments.algorithms.spike_steer.descend11 evaluate --seed 1 --tag _careful --n-test 64 --test-base 2098900000 --workers 30
    python -m experiments.algorithms.spike_steer.descend11 frozen9 --workers 30      # v9 selected centres on the v10 blind set
(default --output runs/spike_steer/v10)
"""
import argparse
import json
import sys
import types
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import bench11 as T11
from . import descend8 as D8
from . import descend9 as D9
from . import descend9b as D9b

OUT = 'runs/spike_steer/v10'
TEST_BASE, N_TEST = 2_098_900_000, 64


def install():
    view = types.SimpleNamespace(**{k: getattr(T, k) for k in dir(T) if not k.startswith('__')})
    view.run_grouped, view.run_jobs, view.run_episode9, view.Episode = T11.run_grouped, T11.run_jobs, T10.run_episode10, T10.Episode
    D9.T = view
    D9b.T = view
    D8.N_IN = T11.N_IN
    return view


def collect_job(args):
    t, c, law, sched = args
    pol = T11.Traced11(law, sched)
    res = T10.run_episode10(B._ENV, t, c, pol)
    return dict(task=t, score=res['score'], flown=res['flown'], log=np.array(pol.log, np.float32))


def stage_collect(a):
    out = Path(a.output)
    base = D9.load(out)
    with E.pool(a.workers) as ex:
        for split, off, n in (('train', D9.OFF['collect'], D9.N_COLLECT), ('val', D9.OFF['collect_val'], D9.N_VAL_IMIT)):
            jobs = [(t, c, base['law'], base['opponent'][t][T.klass(c)]) for t in D9.TASKS for c in T.cases9(t, T.BASES['train'], n, offset=off)]
            rows = list(ex.map(collect_job, jobs, chunksize=1))
            L = np.array([len(r['log']) for r in rows])
            X = np.zeros((len(rows), int(L.max()), T11.N_IN+4), np.float32)
            for i, r in enumerate(rows):
                X[i, :L[i]] = r['log']
            np.savez_compressed(out/f'collect_{split}.npz', X=X, lengths=L, tasks=np.array([r['task'] for r in rows]))
            print(json.dumps(dict(split=split, episodes=len(rows), flown=sum(r['flown'] for r in rows))), flush=True)


def stage_frozen9(a):
    """Attribution arm: the five plan-v9 selected centres (25 inputs, zero-padded = identical) on the v10 blind set."""
    out = Path(a.output)
    base = D9.load(out)
    cs = {t: T.cases9(t, TEST_BASE, N_TEST) for t in D9.TASKS}
    src = Path('artifacts/spike_steer/plan_v9')
    specs = []
    for s in range(1, 6):
        f = src/f'snn/s{s}/selected.json'
        if f.exists():
            specs.append((f'snn9_s{s}', T11.widen(json.loads(f.read_text()))))
    from . import cache8 as _C8  # noqa: F401
    with E.pool(a.workers) as ex:
        of = out/'test_opponent.json'
        if of.exists():
            opp = json.loads(of.read_text())
        else:
            opp = T11.run_jobs(ex, [('opp', t, c, 'classical', dict(law=base['law'], schedule=base['opponent'][t][T.klass(c)]))
                                    for t in D9.TASKS for c in cs[t]])
            of.write_text(json.dumps(opp))
        rows = T11.run_grouped(ex, [(n, t, c, sp) for n, sp in specs for t in D9.TASKS for c in cs[t]], chunk=max(1, len(specs)))
    for n, _ in specs:
        rr = [r for r in rows if r['controller'] == n]
        rep = D9.report(rr, opp)
        doc = dict(arm='snn9_frozen', seed=int(n.rsplit('_s', 1)[1]), machine=a.machine, plan='v10',
                   note='plan-v9 selected centre, frozen, on the v10 blind set', report=rep,
                   crashes=sum(not r['flown'] for r in rr), rows=rr)
        (out/f'evaluate_{n}.json').write_text(json.dumps(doc, indent=1))
        print(n, json.dumps({t: round(rep[t]['ratio'], 3) for t in D9.TASKS}), 'mean', round(rep['manoeuvre_mean'], 3),
              'crashes', doc['crashes'], flush=True)


def main():
    install()
    if '--output' not in sys.argv:
        sys.argv += ['--output', OUT]
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage in ('collect', 'train', 'frozen9'):
        p = argparse.ArgumentParser()
        p.add_argument('stage')
        p.add_argument('--output', default=OUT)
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--workers', type=int, default=8)
        p.add_argument('--threads', type=int, default=1)
        p.add_argument('--epochs', type=int, default=40)
        p.add_argument('--machine', default='CloudPC-2')
        a = p.parse_args()
        E.load_signs()
        B.DASH_M['value'] = .15
        {'collect': stage_collect, 'train': D9.stage_train, 'frozen9': stage_frozen9}[stage](a)
        return
    D9b.main()


if __name__ == '__main__':
    main()
