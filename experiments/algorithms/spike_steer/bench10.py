"""Plan v9 scoring: bench9 tasks (evade / redirect / dash) with a SYMMETRIC altitude term.

bench9 / bench8 charge 0.03 per cm of altitude LOSS only (pos0.z - min z inside the task window), so climbing was free;
the v8 analysis showed ES exploiting this: learned pulses add lift, the learned layer (which does not observe height)
handed back at +41 cm climbing at +40 cm/s, and the PID could not catch it (upward height escape). v9 replaces the loss
term by 0.03 per cm of the largest |z - z(t0)| from the command onset to the end of the episode (up or down, the whole
remaining flight, not only the task window). Everything else is bench9 unchanged; the classical opponents keep their
bench9 schedules.

    python -m experiments.algorithms.spike_steer.bench10 selftest
"""
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T

ALT_W = .03                                    # per cm, as bench9's loss term


def score10(s, trace=False):
    res = T.score9(s, trace)
    a = np.array(s['rows'])
    t0 = s['c']['t0']
    pos0 = s['pos0'] if s['pos0'] is not None else a[0, 3:6]
    after = a[a[:, 0] >= t0]
    dz_abs = float(100*np.abs(after[:, 5]-pos0[2]).max()) if len(after) else 0.
    dz_loss = float(res.get('height_loss_cm') or 0.)
    res.update(score_bench9=res['score'], alt_dev_cm=dz_abs, score=float(res['score']-ALT_W*dz_loss+ALT_W*dz_abs))
    return res


class Episode(T.Episode):
    def finish(self, trace=False):
        return score10(self.s, trace)


def run_episode10(env, task, c, policy, trace=False):
    ep = Episode(env, task, c, policy)
    while not ep.s['done']:
        ep.tick()
    return ep.finish(trace)


def job(args):
    name, task, c, kind, arg = args[:5]
    trace = args[5] if len(args) > 5 else False
    if kind == 'classical':
        pol = T.Classical9(arg.get('law'), arg.get('schedule'))
    else:
        from .descend8 import Learned8
        pol = Learned8(arg)
    return dict(run_episode10(B._ENV, task, c, pol, trace), controller=name)


def run_jobs(ex, jobs):
    return list(ex.map(job, jobs, chunksize=1))


def run_cases(env, task, c, specs, trace=False):
    from . import cache8 as C8
    from .descend8 import Learned8, Net8
    ep = Episode(env, task, c, Learned8(specs[0][1]))
    while not ep.s['done'] and ep.t() < c['t0']-1e-9:
        ep.tick()
    base = ep.s
    out = []
    for name, spec in specs:
        st = C8.snapshot(base)
        if not base['done']:
            pol = st['policy']
            pol.spec, pol.net, pol.twitch = spec, Net8(spec), 0
            for (t, x) in st['xs']:
                pol.replay(t, x)
        e = Episode.__new__(Episode)
        e.s = st
        while not e.s['done']:
            e.tick()
        out.append(dict(e.finish(trace), controller=name))
    return out


def job_group(args):
    task, c, specs = args[:3]
    return run_cases(B._ENV, task, c, specs, args[3] if len(args) > 3 else False)


def run_grouped(ex, items, chunk=8):
    groups = {}
    for k, (name, task, c, spec) in enumerate(items):
        groups.setdefault((task, c['seed']), (task, c, []))[2].append((k, name, spec))
    jobs, index = [], []
    for task, c, lst in groups.values():
        for i in range(0, len(lst), chunk):
            part = lst[i:i+chunk]
            jobs.append((task, c, [(n, s) for _, n, s in part]))
            index.append([k for k, _, _ in part])
    out = [None]*len(items)
    for idx, rows in zip(index, ex.map(job_group, jobs, chunksize=1)):
        for k, r in zip(idx, rows):
            out[k] = r
    return out


def selftest():
    import json
    from pathlib import Path
    E.load_signs()
    B.DASH_M['value'] = .15
    B.worker_init(str(E.SOURCE), .15)
    base = json.loads(Path('artifacts/spike_steer/plan_v7/baseline.json').read_text())
    spec = json.loads(Path('artifacts/spike_steer/plan_v7/full/snn/s5/selected.json').read_text())
    from .descend8 import Learned8
    c = [c for c in T.cases9('dash', 2098500000, 64) if c['seed'] == 2098500015][0]
    for name, pol in (('classical', T.Classical9(base['law'], base['opponent']['dash'][T.klass(c)])), ('snn_es_s5', Learned8(spec))):
        r = run_episode10(B._ENV, 'dash', c, pol)
        print(name, 'bench9 %.3f -> bench10 %.3f' % (r['score_bench9'], r['score']), 'alt dev %.1f cm' % r['alt_dev_cm'], r['termination'])
    print('bench10 selftest ok')


if __name__ == '__main__':
    selftest()
