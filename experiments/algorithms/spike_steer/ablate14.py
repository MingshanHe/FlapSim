"""Plan v14 (artifacts/spike_steer/plan_v14/preregistration_v14.json): the cheap checks asked for by the round-2 review of the
paper draft. Frozen controllers, no training.

  d1  analysis only: paired delta of each SNN arm relative to the classical + ES centres (rows of v11 nominal and of every
      v12 / v13 scenario); answers "parity with the same-budget classical?"
  d2  the opponent with its base-law pulses NOT suppressed during schedule segments (Classical9 keeps firing), on the v11
      nominal blind set, the primary kick set and the x1.5 turbulence set; answers "is the opponent's deficit open-loop
      timing or pulses-off?"
  d3  channel ablation of the frozen v11 centres under the primary hit: pulses suppressed while in charge (nopulse) or
      set-points zeroed while in charge (noref), seeds 1-5; answers "does the robustness travel with the pulse channel?"
  d4  the imitation starts of seeds 3-5 under the primary hit and the x1.5 turbulence; answers "is robust-before-ES one
      start or the architecture?"
  d5  (blind12 test 'kick_rep') replication of the primary scenario on a fresh base with the eight main arms.

    python -m experiments.algorithms.spike_steer.ablate14 run --workers 30
    python -m experiments.algorithms.spike_steer.ablate14 report
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend8 as D8
from . import descend9 as D9
from .blind12 import load_arms, cases12, TESTS as T12
from .blind13 import cases13, TESTS as T13
from .evaluate_v8 import clopper_pearson
from .paired_report import paired

OUT = Path('runs/spike_steer/v14')
ART = Path('artifacts/spike_steer/plan_v14')
A11 = Path('artifacts/spike_steer/plan_v11')


class ClassicalUnsuppressed(T.Classical9):
    """The opponent, but the base-law pulses keep firing while a schedule segment is active."""

    def act(self, env, now, t, x, cmd, rel):
        pulse = False
        if self.steer:
            pulse, _ = self.steer.step(env, now, t)
        ref, _ = (self.schedule.ref9(rel, cmd) if self.schedule else (np.zeros(2), False))
        return pulse, ref


class NoPulse(D8.Learned8):
    def net_step(self, t, x):
        fire, ref = super().net_step(t, x)
        return False, ref


class NoRef(D8.Learned8):
    def net_step(self, t, x):
        fire, ref = super().net_step(t, x)
        return fire, np.zeros(D8.N_G)


def job(args):
    name, task, c, kind, arg = args
    if kind == 'classical_unsup':
        pol = ClassicalUnsuppressed(arg['law'], arg['schedule'])
    elif kind == 'nopulse':
        pol = NoPulse(arg)
    elif kind == 'noref':
        pol = NoRef(arg)
    elif kind == 'learned':
        pol = D8.Learned8(arg)
    else:
        pol = T.Classical9(arg['law'], arg['schedule'])
    return dict(T10.run_episode10(B._ENV, task, c, pol), controller=name)


def case_sets():
    K = T12['kick']
    return dict(nominal={t: T.cases9(t, 2_099_100_000, 64) for t in D9.TASKS},
                kick={t: cases12(t, K['base'], 64, K['condition'], True, K['mag']) for t in D9.TASKS},
                turb15={t: cases13(t, T13['turb15']) for t in D9.TASKS})


def stage_run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    arms, law = load_arms()
    opp = arms[0][2]
    cs = case_sets()
    specs = {n: s for n, k, s in arms if k == 'learned'}
    bc = {}
    for s in range(1, 6):
        f = (A11/f'snn/s{s}/bc.json') if s <= 3 else (A11/f'homepc_snn/s{s}/bc.json')
        if f.exists():
            bc[s] = json.loads(f.read_text())
    jobs = []
    for setname in ('nominal', 'kick', 'turb15'):                                  # d2
        jobs += [(f'{setname}/cls_unsup', t, c, 'classical_unsup', dict(law=law, schedule=opp[t][T.klass(c)])) for t in D9.TASKS for c in cs[setname][t]]
    for s in range(1, 6):                                                           # d3
        if f'snn11_s{s}' in specs:
            for kind in ('nopulse', 'noref'):
                jobs += [(f'kick/snn11_s{s}_{kind}', t, c, kind, specs[f'snn11_s{s}']) for t in D9.TASKS for c in cs['kick'][t]]
    for s in (2, 3, 4, 5):                                                          # d4 (seed 1 exists in v12/v13)
        if s in bc:
            for setname in ('kick', 'turb15'):
                jobs += [(f'{setname}/snn_bc_s{s}', t, c, 'learned', bc[s]) for t in D9.TASKS for c in cs[setname][t]]
    names = sorted({j[0] for j in jobs})
    done = {n for n in names if (OUT/(n.replace('/', '__')+'.json')).exists()}
    jobs = [j for j in jobs if j[0] not in done]
    print('episodes', len(jobs), 'arms', [n for n in names if n not in done], flush=True)
    with E.pool(a.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    for n in names:
        if n in done:
            continue
        rr = [r for r in rows if r['controller'] == n]
        (OUT/(n.replace('/', '__')+'.json')).write_text(json.dumps(rr))
        print('%-28s crashes %d' % (n, sum(not r['flown'] for r in rr)), flush=True)


def _rows(setname, arm):
    src = {'nominal': Path('runs/spike_steer/v11'), 'kick': Path('runs/spike_steer/v12/kick'), 'turb15': Path('runs/spike_steer/v13/turb15')}[setname]
    if setname == 'nominal':
        f = {'classical': src/'test_opponent.json', 'cls_es_s1': src/'evaluate_cls_es_s1_v11.json', 'cls_es_s2': A11/'homepc/evaluate_cls_es_s2_v11.json'}
        for s in range(1, 6):
            f[f'snn11_s{s}'] = (src/f'evaluate_snn_es_s{s}_v11.json') if s <= 3 else (A11/f'homepc/evaluate_snn_es_s{s}_v11.json')
            f[f'snn_bc_s{s}'] = (src/f'evaluate_snn_bc_s{s}_v11.json') if s <= 3 else (A11/f'homepc/evaluate_snn_bc_s{s}_v11.json')
        p = f.get(arm)
        if p is None or not p.exists():
            return None
        d = json.loads(p.read_text())
        return d if isinstance(d, list) else d['rows']
    p = src/f'rows_{arm}.json'
    return json.loads(p.read_text()) if p.exists() else None


def stage_report(a):
    ART.mkdir(parents=True, exist_ok=True)
    out = {}
    # d1: SNN vs classical + ES (seed 1) on every set
    d1 = {}
    sets = [('v11', 'nominal')]+[('v12', t) for t in ('kick2', 'kick', 'kick45', 'kick_early', 'kick_late', 'ood')]+[('v13', t) for t in ('turb125', 'turb15', 'noise5', 'delay3', 'kick_pre', 'kick_double', 'kick_rollaxis', 'kick_pitchaxis')]
    for plan, t in sets:
        if t == 'nominal':
            ref = _rows('nominal', 'cls_es_s1')
            arms_ = {f'snn11_s{s}': _rows('nominal', f'snn11_s{s}') for s in range(1, 6)}
        else:
            d = Path(f'runs/spike_steer/{plan}/{t}')
            ref = json.loads((d/'rows_cls_es_s1.json').read_text())
            arms_ = {f'snn11_s{s}': json.loads((d/f'rows_snn11_s{s}.json').read_text()) for s in range(1, 6)}
        d1[t] = {k: dict(delta=(p := paired(ref, v))['delta'], ci=p['delta_ci95'], crashes=p['crashes'], ref_crashes=p['classical_crashes']) for k, v in arms_.items() if v}
        print('d1 %-14s SNN vs classical+ES s1: ' % t + '  '.join('%s %+.3f [%+.2f,%+.2f]' % (k[-2:], v['delta'], *v['ci']) for k, v in d1[t].items()), flush=True)
    out['d1_snn_vs_cls_es_s1'] = d1
    # d2-d4 from the new rows
    res = {}
    for f in sorted(OUT.glob('*.json')):
        if f.name == 'summary.json':
            continue
        setname, arm = f.stem.split('__')
        rows = json.loads(f.read_text())
        opp = _rows(setname, 'classical')
        p = paired(opp, rows)
        res[f.stem] = dict(set=setname, arm=arm, crashes=p['crashes'], ci95=clopper_pearson(p['crashes'], len(rows)), classical_crashes=p['classical_crashes'],
                           delta=p['delta'], delta_ci95=p['delta_ci95'], wins=p['wins'], per_task={t_: p['per_task'][t_]['delta'] for t_ in D9.TASKS})
        print('%-28s crashes %3d (classical %3d)  Δ %+.3f [%+.3f, %+.3f]  wins %d' % (f.stem, p['crashes'], p['classical_crashes'], p['delta'], *p['delta_ci95'], p['wins']), flush=True)
    out['d2_d3_d4'] = res
    (OUT/'summary.json').write_text(json.dumps(out, indent=1))
    (ART/'summary.json').write_text(json.dumps(out, indent=1))


def stage_unsup_all(a):
    """Amendment 1: the un-suppressed opponent on every remaining v12 / v13 scenario (rows saved as <test>__cls_unsup.json)."""
    OUT.mkdir(parents=True, exist_ok=True)
    arms, law = load_arms()
    opp = arms[0][2]
    sets = {}
    for t, cfg in T12.items():
        if t in ('kick', 'ood') or (OUT/f'{t}__cls_unsup.json').exists():
            if t == 'ood' and not (OUT/f'{t}__cls_unsup.json').exists():
                sets[t] = {task: cases12(task, cfg['base'], 64, cfg['condition'], cfg['kick'], cfg.get('mag'), cfg.get('kick_t')) for task in D9.TASKS}
            continue
        sets[t] = {task: cases12(task, cfg['base'], 64, cfg['condition'], cfg['kick'], cfg.get('mag'), cfg.get('kick_t')) for task in D9.TASKS}
    for t, cfg in T13.items():
        if t in ('turb15', 'turb2', 'turb3') or (OUT/f'{t}__cls_unsup.json').exists():
            continue
        sets[t] = {task: cases13(task, cfg) for task in D9.TASKS}
    jobs = [(f'{t}/cls_unsup', task, c, 'classical_unsup', dict(law=law, schedule=opp[task][T.klass(c)])) for t, cs in sets.items() for task in D9.TASKS for c in cs[task]]
    print('unsup_all sets', list(sets), 'episodes', len(jobs), flush=True)
    with E.pool(a.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    for t in sets:
        rr = [r for r in rows if r['controller'] == f'{t}/cls_unsup']
        (OUT/f'{t}__cls_unsup.json').write_text(json.dumps(rr))
        src = Path('runs/spike_steer/v12')/t if t in T12 else Path('runs/spike_steer/v13')/t
        opp_rows = json.loads((src/'rows_classical.json').read_text())
        p = paired(opp_rows, rr)
        print('%-16s cls_unsup crashes %3d (classical %3d)  Δ %+.3f [%+.3f, %+.3f]  wins %d' % (t, p['crashes'], p['classical_crashes'], p['delta'], *p['delta_ci95'], p['wins']), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['run', 'report', 'unsup_all'])
    p.add_argument('--workers', type=int, default=30)
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    {'run': stage_run, 'report': stage_report, 'unsup_all': stage_unsup_all}[a.stage](a)


if __name__ == '__main__':
    main()
