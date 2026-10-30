"""Plan v15: generalisation of the frozen controllers to changed plants (no training, the arms of blind12.load_arms).
Every test flies the SAME stress cases (one seed base, as bench9 / blind13: stress condition, no hit) on a plant with one
or a few physical parameters changed; 'nominal' is the unchanged plant on those cases, so each arm's own degradation
is measured on identical seeds as well as its paired difference to the classical opponent on the same plant.

Plant parameters (applied once per worker, after the env is built; mass arrays through env.original_mass so that
every reset restores the changed plant):
  lower         base_link + pitch_link mass and inertia x f (the env's own lower_mass_scale; 'ood' = 1.125)
  com_dz        base_link centre of mass moved along body z by dz (m, negative = lower)
  wing_mass     both wings' mass and inertia x f
  span          (left, right) blade-element span x f   (aero only; the wing's mass properties are unchanged)
  chord         blade-element chord x f (planform leading / trailing edges scaled about the axis)
  feather       passive feathering-hinge range x f (wing rotation amplitude, a stiffness proxy)
  density       air density x f
  voltage       motor supply-voltage limit x f (battery sag)
  kt            motor torque constant and back-EMF x f
  tau           actuator lag (action_tau) x f

    python -m experiments.algorithms.spike_steer.blind15 selftest
    python -m experiments.algorithms.spike_steer.blind15 probe --test span95 --workers 19     # classical only, 16 cases
    python -m experiments.algorithms.spike_steer.blind15 run --test span95 --workers 19
    python -m experiments.algorithms.spike_steer.blind15 report --test span95
    python -m experiments.algorithms.spike_steer.blind15 table
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import json
from pathlib import Path
import numpy as np
from flapsim.rl.native import Wing, V3, pointer
from . import bench as B
from . import bench8 as E
from . import bench9 as T
from . import bench10 as T10
from . import descend9 as D9
from .blind12 import load_arms
from .blind13 import cases13
from .evaluate_v8 import clopper_pearson
from .paired_report import paired, ratios

OUT = Path('runs/spike_steer/v15')
ART = Path('artifacts/spike_steer/plan_v15')
BASE = 2_103_000_000
N = 64
TESTS = dict(
    nominal=dict(),
    lower125=dict(lower=1.25),
    lower_com=dict(lower=1.125, com_dz=-.003),
    wing_mass130=dict(wing_mass=1.3),
    span95=dict(span=(.95, .95)),
    span105=dict(span=(1.05, 1.05)),
    chord90=dict(chord=.9),
    asym97=dict(span=(.97, 1.)),
    feather80=dict(feather=.8),
    density90=dict(density=.9),
    volt90=dict(voltage=.9),
    kt90=dict(kt=.9),
    lag150=dict(tau=1.5),
    unit_b=dict(lower=1.125, wing_mass=1.15, span=(.97, .98), voltage=.95, tau=1.25, density=.95),
    # amendment 1 (2026-10-26, after the first results): dose points for the lower-body curve and smaller doses of the
    # two directions that were outside the thrust range
    lower110=dict(lower=1.10),
    lower1175=dict(lower=1.175),
    span102=dict(span=(1.02, 1.02)),
    feather90=dict(feather=.9),
)


def apply_plant(env, q):
    m = env.sim.model
    if 'com_dz' in q:
        m.body_ipos[m.body('base_link').id, 2] += q['com_dz']
    for name in ('left_wing', 'right_wing'):
        i = m.body(name).id
        env.original_mass[i] *= q.get('wing_mass', 1.)
        env.original_inertia[i] *= q.get('wing_mass', 1.)
    if 'feather' in q:
        for j in ('left_feathering_joint', 'right_feathering_joint'):
            m.jnt_range[m.joint(j).id] *= q['feather']
    be = env.sim.backend
    fl = be.fluid
    if 'span' in q or 'chord' in q or 'density' in q:
        wings = []
        for w, f in zip(fl.config.wings, q.get('span', (1., 1.))):
            c = q.get('chord', 1.)
            pf = [[r[0], r[1]*c, r[2]*c] for r in w.planform]
            wings.append(dataclasses.replace(w, span=w.span*f, chord=w.chord*c, chord_offset=w.chord_offset*c, planform=pf))
        fl.config = dataclasses.replace(fl.config, wings=tuple(wings), density=fl.config.density*q.get('density', 1.))
        fl.bind(m)
        be.arrays, records = [], []
        for wing, body, root, points, _, areas, rot in fl.elements:     # as NativeCPUBackend.__init__
            points, areas = np.ascontiguousarray(points), np.ascontiguousarray(areas)
            be.arrays.extend([points, areas])
            records.append(Wing(body, root, len(areas), pointer(points), pointer(areas), V3(*wing.normal_axis),
                                V3(*wing.chord_axis), V3(*wing.span_axis), rot))
        be.wings = (Wing*len(records))(*records)
    be.motor.voltage *= q.get('voltage', 1.)
    be.motor.kt *= q.get('kt', 1.)
    be.motor.ke *= q.get('kt', 1.)
    env.p['action_tau'] *= q.get('tau', 1.)


def _init(test):
    E._init(str(E.SOURCE), .15, dict(E.SIGNS))
    apply_plant(B._ENV, TESTS[test])


def pool(workers, test):
    if not (E.SOURCE/'robot_upper7_lower8.xml').exists():
        E.prepare_source(E.SOURCE)
    return ProcessPoolExecutor(workers, initializer=_init, initargs=(test,))


def cases15(task, test, n=N):
    q = TESTS[test]
    return cases13(task, dict(base=BASE, set=dict(lower_mass_scale=q['lower']) if 'lower' in q else {}), n)


def stage_probe(a):
    """Classical opponent only, 16 cases per task: is the changed plant still flyable (level calibration)."""
    _, law = load_arms()
    opp = load_arms()[0][0][2]
    with pool(a.workers, a.test) as ex:
        rows = T10.run_jobs(ex, [('classical', task, c, 'classical', dict(law=law, schedule=opp[task][T.klass(c)]), False)
                                 for task in D9.TASKS for c in cases15(task, a.test, 16)])
    print(a.test, TESTS[a.test], 'crashes', sum(not r['flown'] for r in rows), 'of', len(rows),
          'mean score', {t: round(float(np.mean([r['score'] for r in rows if r['task'] == t])), 3) for t in D9.TASKS}, flush=True)


def stage_run(a):
    from . import cache8 as _C8  # noqa: F401
    d = OUT/a.test
    d.mkdir(parents=True, exist_ok=True)
    arms, law = load_arms()
    want = set(a.arms.split(',')) if a.arms else None
    arms = [x for x in arms if (want is None or x[0] in want) and not (d/f'rows_{x[0]}.json').exists()]
    cs = {task: cases15(task, a.test) for task in D9.TASKS}
    (d/'cases.json').write_text(json.dumps(dict(plant=TESTS[a.test], cases=cs)))
    print(a.test, TESTS[a.test], 'arms', [x[0] for x in arms], flush=True)
    with pool(a.workers, a.test) as ex:
        for name, kind, payload in arms:
            if kind == 'classical':
                rows = T10.run_jobs(ex, [(name, task, c, 'classical', dict(law=law, schedule=payload[task][T.klass(c)]), False)
                                         for task in D9.TASKS for c in cs[task]])
                _save(d, name, rows)
        learned = [(name, spec) for name, kind, spec in arms if kind == 'learned']
        if learned:
            rows = T10.run_grouped(ex, [(name, task, c, spec) for name, spec in learned for task in D9.TASKS for c in cs[task]], chunk=len(learned))
            for name, _ in learned:
                _save(d, name, [r for r in rows if r['controller'] == name])


def _save(d, name, rows):
    (d/f'rows_{name}.json').write_text(json.dumps(rows))
    print(name, 'crashes', sum(not r['flown'] for r in rows), flush=True)


def _abs(rows, nom_opp):
    """Mean score ratio on the classical opponent's NOMINAL-plant medians (one scale across plants)."""
    med = {t: {cl: float(np.median([r['score'] for r in nom_opp if r['task'] == t and r['cls'] == cl])) for cl in T.CLASSES[t]} for t in D9.TASKS}
    a = ratios(rows, med)
    return float(np.mean([a[t].mean() for t in D9.TASKS]))


def stage_report(a):
    d, d0 = OUT/a.test, OUT/'nominal'
    opp = json.loads((d/'rows_classical.json').read_text())
    nom_opp = json.loads((d0/'rows_classical.json').read_text())
    oc = sum(not r['flown'] for r in opp)
    S = dict(plan='v15', test=a.test, plant=TESTS[a.test], cases_per_task=N, machine=a.machine,
             classical=dict(crashes=oc, n=len(opp), ci95=clopper_pearson(oc, len(opp)), abs=_abs(opp, nom_opp)))
    for f in sorted(d.glob('rows_*.json')):
        name = f.stem[len('rows_'):]
        if name == 'classical':
            continue
        rows = json.loads(f.read_text())
        p = paired(opp, rows)
        S[name] = dict(mean=p['mean'], classical_self=p['classical_self'], delta=p['delta'], delta_ci95=p['delta_ci95'], wins=p['wins'],
                       crashes=p['crashes'], ci95=clopper_pearson(p['crashes'], len(rows)), abs=_abs(rows, nom_opp),
                       crash_cases=[(r['task'], r['cls'], r['termination']) for r in rows if not r['flown']])
        f0 = d0/f.name
        if f0.exists() and a.test != 'nominal':
            S[name]['abs_nominal'] = _abs(json.loads(f0.read_text()), nom_opp)
        print('%-12s abs %.3f%s  Δ vs classical %+.3f [%+.3f, %+.3f]  wins %3d  crashes %2d' % (
            name, S[name]['abs'], '  (nominal %.3f)' % S[name]['abs_nominal'] if 'abs_nominal' in S[name] else '',
            p['delta'], *p['delta_ci95'], p['wins'], p['crashes']), flush=True)
    print('classical    abs %.3f  crashes %d' % (S['classical']['abs'], oc))
    (d/'summary.json').write_text(json.dumps(S, indent=1))
    ART.mkdir(parents=True, exist_ok=True)
    (ART/f'summary_{a.test}.json').write_text(json.dumps(S, indent=1))


def stage_table(a):
    arms = ['classical', 'cls_es_s1', 'cls_es_s2', 'snn11_s1', 'snn11_s2', 'snn11_s3', 'snn11_s4', 'snn11_s5']
    lines = ['| plant | change | ' + ' | '.join(arms) + ' |', '|' + '---|'*(len(arms)+2)]
    for t, q in TESTS.items():
        f = OUT/t/'summary.json'
        if not f.exists():
            continue
        S = json.loads(f.read_text())
        cells = ['%d / %.2f' % (S[x]['crashes'], S[x]['abs']) if x in S else '–' for x in arms]
        lines.append(f'| {t} | {q} | ' + ' | '.join(cells) + ' |')
    txt = '\n'.join(lines)
    print(txt)
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'crash_table.md').write_text('crashes of 192 / mean score ratio on the nominal-plant classical medians (lower is better)\n\n'+txt+'\n')


def selftest():
    """Each plant change reaches the physics: one 0.6 s classical hover per test, compared with nominal."""
    E.load_signs()
    _, law = load_arms()
    opp = load_arms()[0][0][2]
    ref = None
    for t in TESTS:
        c = cases15('dash', t, 1)[0]
        _init(t)
        env = B._ENV
        r = T10.job(('classical', 'dash', dict(c, seconds=.6), 'classical', dict(law=law, schedule=opp['dash'][T.klass(c)]), True))
        tr = np.array(r['trace'])
        sig = float(np.abs(tr[:, 1:10]).sum())
        print('%-13s %-70s trace |sum| %.6f  motor V %.3f kt %.6f tau %.4f wing span %.4f' % (
            t, TESTS[t], sig, env.sim.backend.motor.voltage, env.sim.backend.motor.kt, env.p['action_tau'], env.sim.backend.fluid.config.wings[0].span), flush=True)
        if t == 'nominal':
            ref = sig
        else:
            assert sig != ref, t
    print('blind15 selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'probe', 'run', 'report', 'table'])
    p.add_argument('--test', choices=list(TESTS), default='nominal')
    p.add_argument('--arms', default='')
    p.add_argument('--workers', type=int, default=19)
    p.add_argument('--machine', default='HomePC')
    a = p.parse_args()
    E.load_signs()
    B.DASH_M['value'] = .15
    if a.stage == 'selftest':
        selftest()
        return
    {'probe': stage_probe, 'run': stage_run, 'report': stage_report, 'table': stage_table}[a.stage](a)


if __name__ == '__main__':
    main()
