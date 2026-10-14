"""Research plan v4, round 2: command-gated descending layer (pre-registered before any round-2 result).

Round 1 (descend.py finetune on the 4-task composite) traded hover for the manoeuvres: the ES centre reached dash
0.88-0.90 and nod ~1.1 but hover 1.37-1.50, so H1 failed. Round 2 keeps the three-layer architecture but gates the
learned layer the way insect descending neurons gate reflexes: the learned pulses and pitch reference act only while a
supervisor command is on (nod/dash) plus a short hand-back tail; otherwise the reflex PID + hover pulse law (the
classical hover controller, `base_law`) runs unchanged. Hover and upset therefore equal the classical controller by
construction (ratio 1.0 when base_law is the task's opponent) and the ES objective is the nod+dash composite only.

Stages (round-2 directories are built from round-1 outputs or from artifacts/spike_steer/plan_v4):
    python -m experiments.algorithms.spike_steer.descend_gated prepare  --output runs/spike_steer/bench_r2 [--source runs/spike_steer/bench]
    python -m experiments.algorithms.spike_steer.descend_gated finetune --output runs/spike_steer/bench_r2 --seed 1 --generations 60 --hours 6 --workers 30
    python -m experiments.algorithms.spike_steer.descend_gated evaluate --output runs/spike_steer/bench_r2 --seed 1 --variants intact,clamp_fast,nodelay_all,noref,nopulse
    python -m experiments.algorithms.spike_steer.descend_gated selftest
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_ann as A
from . import pilot_v3 as V

GATE = 'command'
TAIL_S = 1.0                     # learned layer stays in charge this long after the command drops (braking / hand-back)
WEIGHTS_R2 = dict(hover=.25, nod=.25, dash=.25, upset=.25)
ES_R2 = dict(cases_per_task=4, generations=60)
TRAIN_TASKS = ('nod', 'dash')


# ----------------------------------------------------------------------------------------- policy
class GatedLearned(B.Learned):
    """bench.Learned whose outputs are applied only while the supervisor command is on (+ TAIL_S); otherwise the
    reflex PID + pulse law `base_law` of the spec runs exactly as bench.Classical(law) does."""

    def reset(self):
        super().reset()
        law = self.spec.get('base_law')
        self.steer = V.SteerV3(None, law, False) if law else None
        if self.steer:
            self.steer.reset()
        self.off_t = None
        self.was_on = False
        self.tail = float(self.spec.get('tail_s', TAIL_S))

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = super().act(env, now, t, x, cmd, rel)          # net ticks every step (keeps its state)
        law_pulse = False
        if self.steer:
            law_pulse, _ = self.steer.step(env, now, t)             # pulse law ticks every step (keeps the PLL)
        on = bool(cmd.any())
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        if on or (self.off_t is not None and t-self.off_t < self.tail):
            return fire, ref
        return bool(law_pulse), 0.


_Learned = B.Learned


def dispatch_learned(spec):
    return GatedLearned(spec) if spec.get('gate') == GATE else _Learned(spec)


def _worker_init(source, dash_m):
    install()
    B.worker_init(source, dash_m)


def _pool(workers, dash_m=None):
    from concurrent.futures import ProcessPoolExecutor
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(str(source), dash_m if dash_m is not None else B.DASH_M['value']))


def install():
    """Gated dispatch (+ the ANN dispatch) in this process and in every pool worker (fork or spawn)."""
    B.Learned = dispatch_learned
    B.DescendingNet = A.dispatch
    B.pool = _pool


# ----------------------------------------------------------------------------------------- round-2 directories
def prepare(a):
    out = Path(a.output)
    src = Path(a.source) if a.source else None
    if src is None:
        for cand in ('runs/spike_steer/bench', 'artifacts/spike_steer/plan_v4'):
            if (Path(cand)/'baseline.json').exists():
                src = Path(cand)
                break
    base = json.loads((src/'baseline.json').read_text())
    (out/'upset_v2').mkdir(parents=True, exist_ok=True)
    (out/'train'/'upset_v2').mkdir(parents=True, exist_ok=True)
    for d in (out, out/'train'):
        shutil.copy(src/'upset_v2'/'calibrate.json', d/'upset_v2'/'calibrate.json')
    base_law_name = a.base_law
    base_law = base['arms_full']['hover'][base_law_name]['law'] if base_law_name == 'law_tuned' else base['arms_full']['upset'][base_law_name]['law']
    # evaluation baseline: same opponents, normalisers and test rows; round-2 weights
    ev = dict(base)
    ev['prereg'] = dict(base['prereg'], weights=dict(WEIGHTS_R2), round=2)
    (out/'baseline.json').write_text(json.dumps(ev))
    # training baseline: ES objective = nod/dash only
    tr = dict(base)
    tr['opponent'] = {t: base['opponent'][t] for t in TRAIN_TASKS}
    tr['prereg'] = dict(base['prereg'], weights={t: 1./len(TRAIN_TASKS) for t in TRAIN_TASKS}, round=2)
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    seeds = []
    for s in (1, 2, 3):
        f = src/f'imitate_spec_s{s}.json'
        if f.exists():
            spec = json.loads(f.read_text())
            spec.update(gate=GATE, tail_s=TAIL_S, base_law=base_law, base_law_name=base_law_name)
            (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
            seeds.append(s)
    prereg = dict(
        version='plan v4 round 2 (command-gated descending layer)', time=__import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        written_before='any round-2 fine-tune or blind test',
        weights=WEIGHTS_R2, weights_reason='four tasks weighted equally; set by the user before round 2 (round 1 used 40/20/20/20)',
        architecture=dict(gate=GATE, tail_s=TAIL_S, base_law=base_law_name,
                          note='learned pulses + pitch reference act only while the nod/dash command is on and for tail_s after it drops; '
                               'otherwise reflex PID + base_law pulse law run exactly as bench.Classical(law). Hover and upset are therefore '
                               'identical to the classical controller base_law (ratio 1.0 when base_law is the task opponent).'),
        training=dict(method='descend.py finetune (OpenAI-ES, pop 32, sigma .03, Adam .02)', objective='mean ratio over nod and dash on fresh training cases',
                      cases_per_task=ES_R2['cases_per_task'], generations=ES_R2['generations'], start='round-1 imitation spec of the same seed',
                      selection='lowest validation composite (nod/dash validation cases)'),
        test_sets='identical to round 1 (baseline.json test_rows / preregistration.json)',
        hypotheses=dict(
            H1='composite (25/25/25/25) <= 0.95 in >= 2 of 3 seeds AND T1 ratio <= 1.05 (T1 is 1.0 by construction when base_law = law_tuned)',
            H1_prime='non-inferiority + superiority: every task ratio <= 1.05 AND at least one of T2/T3 <= 0.90 with paired sign-test p < 0.05, in >= 2 of 3 seeds',
            H2='>= 2 of T2, T3, T4 <= 0.90 in >= 2 of 3 seeds (T4 is 1.0 by construction unless base_law = pulse_tuned, then T1 moves instead)',
            H3='ablations (clamp_fast, nodelay_all, noref, nopulse) raise the composite by >= 10 % or silence the pulses',
            H4='ANN of the same size through the same gated pipeline matches the SNN (no SNN performance claim)',
            secondary='per-case median ratio and paired win fraction are reported next to the mean ratio for every task (the nod mean is dominated by single misses)'),
        seeds=seeds, source=str(src))
    (out/'preregistration_r2.json').write_text(json.dumps(prereg, indent=1))
    print(json.dumps(dict(prepared=str(out), base_law=base_law_name, seeds=seeds, weights=WEIGHTS_R2)), flush=True)


def finetune(a):
    install()
    Dn.ES['cases_per_task'] = ES_R2['cases_per_task']
    a.output = str(Path(a.output)/'train')
    a.spec = a.spec or str(Path(a.output)/f'imitate_spec_s{a.seed}.json')
    Dn.finetune(a)


def evaluate(a):
    install()
    out = Path(a.output)
    a.spec = a.spec or str(out/'train'/f'finetune_s{a.seed}{a.tag_train}'/'spec_best.json')
    a.tag = a.tag or f'r2_s{a.seed}'
    B.stage_evaluate(a)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    install()
    spec = dict(Dn.DescendingSNN(hidden=2, seed=0).export(), gate=GATE, base_law=None, tail_s=.5)
    pol = B.Learned(spec)
    assert isinstance(pol, GatedLearned)
    pol.reset()
    x = np.zeros(B.N_IN)
    # command off: reflex only (no law -> no pulse, ref 0)
    assert pol.act(None, 0., 2., x, np.zeros(2), 0.) == (False, 0.)
    # command on: learned output passes through (ref may be nonzero)
    f, r = pol.act(None, 0., 2.002, x, np.array([1., 0.]), .002)
    assert isinstance(f, (bool, np.bool_)) and isinstance(r, float)
    # command off again: tail keeps the learned layer in charge for tail_s, then hands back
    pol.act(None, 0., 2.004, x, np.zeros(2), .004)
    assert pol.off_t == 2.004
    assert pol.act(None, 0., 2.6, x, np.zeros(2), .6) == (False, 0.)
    plain = B.Learned(Dn.DescendingSNN(hidden=2, seed=0).export())
    assert type(plain) is _Learned
    print('selftest ok (gated dispatch, tail hand-back, plain Learned untouched)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'prepare', 'finetune', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/bench_r2')
    p.add_argument('--source', default=None)
    p.add_argument('--base-law', default='law_tuned', choices=['law_tuned', 'pulse_tuned'])
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tag', default='')
    p.add_argument('--tag-train', default='')
    p.add_argument('--spec', default=None)
    p.add_argument('--variants', default='')
    p.add_argument('--generations', type=int, default=ES_R2['generations'])
    p.add_argument('--hours', type=float, default=6.)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    B.QUICK = a.quick
    if a.stage == 'finetune':
        a.tag = a.tag_train
    dict(selftest=lambda a: selftest(), prepare=prepare, finetune=finetune, evaluate=evaluate)[a.stage](a)


if __name__ == '__main__':
    main()
