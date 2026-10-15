"""Research plan v4, round 3: command-gated descending layer + reflex envelope (pre-registered before any round-3 result).

Round 2 (descend_gated.py) won 15 of 16 blind cases per manoeuvre in every seed but lost the composite to single tilt
crashes: the learned layer, in sole charge while the nod/dash command is on, occasionally pitches past 76 deg and the
episode terminates at 80 deg. Round 2's righting reflex (amendment 1) only acts while the command is OFF. Round 3 adds
three things and changes nothing else (same sensors, actuators, control rate, limits, opponents, hover/upset layers):

  1. reflex envelope while the command is on: once the AHRS tilt exceeds `cmd_envelope_deg` the reflex layer takes
     over for the rest of the episode (upset-tuned pulse law, pitch reference 0), exactly as it does when the command
     is off -- a reflex that overrides the descending command (insect descending neurons are gated the same way).
  2. ES objective with a tilt margin: per task, mean score ratio + mean over cases of
     max(0, tilt_peak_deg - (cmd_envelope_deg - MARGIN_DEG)) / 10  (one ratio unit per 10 deg beyond the soft margin).
  3. selection (reselect_r3.py): lowest penalised composite on 16 validation cases among centres with zero crashes AND
     no envelope trigger (true tilt peak below cmd_envelope_deg on all 16 cases).
  Fresh blind set for nod/dash (round 2's blind set was looked at ~8 times); hover/upset blind sets unchanged.

Stages:
    python -m experiments.algorithms.spike_steer.descend_r3 diag     --workers 12 --cases 64 --envelopes none,45,50,60 --specs <spec>[,<spec>...]
    python -m experiments.algorithms.spike_steer.descend_r3 prepare  --output runs/spike_steer/bench_r3 --source runs/spike_steer/bench_r2 --envelope 50
    python -m experiments.algorithms.spike_steer.descend_r3 finetune --output runs/spike_steer/bench_r3 --seed 1 --generations 60 --hours 9 --workers 30
    python -m experiments.algorithms.spike_steer.descend_r3 evaluate --output runs/spike_steer/bench_r3 --seed 1 --spec <spec> --tag r3_s1 --variants intact,clamp_fast,nodelay_all,noref,nopulse
    python -m experiments.algorithms.spike_steer.descend_r3 selftest
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from . import bench as B
from . import descend as Dn
from . import descend_ann as A
from . import descend_gated as G
from . import maneuver_v2 as M

ENVELOPE_DEG = 50.               # default; the pre-registered value is written into every round-3 spec (cmd_envelope_deg)
MARGIN_DEG = 5.                  # soft margin below the envelope where the ES penalty starts
PENALTY_PER_DEG = .1             # ratio units per degree beyond the soft margin
TEST_BASE_R3 = dict(nod=2_096_400_000, dash=2_096_400_000)   # fresh blind seeds (round 2 test: 2_096_100_000; ES train < 2_096_310_000)
DIAG_OFFSET = 70_000             # fresh non-blind cases for the envelope diagnostic (ES: 100_000+; val: 60_000)
SEEDS = (1, 2, 3, 4, 5)
_cases0 = B.cases
_composite0 = Dn.composite_of


# ----------------------------------------------------------------------------------------- policy
class EnvelopedLearned(G.GatedLearned):
    """GatedLearned + reflex envelope: while the learned layer is in charge, an AHRS tilt above cmd_envelope_deg hands
    control to the reflex layer (upset law, reference 0) for the rest of the episode."""

    def reset(self):
        super().reset()
        self.env_thr = math.radians(float(self.spec['cmd_envelope_deg']))
        self.envelope = False
        self.envelope_t = None

    def act(self, env, now, t, x, cmd, rel):
        fire, ref = G._Learned.act(self, env, now, t, x, cmd, rel)   # net ticks every step (keeps its state)
        law_pulse = False
        if self.steer:
            law_pulse, _ = self.steer.step(env, now, t)
        on = bool(cmd.any())
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        learned_in_charge = on or (self.off_t is not None and t-self.off_t < self.tail)
        tilt = math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) if env is not None else 0.
        up_pulse = None
        if self.steer_up:
            up_pulse, _ = self.steer_up.step(env, now, t)
        if not self.upset and not learned_in_charge and tilt > self.tilt_thr:
            self.upset = True                                       # righting reflex (round-2 amendment 1)
        if not self.envelope and learned_in_charge and tilt > self.env_thr:
            self.envelope, self.envelope_t = True, t               # reflex envelope (round 3): latched
        if (self.upset or self.envelope) and up_pulse is not None:
            law_pulse = up_pulse
        if learned_in_charge and not self.envelope:
            return fire, ref
        return bool(law_pulse), 0.


def dispatch_learned(spec):
    if spec.get('gate') == G.GATE and spec.get('cmd_envelope_deg') is not None:
        return EnvelopedLearned(spec)
    return G.dispatch_learned(spec)


def cases_r3(task, split, levels=None, n=None, offset=0):
    if task in ('nod', 'dash') and split == 'test':
        return M.cases(TEST_BASE_R3[task], n or 16, task)
    return _cases0(task, split, levels, n, offset)


def composite_r3(rows, tasks, normalisers, weights):
    """Round-3 ES / validation objective: mean ratio + tilt-margin penalty (ratio units), per task, weighted."""
    scores = {t: float(np.mean([r['score'] for r in rows if r['task'] == t])) for t in tasks}
    pen = {}
    for t in tasks:
        rs = [r for r in rows if r['task'] == t]
        pen[t] = float(np.mean([max(0., (r.get('tilt_peak_deg') or 0.)-(ENVELOPE['value']-MARGIN_DEG))*PENALTY_PER_DEG for r in rs])) if rs else 0.
    tot = sum(weights.values())
    comp = float(sum(weights[t]*(scores[t]/normalisers[t]+pen[t]) for t in weights)/tot)
    return comp, scores


ENVELOPE = dict(value=ENVELOPE_DEG)   # set from the spec / prereg before finetune (penalty threshold)


def _worker_init(source, dash_m):
    install()
    B.worker_init(source, dash_m)


def _pool(workers, dash_m=None):
    from concurrent.futures import ProcessPoolExecutor
    source = Path('runs/neuro_gates/scratch/source')
    if not (source/'robot_upper7_lower8.xml').exists():
        B.prepare_source(source)
    return ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(str(source), dash_m if dash_m is not None else B.DASH_M['value']))


def install(fresh_test=True):
    """Round-3 dispatch in this process and in every pool worker; fresh nod/dash blind set; penalised ES objective."""
    B.Learned = dispatch_learned
    B.DescendingNet = A.dispatch
    B.pool = _pool
    B.cases = cases_r3 if fresh_test else _cases0
    Dn.composite_of = composite_r3


# ----------------------------------------------------------------------------------------- diagnostic (non-blind)
def diag(a):
    """Crash rate / mean ratio of existing round-2 centres with and without the envelope on fresh training-split cases
    (never the blind set). Used only to choose cmd_envelope_deg before the round-3 pre-registration."""
    install(fresh_test=False)
    out = Path(a.output)
    base_tr = json.loads((out/'train'/'baseline.json').read_text())
    norm = base_tr['normalisers']
    tasks = ('nod', 'dash')
    cases = {t: _cases0(t, 'train', None, n=a.cases, offset=Dn.COLLECT_OFFSET['val']+DIAG_OFFSET) for t in tasks}
    envs = [None if e == 'none' else float(e) for e in a.envelopes.split(',')]
    specs = [(Path(s).stem+'@'+Path(s).parent.name, json.loads(Path(s).read_text())) for s in a.specs.split(',')]
    report = []
    with B.pool(a.workers, base_tr['prereg']['dash_m'] or .25) as ex:
        for label, spec0 in specs:
            for e in envs:
                spec = dict(spec0)
                if e is None:
                    spec.pop('cmd_envelope_deg', None)
                else:
                    spec['cmd_envelope_deg'] = e
                rows = B.run_jobs(ex, [(label, t, c, 'learned', spec) for t in tasks for c in cases[t]])
                entry = dict(spec=label, envelope=e)
                for t in tasks:
                    rs = [r for r in rows if r['task'] == t]
                    fl = [r for r in rs if r['flown']]
                    entry[t] = dict(episodes=len(rs), crashes=sum(not r['flown'] for r in rs), ratio=float(np.mean([r['score'] for r in rs]))/norm[t],
                                    ratio_flown=float(np.mean([r['score'] for r in fl]))/norm[t] if fl else None,
                                    tilt_peak_max=float(max(r['tilt_peak_deg'] or 0. for r in rs)),
                                    over_envelope=sum((r['tilt_peak_deg'] or 0.) > (e or 1e9) for r in rs),
                                    hits=sum(r.get('t30' if t == 'nod' else 't_reach') is not None for r in rs))
                report.append(entry)
                print(json.dumps(entry), flush=True)
    Path(a.diag_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.diag_out).write_text(json.dumps(dict(cases=a.cases, offset=Dn.COLLECT_OFFSET['val']+DIAG_OFFSET, rows=report), indent=1))
    print('wrote', a.diag_out, flush=True)


# ----------------------------------------------------------------------------------------- round-3 directories
def prepare(a):
    install()
    ENVELOPE['value'] = a.envelope
    out, src = Path(a.output), Path(a.source)
    base_ev = json.loads((src/'baseline.json').read_text())
    base_tr = json.loads((src/'train'/'baseline.json').read_text())
    (out/'upset_v2').mkdir(parents=True, exist_ok=True)
    (out/'train'/'upset_v2').mkdir(parents=True, exist_ok=True)
    for d in (out, out/'train'):
        shutil.copy(src/'upset_v2'/'calibrate.json', d/'upset_v2'/'calibrate.json')
    # fresh nod/dash blind set: the classical arms are re-run on it (same arms, same tuning -> new test rows / normalisers)
    levels = B.upset_levels(out)
    tasks = list(base_ev['opponent'])
    test_rows, table = dict(base_ev['test_rows']), {t: dict(v) for t, v in base_ev['table'].items()}
    with B.pool(a.workers, base_ev['prereg']['dash_m'] or .25) as ex:
        for tname in ('nod', 'dash'):
            tc = B.cases(tname, 'test', levels, n=4 if B.QUICK else None)
            rows = B.run_jobs(ex, [(name, tname, c, 'classical', arg) for name, arg in base_ev['arms_full'][tname].items() for c in tc])
            test_rows[tname] = rows
            table[tname] = {name: B.summary([r for r in rows if r['controller'] == name]) for name in base_ev['arms_full'][tname]}
            for name, s in table[tname].items():
                print(f"{tname:6s} {name:13s} flown {s['flown']}/{s['episodes']} score {s['score']:.3f}", flush=True)
    normalisers = {t: table[t][base_ev['opponent'][t]]['score'] for t in tasks}
    ev = dict(base_ev, test_rows=test_rows, table=table, normalisers=normalisers)
    ev['prereg'] = dict(base_ev['prereg'], round=3, normalisers=normalisers, test_sets={t: [c['seed'] for c in B.cases(t, 'test', levels)] for t in tasks})
    (out/'baseline.json').write_text(json.dumps(ev))
    tr = dict(base_tr, normalisers={t: normalisers[t] for t in base_tr['opponent']})
    tr['prereg'] = dict(base_tr['prereg'], round=3, envelope_deg=a.envelope)
    (out/'train'/'baseline.json').write_text(json.dumps(tr))
    seeds = []
    for s in SEEDS:
        for f in (src/'train'/f'imitate_spec_s{s}.json', src/'train'/f'imitate_spec_s{s}.json.hold'):
            if f.exists():
                spec = json.loads(f.read_text())
                spec.update(cmd_envelope_deg=a.envelope)
                (out/'train'/f'imitate_spec_s{s}.json').write_text(json.dumps(spec, indent=1))
                seeds.append(s)
                break
    prereg = dict(
        version='plan v4 round 3 (command-gated descending layer + reflex envelope)', time=time.strftime('%Y-%m-%d %H:%M:%S'),
        written_before='any round-3 fine-tune or blind test',
        weights=G.WEIGHTS_R2,
        architecture=dict(gate=G.GATE, tail_s=G.TAIL_S, base_law=base_ev['prereg'].get('base_law', 'law_tuned'), upset_tilt_deg=G.UPSET_TILT_DEG,
                          cmd_envelope_deg=a.envelope,
                          note='as round 2 plus the reflex envelope: while the learned layer is in charge (command on or tail), an AHRS tilt '
                               'above cmd_envelope_deg hands control to the reflex layer (pulse_tuned law, pitch reference 0) for the rest of '
                               'the episode. Hover and upset are unchanged by construction (envelope acts only while the command is on).'),
        envelope_choice=a.envelope_note,
        training=dict(method='descend.py finetune (OpenAI-ES, pop 32, sigma .03, Adam .02), 4 cases per task per candidate, 60 generations',
                      objective=f'per task: mean score ratio + mean over cases of max(0, tilt_peak_deg - (cmd_envelope_deg - {MARGIN_DEG})) * {PENALTY_PER_DEG}; '
                                'nod and dash weighted equally (hover/upset are classical by construction)',
                      start='round-2 gated start spec of the same seed (imitation + base_law + upset law) with cmd_envelope_deg added',
                      selection='reselect_r3.py: lowest penalised composite on 16 validation cases (8 per task) among centres with zero crashes '
                                'and true tilt peak below cmd_envelope_deg on all 16 cases; the round-2 8-case rule is also reported',
                      machines='every search on CloudPC when possible; a HomePC search is a candidate source only (its centres are re-validated '
                               'and selected on CloudPC, because the same spec scores differently across machines - measured in round 2)'),
        test_sets=dict(nod_dash=f'fresh blind set, maneuver_v2 base {TEST_BASE_R3["nod"]} (16 cases per task; round 2 used 2_096_100_000)',
                       hover_upset='unchanged from rounds 1-2', normalisers=normalisers,
                       note='the classical opponents (same arms, same tuning) are re-run on the fresh set; the learned layer never sees it before the blind test'),
        hypotheses=dict(
            H1='composite (25/25/25/25) <= 0.95 in >= 3 of 5 seeds AND T1 ratio <= 1.05',
            H1_prime='non-inferiority + superiority: every task ratio <= 1.05 AND at least one of T2/T3 <= 0.90 with paired sign-test p < 0.05, in >= 3 of 5 seeds',
            H2='>= 2 of T2, T3, T4 <= 0.90 in >= 3 of 5 seeds',
            H3='ablations (clamp_fast, nodelay_all, noref, nopulse) raise the composite by >= 10 % or silence the pulses',
            H5='crash rate: 0 crashes in the 32 nod+dash blind episodes in >= 3 of 5 seeds (round 2: 1 crash per 16 in 4 of 5 selected centres)',
            secondary='per-case median ratio, paired win fraction and envelope-trigger count are reported next to the mean ratio for every task'),
        seeds=seeds, source=str(src))
    (out/'preregistration_r3.json').write_text(json.dumps(prereg, indent=1))
    print(json.dumps(dict(prepared=str(out), envelope=a.envelope, seeds=seeds, normalisers=normalisers)), flush=True)


def finetune(a):
    install()
    out = Path(a.output)
    ENVELOPE['value'] = json.loads((out/'train'/'baseline.json').read_text())['prereg']['envelope_deg']
    Dn.ES['cases_per_task'] = G.ES_R2['cases_per_task']
    a.output = str(out/'train')
    a.spec = a.spec or str(Path(a.output)/f'imitate_spec_s{a.seed}.json')
    Dn.finetune(a)


def evaluate(a):
    install()
    out = Path(a.output)
    a.spec = a.spec or str(out/'train'/f'finetune_s{a.seed}{a.tag_train}'/'spec_best.json')
    a.tag = a.tag or f'r3_s{a.seed}'
    B.stage_evaluate(a)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    install()

    class FakeEnv:
        last_reading = np.zeros(3)
    spec = dict(Dn.DescendingSNN(hidden=2, seed=0).export(), gate=G.GATE, base_law=None, tail_s=.5, cmd_envelope_deg=50.)
    pol = B.Learned(spec)
    assert isinstance(pol, EnvelopedLearned)
    pol.reset()
    env, x = FakeEnv(), np.zeros(B.N_IN)
    assert pol.act(env, 0., 2., x, np.zeros(2), 0.) == (False, 0.)                      # command off: reflex only
    f, r = pol.act(env, 0., 2.002, x, np.array([1., 0.]), .002)                       # command on: learned output
    assert isinstance(f, (bool, np.bool_)) and isinstance(r, float) and not pol.envelope
    env.last_reading = np.array([0., math.radians(55.), 0.])                           # tilt 55 > 50 while on -> envelope latched
    assert pol.act(env, 0., 2.004, x, np.array([1., 0.]), .004) == (False, 0.) and pol.envelope and pol.envelope_t == 2.004
    env.last_reading = np.zeros(3)
    assert pol.act(env, 0., 2.006, x, np.array([1., 0.]), .006) == (False, 0.)      # stays with the reflex layer
    gated = B.Learned(dict(spec, cmd_envelope_deg=None))
    assert type(gated) is G.GatedLearned
    plain = B.Learned(Dn.DescendingSNN(hidden=2, seed=0).export())
    assert type(plain) is G._Learned
    # penalised objective: 2 rows per task, tilt 40/60 with envelope 50 (soft 45): pen = (0 + 1.5)/2 = .75 on the second task only
    ENVELOPE['value'] = 50.
    rows = [dict(task='nod', score=1., tilt_peak_deg=40.), dict(task='nod', score=1., tilt_peak_deg=44.),
            dict(task='dash', score=2., tilt_peak_deg=40.), dict(task='dash', score=2., tilt_peak_deg=60.)]
    comp, sc = Dn.composite_of(rows, ['nod', 'dash'], dict(nod=1., dash=2.), dict(nod=.5, dash=.5))
    assert abs(comp-(1.+(1.+.75))/2) < 1e-9, comp
    # fresh blind set differs from round 2's
    assert B.cases('nod', 'test')[0]['seed'] != _cases0('nod', 'test')[0]['seed'] and len(B.cases('dash', 'test')) == 16
    assert B.cases('hover', 'test')[0]['seed'] == _cases0('hover', 'test')[0]['seed']
    print('selftest ok (envelope latch, dispatch, penalised objective, fresh blind set)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'diag', 'prepare', 'finetune', 'evaluate'])
    p.add_argument('--output', default='runs/spike_steer/bench_r3')
    p.add_argument('--source', default='runs/spike_steer/bench_r2')
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--tag', default='')
    p.add_argument('--tag-train', default='')
    p.add_argument('--spec', default=None)
    p.add_argument('--variants', default='')
    p.add_argument('--generations', type=int, default=G.ES_R2['generations'])
    p.add_argument('--hours', type=float, default=9.)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--envelope', type=float, default=ENVELOPE_DEG)
    p.add_argument('--envelope-note', default='')
    p.add_argument('--envelopes', default='none,45,50,60')
    p.add_argument('--cases', type=int, default=64)
    p.add_argument('--specs', default='')
    p.add_argument('--diag-out', default='runs/spike_steer/r3_diag/diag_envelope.json')
    a = p.parse_args()
    B.QUICK = a.quick
    if a.stage == 'finetune':
        a.tag = a.tag_train
    dict(selftest=lambda a: selftest(), diag=diag, prepare=prepare, finetune=finetune, evaluate=evaluate)[a.stage](a)


if __name__ == '__main__':
    main()
