"""Spike-timed steering v5: the v4 pipeline from NEAT's canonical start instead of the v1 warm start.

Why (artifacts/spike_steer/v4/analysis.md, sections 5-7): a two-neuron comparator SNN inside NEAT's gene bounds
ties the classical law and is robust under mutation, yet v1-v4 never found it. Their dense six-hidden-unit
initialisation plus the v1 warm start locked the population into a "slow-membrane anticipation" basin. The
canonical NEAT start (inputs wired directly to the outputs, no hidden units, small random weights) lies next
to the comparator basin.

Changes relative to v4, and nothing else:
  * population created by neat (initial_connection = full_direct, num_hidden = 0), tau log-uniform 0.5-5 ms;
  * mutation rates for a cold start (weights/biases: rate 0.5, power 0.5/0.3; init stdev 1.0; delays start at 0);
  * no D1 (nothing to warm-start); law_ext reused from v3; the case streams are those of v3/v4 seed 3.
Pre-registered (written before the test stage): the v3 pass rule, plus the prediction that the champion is
comparator-like (J <= 1.05 x law_tuned on the old 16 cases; tau_m of the champion <= 2 ms; <= 2 hidden units).

    python -m experiments.algorithms.spike_steer.pilot_v5 selftest
    python -m experiments.algorithms.spike_steer.pilot_v5 all --quick --output runs/spike_steer/pilot_v5_quick
    python -m experiments.algorithms.spike_steer.pilot_v5 all --output runs/spike_steer/pilot_v5_s3 --seed 3
"""
import argparse
import json
from pathlib import Path
import random
import re
import tempfile
import time
import neat
import numpy as np
from . import pilot as P
from . import pilot_v3 as V
from . import pilot_v4 as W
from ..wingbeat_snn.evolve import plain_tanh

V3 = W.V3
OUTPUTS = 2
N_OUT = 2  # output keys of the genome: 2 (untied pair) or 1 (--tied: the second pool is the mirror of the first)
COLD = {  # cold-start settings; everything else stays as in neat_template_v3.ini
    'initial_connection': 'full_direct', 'num_hidden': '0',
    # silent-at-rest prior: with bias ~ N(-2, 0.5) and weights ~ N(0, 0.5) a random genome fires sparsely instead
    # of continuously (the smoke run with bias N(0, 1), weights N(0, 1) crashed every genome by over-pulsing)
    'bias_init_mean': '-2.0', 'bias_init_stdev': '0.5', 'bias_mutate_power': '0.3', 'bias_mutate_rate': '0.5',
    'weight_init_stdev': '0.5', 'weight_mutate_power': '0.5', 'weight_mutate_rate': '0.5',
    'delay_init_mean': '0.0', 'delay_init_stdev': '0.0'}


def make_config(out, population, outputs=OUTPUTS):
    text = (Path(V.__file__).with_name('neat_template_v3.ini').read_text()
            .replace('[DelayGenome]', '[TauGenome]')
            .replace('{population}', str(population)).replace('{inputs}', str(P.INPUTS))
            .replace('{outputs}', str(outputs)))
    for key, value in COLD.items():
        text, n = re.subn(rf'^{key}\s*=.*$', f'{key} = {value}', text, flags=re.M)
        assert n == 1, key
    path = Path(out)/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(W.TauGenome, neat.DefaultReproduction, V.AdaptiveSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def silent_spec():
    """Two output neurons, no edges, bias -3: never fires, i.e. the retuned PID alone (reference arm)."""
    return dict(kind='snn', inputs=P.INPUTS, outputs=[0, 1], tau_m=.001, tau_s=.0005, spike_scale=P.SPEC['spike_scale'],
                nodes=[dict(id=0, bias=-3.), dict(id=1, bias=-3.)], edges=[])


def strip_output_loops(pop):
    """neat's full_direct start adds an output->output self-loop per output when feed_forward is False; the graph
    convention since v1 is that outputs are never sources (pilot_v3 mutate_add_connection), so drop them."""
    for g in pop.population.values():
        for key in [k for k in g.connections if k[0] >= 0 and k[0] < N_OUT]:
            del g.connections[key]


def structure(g):
    hidden = [k for k in g.nodes if k >= N_OUT]
    enabled = [c for c in g.connections.values() if c.enabled]
    return dict(hidden=len(hidden), edges=len(enabled), direct=sum(1 for c in enabled if c.key[0] < 0 and c.key[1] < N_OUT),
                delayed=sum(1 for c in enabled if round(c.delay) > 0))


def stage_train(executor, out, a):
    """pilot_v4.stage_train from a cold start: neat creates the population; history logs tau and structure."""
    global N_OUT
    random.seed(a.seed)
    tied = bool(getattr(a, 'tied', False))
    N_OUT = W.OUTPUT_KEYS = 1 if tied else OUTPUTS
    balanced = getattr(a, 'fitness', 'mean') == 'balanced'
    cfg = make_config(out, a.population, N_OUT)
    pop = neat.Population(cfg)  # TauGenome.configure_new: full_direct wiring, tau log-uniform over TAU_WARM
    strip_output_loops(pop)
    pop.species.speciate(cfg, pop.population, 0)
    validation = V.stratified(V.BASES['validation'], V.n(8), 6.)
    seen, validated, history, start = {}, [], [], time.monotonic()

    def score(genomes, config):
        generation = len(history)
        batch = V.stratified(V.BASES['train']+a.seed*100_000+generation*1_000, a.cases, 6.)
        specs = {gid: W.export(g, tied) for gid, g in genomes}
        rows = V.run_all(executor, [(str(gid), c, specs[gid], None, False) for gid, _ in genomes for c in batch])
        for k, (gid, g) in enumerate(genomes):
            js = [r['objective'] for r in rows[k*a.cases:(k+1)*a.cases]]
            # balanced: half mean, half worst case over the 4 sign-stratified cases, so that a one-direction
            # solution (v5 seed 3: push-up only, J ratio 1.30) cannot plateau on the mean
            g.fitness = -float(.5*np.mean(js)+.5*np.max(js)) if balanced else -float(np.mean(js))
            seen.setdefault(gid, []).extend(js)
        best = max(genomes, key=lambda x: x[1].fitness)[1]
        row = dict(generation=generation, fitness_max=best.fitness,
                   fitness_mean=float(np.mean([g.fitness for _, g in genomes])), species=len(pop.species.species),
                   threshold=cfg.species_set_config.compatibility_threshold, wall_s=round(time.monotonic()-start, 1),
                   tau_ms=W.tau_summary(genomes), best_structure=structure(best),
                   hidden_mean=float(np.mean([structure(g)['hidden'] for _, g in genomes])))
        if generation % a.validate_every == a.validate_every-1:
            ranked = sorted(genomes, key=lambda x: (len(seen[x[0]]) < 8, float(np.mean(seen[x[0]]))
                                                    if len(seen[x[0]]) >= 8 else -x[1].fitness))
            top = [gid for gid, _ in ranked[:a.validate_top]]
            vrows = V.run_all(executor, [(str(gid), c, specs[gid], None, False) for gid in top for c in validation])
            for gid in top:
                s = V.stats([r for r in vrows if r['controller'] == str(gid)])
                validated.append(dict(generation=generation, genome=gid, flown=s['flown'], J=s['J'], spec=specs[gid],
                                      structure=structure(dict(genomes)[gid])))
            row['validation'] = [dict(genome=v['genome'], flown=v['flown'], J=round(v['J'], 3),
                                      tau_ms=[round(1e3*v['spec']['tau_m'], 3), round(1e3*v['spec']['tau_s'], 3)],
                                      **v['structure']) for v in validated if v['generation'] == generation]
        k = len(pop.species.species)
        factor = 1.5 if k > 10 else 1.3 if k > 6 else 1.15 if k > 4 else (.87 if k < 3 else 1.)
        cfg.species_set_config.compatibility_threshold = float(np.clip(
            cfg.species_set_config.compatibility_threshold*factor, .02, 4.))
        history.append(row)
        with (out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    pop.run(score, a.generations)
    unique = {}
    for v in sorted(validated, key=lambda v: (-v['flown'], v['J'])):
        unique.setdefault(v['genome'], v)
    best = list(unique.values())[:3]
    (out/'validated.json').write_text(json.dumps([dict(v, spec=None) for v in validated], indent=1))
    return dict(candidates=[v['spec'] for v in best], source=silent_spec(), tied=tied)


def stage_test(executor, out, calibrated, raw, law_ext):
    rule = dict(only='snn_v5 (calibrated, validation-selected)',
                pass_rule=['flies 16/16 on the old 16-case test',
                           'J_snn <= 1.05 x J_law_tuned (v2 108-point law), both re-run here',
                           '>= 20 onsets for need-up and need-down; >= 70 % within +-67.5 deg of 135 / 315 deg'],
                robust_pass='the same on the fresh 16-case test (v3 test3 stream)',
                prediction=['the champion is comparator-like: tau_m <= 2 ms and <= 2 hidden units',
                            'J ratio <= 1.05 on the old 16 cases (the in-bounds hand-built comparator scored 1.021)'],
                changes_from_v4=['canonical NEAT start (full_direct, no hidden units, no warm start)',
                                 'cold-start mutation rates (COLD)', 'no D1; law_ext reused from v3'],
                search_fix=dict(fitness=getattr(stage_test, 'fitness', 'mean'), tied=bool(calibrated.get('tied', False))),
                registered_before_test=True, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out/'preregistration.json').write_text(json.dumps(rule, indent=1))
    law_tuned = json.loads(V.V2_LAW.read_text())['best']
    v3 = json.loads((V3/'selected_calibrated.json').read_text())
    arms = [('snn_v5', calibrated, None, False), ('snn_v5_raw', raw, None, False), ('snn_v3', v3, None, False),
            ('law_tuned', None, law_tuned, False), ('law_ext', None, law_ext, False), ('pid_retuned', None, None, False)]
    sets = dict(old16=P.cases('test', 16, 8.)[:V.n(16, 2)], fresh16=V.consecutive(V.BASES['test3'], V.n(16, 2), 8.))
    report, all_rows = {}, {}
    for label, cases in sets.items():
        rows = V.run_all(executor, [(a, c, s, l, x) for a, s, l, x in arms for c in cases])
        all_rows[label] = rows
        by = {a: [r for r in rows if r['controller'] == a] for a, *_ in arms}
        table = {a: V.stats(v) for a, v in by.items()}
        snn, law = table['snn_v5'], table['law_tuned']
        dirs = {a: V.directions(by[a]) for a in ('snn_v5', 'snn_v5_raw', 'snn_v3')}
        verdict = dict(flown=snn['flown'] == len(cases), J_ratio=snn['J']/law['J'], directions=dirs['snn_v5'])
        verdict['J_ok'] = verdict['J_ratio'] <= 1.05
        verdict['pass'] = bool(verdict['flown'] and verdict['J_ok'] and dirs['snn_v5']['pass'])
        report[label] = dict(table=table, directions=dirs,
                             paired_vs_law_tuned={a: V.paired(by[a], by['law_tuned']) for a in by if a != 'law_tuned'},
                             strata={a: V.strata(by[a]) for a in ('snn_v5', 'law_tuned')}, verdict=verdict)
        print(label, json.dumps({a: dict(flown=t['flown'], J=round(t['J'], 3), pitch=round(t['pitch_rms_deg'] or 0, 2),
                                         height=round(t['height_rmse_cm'] or 0, 1)) for a, t in table.items()}, indent=1),
              flush=True)
    hidden = [n['id'] for n in calibrated['nodes'] if n['id'] not in calibrated['outputs']]
    report['overall'] = dict(PASS=report['old16']['verdict']['pass'], ROBUST_PASS=report['fresh16']['verdict']['pass'],
                             tau_ms=[1e3*calibrated['tau_m'], 1e3*calibrated['tau_s']], hidden=len(hidden),
                             edges=sum(e['enabled'] for e in calibrated['edges']), calib=calibrated.get('calib'),
                             prediction_ok=bool(1e3*calibrated['tau_m'] <= 2. and len(hidden) <= 2
                                                and report['old16']['verdict']['J_ok']))
    (out/'test.json').write_text(json.dumps(dict(report=report, old_rows=all_rows['old16'], fresh_rows=all_rows['fresh16']),
                                            indent=1))
    print(json.dumps(report['overall'], indent=1), flush=True)
    return report


def selftest():
    """No simulation: cold-start config, population structure, tau range, export round trip, silent reference."""
    W.OUTPUT_KEYS = OUTPUTS
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp, 16)
        text = (Path(tmp)/'neat.ini').read_text()
        assert 'initial_connection = full_direct' in text and 'num_hidden = 0' in text and '[TauGenome]' in text
        random.seed(0)
        pop = neat.Population(cfg)
        assert any(k[0] in (0, 1) for g in pop.population.values() for k in g.connections)  # neat adds output loops
        strip_output_loops(pop)
        pop.species.speciate(cfg, pop.population, 0)
        gs = list(pop.population.values())
        assert len(gs) == 16
        for g in gs:
            st = structure(g)
            assert st['hidden'] == 0 and st['edges'] == P.INPUTS*OUTPUTS and st['direct'] == P.INPUTS*OUTPUTS, st
            assert W.TAU_WARM[0]-1e-12 <= g.tau_m <= W.TAU_WARM[1]+1e-12 and g.tau_s < g.tau_m
            assert all(abs(c.weight) <= 5. for c in g.connections.values()) and all(abs(n.bias) <= 3. for n in g.nodes.values())
        taus = sorted(g.tau_m for g in gs)
        assert taus[0] < .001 and taus[-1] > .003, taus  # spread over the warm range
        spec = W.export(gs[0], False)
        assert spec['outputs'] == [0, 1] and len(spec['edges']) == 24 and all(n['id'] in (0, 1) for n in spec['nodes'])
        net = P.SpikingGraph(spec)
        for _ in range(10):
            net.snn_substep(np.random.default_rng(1).random(P.INPUTS))
        # a few mutate rounds: genes stay finite (a gain-preserving tau change may push nominal weights past the
        # bounds; neat clamps them at that gene's next mutation) and outputs never become sources
        for g in gs:
            for _ in range(5):
                g.mutate(cfg.genome_config)
            assert all(np.isfinite(c.weight) for c in g.connections.values()) and all(np.isfinite(n.bias) for n in g.nodes.values())
            assert not any(0 <= k[0] < OUTPUTS for k in g.connections)
        # tied path: one output key, mirrored at export into two pools
        global N_OUT
        N_OUT = W.OUTPUT_KEYS = 1
        cfg1 = make_config(tmp, 8, 1)
        pop1 = neat.Population(cfg1)
        strip_output_loops(pop1)
        g1 = next(iter(pop1.population.values()))
        assert structure(g1)['direct'] == P.INPUTS and structure(g1)['hidden'] == 0
        m = W.export(g1, True)
        assert m['tied'] and m['outputs'] == [0, V.MIRROR_OFFSET] and len(m['edges']) == 2*P.INPUTS
        st = V.SteerV3(m)
        st.reset()
        for _ in range(20):
            st.net.snn_substep(np.random.default_rng(2).random(P.INPUTS))
        N_OUT = W.OUTPUT_KEYS = OUTPUTS
        s = silent_spec()
        net = P.SpikingGraph(s)
        fired = np.zeros(2, bool)
        for _ in range(50):
            fired |= net.snn_substep(np.ones(P.INPUTS))
        assert not fired.any()
        assert len(pop.species.species) <= V.AdaptiveSpeciesSet.MAX_SPECIES
    print('selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all'])
    p.add_argument('--output', default='runs/spike_steer/pilot_v5_s3')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=16)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=4)
    p.add_argument('--validate-top', type=int, default=3)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--fitness', default='mean', choices=['mean', 'balanced'], help='balanced: 0.5 mean + 0.5 worst case per batch')
    p.add_argument('--tied', action='store_true', help='mirror-tied antagonist pair: evolve one pool, mirror it')
    a = p.parse_args()
    V.QUICK = a.quick
    stage_test.fitness = a.fitness
    if a.quick:
        a.population, a.generations, a.validate_every, a.validate_top = 8, 2, 1, 2
    if a.stage == 'selftest':
        selftest()
        return
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    law_ext = json.loads((V3/'law_ext.json').read_text())['best']
    V.CALIB_GRID = W.CALIB_GRID
    choice = dict(tied=a.tied, source='cold', lockout=False)
    print(json.dumps(dict(choice=choice, fitness=a.fitness, law_ext=law_ext, cold=COLD)), flush=True)
    with P.pool(a.workers) as executor:
        trained = stage_train(executor, out, a)
        raw = V.stage_select(executor, out, trained, choice)
        calibrated = V.stage_calibrate(executor, out, raw, choice)
        stage_test(executor, out, calibrated, raw, law_ext)


if __name__ == '__main__':
    main()
