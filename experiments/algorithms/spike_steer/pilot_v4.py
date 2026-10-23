"""Spike-timed steering v4: the v3 pipeline with the LIF time constants as evolvable genome-level genes.

Why (artifacts/spike_steer/v3/analysis.md, section 5): v3's fixed tau_m = 5 ms / tau_s = 2.5 ms cannot resolve
the 5 ms (45 deg) phase window, so the evolved circuit fired softly and late. A hand-built law-SNN with
tau = 0.5 ms passes the pre-registered J bar; the same network with v3's tau does not.

Changes relative to v3, and nothing else:
  * TauGenome: tau_m and tau_s are genome genes (log-space mutation, bounds 0.3-10 ms). The warm population
    starts spread log-uniformly over 0.5-5 ms; genome 0 is the exact v3 source (5 / 2.5 ms).
    A tau change is gain-preserving: all weights and biases are scaled by the ratio of the one-tick step
    responses s(tau_old)/s(tau_new), so the tau gene sets the temporal sharpness, not the effective gain.
    (Without this, v3 weights at tau = 0.5 ms see a 6.6x larger membrane response and over-fire; smoke run.)
  * The calibration grid reaches g_theta 2.5 and twitch 1 (v3's champion sat on the grid boundary).
  * D1 (warm-start choice) and law_ext are taken from the v3 run (same cases); the training, validation,
    selection and test case streams are those of v3 seed 3.
Pre-registered pass rule (written before the test stage): identical to v3.

    python -m experiments.algorithms.spike_steer.pilot_v4 selftest
    python -m experiments.algorithms.spike_steer.pilot_v4 all --quick --output runs/spike_steer/pilot_v4_quick
    python -m experiments.algorithms.spike_steer.pilot_v4 all --output runs/spike_steer/pilot_v4_s3 --seed 3
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import tempfile
import time
import neat
import numpy as np
from . import pilot as P
from . import pilot_v3 as V
from ..wingbeat_snn.evolve import DelayGenome, plain_tanh

V3 = V.ROOT/'artifacts/spike_steer/v3'
TAU_BOUNDS = (.0003, .01)        # s
TAU_WARM = (.0005, .005)         # warm population spread (log-uniform), genome 0 keeps the source's tau
TAU_MUTATE_RATE, TAU_MUTATE_POWER, TAU_DISTANCE = .3, .5, .5
CALIB_GRID = [dict(g_theta=a, g_q=b, d_out=c, twitch=d) for a in (1., 1.6, 2.5) for b in (.5, 1., 2.)
              for c in (-.15, 0., .15) for d in (1, 2, 3)]
OUTPUT_KEYS = 1
NEURAL_DT = .001  # wingbeat_snn.networks: 1 ms neural step, two substeps per 2 ms tick


def step_response(tau_m, tau_s):
    """Membrane voltage after one control tick (two 1 ms substeps) for a unit constant drive, from rest."""
    a_m, a_s = math.exp(-NEURAL_DT/tau_m), math.exp(-NEURAL_DT/tau_s)
    return (1.-a_m)*(a_m*(1.-a_s)+1.-a_s*a_s)


def set_tau(g, tau_m, tau_s):
    """Change a genome's time constants and rescale weights and biases so the one-tick response is unchanged."""
    k = step_response(g.tau_m, g.tau_s)/step_response(tau_m, tau_s)
    for n in g.nodes.values():
        n.bias *= k
    for c in g.connections.values():
        c.weight *= k
    g.tau_m, g.tau_s = float(tau_m), float(tau_s)


class TauGenome(DelayGenome):
    """DelayGenome plus two genome-level genes: the LIF membrane and synaptic time constants (seconds)."""

    def __init__(self, key):
        super().__init__(key)
        self.tau_m, self.tau_s = float(P.SPEC['tau_m']), float(P.SPEC['tau_s'])

    @staticmethod
    def clip(v):
        return float(min(TAU_BOUNDS[1], max(TAU_BOUNDS[0], v)))

    def configure_new(self, config):
        super().configure_new(config)
        self.tau_m = self.clip(math.exp(random.uniform(*map(math.log, TAU_WARM))))
        self.tau_s = self.clip(self.tau_m*random.uniform(.3, .7))

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.tau_m = float(random.choice((genome1, genome2)).tau_m)
        self.tau_s = float(random.choice((genome1, genome2)).tau_s)

    def mutate(self, config):
        super().mutate(config)
        new = [self.tau_m, self.tau_s]
        for i in range(2):
            if random.random() < TAU_MUTATE_RATE:
                new[i] = self.clip(new[i]*math.exp(random.gauss(0., TAU_MUTATE_POWER)))
        if new != [self.tau_m, self.tau_s]:
            set_tau(self, *new)

    def distance(self, other, config):
        tau = abs(math.log(self.tau_m/other.tau_m))+abs(math.log(self.tau_s/other.tau_s))
        return super().distance(other, config)+TAU_DISTANCE*tau


def make_config(out, population, outputs):
    text = (Path(V.__file__).with_name('neat_template_v3.ini').read_text()
            .replace('[DelayGenome]', '[TauGenome]')
            .replace('{population}', str(population)).replace('{inputs}', str(P.INPUTS))
            .replace('{outputs}', str(outputs)))
    path = Path(out)/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(TauGenome, neat.DefaultReproduction, V.AdaptiveSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def genome_from_spec(cfg, key, spec, direct_need=True):
    """pilot_v3.genome_from_spec for a TauGenome; tau comes from the spec (pilot.SPEC if absent)."""
    gc = cfg.genome_config
    g = TauGenome(key)
    for n in spec['nodes']:
        node = g.create_node(gc, n['id'])
        node.bias = float(n['bias'])
        g.nodes[n['id']] = node
    edges = [dict(e) for e in spec['edges']]
    if direct_need:
        have = {(e['source'], e['target']) for e in edges}
        for o in spec['outputs']:
            for i in V.NEED_INPUTS:
                if (-(i+1), o) not in have:
                    edges.append(dict(source=-(i+1), target=o, weight=0., delay=0., enabled=True))
    for e in edges:
        key2 = (e['source'], e['target'])
        conn = g.create_connection(gc, *key2, gc.innovation_tracker.get_innovation_number(*key2, 'shared_initial'))
        conn.weight, conn.delay, conn.enabled = float(e['weight']), float(e['delay']), bool(e['enabled'])
        g.connections[key2] = conn
    g.tau_m = g.clip(spec.get('tau_m', P.SPEC['tau_m']))
    g.tau_s = g.clip(spec.get('tau_s', P.SPEC['tau_s']))
    return g


def export(g, tied, calib=None):
    spec = dict(kind='snn', inputs=P.INPUTS, genome_id=g.key, tau_m=float(g.tau_m), tau_s=float(g.tau_s),
                spike_scale=P.SPEC['spike_scale'], outputs=list(range(OUTPUT_KEYS)),
                nodes=[dict(id=k, bias=float(n.bias)) for k, n in sorted(g.nodes.items())],
                edges=[dict(source=a, target=b, weight=float(e.weight), delay=float(round(e.delay)),
                            enabled=bool(e.enabled)) for (a, b), e in sorted(g.connections.items())])
    if tied:
        spec = V.mirror_spec(spec)
    if calib:
        spec['calib'] = dict(calib)
    return spec


def warm_population(pop, cfg, source, tied):
    """Genome 0 is the exact source; the others get a log-uniform tau and two gentle mutate() rounds."""
    gc = cfg.genome_config
    keys = sorted(pop.population)
    top = max([n['id'] for n in source['nodes']]+[31])
    gc.node_indexer = itertools.count(top+1)
    for i, k in enumerate(keys):
        g = genome_from_spec(cfg, k, source)
        if i > 0:
            tau_m = g.clip(math.exp(random.uniform(*map(math.log, TAU_WARM))))
            set_tau(g, tau_m, g.clip(tau_m*random.uniform(.3, .7)))
            for _ in range(2):
                g.mutate(gc)
        pop.population[k] = g
    distances = [pop.population[a].distance(pop.population[b], gc) for a, b in itertools.combinations(keys, 2)]
    cfg.species_set_config.compatibility_threshold = max(.02, float(np.median(distances)))
    pop.species = V.AdaptiveSpeciesSet(cfg.species_set_config, pop.reporters)
    pop.species.speciate(cfg, pop.population, 0)
    return cfg.species_set_config.compatibility_threshold


def tau_summary(genomes):
    tm = np.array([g.tau_m for _, g in genomes])*1e3
    ts = np.array([g.tau_s for _, g in genomes])*1e3
    best = max(genomes, key=lambda x: x[1].fitness if x[1].fitness is not None else -1e9)[1]
    return dict(best=[round(best.tau_m*1e3, 3), round(best.tau_s*1e3, 3)], median_m=round(float(np.median(tm)), 3),
                range_m=[round(float(tm.min()), 3), round(float(tm.max()), 3)], median_s=round(float(np.median(ts)), 3))


def stage_train(executor, out, a, choice):
    """pilot_v3.stage_train with TauGenome and a tau summary in every history row."""
    global OUTPUT_KEYS
    source = V.source_spec(choice['source'])
    OUTPUT_KEYS = len(source['outputs'])
    tied = choice['tied']
    random.seed(a.seed)
    cfg = make_config(out, a.population, OUTPUT_KEYS)
    pop = neat.Population(cfg)
    threshold0 = warm_population(pop, cfg, source, tied)
    validation = V.stratified(V.BASES['validation'], V.n(8), 6.)
    seen, validated, history, start = {}, [], [], time.monotonic()

    def score(genomes, config):
        generation = len(history)
        batch = V.stratified(V.BASES['train']+a.seed*100_000+generation*1_000, a.cases, 6.)
        specs = {gid: export(g, tied) for gid, g in genomes}
        rows = V.run_all(executor, [(str(gid), c, specs[gid], None, choice['lockout'])
                                    for gid, _ in genomes for c in batch])
        for k, (gid, g) in enumerate(genomes):
            js = [r['objective'] for r in rows[k*a.cases:(k+1)*a.cases]]
            g.fitness = -float(np.mean(js))
            seen.setdefault(gid, []).extend(js)
        row = dict(generation=generation, fitness_max=max(g.fitness for _, g in genomes),
                   fitness_mean=float(np.mean([g.fitness for _, g in genomes])), species=len(pop.species.species),
                   threshold=cfg.species_set_config.compatibility_threshold, wall_s=round(time.monotonic()-start, 1),
                   tau_ms=tau_summary(genomes))
        if generation % a.validate_every == a.validate_every-1:
            ranked = sorted(genomes, key=lambda x: (len(seen[x[0]]) < 8, float(np.mean(seen[x[0]]))
                                                    if len(seen[x[0]]) >= 8 else -x[1].fitness))
            top = [gid for gid, _ in ranked[:a.validate_top]]
            vrows = V.run_all(executor, [(str(gid), c, specs[gid], None, choice['lockout'])
                                         for gid in top for c in validation])
            for gid in top:
                s = V.stats([r for r in vrows if r['controller'] == str(gid)])
                validated.append(dict(generation=generation, genome=gid, flown=s['flown'], J=s['J'], spec=specs[gid]))
            row['validation'] = [dict(genome=v['genome'], flown=v['flown'], J=round(v['J'], 3),
                                      tau_ms=[round(1e3*v['spec']['tau_m'], 3), round(1e3*v['spec']['tau_s'], 3)])
                                 for v in validated if v['generation'] == generation]
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
    (out/'validated.json').write_text(json.dumps([dict(v, spec=None, tau_ms=[1e3*v['spec']['tau_m'], 1e3*v['spec']['tau_s']])
                                                  for v in validated], indent=1))
    return dict(candidates=[v['spec'] for v in best], source=export(genome_from_spec(cfg, -1, source, False), tied),
                threshold0=threshold0, tied=tied)


def stage_test(executor, out, calibrated, raw, law_ext, choice):
    rule = dict(only='snn_v4 (calibrated, validation-selected)',
                pass_rule=['flies 16/16 on the old 16-case test',
                           'J_snn <= 1.05 x J_law_tuned (v2 108-point law), both re-run here',
                           '>= 20 onsets for need-up and need-down; >= 70 % within +-67.5 deg of 135 / 315 deg'],
                robust_pass='the same on the fresh 16-case test (v3 test3 stream)',
                changes_from_v3=['tau_m, tau_s evolvable (0.3-10 ms)', 'calibration grid g_theta<=2.5, twitch>=1',
                                 'D1 choice and law_ext reused from v3'],
                registered_before_test=True, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out/'preregistration.json').write_text(json.dumps(rule, indent=1))
    law_tuned = json.loads(V.V2_LAW.read_text())['best']
    v3 = json.loads((V3/'selected_calibrated.json').read_text())
    lo = choice['lockout']
    arms = [('snn_v4', calibrated, None, lo), ('snn_v4_raw', raw, None, lo), ('snn_v3', v3, None, False),
            ('law_tuned', None, law_tuned, False), ('law_ext', None, law_ext, False), ('pid_retuned', None, None, False)]
    sets = dict(old16=P.cases('test', 16, 8.)[:V.n(16, 2)], fresh16=V.consecutive(V.BASES['test3'], V.n(16, 2), 8.))
    report, all_rows = {}, {}
    for label, cases in sets.items():
        rows = V.run_all(executor, [(a, c, s, l, x) for a, s, l, x in arms for c in cases])
        all_rows[label] = rows
        by = {a: [r for r in rows if r['controller'] == a] for a, *_ in arms}
        table = {a: V.stats(v) for a, v in by.items()}
        snn, law = table['snn_v4'], table['law_tuned']
        dirs = {a: V.directions(by[a]) for a in ('snn_v4', 'snn_v4_raw', 'snn_v3')}
        verdict = dict(flown=snn['flown'] == len(cases), J_ratio=snn['J']/law['J'], directions=dirs['snn_v4'])
        verdict['J_ok'] = verdict['J_ratio'] <= 1.05
        verdict['pass'] = bool(verdict['flown'] and verdict['J_ok'] and dirs['snn_v4']['pass'])
        report[label] = dict(table=table, directions=dirs,
                             paired_vs_law_tuned={a: V.paired(by[a], by['law_tuned']) for a in by if a != 'law_tuned'},
                             strata={a: V.strata(by[a]) for a in ('snn_v4', 'law_tuned')}, verdict=verdict)
        print(label, json.dumps({a: dict(flown=t['flown'], J=round(t['J'], 3), pitch=round(t['pitch_rms_deg'] or 0, 2),
                                         height=round(t['height_rmse_cm'] or 0, 1)) for a, t in table.items()}, indent=1),
              flush=True)
    report['overall'] = dict(PASS=report['old16']['verdict']['pass'], ROBUST_PASS=report['fresh16']['verdict']['pass'],
                             tau_ms=[1e3*calibrated['tau_m'], 1e3*calibrated['tau_s']], calib=calibrated.get('calib'))
    (out/'test.json').write_text(json.dumps(dict(report=report, old_rows=all_rows['old16'], fresh_rows=all_rows['fresh16']),
                                            indent=1))
    print(json.dumps(report['overall'], indent=1), flush=True)
    return report


def selftest():
    """No simulation: tau genes round-trip through spec export, stay in bounds under mutation, cross over, enter
    the distance, and a tau 0.5 ms spec drives SteerV3's network sharply (one-tick response)."""
    global OUTPUT_KEYS
    OUTPUT_KEYS = 2
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp, 8, 2)
        pop = neat.Population(cfg)  # its reproduction installs the innovation tracker on genome_config
        src = V.source_spec('v1_pair')
        g = genome_from_spec(cfg, 0, src)
        assert (g.tau_m, g.tau_s) == (P.SPEC['tau_m'], P.SPEC['tau_s'])
        spec = export(g, False)
        assert spec['tau_m'] == g.tau_m and spec['outputs'] == [0, 1] and len(spec['edges']) == len(g.connections)
        g2 = genome_from_spec(cfg, 1, dict(spec, tau_m=.0004, tau_s=.0003))
        assert (g2.tau_m, g2.tau_s) == (.0004, .0003)
        random.seed(0)
        for _ in range(200):
            g2.mutate(cfg.genome_config)
            assert TAU_BOUNDS[0] <= g2.tau_m <= TAU_BOUNDS[1] and TAU_BOUNDS[0] <= g2.tau_s <= TAU_BOUNDS[1]
        g.fitness, g2.fitness = -5., -6.
        child = TauGenome(2)
        child.configure_crossover(g, g2, cfg.genome_config)
        assert child.tau_m in (g.tau_m, g2.tau_m) and child.tau_s in (g.tau_s, g2.tau_s)
        d_same = g.distance(genome_from_spec(cfg, 3, spec), cfg.genome_config)
        d_tau = g.distance(genome_from_spec(cfg, 4, dict(spec, tau_m=spec['tau_m']/math.e)), cfg.genome_config)
        assert abs(d_tau-d_same-TAU_DISTANCE) < 1e-9, (d_same, d_tau)
        # gain-preserving tau change: same output voltage after one tick for a sub-threshold constant input
        assert abs(step_response(.005, .0025)-.1488) < 1e-3 and abs(step_response(.0005, .00025)-.979) < 1e-3
        g_slow = genome_from_spec(cfg, 5, src)
        g_fast = genome_from_spec(cfg, 6, src)
        set_tau(g_fast, .0005, .00025)
        x = np.full(P.INPUTS, .02)
        nets = [P.SpikingGraph(export(gg, False)) for gg in (g_slow, g_fast)]
        for net in nets:
            net.snn_substep(x), net.snn_substep(x)
            assert net.spikes == 0
        v = [net.voltage[net.outputs] for net in nets]
        assert np.allclose(v[0], v[1], rtol=1e-6, atol=1e-9), v
        # warm population: genome 0 exact, the rest spread over TAU_WARM
        warm_population(pop, cfg, src, False)
        taus = sorted(x.tau_m for x in pop.population.values())
        assert any(abs(t-P.SPEC['tau_m']) < 1e-12 for t in taus) and taus[0] < .002 and len(pop.species.species) <= 5
        # calibration grid contains v3's default entry (pilot_v3.stage_calibrate indexes it)
        assert dict(g_theta=1., g_q=1., d_out=0., twitch=3) in CALIB_GRID and len(CALIB_GRID) == 81
        # sharpness: a fast-tau output neuron driven at 3x threshold fires within one tick; the slow one does not
        for tau, expect in ((.0005, True), (.005, False)):
            net = P.SpikingGraph(dict(kind='snn', inputs=1, outputs=[0], tau_m=tau, tau_s=tau/2, spike_scale=5.,
                                      nodes=[dict(id=0, bias=0.)], edges=[dict(source=-1, target=0, weight=3., delay=0., enabled=True)]))
            fired = net.snn_substep(np.ones(1)) | net.snn_substep(np.ones(1))
            assert bool(fired[0]) == expect, (tau, fired)
    print('selftest ok')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['selftest', 'all'])
    p.add_argument('--output', default='runs/spike_steer/pilot_v4_s3')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=16)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=4)
    p.add_argument('--validate-top', type=int, default=3)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    V.QUICK = a.quick
    if a.quick:
        a.population, a.generations, a.validate_every, a.validate_top = 8, 2, 1, 2
    if a.stage == 'selftest':
        selftest()
        return
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    choice = json.loads((V3/'d1.json').read_text())['choice']          # v3 D1: untied v1 pair, no lockout
    law_ext = json.loads((V3/'law_ext.json').read_text())['best']       # v3 law_ext, same 8 tuning cases
    V.CALIB_GRID = CALIB_GRID                                           # pilot_v3.stage_calibrate reads this global
    print(json.dumps(dict(choice=choice, law_ext=law_ext)), flush=True)
    with P.pool(a.workers) as executor:
        trained = stage_train(executor, out, a, choice)
        raw = V.stage_select(executor, out, trained, choice)
        calibrated = V.stage_calibrate(executor, out, raw, choice)
        stage_test(executor, out, calibrated, raw, law_ext, choice)


if __name__ == '__main__':
    main()
