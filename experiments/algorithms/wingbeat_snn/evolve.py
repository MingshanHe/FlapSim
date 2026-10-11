"""NEAT with evolvable synaptic delays for wingbeat-synchronous controllers.

Arms (all IMU-only, phasor output around the PLL phase, same budget):
  snn_neat_delay  B5  LIF graph, structure + weights + integer delays (0-20 ms)
  snn_fixed       B4  LIF graph, fixed dense topology, weights only, delays 0
  ann_neat        B3  recurrent tanh graph, structure + weights, no delays

Every genome of a generation is scored on the same cases (paired evaluation).

    python -m experiments.algorithms.wingbeat_snn.evolve --method snn_neat_delay --seed 1 \
      --scratch <dir> --output runs/wingbeat_snn/pilot/snn_neat_delay_seed1
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import time
import neat
from neat.attributes import FloatAttribute
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig
import numpy as np
from .networks import MAX_DELAY
from .plant import A_MAX, SPLITS, cases
from . import evaluate as E

METHODS = dict(snn_neat_delay=dict(kind='snn', structure=True, delays=True, hidden=4),
               snn_fixed=dict(kind='snn', structure=False, delays=False, hidden=8),
               ann_neat=dict(kind='ann', structure=True, delays=False, hidden=4))
INPUTS, OUTPUTS = 11, 2  # IMU (8) + PLL sin, cos, notched pitch rate; phasor (a, b)


class DelayConnectionGene(DefaultConnectionGene):
    _gene_attributes = DefaultConnectionGene._gene_attributes+[
        FloatAttribute('delay', init_mean=0., init_stdev=0., min_value=0., max_value=float(MAX_DELAY),
                       mutate_rate=0., mutate_power=3., replace_rate=0.)]

    def distance(self, other, config):
        return super().distance(other, config)+.05*abs(self.delay-other.delay)


class DelayGenome(neat.DefaultGenome):
    @classmethod
    def parse_config(cls, params):
        params['node_gene_type'] = DefaultNodeGene
        params['connection_gene_type'] = DelayConnectionGene
        return DefaultGenomeConfig(params, cls.__name__)

    def mutate_add_node(self, config):
        if len(self.nodes) < 30:
            super().mutate_add_node(config)

    def mutate_add_connection(self, config):
        """Sources: inputs or hidden; targets: hidden or outputs; outputs only read hidden units."""
        hidden = sorted(set(self.nodes)-set(config.output_keys))
        if not hidden:
            return
        target = random.choice(sorted(self.nodes))
        source = random.choice(hidden if target in config.output_keys else list(config.input_keys)+hidden)
        key = (source, target)
        if key in self.connections:
            return
        innovation = config.innovation_tracker.get_innovation_number(source, target, 'add_connection')
        self.connections[key] = self.create_connection(config, source, target, innovation)


def plain_tanh(x):
    return math.tanh(x)


def make_config(out, method, population):
    m = METHODS[method]
    structural = dict(conn_add_prob=.2, conn_delete_prob=.05, node_add_prob=.1, node_delete_prob=.03,
                      enabled_mutate_rate=.01) if m['structure'] else \
        dict(conn_add_prob=0., conn_delete_prob=0., node_add_prob=0., node_delete_prob=0., enabled_mutate_rate=0.)
    text = f"""[NEAT]
fitness_criterion = max
fitness_threshold = 1e9
pop_size = {population}
reset_on_extinction = False
no_fitness_termination = True

[DelayGenome]
num_inputs = {INPUTS}
num_outputs = {OUTPUTS}
num_hidden = 0
feed_forward = False
initial_connection = unconnected
activation_default = plain_tanh
activation_mutate_rate = 0.0
activation_options = plain_tanh
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 0.1
bias_init_type = gaussian
bias_max_value = 3.0
bias_min_value = -3.0
bias_mutate_power = 0.05
bias_mutate_rate = 0.3
bias_replace_rate = 0.01
response_init_mean = 1.0
response_init_stdev = 0.0
response_init_type = gaussian
response_max_value = 1.0
response_min_value = 1.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 0.3
weight_init_type = gaussian
weight_max_value = 5.0
weight_min_value = -5.0
weight_mutate_power = 0.1
weight_mutate_rate = 0.6
weight_replace_rate = 0.01
enabled_default = True
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
single_structural_mutation = True
structural_mutation_surer = default
delay_mutate_rate = {.3 if m['delays'] else 0.}
delay_replace_rate = {.03 if m['delays'] else 0.}
{chr(10).join(f'{k} = {v}' for k, v in structural.items())}

[DefaultSpeciesSet]
compatibility_threshold = 2.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.3
min_species_size = 2
"""
    path = out/'neat.ini'
    path.write_text(text)
    cfg = neat.Config(DelayGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    return cfg


def initialize(pop, cfg, method, seed, readout_std=.05):
    """Shared start: hidden units read all inputs; genome 0 has zero readout (= the base controller)."""
    m = METHODS[method]
    rng = np.random.default_rng(seed)
    gc = cfg.genome_config
    hidden = list(range(OUTPUTS, OUTPUTS+m['hidden']))
    gc.node_indexer = itertools.count(OUTPUTS+m['hidden'])
    for index, g in enumerate(pop.population.values()):
        g.nodes.clear()
        g.connections.clear()
        for k in list(gc.output_keys)+hidden:
            node = g.create_node(gc, k)
            node.bias = 0. if k in gc.output_keys else (float(rng.normal(.8, .2)) if m['kind'] == 'snn'
                                                         else float(rng.normal(0., .1)))
            g.nodes[k] = node
        pairs = [(i, h) for i in gc.input_keys for h in hidden]+[(h, o) for h in hidden for o in gc.output_keys]
        if not m['structure']:
            pairs += [(a, b) for a in hidden for b in hidden]
        for source, target in pairs:
            innovation = gc.innovation_tracker.get_innovation_number(source, target, 'shared_initial')
            edge = g.create_connection(gc, source, target, innovation)
            readout = target in gc.output_keys
            edge.weight = 0. if (readout and index == 0) else float(rng.normal(0., readout_std if readout else .5))
            edge.delay = float(rng.integers(0, 9)) if m['delays'] else 0.
            g.connections[(source, target)] = edge
    pop.species.speciate(cfg, pop.population, 0)


def export(g, method, residual=False, spike_scale=1., residual_scale=.5):
    m = METHODS[method]
    extra = dict(residual_on='B2plus', residual_scale=residual_scale) if residual else {}
    return dict(kind=m['kind'], method=method, mode='phasor', inputs=INPUTS, outputs=list(range(OUTPUTS)),
                tau_m=.01, tau_s=.005, spike_scale=spike_scale if m['kind'] == 'snn' else 1., genome_id=g.key, **extra,
                nodes=[dict(id=k, bias=float(n.bias)) for k, n in sorted(g.nodes.items())],
                edges=[dict(source=a, target=b, weight=float(e.weight), delay=float(round(e.delay)),
                            enabled=bool(e.enabled)) for (a, b), e in sorted(g.connections.items())])


def complexity(spec):
    enabled = [e for e in spec['edges'] if e['enabled']]
    return dict(nodes=len(spec['nodes']), connections=len(enabled),
                delayed=sum(e['delay'] > 0 for e in enabled), mean_delay_ms=float(np.mean([e['delay'] for e in enabled])) if enabled else 0.)


def fitness(rows):
    """Mean v3 hover return per second, minus a small drive-activity cost (lift and power)."""
    return float(np.mean([r['return_sum']/r['seconds']-.2*r['drive_activity']/A_MAX for r in rows]))


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    cfg = make_config(out, args.method, args.population)
    pop = neat.Population(cfg)
    initialize(pop, cfg, args.method, args.seed, args.readout_std)
    validation = cases('validation', args.validation_cases, args.seconds)
    best, history, start = None, [], time.monotonic()
    with E.pool(Path(args.scratch)/'source', args.workers) as executor:
        def score(genomes, config):
            nonlocal best
            generation = len(history)
            batch = cases('training', args.cases, args.seconds, offset=args.seed*10_000+generation*args.cases)
            peaks = [args.levels[(generation*args.cases+i) % len(args.levels)] for i in range(args.cases)]
            specs = {gid: export(g, args.method, args.residual, args.spike_scale) for gid, g in genomes}
            tasks = [(dict(name=str(gid), kind='graph', policy=specs[gid], phase_inputs=True), c, p)
                     for gid, _ in genomes for c, p in zip(batch, peaks)]
            rows = list(executor.map(E.run_task, tasks))
            for k, (gid, g) in enumerate(genomes):
                g.fitness = fitness(rows[k*args.cases:(k+1)*args.cases])
            champion_id, champion = max(genomes, key=lambda x: x[1].fitness)
            row = dict(generation=generation, fitness_max=champion.fitness,
                       fitness_mean=float(np.mean([g.fitness for _, g in genomes])),
                       species=len(pop.species.species), peaks=peaks, wall_s=time.monotonic()-start,
                       champion=complexity(specs[champion_id]))
            if (generation+1) % args.validate_every == 0 or generation == 0:
                spec = specs[champion_id]
                vrows = E.evaluate(executor, [dict(name='champion', kind='graph', policy=spec, phase_inputs=True)],
                                   validation, args.validation_levels)
                key = [sum(r['termination'] == 'time_limit' for r in vrows), fitness(vrows)]
                row['validation'] = dict(completed=key[0], episodes=len(vrows), fitness=key[1])
                if best is None or key > best['key']:
                    best = dict(key=key, generation=generation, spec=spec)
                    (out/'best_policy.json').write_text(json.dumps(spec, indent=1))
            history.append(row)
            with (out/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        pop.run(score, args.generations)
    result = dict(method=args.method, seed=args.seed, generations=args.generations, population=args.population,
                  cases_per_generation=args.cases, seconds=args.seconds, levels=args.levels,
                  residual=args.residual, readout_std=args.readout_std, spike_scale=args.spike_scale,
                  selected_generation=best['generation'], selected_validation=best['key'],
                  selected_complexity=complexity(best['spec']), wall_s=time.monotonic()-start)
    (out/'results.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=sorted(METHODS), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--scratch', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--seconds', type=float, default=7.)
    p.add_argument('--levels', type=float, nargs='+', default=[0., 1., 1.5, 2.])
    p.add_argument('--validation-cases', type=int, default=8)
    p.add_argument('--validation-levels', type=float, nargs='+', default=[1.5, 2.])
    p.add_argument('--validate-every', type=int, default=5)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--residual', action='store_true', help='learn a bounded residual on the B2+ hand law')
    p.add_argument('--readout-std', type=float, default=.05)
    p.add_argument('--spike-scale', type=float, default=1.)
    train(p.parse_args())


if __name__ == '__main__':
    main()
