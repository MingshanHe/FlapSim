"""Import the exact actor into NEAT and freeze neuronal time constants."""
import copy
import itertools
import math
import random
import numpy as np
import neat
from ..snn_hover.evolution import configuration, export as legacy_export


def create_population(output, spec, seed, population, fixed_topology=False):
    random.seed(seed)
    kind = spec['kind']
    cfg = configuration(output, 'snn_fixed_tau' if kind == 'snn' else 'ann_neat', population)
    # Rebuild the NEAT config with the task's declared observation dimension.
    path = output/'neat.ini'
    text = path.read_text()
    import re
    text = re.sub(r'(?m)^num_inputs\s*=.*$', f"num_inputs = {spec['inputs']}", text)
    path.write_text(text)
    cfg = neat.Config(cfg.genome_type, cfg.reproduction_type, cfg.species_set_type,
                      cfg.stagnation_type, str(path))
    from ..topology_hover.evolve import plain_tanh
    cfg.genome_config.activation_defs.add('plain_tanh', plain_tanh)
    if fixed_topology:
        gc = cfg.genome_config
        gc.node_add_prob = gc.node_delete_prob = gc.conn_add_prob = gc.conn_delete_prob = 0.
        gc.enabled_mutate_rate = 0.
    pop = neat.Population(cfg)
    gc = cfg.genome_config
    gc.node_indexer = itertools.count(max(n['id'] for n in spec['nodes'])+1)
    rng = np.random.default_rng(seed)
    for index, genome in enumerate(pop.population.values()):
        genome.nodes.clear()
        genome.connections.clear()
        for node in spec['nodes']:
            gene = genome.create_node(gc, node['id'])
            gene.bias = node['bias'] + (float(rng.normal(0, .005)) if index else 0.)
            gene.log_tau_m = math.log(.02)
            gene.log_tau_s = math.log(.01)
            gene.activation = 'plain_tanh'
            gene.response = 1.
            genome.nodes[node['id']] = gene
        for edge in spec['edges']:
            a, b = edge['source'], edge['target']
            innovation = gc.innovation_tracker.get_innovation_number(a, b, 'shared_initial')
            gene = genome.create_connection(gc, a, b, innovation)
            gene.weight = edge['weight'] + (float(rng.normal(0, .005)) if index else 0.)
            gene.enabled = edge['enabled']
            genome.connections[(a, b)] = gene
        genome.parents = []
        genome.mutation_event = {}
    # Initial speciation must see imported weights/topology, not empty genomes.
    pop.species.speciate(cfg, pop.population, 0)
    return cfg, pop


def export(genome, kind, inputs=11):
    result = legacy_export(genome, 'snn_fixed_tau' if kind == 'snn' else 'ann_neat')
    result.update(inputs=inputs, format='flapsim-pid-compare-v1', kind=kind, control_dt=.002,
                  neural_dt=.001, substeps=2 if kind == 'snn' else 1, fixed_tau=True,
                  background=1.2, spike_scale=20.)
    return result
