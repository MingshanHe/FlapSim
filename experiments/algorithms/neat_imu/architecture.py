"""Explicit hidden-layer seeding and powered trim initial conditions."""
import copy
import numpy as np
from . import activation,robust
from .hover_initial import HoverInitialEnv


def init_worker(settings):
    activation.WORKER_MAPPING=activation.signed_map(settings['trim'])
    if 'model_path' in settings:
        from .mass_model import MassVariantEnv
        robust.ENV=MassVariantEnv(**settings)
    else:
        robust.ENV=HoverInitialEnv(**settings)
    robust.ResidualPolicy=activation.MappedPolicy


def trim_network(trim):
    payload=activation.gyro_network(activation.signed_map(trim),0.,0.,activation='clamped')
    payload['nodes'][0]['bias']=trim['command'][1]/.3
    payload['nodes'][1]['bias']=trim['command'][2]/.3
    return payload


def add_hidden(genome,config,count,rng,*,perturb=False):
    """Shared homologous 6 -> H -> 3 seed layer plus the existing skip graph.

    Zero outgoing weights preserve the seed controller exactly until mutation.
    A subset of initial genomes may use small outgoing weights for exploration.
    """
    if not count: return
    cfg=config.genome_config
    if not hasattr(cfg,'initial_hidden_keys'):
        cfg.initial_hidden_keys=[next(cfg.node_indexer) for _ in range(count)]
    if len(cfg.initial_hidden_keys)!=count: raise ValueError('hidden count changed within population')
    for index,key in enumerate(cfg.initial_hidden_keys):
        if key in genome.nodes: raise ValueError('initial hidden node collides with a parent')
        node=genome.create_node(cfg,key)
        node.activation,node.aggregation,node.bias,node.response='tanh','sum',0.,1.
        genome.nodes[key]=node
        for source in cfg.input_keys:
            edge=(source,key)
            innovation=cfg.innovation_tracker.get_innovation_number(*edge,'initial_hidden')
            gene=genome.create_connection(cfg,*edge,innovation)
            gene.weight=float(rng.normal(0.,.2)); gene.enabled=True
            genome.connections[edge]=gene
        for target in cfg.output_keys:
            edge=(key,target)
            innovation=cfg.innovation_tracker.get_innovation_number(*edge,'initial_hidden')
            gene=genome.create_connection(cfg,*edge,innovation)
            gene.weight=float(rng.normal(0.,.02)) if perturb else 0.
            gene.enabled=True; genome.connections[edge]=gene


def initialize(genome,config,mapping,rng,parent=None):
    activation.initialize(genome,config,mapping,rng,parent)
    add_hidden(genome,config,getattr(config.genome_config,'hidden_seed_count',0),rng,
               perturb=parent is None)
