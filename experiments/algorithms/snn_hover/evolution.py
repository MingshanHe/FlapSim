"""NEAT genotype with audited log-time-constant inheritance and mutations."""
import itertools,math,random
from pathlib import Path
import neat
import numpy as np
from neat.attributes import FloatAttribute
from neat.genes import DefaultNodeGene,DefaultConnectionGene
from neat.genome import DefaultGenomeConfig
from neat.config import ConfigParameter,DefaultClassConfig
from ..topology_hover.evolve import AuditGenome,plain_tanh

METHODS=['snn_neat','snn_fixed_tau','snn_fixed_topology','leaky_ann_neat','ann_neat']

class CapacitySpeciesSet(neat.DefaultSpeciesSet):
    """Keep at least half the population available for offspring at elitism2.

    Default NEAT preserves an existing representative even after raising the
    compatibility threshold. A hard population-capacity invariant therefore
    needs explicit merges, not repeated threshold tuning. Only species labels
    change; every genome and all its genes remain intact.
    """
    @classmethod
    def parse_config(cls,params):
        return DefaultClassConfig(params,[ConfigParameter('compatibility_threshold',float),ConfigParameter('max_species',int)],cls.__name__)
    def speciate(self,config,population,generation):
        super().speciate(config,population,generation);self.merge_events=[]
        while len(self.species)>self.species_set_config.max_species:
            pairs=itertools.combinations(sorted(self.species),2)
            distance,keep,drop=min((self.species[a].representative.distance(self.species[b].representative,config.genome_config),a,b) for a,b in pairs)
            a,b=self.species[keep],self.species.pop(drop)
            moved=sorted(b.members);a.members.update(b.members);a.last_improved=max(a.last_improved,b.last_improved)
            for gid in moved:self.genome_to_species[gid]=keep
            self.merge_events.append(dict(speciation_generation=generation,keep=keep,drop=drop,representative_distance=distance,moved_genomes=moved))

class TemporalNode(DefaultNodeGene):
    _gene_attributes=DefaultNodeGene._gene_attributes+[
        FloatAttribute('log_tau_m',init_mean=math.log(.02),init_stdev=0.,min_value=math.log(.005),max_value=math.log(.2),mutate_rate=.2,mutate_power=.25,replace_rate=.01),
        FloatAttribute('log_tau_s',init_mean=math.log(.01),init_stdev=0.,min_value=math.log(.002),max_value=math.log(.1),mutate_rate=.2,mutate_power=.25,replace_rate=.01)]
    def distance(self,other,config):
        return super().distance(other,config)+.5*(abs(self.log_tau_m-other.log_tau_m)/math.log(40)+abs(self.log_tau_s-other.log_tau_s)/math.log(50))

class TemporalGenome(AuditGenome):
    @classmethod
    def parse_config(cls,params):
        params['node_gene_type']=TemporalNode;params['connection_gene_type']=DefaultConnectionGene
        return DefaultGenomeConfig(params,cls.__name__)
    def mutate_add_node(self,config):
        if len(self.nodes)<35:super().mutate_add_node(config)
    def mutate_add_connection(self,config):
        hidden=sorted(set(self.nodes)-set(config.output_keys))
        if not hidden:return
        target=random.choice(sorted(self.nodes))
        source=random.choice(hidden if target in config.output_keys else list(config.input_keys)+hidden)
        key=(source,target)
        if key in self.connections:
            if config.check_structural_mutation_surer():self.connections[key].enabled=True
            return
        inn=config.innovation_tracker.get_innovation_number(source,target,'add_connection')
        self.connections[key]=self.create_connection(config,source,target,inn)
    def mutate(self,config):
        before={k:(n.bias,n.log_tau_m,n.log_tau_s) for k,n in self.nodes.items()}
        super().mutate(config)
        self.mutation_event['changed_nodes']=[dict(id=k,before=list(before[k]),after=[n.bias,n.log_tau_m,n.log_tau_s]) for k,n in self.nodes.items() if k in before and before[k]!=(n.bias,n.log_tau_m,n.log_tau_s)]

def configuration(out,method,population):
    base=(Path(__file__).parents[1]/'neat_imu/config.ini').read_text();fixed=method=='snn_fixed_topology';ft=method in ('snn_fixed_tau','ann_neat')
    change=dict(pop_size=population,num_inputs=11,feed_forward='False',initial_connection='unconnected',activation_default='plain_tanh',activation_options='plain_tanh',
        bias_init_stdev=.01,bias_mutate_power=.04,weight_init_stdev=.15,weight_mutate_power=.1,
        conn_add_prob=0 if fixed else .30,conn_delete_prob=0 if fixed else .06,node_add_prob=0 if fixed else .20,node_delete_prob=0 if fixed else .03,
        enabled_mutate_rate=0 if fixed else .015,compatibility_threshold=1.,max_stagnation=15)
    lines=[]
    for line in base.splitlines():
        key=line.split('=')[0].strip();lines.append(f'{key} = {change[key]}' if key in change else line)
    text='\n'.join(lines).replace('[DefaultGenome]','[TemporalGenome]')
    additions='\n'.join(f'log_tau_{tag}_{key} = {val}' for tag in ['m','s'] for key,val in [('mutate_rate',0 if ft else .2),('replace_rate',0 if ft else .01)])
    text=text.replace('[DefaultSpeciesSet]',additions+f'\n\n[CapacitySpeciesSet]\nmax_species = {max(1,population//4)}')
    p=out/'neat.ini';p.write_text(text)
    cfg=neat.Config(TemporalGenome,neat.DefaultReproduction,CapacitySpeciesSet,neat.DefaultStagnation,str(p))
    cfg.genome_config.activation_defs.add('plain_tanh',plain_tanh);return cfg

def initialize(pop,cfg,seed):
    rng=np.random.default_rng(seed);gc=cfg.genome_config;hidden=list(range(3,7));gc.node_indexer=itertools.count(7)
    for index,g in enumerate(pop.population.values()):
        for k in hidden:g.nodes[k]=g.create_node(gc,k)
        for target in sorted(g.nodes):
            sources=hidden if target in gc.output_keys else list(gc.input_keys)+[target]
            for source in sources:
                inn=gc.innovation_tracker.get_innovation_number(source,target,'shared_initial')
                edge=g.create_connection(gc,source,target,inn);edge.weight=float(rng.normal(0,.02 if target in gc.output_keys else .3))
                if index==0 and target in gc.output_keys:edge.weight=0.
                g.connections[(source,target)]=edge
            if index==0:g.nodes[target].bias=0.
        g.parents=[];g.mutation_event={}

def export(g,method):
    return dict(kind='graph' if method=='ann_neat' else 'leaky_ann' if method=='leaky_ann_neat' else 'snn',inputs=11,outputs=[0,1,2],
        genome_id=g.key,fixed_tau=method in ('snn_fixed_tau','ann_neat'),method=method,
        nodes=[dict(id=k,bias=n.bias,tau_m=math.exp(n.log_tau_m),tau_s=math.exp(n.log_tau_s),activation='plain_tanh' if method=='ann_neat' else ('lif' if method!='leaky_ann_neat' else 'leaky_tanh') if k>=3 else 'leaky_tanh_readout') for k,n in sorted(g.nodes.items())],
        edges=[dict(source=a,target=b,weight=e.weight,enabled=e.enabled,innovation=e.innovation) for (a,b),e in sorted(g.connections.items())])
