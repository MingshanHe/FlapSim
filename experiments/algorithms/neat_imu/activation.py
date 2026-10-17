"""Signed full-range gimbal control and activation/time-constant evolution."""
import copy
import math
import random
import numpy as np
import neat
from gymnasium import spaces
from . import robust
from .residual import ResidualPolicy, action_map


def operational_trim(trim):
    result = copy.deepcopy(trim)
    result['command'] = [trim['command'][0], 0., 0.]
    result['hover_attitude_rad'] = [0., 0., 0.]
    return result


def signed_map(trim):
    return action_map(operational_trim(trim), angle_span=.3)


class SignedHoverEnv(robust.RobustEnv):
    def __init__(self, trim, sensor_inputs=3, **kwargs):
        if sensor_inputs not in (3, 6):
            raise ValueError('signed IMU contract supports 3 angles or 3 angles + 3 gyro axes')
        self.sensor_inputs = sensor_inputs
        super().__init__(operational_trim(trim), angle_span=.3, **kwargs)
        self.observation_space = spaces.Box(-10., 10., (sensor_inputs,), dtype=np.float32)
        np.testing.assert_allclose(self.sim.model.actuator_ctrlrange, [[-.3,.3],[-.3,.3]])

    def _observation(self):
        values = self.angles()/.35
        if self.sensor_inputs == 6:
            rotation = self.sim.data.xmat[self.body].reshape(3,3)
            values = np.r_[values, self.sim.data.cvel[self.body,:3] @ rotation/10.]
        return np.clip(values, -10, 10).astype(np.float32)

    def sensor(self, obs):
        if self.sensor_inputs == 3:
            return super().sensor(obs)
        previous = self.previous_observation
        self.previous_observation = obs.copy()
        measured = previous if self.case.get('delay_steps', 0) else obs
        sigma = np.r_[np.full(3, math.radians(self.case.get('noise_degrees', 0.))/.35),
                      np.full(3, self.case.get('noise_gyro_rad_s', 0.)/10.)]
        return (measured+self.sensor_rng.normal(0., sigma)).astype(np.float32)


WORKER_MAPPING = None


class MappedPolicy(ResidualPolicy):
    """Evaluate an older policy with its own physical action units intact."""
    def __init__(self, payload, target_map=None):
        super().__init__(payload)
        target = target_map if target_map is not None else WORKER_MAPPING
        self.target_center, self.target_span = np.array(target['center']), np.array(target['span'])
        self.source_center = np.array(self.payload['action_map']['center'])
        self.source_span = np.array(self.payload['action_map']['span'])
        self.same_units = (np.array_equal(self.target_center, self.source_center)
                           and np.array_equal(self.target_span, self.source_span))

    def reset(self, obs):
        return super().reset(np.asarray(obs)[:len(self.payload['inputs'])])

    def act(self, obs):
        action = super().act(np.asarray(obs)[:len(self.payload['inputs'])])
        if self.same_units:
            return action
        return (self.source_center+self.source_span*action-self.target_center)/self.target_span


def init_worker(settings):
    global WORKER_MAPPING
    WORKER_MAPPING = signed_map(settings['trim'])
    robust.ENV = SignedHoverEnv(**settings)
    robust.ResidualPolicy = MappedPolicy


rollout = robust.rollout


class ActivationMemoryGenome(robust.StableMemoryGenome):
    """Bounded hidden activations, signed outputs, and stable evolvable memory."""
    def repair_memory(self, config):
        taus = {key: (self.nodes[key].time_constant if key in self.nodes
                      else -.01/math.log(rho)) for key, (_,rho) in robust.MEMORY.items()}
        super().repair_memory(config)
        for key, (source, _) in robust.MEMORY.items():
            tau = float(np.clip(taus[key], .01, .15))
            self.nodes[key].time_constant = tau
            rho = math.exp(-.01/tau)
            self.connections[(key,key)].weight = rho
            self.connections[(source,key)].weight = 1-rho
        if config.num_inputs == 6:
            # Optional causal gyro filtering; raw gyro inputs remain available.
            for key in list(self.connections):
                if key[1] in (9,10,11) and key[0] not in (key[1],5-key[1]):
                    del self.connections[key]
            for key in (9,10,11):
                if key not in self.nodes:
                    self.nodes[key] = self.create_node(config,key)
                    self.nodes[key].time_constant = .015
                node = self.nodes[key]
                node.activation, node.aggregation, node.bias, node.response = 'identity','sum',0.,1.
                node.time_constant = float(np.clip(node.time_constant,.01,.15))
                rho = math.exp(-.01/node.time_constant)
                for source,weight in ((5-key,1-rho),(key,rho)):
                    edge=(source,key)
                    if edge not in config.memory_innovations:
                        config.memory_innovations[edge]=config.innovation_tracker.get_innovation_number(*edge,'gyro_memory')
                    if edge not in self.connections:
                        self.connections[edge]=self.create_connection(config,*edge,config.memory_innovations[edge])
                    gene=self.connections[edge]
                    gene.weight,gene.enabled,gene.innovation=weight,True,config.memory_innovations[edge]
        for key in config.output_keys:
            if key in self.nodes and self.nodes[key].activation not in getattr(config,'signed_output_activations',('tanh','clamped')):
                self.nodes[key].activation = 'tanh'

    def mutate(self, config):
        activations = {key:self.nodes[key].activation for key in config.output_keys}
        super().mutate(config)
        for key in config.output_keys:
            self.nodes[key].activation = (random.choice(getattr(config,'signed_output_activations',('tanh','clamped'))) if random.random()<getattr(config,'output_activation_mutate_rate',.05) else activations[key])

    def mutate_add_node(self, config):
        reserved=set(robust.MEMORY)|({9,10,11} if config.num_inputs==6 else set())
        protected={k:v for k,v in self.connections.items() if k[1] in reserved}
        for key in protected: del self.connections[key]
        try: neat.DefaultGenome.mutate_add_node(self,config)
        finally: self.connections.update(protected)

    def mutate_delete_node(self, config):
        reserved=set(config.output_keys)|set(robust.MEMORY)|({9,10,11} if config.num_inputs==6 else set())
        candidates=[key for key in self.nodes if key not in reserved]
        if not candidates: return -1
        key=random.choice(candidates)
        del self.nodes[key]
        self.connections={edge:gene for edge,gene in self.connections.items() if key not in edge}
        return key


def initialize(genome, config, mapping, rng, parent=None):
    if parent is None and config.genome_config.num_inputs == 6:
        parent = gyro_network(mapping,float(rng.choice([.05,.2,.8,2.])),float(rng.choice([.002,.01,.04,.12])),
                              bias_alpha=float(rng.uniform(-.2,.2)),bias_motor=float(rng.normal(0.,.2)),
                              signs=tuple(rng.choice([-1.,1.],2)),activation=str(rng.choice(['tanh','clamped'])))
    if parent is not None and config.genome_config.num_inputs==6 and not parent.get('gyro_memory'):
        parent=copy.deepcopy(parent)
        remap={n['key']:n['key']+100 for n in parent['nodes'] if n['key']>=9}
        for node in parent['nodes']:
            node['key']=remap.get(node['key'],node['key'])
            node['links']=[[remap.get(source,source),w] for source,w in node['links']]
    robust.initialize(genome, config, mapping, rng, parent)
    for key in config.genome_config.output_keys:
        if key not in genome.nodes:
            genome.nodes[key] = genome.create_node(config.genome_config, key)
            genome.nodes[key].bias = 0.
    if parent is not None and parent.get('memory_design'):
        restored_memory=set(robust.MEMORY)|({9,10,11} if parent.get('gyro_memory') else set())
        for node in parent['nodes']:
            key = node['key']
            if key in restored_memory:
                rho = next((w for source,w in node['links'] if source == key), 0.)
                if 0 < rho < 1:
                    genome.nodes[key].time_constant = float(np.clip(-.01/math.log(rho), .01, .15))
    genome.repair_memory(config.genome_config)


def export(genome, config, mapping):
    payload = robust.export(genome, config, mapping)
    payload['memory_design'] = 'stable unity-gain EMA; evolved time constants between 10 and 150 ms'
    payload['memory_time_constants_s'] = {str(key):genome.nodes[key].time_constant for key in robust.MEMORY}
    if len(payload['inputs']) == 6:
        payload['input_names'] += ['gyro_x', 'gyro_y', 'gyro_z']
        payload['gyro_scale_rad_s'] = 10.
        payload['gyro_memory'] = 'three stable unity-gain EMA nodes, 10..150 ms'
        payload['memory_time_constants_s'].update({str(k):genome.nodes[k].time_constant for k in (9,10,11)})
    return payload


def cases(seeds, seconds=8., stress=False):
    result = robust.cases(seeds, seconds, stress)
    for case in result:
        case['noise_gyro_rad_s'] = .03 if stress else 0.
    return result


def gyro_network(mapping, kp, kd, *, signs=(-1.,1.), bias_alpha=0., bias_motor=0., activation='tanh'):
    """Six measured inputs, direct PD-like links as an evolvable NEAT seed."""
    nodes = []
    slope = 2.5 if activation == 'tanh' else 1.
    for output, axis, sign in ((0,1,signs[0]), (1,0,signs[1])):
        scale = 1./(slope*mapping['span'][output])
        nodes.append(dict(key=output, activation=activation, aggregation='sum',
                          bias=bias_alpha*scale if output == 0 else 0., response=1.,
                          links=[[-axis-1, sign*kp*.35*scale],[-axis-4, sign*kd*10.*scale]]))
    nodes.append(dict(key=2,activation=activation,aggregation='sum',bias=bias_motor/slope,
                      response=1.,links=[[-3,0.]]))
    return dict(format='flapsim-neat-residual-v1', recurrent=True, inputs=[-1,-2,-3,-4,-5,-6],
                outputs=[0,1,2], nodes=nodes, action_map=mapping, input_names=['roll','pitch','yaw','gyro_x','gyro_y','gyro_z'],
                input_scale_rad=.35, gyro_scale_rad_s=10., output_names=['alpha','beta','motor_speed'],priming_steps=100)
