"""Constrained-memory NEAT and reproducible disturbed flight evaluations.

Only three attitude samples enter the network. Six reserved recurrent nodes
are stable exponential memories; all control connections and other nodes evolve.
"""
from dataclasses import replace
import copy
import math
import random

import mujoco
import neat
import numpy as np

from .residual import ResidualHoverEnv, ResidualPolicy, encode, seed_genome

MEMORY = {3+3*bank+axis: (-axis-1, math.exp(-.01/tau))
          for bank, tau in enumerate((.02, .08)) for axis in range(3)}


class BoundedSpeciesSet(neat.DefaultSpeciesSet):
    """Keep useful niches without violating the finite offspring budget.

Merging the closest representatives preserves every genome. Existing species
history is retained, so threshold adaptation cannot reset stagnation each step.
"""
    def speciate(self, config, population, generation):
        super().speciate(config, population, generation)
        limit = max(2, min(8, config.pop_size//6))
        self.last_merges = 0
        while len(self.species) > limit:
            keys = sorted(self.species)
            _, keep, remove = min((self.species[a].representative.distance(self.species[b].representative, config.genome_config), a, b)
                                  for i, a in enumerate(keys) for b in keys[i+1:])
            first, second = self.species[keep], self.species.pop(remove)
            first.members.update(second.members)
            first.last_improved = max(first.last_improved, second.last_improved)
            for gid in second.members:
                self.genome_to_species[gid] = keep
            self.last_merges += 1


class StableMemoryGenome(neat.DefaultGenome):
    """NEAT with six protected, unity-DC-gain attitude-memory nodes."""
    def repair_memory(self, config):
        if not hasattr(config, 'memory_innovations'):
            config.memory_innovations = {}
        for key in list(self.connections):
            source, target = key
            if target in MEMORY and source not in (MEMORY[target][0], target):
                del self.connections[key]
        for target, (source, rho) in MEMORY.items():
            if target not in self.nodes:
                self.nodes[target] = self.create_node(config, target)
            node = self.nodes[target]
            node.activation, node.aggregation = 'identity', 'sum'
            node.bias, node.response = 0., 1.
            for incoming, weight in ((source, 1-rho), (target, rho)):
                key = (incoming, target)
                if key not in config.memory_innovations:
                    config.memory_innovations[key] = config.innovation_tracker.get_innovation_number(*key, 'stable_memory')
                if key not in self.connections:
                    self.connections[key] = self.create_connection(config, *key, config.memory_innovations[key])
                gene = self.connections[key]
                gene.innovation = config.memory_innovations[key]
                gene.weight, gene.enabled = weight, True

    def configure_new(self, config):
        super().configure_new(config)
        self.repair_memory(config)

    def configure_crossover(self, first, second, config):
        super().configure_crossover(first, second, config)
        self.repair_memory(config)

    def mutate(self, config):
        super().mutate(config)
        self.repair_memory(config)

    def mutate_add_node(self, config):
        protected = {k: v for k, v in self.connections.items() if k[1] in MEMORY}
        for key in protected:
            del self.connections[key]
        try:
            super().mutate_add_node(config)
        finally:
            self.connections.update(protected)

    def mutate_delete_node(self, config):
        candidates = [k for k in self.nodes if k not in config.output_keys and k not in MEMORY]
        if not candidates:
            return -1
        key = random.choice(candidates)
        del self.nodes[key]
        self.connections = {k: c for k, c in self.connections.items() if key not in k}
        return key


def export(genome, config, mapping):
    payload = encode(genome, config, mapping)
    payload['priming_steps'] = 100
    payload['memory_design'] = 'protected EMA, 20 and 80 ms per attitude axis'
    return payload


def initialize(genome, config, mapping, rng, parent=None):
    """A diverse start: previous NEAT graph, asymmetric gain graphs, dense graphs."""
    if parent is not None:
        payload = copy.deepcopy(parent)
        remap = ({} if payload.get('memory_design') else
                 {n['key']: n['key']+6 for n in payload['nodes'] if n['key'] >= 3})
        for node in payload['nodes']:
            node['key'] = remap.get(node['key'], node['key'])
            node['links'] = [[remap.get(source, source), w] for source, w in node['links']]
        seed_genome(genome, config, payload)
        for key, gene in genome.connections.items():
            if key[1] not in MEMORY:
                gene.innovation = config.genome_config.innovation_tracker.get_innovation_number(*key, 'control_seed')
        genome.repair_memory(config.genome_config)
        return
    cfg = config.genome_config
    genome.connections = {k: v for k, v in genome.connections.items() if k[1] in MEMORY}
    for output in cfg.output_keys:
        genome.nodes[output].bias = float(rng.normal(0., .07 if output == 0 else .02 if output == 1 else .12))
        for source in cfg.input_keys+list(MEMORY):
            key = (source, output)
            innovation = cfg.innovation_tracker.get_innovation_number(*key, 'control_seed')
            gene = genome.create_connection(cfg, *key, innovation)
            gene.weight, gene.enabled = 0., True
            genome.connections[key] = gene
    if rng.random() < .25:
        for key, gene in genome.connections.items():
            if key[1] not in MEMORY:
                gene.weight = float(rng.normal(0., .25))
    else:
        for output, axis, sign in ((0, 1, -1.), (1, 0, 1.)):
            kp = float(rng.choice([.1, .3, .8, 2.]))
            kd = float(rng.choice([0., .005, .02, .05, .1]))
            bank = int(rng.integers(2))
            tau = (.02, .08)[bank]
            if rng.random() < .15:
                sign *= -1
            scale = .35/(2.5*mapping['span'][output])
            genome.connections[(-axis-1, output)].weight = sign*(kp+kd/tau)*scale
            genome.connections[(3+3*bank+axis, output)].weight = -sign*kd/tau*scale
        for source in cfg.input_keys:
            genome.connections[(source, 2)].weight = float(rng.normal(0., .15))
    genome.repair_memory(cfg)


class RobustEnv(ResidualHoverEnv):
    """Disturbances affect initial state, attitude readings, and motor command."""
    def configure_case(self, case):
        self.case = case
        self.initial_degrees = case['degrees']
        self.set_seconds(case['seconds'])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, 9841]))
        rate = self.case.get('angular_rate', 0.)
        if rate:
            qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
            qvel[self.free_dof+3:self.free_dof+6] += self.sensor_rng.uniform(-rate, rate, 3)
            self.sim.reset(qpos=qpos, qvel=qvel)
            self.sim.set_speed(self.center[2])
            self.sim.set_gimbal(pitch=self.center[0], roll=self.center[1])
            obs = self._last_observation = self._observation()
        spread = self.case.get('motor_uncertainty', 0.)
        self.motor_gain = 1.+self.sensor_rng.uniform(-spread, spread)
        self.previous_observation = obs.copy()
        return self.sensor(obs), info

    def sensor(self, obs):
        previous = self.previous_observation
        self.previous_observation = obs.copy()
        measured = previous if self.case.get('delay_steps', 0) else obs
        sigma = math.radians(self.case.get('noise_degrees', 0.))/.35
        return (measured+self.sensor_rng.normal(0., sigma, 3)).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float).copy()
        # Bounded actuator command uncertainty, not an added lift or force.
        speed = (self.center[2]+self.span[2]*np.clip(action[2], -1, 1))*self.motor_gain
        action[2] = np.clip((speed-self.center[2])/self.span[2], -1, 1)
        obs, reward, terminated, truncated, info = super().step(action)
        return self.sensor(obs), reward, terminated, truncated, info


ENV = None


def init_worker(settings):
    global ENV
    ENV = RobustEnv(**settings)


def rollout(task):
    payload, cases, trace = task
    policy = ResidualPolicy(payload)
    records = []
    for case in cases:
        ENV.configure_case(case)
        obs, _ = ENV.reset(seed=case['seed'])
        policy.reset(obs)
        quality, tilt_squared, rows, action_change = 0., 0., [], 0.
        old_action = np.zeros(3)
        while True:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = ENV.step(action)
            tilt = math.acos(float(np.clip(ENV.sim.data.xmat[ENV.body, 8], -1, 1)))
            quality += math.exp(-(tilt/.35)**2-(info['position_error']/.25)**2)
            tilt_squared += tilt**2
            action_change += float(np.sum((ENV.filtered_action-old_action)**2))
            old_action = ENV.filtered_action.copy()
            if trace:
                rows.append([info['elapsed'], *np.rad2deg(info['rpy']), info['alpha'], info['beta'],
                             *info['position'], info['motor_command']])
            if terminated or truncated:
                break
        # Fixed reference horizon avoids artificial fitness drops when the
        # curriculum length changes and NEAT checks species stagnation.
        fitness = (ENV.steps+2.*quality-.03*action_change)/800.
        records.append(dict(seed=case['seed'], case=case, fitness=fitness,
                            steps=ENV.steps, elapsed=info['elapsed'], success=info['is_success'],
                            tracking_rmse_m=info['tracking_rmse'], termination=info['termination_reason'],
                            tilt_rmse_deg=math.degrees(math.sqrt(tilt_squared/ENV.steps)),
                            stable_seconds=.01*quality, trace=rows))
    return records


def robust_score(rows):
    fitness = sorted(r['fitness'] for r in rows)
    count = max(1, math.ceil(len(fitness)/4))
    return .5*float(np.mean(fitness))+.5*float(np.mean(fitness[:count]))


def summarize(rows):
    elapsed = [r['elapsed'] for r in rows]
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                mean_elapsed=float(np.mean(elapsed)), p10_elapsed=float(np.quantile(elapsed, .1)),
                min_elapsed=min(elapsed), mean_tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in rows])),
                mean_tilt_rmse_deg=float(np.mean([r['tilt_rmse_deg'] for r in rows])),
                mean_stable_seconds=float(np.mean([r['stable_seconds'] for r in rows])),
                robust_score=robust_score(rows), control_steps=sum(r['steps'] for r in rows),
                termination_counts={k: sum(r['termination'] == k for r in rows)
                                    for k in sorted({r['termination'] for r in rows})})


def cases(seeds, seconds=8., stress=False):
    return [dict(seed=int(seed), seconds=seconds, degrees=10. if stress else 5.,
                 angular_rate=.25 if stress else 0., noise_degrees=.25 if stress else 0.,
                 motor_uncertainty=.03 if stress else 0., delay_steps=int(stress)) for seed in seeds]
