"""Bounded trim-relative controls and a three-input recurrent NEAT phenotype."""
from dataclasses import replace
import json
from pathlib import Path
import math

import neat
import mujoco
import numpy as np
from .environment import ImuAttitudeEnv
from flapsim import Simulator
from flapsim.kinematics import euler_matrix
from flapsim.rl.native import accelerate
from flapsim.rl.native_fast import accelerate_fast


class ResidualHoverEnv(ImuAttitudeEnv):
    def __init__(self, trim, *, seconds=4., initial_degrees=5., speed_fraction=.1,
                 angle_span=.2, warmup_seconds=.1, render_mode=None, simulation_config=None):
        self.trim = dict(trim)
        self.center = np.array([trim['command'][1], trim['command'][2], trim['command'][0]])
        self.span = np.array([angle_span, angle_span, self.center[2]*speed_fraction])
        if (not np.isfinite(self.span).all() or np.any(self.span <= 0) or
                np.any(np.abs(self.center[:2])+self.span[:2] > .3) or
                self.center[2]-self.span[2] < 0 or self.center[2]+self.span[2] > 5000):
            raise ValueError('invalid trim/residual limits')
        super().__init__(seconds=seconds, motor_speed=self.center[2], initial_degrees=initial_degrees,
                         warmup_seconds=warmup_seconds, render_mode=render_mode, inputs=3, simulation_config=simulation_config)
        if warmup_seconds and np.any(self.center[:2]):
            source = self.source_config
            fixed = accelerate_fast(accelerate(Simulator(replace(source,
                scene=replace(source.scene, fixed_base=True, orientation=(1., 0., 0., 0.)),
                gimbal=replace(source.gimbal, angles=tuple(self.center[:2]))))))
            fixed.set_speed(self.center[2])
            fixed.set_gimbal(pitch=self.center[0], roll=self.center[1])
            fixed.step(round(warmup_seconds/fixed.model.opt.timestep))
            if any(w.number for w in fixed.data.warning):
                raise RuntimeError('Trimmed warmup is unstable')
            for joint in self.internal:
                self._initial_qpos[joint.qposadr[0]] = fixed.data.joint(joint.name).qpos[0]
                self._initial_qvel[joint.dofadr[0]] = fixed.data.joint(joint.name).qvel[0]
        # Smooth physical targets linearly here; base action smoothing is off so
        # its asymmetric full-range motor mapping cannot distort +/- residuals.
        self.config = replace(self.config, action_tau=0.)
        self.filtered_action = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        target = np.asarray(self.trim.get('hover_attitude_rad', [0., 0., 0.]))
        if np.any(target):
            qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
            quaternion = np.zeros(4)
            rpy = self.angles()+target
            mujoco.mju_mat2Quat(quaternion, euler_matrix(rpy).ravel())
            qpos[self.free_qpos+3:self.free_qpos+7] = quaternion
            self.sim.reset(qpos=qpos, qvel=qvel)
            self.sim.set_speed(self.center[2])
            obs = self._last_observation = self._observation()
            info['rpy'] = rpy
        self.filtered_action = np.zeros(3)
        self.applied_action[1:] = self.center[:2]/.3
        self.sim.set_gimbal(pitch=self.center[0], roll=self.center[1])
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('expected alpha, beta, speed residual in [-1,1]')
        previous = self.filtered_action.copy()
        self.filtered_action += (1-math.exp(-.01/.02))*(np.clip(action, -1, 1)-self.filtered_action)
        physical = self.center+self.span*self.filtered_action
        delta = physical[2]-self.center[2]
        denominator = 5000-self.center[2] if delta >= 0 else self.center[2]
        legacy = np.r_[physical[:2]/.3, delta/denominator]
        obs, _, terminated, truncated, info = super().step(legacy)
        position, velocity, rotation = self._state()
        tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
        omega = self.sim.data.cvel[self.body, :3] @ rotation
        # State used by the objective is not made available to the controller.
        quality = math.exp(-(info['position_error']/.15)**2-(tilt/.25)**2)
        reward = (1. + 2.*quality + .5*math.exp(-(np.linalg.norm(velocity)/.4)**2)
                  + .5*math.exp(-(tilt/.2)**2)
                  - .01*min(float(np.sum((omega/10.)**2)), 100.)
                  - .02*float(np.sum((self.filtered_action-previous)**2)))
        if terminated:
            reward -= 10.
        if info['termination_reason'] == 'numerical':
            reward = -25.
        info['residual_action'] = self.filtered_action.copy()
        return obs, reward, terminated, truncated, info


def action_map(trim, speed_fraction=.1, angle_span=.2):
    return dict(center=[trim['command'][1], trim['command'][2], trim['command'][0]],
                span=[angle_span, angle_span, trim['command'][0]*speed_fraction],
                units=['rad', 'rad', 'rad/s'], smoothing_tau_s=.02,
                observation_offset_rad=trim.get('hover_attitude_rad', [0., 0., 0.]))


def gain_network(kp, kd, tau, mapping, *, bias=0.):
    """A causal PD-like initial recurrent graph, not a separate PID controller.

    Three clamped memory nodes remember exponentially filtered IMU angles.
    All their connections can subsequently evolve under ordinary NEAT.
    """
    nodes = []
    leak = math.exp(-.01/tau)
    for i in range(3):
        nodes.append(dict(key=3+i, activation='clamped', aggregation='sum', bias=0., response=1.,
                          links=[[-i-1, 1-leak], [3+i, leak]]))
    for output, axis, sign in ((0, 1, -1.), (1, 0, 1.)):
        scale = .35/(2.5*mapping['span'][output])
        nodes.append(dict(key=output, activation='tanh', aggregation='sum', bias=0., response=1.,
                          links=[[-axis-1, sign*(kp+kd/tau)*scale],
                                 [3+axis, -sign*kd/tau*scale]]))
    nodes.append(dict(key=2, activation='tanh', aggregation='sum', bias=bias, response=1., links=[[-3, 0.]]))
    return dict(format='flapsim-neat-residual-v1', recurrent=True, inputs=[-1, -2, -3],
                outputs=[0, 1, 2], nodes=nodes, action_map=mapping,
                input_names=['roll', 'pitch', 'yaw'], input_scale_rad=.35,
                output_names=['alpha', 'beta', 'motor_speed'], priming_steps=20)


def encode(genome, config, mapping):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    return dict(format='flapsim-neat-residual-v1', recurrent=True,
                inputs=net.input_nodes, outputs=net.output_nodes,
                nodes=[dict(key=key, activation=genome.nodes[key].activation,
                            aggregation=genome.nodes[key].aggregation,
                            bias=bias, response=response, links=links)
                       for key, _, _, bias, response, links in net.node_evals],
                action_map=mapping, input_names=['roll', 'pitch', 'yaw'],
                input_scale_rad=.35, output_names=['alpha', 'beta', 'motor_speed'], priming_steps=20)


class ResidualPolicy:
    def __init__(self, payload):
        self.payload = json.loads(Path(payload).read_text()) if isinstance(payload, (str, Path)) else payload
        if self.payload['format'] != 'flapsim-neat-residual-v1':
            raise ValueError('wrong policy format')
        acts, aggs = neat.activations.ActivationFunctionSet(), neat.aggregations.AggregationFunctionSet()
        from .signed_activations import register
        register(acts)
        self.net = neat.nn.RecurrentNetwork(self.payload['inputs'], self.payload['outputs'], [
            (n['key'], acts.get(n['activation']), aggs.get(n['aggregation']), n['bias'], n['response'], n['links'])
            for n in self.payload['nodes']])

    def reset(self, observation):
        self.net.reset()
        observation = self.normalize(observation)
        # Only the current initial observation is repeated; no future samples.
        for _ in range(self.payload['priming_steps']):
            self.net.activate(observation)

    def act(self, observation):
        observation = self.normalize(observation)
        return np.clip(self.net.activate(observation), -1, 1)

    def normalize(self, observation):
        observation = np.asarray(observation, dtype=float).copy()
        if observation.shape != (len(self.payload['inputs']),):
            raise ValueError('observation dimensions do not match policy inputs')
        observation[:3] -= np.asarray(self.payload['action_map'].get('observation_offset_rad', [0, 0, 0]))/.35
        return observation

    def commands(self, observation):
        return np.array(self.payload['action_map']['center'])+np.array(self.payload['action_map']['span'])*self.act(observation)


def seed_genome(genome, config, payload):
    """Install a common initial graph with globally shared innovation numbers."""
    cfg = config.genome_config
    genome.nodes.clear()
    genome.connections.clear()
    for spec in payload['nodes']:
        node = genome.create_node(cfg, spec['key'])
        node.activation, node.aggregation = spec['activation'], spec['aggregation']
        node.bias, node.response = spec['bias'], spec['response']
        genome.nodes[spec['key']] = node
        for source, weight in spec['links']:
            innovation = cfg.innovation_tracker.get_innovation_number(source, spec['key'], 'residual_seed')
            conn = genome.create_connection(cfg, source, spec['key'], innovation)
            conn.weight, conn.enabled = weight, True
            genome.connections[conn.key] = conn
