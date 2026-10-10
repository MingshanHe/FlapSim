"""Identical exported graph semantics in NumPy and differentiable Torch.

No output feedback or input-output shortcuts. SNN integrates twice per control
tick; ANN updates its synchronous recurrent state once per control tick.
"""
import copy
import numpy as np
import torch
from torch import nn
from ..topology_hover.networks import graph_arrays
from ..snn_hover.networks import TemporalPolicy
from ..topology_hover.networks import NumpyPolicy


def initial_spec(kind, seed, hidden=16, zero_readout=False, inputs=11):
    rng = np.random.default_rng(seed)
    keys = list(range(3+hidden))
    nodes = [dict(id=k, bias=0., tau_m=.02, tau_s=.01,
                  activation='lif' if kind == 'snn' and k >= 3 else 'plain_tanh') for k in keys]
    edges = []
    for target in keys:
        sources = list(range(3, hidden+3)) if target < 3 else list(range(-inputs, 0))+list(range(3, hidden+3))
        for source in sources:
            scale = .15 if source < 0 else .04
            weight = 0. if zero_readout and target < 3 else float(rng.normal(0, scale))
            edges.append(dict(source=source, target=target, weight=weight, enabled=True, innovation=len(edges)))
    return dict(format='flapsim-pid-compare-v1', kind=kind, inputs=inputs, outputs=[0, 1, 2],
                nodes=nodes, edges=edges, fixed_tau=True, control_dt=.002, neural_dt=.001,
                substeps=2 if kind == 'snn' else 1, background=1.2, spike_scale=20.)


class Spikes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient/(1.+5.*voltage.abs()).square()


class Actor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.kind = spec['kind']
        self.inputs = spec['inputs']
        weight, bias, mask, out = graph_arrays(spec)
        weight = weight.astype(np.float64)
        bias = np.array([n['bias'] for n in spec['nodes']], dtype=np.float64)
        index = {n['id']: i for i, n in enumerate(spec['nodes'])}
        for e in spec['edges']:
            if e['enabled']:
                source = -e['source']-1 if e['source'] < 0 else self.inputs+index[e['source']]
                weight[index[e['target']], source] = e['weight']
        self.weight = nn.Parameter(torch.from_numpy(weight))
        self.bias = nn.Parameter(torch.from_numpy(bias))
        self.register_buffer('mask', torch.from_numpy(mask.astype(np.float64)))
        self.out = out
        self.count = len(bias)
        self.state_size = self.count*(3 if self.kind == 'snn' else 1)
        hidden = np.ones(self.count, dtype=np.float64)
        hidden[out] = 0.
        self.register_buffer('hidden_mask', torch.tensor(hidden))
        self.register_buffer('am', torch.tensor(np.exp(-.001/np.array([n['tau_m'] for n in spec['nodes']])), dtype=torch.float64))
        self.register_buffer('asyn', torch.tensor(np.exp(-.001/np.array([n['tau_s'] for n in spec['nodes']])), dtype=torch.float64))

    def initial_state(self, batch):
        return self.weight.new_zeros((batch, self.state_size))

    def forward(self, observation, state):
        weight = self.weight*self.mask
        if self.kind == 'ann':
            raw = nn.functional.linear(torch.cat((observation, state), -1), weight, self.bias)
            return raw[:, self.out], torch.tanh(raw)
        current, voltage, signal = state.chunk(3, dim=-1)
        for _ in range(2):
            drive = nn.functional.linear(torch.cat((observation, signal), -1), weight, self.bias)+1.2*self.hidden_mask
            current = self.asyn*current+(1-self.asyn)*drive
            voltage = self.am*voltage+(1-self.am)*current
            spikes = Spikes.apply(voltage-1.)*self.hidden_mask
            voltage = voltage-spikes
            signal = spikes*20.
        return voltage[:, self.out], torch.cat((current, voltage, signal), -1)

    def export(self):
        spec = copy.deepcopy(self.spec)
        weight = self.weight.detach().cpu().numpy()
        bias = self.bias.detach().cpu().numpy()
        index = {n['id']: i for i, n in enumerate(spec['nodes'])}
        for n in spec['nodes']:
            n['bias'] = float(bias[index[n['id']]])
        for e in spec['edges']:
            source = -e['source']-1 if e['source'] < 0 else self.inputs+index[e['source']]
            e['weight'] = float(weight[index[e['target']], source])
        return spec


class SpikingPolicy(TemporalPolicy):
    def act(self, observation):
        for _ in range(2):
            self.substep(observation)
        return np.tanh(self.voltage[self.out])


def policy(spec):
    if spec['kind'] == 'snn':
        return SpikingPolicy(spec)
    return RecurrentPolicy(spec)


class RecurrentPolicy:
    def __init__(self, spec):
        actor = Actor(spec)
        self.weight = (actor.weight*actor.mask).detach().numpy().copy()
        self.bias = actor.bias.detach().numpy().copy()
        self.out = actor.out
        self.reset()

    def reset(self):
        self.state = np.zeros(len(self.bias))

    def act(self, observation):
        self.state = np.tanh(self.weight@np.r_[observation, self.state]+self.bias)
        return self.state[self.out].copy()


def complexity(spec):
    enabled = [e for e in spec['edges'] if e['enabled']]
    return dict(nodes=len(spec['nodes']), hidden=len(spec['nodes'])-3,
                connections=len(enabled), parameters=len(enabled)+len(spec['nodes']),
                state_floats=len(spec['nodes'])*(3 if spec['kind'] == 'snn' else 1),
                internal_updates_per_action=2 if spec['kind'] == 'snn' else 1,
                control_hz=500, fixed_time_constants=True)
