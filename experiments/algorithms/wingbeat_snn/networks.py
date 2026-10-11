"""Graph controllers with one exported format: recurrent tanh ANN and delay SNN.

Spec (JSON): kind 'ann' | 'snn', inputs n, outputs [0, 1], nodes [{id, bias}],
edges [{source, target, weight, delay, enabled}], tau_m, tau_s. Inputs are
negative ids -1..-n. No input->output shortcuts and no output feedback: the
controller output always passes through hidden units.

SNN: current-based LIF at a 1 ms neural step, two substeps per 2 ms control
tick, threshold 1 with subtractive reset. Hidden units spike; outputs are
non-spiking leaky readouts squashed by tanh. A synaptic delay of d ms means
an input sampled d ms earlier, or a spike emitted 1+d steps earlier.
ANN: one synchronous recurrent tanh update per control tick; delays unused.
"""
import math
import numpy as np
from .plant import Controller

NEURAL_DT = .001
MAX_DELAY = 20


class Graph:
    def __init__(self, spec):
        self.kind = spec['kind']
        self.ids = [n['id'] for n in spec['nodes']]
        index = {k: i for i, k in enumerate(self.ids)}
        self.outputs = np.array([index[k] for k in spec['outputs']])
        self.n_inputs = spec['inputs']
        self.bias = np.array([n['bias'] for n in spec['nodes']], dtype=float)
        self.hidden = np.ones(len(self.ids), bool)
        self.hidden[self.outputs] = False
        edges = [e for e in spec['edges'] if e['enabled']]
        def arrays(selected, source):
            return (np.array([source(e) for e in selected], int), np.array([index[e['target']] for e in selected], int),
                    np.array([e['weight'] for e in selected], float),
                    np.array([int(np.clip(round(e.get('delay', 0.)), 0, MAX_DELAY)) for e in selected], int))
        self.inp = arrays([e for e in edges if e['source'] < 0], lambda e: -e['source']-1)
        self.rec = arrays([e for e in edges if e['source'] >= 0], lambda e: index[e['source']])
        self.a_m = math.exp(-NEURAL_DT/spec.get('tau_m', .01))
        self.a_s = math.exp(-NEURAL_DT/spec.get('tau_s', .005))
        self.length = MAX_DELAY+2
        self.reset()

    def reset(self):
        n = len(self.ids)
        self.current, self.voltage, self.state = np.zeros(n), np.zeros(n), np.zeros(n)
        self.inputs = np.zeros((self.length, self.n_inputs))
        self.spike_history = np.zeros((self.length, n))
        self.pointer, self.spikes = 0, 0

    def _drive(self, values_in, values_rec):
        drive = self.bias.copy()
        src, dst, w, _ = self.inp
        if len(w):
            np.add.at(drive, dst, w*values_in)
        src, dst, w, _ = self.rec
        if len(w):
            np.add.at(drive, dst, w*values_rec)
        return drive

    def snn_substep(self, x):
        self.pointer = (self.pointer+1) % self.length
        self.inputs[self.pointer] = x
        p = self.pointer
        src, _, _, d = self.inp
        values_in = self.inputs[(p-d) % self.length, src]
        src, _, _, d = self.rec
        values_rec = self.spike_history[(p-1-d) % self.length, src]
        drive = self._drive(values_in, values_rec)
        self.current = self.a_s*self.current+(1-self.a_s)*drive
        self.voltage = self.a_m*self.voltage+(1-self.a_m)*self.current
        fired = (self.voltage >= 1.) & self.hidden
        self.voltage[fired] -= 1.
        self.spike_history[p] = fired
        self.spikes += int(fired.sum())
        return np.tanh(self.voltage[self.outputs])

    def ann_step(self, x):
        src, _, _, _ = self.inp
        values_in = np.asarray(x)[src]
        src, _, _, _ = self.rec
        self.state = np.tanh(self._drive(values_in, self.state[src]))
        return self.state[self.outputs].copy()


class GraphController(Controller):
    prewarm = True

    def __init__(self, spec, phase_inputs=True):
        self.net = Graph(spec)
        self.mode = spec.get('mode', 'phasor')
        self.phase_inputs = phase_inputs
        expected = 11 if phase_inputs else 8
        if spec['inputs'] != expected:
            raise ValueError(f'spec has {spec["inputs"]} inputs, plant provides {expected}')

    @property
    def spikes(self):
        return self.net.spikes

    def reset(self):
        self.net.reset()

    def act(self, plant, time):
        x = plant.observation(self.phase_inputs)
        if self.net.kind == 'snn':
            first = self.net.snn_substep(x)
            second = self.net.snn_substep(x)
            return (second[0], second[1]) if self.mode == 'phasor' else (first[0], second[0])
        y = self.net.ann_step(x)
        return (y[0], y[1]) if self.mode == 'phasor' else (y[0], y[0])
