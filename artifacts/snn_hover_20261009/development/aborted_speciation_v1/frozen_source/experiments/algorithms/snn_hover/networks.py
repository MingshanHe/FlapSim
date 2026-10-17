"""Causal current-encoded LIF hidden nodes and signed continuous readouts."""
import copy
import numpy as np
from ..topology_hover.networks import NumpyPolicy, complexity as base_complexity

DT=.001
SUBSTEPS=10
BACKGROUND=1.2
SPIKE_SCALE=20.  # A 50 Hz train has unit average drive at a 1 ms neural clock.


class TemporalPolicy:
    def __init__(self,spec):
        self.spec=spec;self.kind=spec['kind'];self.ids=[n['id'] for n in spec['nodes']]
        self.index={k:i for i,k in enumerate(self.ids)};self.out=[self.index[k] for k in spec['outputs']]
        self.hidden=np.array([i for i,k in enumerate(self.ids) if k not in spec['outputs']],dtype=int)
        self.W=np.zeros((len(self.ids),spec['inputs']+len(self.ids)))
        for e in spec['edges']:
            if not e['enabled']:continue
            if e['source'] in spec['outputs'] or (e['source']<0 and e['target'] in spec['outputs']):
                raise ValueError('Hidden nodes alone may drive temporal readouts')
            source=-e['source']-1 if e['source']<0 else spec['inputs']+self.index[e['source']]
            self.W[self.index[e['target']],source]=e['weight']
        self.bias=np.array([n['bias'] for n in spec['nodes']])
        self.am=np.exp(-DT/np.array([n['tau_m'] for n in spec['nodes']]))
        self.asyn=np.exp(-DT/np.array([n['tau_s'] for n in spec['nodes']]))
        self.background=np.zeros(len(self.ids));self.background[self.hidden]=BACKGROUND if self.kind=='snn' else 0.
        self.capture=False;self.reset()

    def reset(self):
        self.current=np.zeros(len(self.ids));self.voltage=np.zeros(len(self.ids));self.signal=np.zeros(len(self.ids))
        self.spike_count=np.zeros(len(self.ids),dtype=np.int64);self.ticks=0;self.internal_trace=[]

    def substep(self,obs):
        drive=self.W@np.r_[obs,self.signal]+self.bias+self.background
        self.current=self.asyn*self.current+(1-self.asyn)*drive
        self.voltage=self.am*self.voltage+(1-self.am)*self.current
        spikes=np.zeros(len(self.ids),dtype=bool);self.signal[:]=0.
        if self.kind=='snn':
            spikes[self.hidden]=self.voltage[self.hidden]>=1.
            self.voltage[spikes]-=1.
            self.signal[spikes]=SPIKE_SCALE
        else:
            self.signal[self.hidden]=np.tanh(self.voltage[self.hidden])
        self.spike_count+=spikes;self.ticks+=1
        if self.capture:self.internal_trace.append([self.ticks*DT,*spikes.astype(float),*self.voltage,*self.current])
        return spikes

    def act(self,obs):
        for _ in range(SUBSTEPS):self.substep(obs)
        return np.tanh(self.voltage[self.out])


def policy(spec):
    return TemporalPolicy(spec) if spec['kind'] in ('snn','leaky_ann') else NumpyPolicy(spec)


def complexity(spec):
    if spec['kind'] not in ('snn','leaky_ann'):return base_complexity(spec)
    graph=copy.deepcopy(spec);graph['kind']='graph';r=base_complexity(graph);base=r['parameters']
    r.update(time_constants=2*len(spec['nodes']),evolved_time_constants=0 if spec.get('fixed_tau') else 2*len(spec['nodes']),
        parameters=base+(0 if spec.get('fixed_tau') else 2*len(spec['nodes'])),stored_scalar_parameters=base+2*len(spec['nodes']),
        state_floats=3*len(spec['nodes']),internal_updates_per_action=SUBSTEPS)
    return r
