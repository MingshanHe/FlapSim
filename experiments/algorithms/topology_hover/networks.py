"""The same synchronous graph dynamics in NumPy evolution and Torch PPO."""
import copy
import math
import numpy as np


def export_genome(g,ninputs):
    return dict(kind='graph',inputs=ninputs,outputs=[0,1,2],genome_id=g.key,
        nodes=[dict(id=k,bias=n.bias,activation=n.activation) for k,n in sorted(g.nodes.items())],
        edges=[dict(source=a,target=b,weight=c.weight,enabled=c.enabled,innovation=c.innovation)
               for (a,b),c in sorted(g.connections.items())])


def graph_arrays(spec):
    keys=[n['id'] for n in spec['nodes']];index={k:i for i,k in enumerate(keys)}
    nin=spec['inputs'];N=len(keys);W=np.zeros((N,nin+N),dtype=np.float32);mask=np.zeros_like(W)
    for e in spec['edges']:
        if not e['enabled']:continue
        source=-e['source']-1 if e['source']<0 else nin+index[e['source']]
        W[index[e['target']],source]=e['weight'];mask[index[e['target']],source]=1
    b=np.array([n['bias'] for n in spec['nodes']],dtype=np.float32)
    return W,b,mask,[index[k] for k in spec['outputs']]


class NumpyPolicy:
    def __init__(self,spec):
        self.spec=spec;self.kind=spec['kind']
        if self.kind=='graph':self.W,self.b,self.mask,self.out=graph_arrays(spec)
        else:self.layers=[(np.array(x['weight'],dtype=np.float32),np.array(x['bias'],dtype=np.float32)) for x in spec['layers']]
        self.reset()
    def reset(self):
        if self.kind=='graph':self.state=np.zeros(len(self.b),dtype=np.float32)
    def act(self,obs):
        if self.kind=='graph':
            self.state=np.tanh(self.W@np.r_[obs,self.state]+self.b)
            return self.state[self.out].copy()
        x=np.asarray(obs,dtype=np.float32)
        for W,b in self.layers:x=np.tanh(W@x+b)
        return x


def random_graph(ninputs,hidden=8,seed=0,reference=None):
    rng=np.random.default_rng(seed)
    if reference is not None:
        # Preserve node count and per-target input/recurrent indegree, without selected connectivity.
        s=copy.deepcopy(reference);s.pop('genome_id',None)
        for n in s['nodes']:n['bias']=0.
        keys=[n['id'] for n in s['nodes']];edges=[]
        for target in keys:
            incoming=[e for e in s['edges'] if e['enabled'] and e['target']==target]
            for sources in (list(range(-ninputs,0)),keys):
                count=sum((e['source']<0)==(sources[0]<0) for e in incoming)
                for source in rng.choice(sources,count,replace=False):
                    edges.append(dict(source=int(source),target=target,weight=float(rng.normal(0,.03)),enabled=True,innovation=len(edges)))
        s['edges']=edges;return s
    keys=list(range(hidden+3));nodes=[dict(id=k,bias=0.,activation='plain_tanh') for k in keys]
    edges=[]
    for target in keys:
        sources=range(-ninputs,hidden+3) if target>=3 else range(3,hidden+3)
        for source in sources:
            edges.append(dict(source=source,target=target,weight=float(rng.normal(0,.03)),enabled=True,innovation=len(edges)))
    return dict(kind='graph',inputs=ninputs,outputs=[0,1,2],nodes=nodes,edges=edges)


def torch_actor(kind,ninputs,seed=0,spec=None,hidden=8):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    class Graph(nn.Module):
        def __init__(self,s):
            super().__init__();self.spec=copy.deepcopy(s)
            W,b,mask,out=graph_arrays(s);self.state_size=len(b);self.out=out
            self.weight=nn.Parameter(torch.tensor(W));self.bias=nn.Parameter(torch.tensor(b))
            self.register_buffer('mask',torch.tensor(mask))
        def forward(self,x,state):
            raw=nn.functional.linear(torch.cat([x,state],-1),self.weight*self.mask,self.bias)
            return raw[:,self.out],torch.tanh(raw)
        def export(self):
            s=copy.deepcopy(self.spec);W=self.weight.detach().cpu().numpy();b=self.bias.detach().cpu().numpy()
            index={n['id']:i for i,n in enumerate(s['nodes'])}
            for n in s['nodes']:n['bias']=float(b[index[n['id']]])
            for e in s['edges']:
                source=-e['source']-1 if e['source']<0 else s['inputs']+index[e['source']]
                e['weight']=float(W[index[e['target']],source])
            return s
    class MLP(nn.Module):
        def __init__(self):
            super().__init__();self.state_size=1
            self.layers=nn.ModuleList([nn.Linear(ninputs,32),nn.Linear(32,32),nn.Linear(32,3)])
            nn.init.normal_(self.layers[-1].weight,0,.01);nn.init.zeros_(self.layers[-1].bias)
        def forward(self,x,state):
            for layer in self.layers[:-1]:x=torch.tanh(layer(x))
            return self.layers[-1](x),state
        def export(self):
            return dict(kind='mlp',inputs=ninputs,layers=[dict(weight=l.weight.detach().cpu().numpy().tolist(),bias=l.bias.detach().cpu().numpy().tolist()) for l in self.layers])
    if kind=='mlp':return MLP()
    return Graph(spec or random_graph(ninputs,hidden,seed))


def complexity(spec):
    if spec['kind']=='graph':
        active=[e for e in spec['edges'] if e['enabled']]
        # Input-to-output reachability, including recurrent paths, excludes disconnected genes.
        reachable=set(range(-spec['inputs'],0));useful=set(spec['outputs'])
        for _ in spec['nodes']:
            reachable.update(e['target'] for e in active if e['source'] in reachable)
            useful.update(e['source'] for e in active if e['target'] in useful)
        effective=[e for e in active if e['source'] in reachable and e['target'] in useful]
        return dict(nodes=len(spec['nodes']),hidden=len(spec['nodes'])-3,connections=len(active),effective_connections=len(effective),
            recurrent_edges=sum(e['source']>=0 for e in active),parameters=len(active)+len(spec['nodes']),state_floats=len(spec['nodes']))
    return dict(parameters=sum(np.size(l['weight'])+len(l['bias']) for l in spec['layers']),
        connections=sum(np.size(l['weight']) for l in spec['layers']),nodes=sum(len(l['bias']) for l in spec['layers']),recurrent_edges=0,state_floats=0)
