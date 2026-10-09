"""Independent numeric, inheritance, reset and plant-parity contracts."""
import copy,math,random
from pathlib import Path
import neat,numpy as np,pytest
from experiments.algorithms.snn_hover.evolution import configuration,initialize,export,METHODS
from experiments.algorithms.snn_hover.networks import TemporalPolicy,policy,complexity
from experiments.algorithms.snn_hover.environment import HoverEnv,case
from experiments.algorithms.topology_hover.environment import HoverEnv as Original

def make(tmp_path,method='snn_neat'):
    tmp_path.mkdir(exist_ok=True);random.seed(42);c=configuration(tmp_path,method,4);p=neat.Population(c);initialize(p,c,42)
    return c,p

def test_neuron_step_matches_scalar_reference(tmp_path):
    c,p=make(tmp_path);s=export(p.population[2],'snn_neat');net=TemporalPolicy(s)
    rng=np.random.default_rng(52);I=np.zeros(7);V=np.zeros(7);signal=np.zeros(7)
    for _ in range(300):
        x=rng.normal(0,.4,11);newI=[];newV=[];z=[]
        for i,n in enumerate(s['nodes']):
            drive=n['bias']+(1.2 if n['id']>=3 else 0.)
            for e in s['edges']:
                if e['enabled'] and e['target']==n['id']:drive+=e['weight']*(x[-e['source']-1] if e['source']<0 else signal[e['source']])
            ai=math.exp(-.001/n['tau_s']);av=math.exp(-.001/n['tau_m'])
            current=ai*I[i]+(1-ai)*drive;voltage=av*V[i]+(1-av)*current
            spike=n['id']>=3 and voltage>=1.;newI.append(current);newV.append(voltage-int(spike));z.append(spike)
        I=np.array(newI);V=np.array(newV);signal=np.array(z)*20.
        actual=net.substep(x)
        np.testing.assert_allclose(net.current,I,atol=1e-12);np.testing.assert_allclose(net.voltage,V,atol=1e-12)
        np.testing.assert_array_equal(actual,z)

def test_signed_actions_and_exact_reset(tmp_path):
    c,p=make(tmp_path);s=export(p.population[1],'snn_neat')
    for n in s['nodes']:
        if n['id'] in [0,1]:n['bias']=1 if n['id']==0 else -1
    q=policy(s);sequence=[q.act(np.zeros(11)) for _ in range(50)]
    assert sequence[-1][0]>.7 and sequence[-1][1]<-.7 and q.spike_count.sum()>0
    q.reset();np.testing.assert_allclose([q.act(np.zeros(11)) for _ in range(50)],sequence,rtol=0,atol=0)

def test_fixed_ablation_initial_populations_identical(tmp_path):
    specs=[]
    for method in ['snn_neat','snn_fixed_tau','snn_fixed_topology']:
        c,p=make(tmp_path/method,method);specs.append([export(g,method) for g in p.population.values()])
    for i in range(4):
        assert specs[0][i]['nodes']==specs[1][i]['nodes']==specs[2][i]['nodes']
        assert specs[0][i]['edges']==specs[1][i]['edges']==specs[2][i]['edges']
        assert complexity(specs[0][i])['parameters']==81

@pytest.mark.parametrize('method',METHODS)
def test_mutation_constraints_and_tau_ablation(tmp_path,method):
    c,p=make(tmp_path,method);g=p.population[2];initial=set(g.connections);nodes=set(g.nodes)
    for _ in range(100):
        g.mutate(c.genome_config);s=export(g,method)
        assert len(g.nodes)<=35
        for n in s['nodes']:
            assert .005-1e-12<=n['tau_m']<=.2+1e-12 and .002-1e-12<=n['tau_s']<=.1+1e-12
            if method in ('snn_fixed_tau','ann_neat'):assert n['tau_m']==pytest.approx(.02) and n['tau_s']==pytest.approx(.01)
        for e in s['edges']:assert e['source'] not in [0,1,2] and not(e['source']<0 and e['target']<3)
        if method=='snn_fixed_topology':assert set(g.connections)==initial and set(g.nodes)==nodes and all(e.enabled for e in g.connections.values())
        assert np.isfinite(policy(s).act(np.zeros(11))).all()

def test_time_constants_inherited_and_affect_response(tmp_path):
    c,p=make(tmp_path);a,b=p.population[1],p.population[2];a.fitness=2.;b.fitness=1.
    a.nodes[3].log_tau_m=math.log(.005);b.nodes[3].log_tau_m=math.log(.2)
    child=c.genome_type(99);child.configure_crossover(a,b,c.genome_config)
    assert child.nodes[3].log_tau_m in [a.nodes[3].log_tau_m,b.nodes[3].log_tau_m]
    x=export(a,'snn_neat');y=copy.deepcopy(x);y['nodes'][3]['tau_m']=.2
    q,r=policy(x),policy(y)
    for _ in range(20):q.act(np.zeros(11));r.act(np.zeros(11))
    assert q.spike_count[3]>r.spike_count[3]

def test_wrapped_plant_matches_previous_state_and_reward():
    a,b=HoverEnv(),Original();c=case(409900001,.05,'stress',2)
    for e in [a,b]:e.configure(c);e.reset(seed=c['seed'])
    for _ in range(5):
        x=a.step([.1,-.2,.3]);y=b.step([.1,-.2,.3])
        np.testing.assert_allclose(a.sim.data.qpos,b.sim.data.qpos,atol=1e-12,rtol=0)
        np.testing.assert_allclose(x[0],y[0],atol=1e-12,rtol=0);assert x[1]==y[1]

@pytest.mark.parametrize('direction',[0,45,90,135,180,225,270,315])
def test_exact_large_angle_release(direction):
    e=HoverEnv();c=case(409900002,.05,'nominal',3);c.update(tilt_deg=30.,direction_deg=direction)
    e.configure(c);obs,_=e.reset(seed=c['seed'])
    assert e.initial_tilt_deg==pytest.approx(30.,abs=1e-8)
    np.testing.assert_allclose(e.sim.data.qvel,0,atol=0)
    assert obs.shape==(11,) and e.minimum_clearance>.03

def test_controllers_do_not_gain_truth_feedback():
    e=HoverEnv();c=case(409900003,.05);e.configure(c);e.reset(seed=c['seed'])
    def forbidden():raise AssertionError('Truth entered policy observation or PID')
    e.air.env._state=forbidden
    assert np.isfinite(e.observation()).all() and np.isfinite(e.pid()).all()
