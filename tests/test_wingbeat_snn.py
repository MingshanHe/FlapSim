"""Wingbeat-synchronous control: plant equivalence, delay-SNN semantics and NEAT constraints."""
import random
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('neat')


def tiny_spec(kind='snn', delay=3, weight=50.):
    # input -1 -> hidden 2 (delay) -> output 0; output 1 unused
    return dict(kind=kind, inputs=1, outputs=[0, 1], tau_m=.01, tau_s=.005,
                nodes=[dict(id=0, bias=0.), dict(id=1, bias=0.), dict(id=2, bias=0.)],
                edges=[dict(source=-1, target=2, weight=weight, delay=delay, enabled=True),
                       dict(source=2, target=0, weight=1., delay=0, enabled=True)])


def test_input_delay_postpones_the_hidden_response():
    from experiments.algorithms.wingbeat_snn.networks import Graph
    for delay in (0, 3, 7):
        g = Graph(tiny_spec(delay=delay))
        trace = []
        for k in range(12):
            g.snn_substep(np.array([1. if k == 0 else 0.]))
            trace.append(g.current[2])
        first = next(i for i, v in enumerate(trace) if v != 0.)
        assert first == delay


def test_spike_delay_counts_one_transmission_step():
    from experiments.algorithms.wingbeat_snn.networks import Graph
    spec = tiny_spec(delay=0, weight=1e4)
    spec['edges'][1]['delay'] = 4
    g = Graph(spec)
    fired_at, readout = None, []
    for k in range(15):
        g.snn_substep(np.array([1. if k == 0 else 0.]))
        if fired_at is None and g.spikes:
            fired_at = k
        readout.append(g.current[0])
    first = next(i for i, v in enumerate(readout) if v != 0.)
    assert first == fired_at+1+4


def test_zero_readout_gives_zero_drive_and_outputs_never_spike():
    from experiments.algorithms.wingbeat_snn.networks import Graph
    spec = tiny_spec(weight=1e4)
    spec['edges'][1]['weight'] = 0.
    g = Graph(spec)
    for k in range(20):
        y = g.snn_substep(np.array([1.]))
        assert np.all(y == 0.)
    assert not g.spike_history[:, [0, 1]].any()


def test_neat_never_adds_input_to_output_edges(tmp_path):
    import neat
    from experiments.algorithms.wingbeat_snn import evolve
    random.seed(0)
    cfg = evolve.make_config(tmp_path, 'snn_neat_delay', 8)
    pop = neat.Population(cfg)
    evolve.initialize(pop, cfg, 'snn_neat_delay', 0)
    gc = cfg.genome_config
    for g in pop.population.values():
        for _ in range(200):
            g.mutate(gc)
        for (a, b), e in g.connections.items():
            assert not (a < 0 and b in gc.output_keys)
            assert a not in gc.output_keys
            assert 0. <= e.delay <= 20.
        spec = evolve.export(g, 'snn_neat_delay')
        assert spec['inputs'] == 11 and all(float(e['delay']).is_integer() for e in spec['edges'])


def test_initial_genome_zero_is_the_pid(tmp_path):
    import neat
    from experiments.algorithms.wingbeat_snn import evolve
    from experiments.algorithms.wingbeat_snn.networks import Graph
    for method in evolve.METHODS:
        cfg = evolve.make_config(tmp_path, method, 6)
        pop = neat.Population(cfg)
        evolve.initialize(pop, cfg, method, 1)
        first = pop.population[min(pop.population)]
        g = Graph(evolve.export(first, method))
        rng = np.random.default_rng(0)
        for _ in range(30):
            x = rng.normal(size=11)
            y = g.snn_substep(x) if g.kind == 'snn' else g.ann_step(x)
            assert np.all(y == 0.)


def test_zero_residual_network_reproduces_b2plus_bitwise(tmp_path):
    import neat
    from experiments.algorithms.snn_pid_compare.common import prepare_source
    from experiments.algorithms.wingbeat_snn import evolve
    from experiments.algorithms.wingbeat_snn.networks import GraphController
    from experiments.algorithms.wingbeat_snn.plant import Plant, HandLaw, cases, gusts
    source = tmp_path/'source'
    prepare_source(source)
    cfg = evolve.make_config(tmp_path, 'snn_neat_delay', 4)
    pop = neat.Population(cfg)
    evolve.initialize(pop, cfg, 'snn_neat_delay', 3, readout_std=.3)
    spec = evolve.export(pop.population[min(pop.population)], 'snn_neat_delay', residual=True, spike_scale=5.)
    c = cases('development', 1, seconds=1.4)[0]
    plant = Plant(source)
    alone = plant.run(c, HandLaw(highpass_s=1.), gusts(1., onsets=(1.1,)))
    q_alone = plant.env.sim.data.qpos.copy()
    with_net = plant.run(c, GraphController(spec), gusts(1., onsets=(1.1,)))
    assert alone['drive_activity'] > 0 and with_net['spikes_per_s'] > 0
    assert np.array_equal(q_alone, plant.env.sim.data.qpos)


def test_torch_snn_exports_to_identical_numpy_graph():
    torch = pytest.importorskip('torch')
    from experiments.algorithms.wingbeat_snn.phase_train import TorchSNN, GraphStep
    model = TorchSNN(hidden=6, seed=3).double()
    with torch.no_grad():
        model.w_out.mul_(20.)  # make the readout non-trivial
    x = np.random.default_rng(0).normal(size=(1, 120, 8))
    with torch.no_grad():
        expected = model(torch.from_numpy(x))[0].numpy()
    step = GraphStep(model.export())
    step.reset()
    got = np.array([step.step(row) for row in x[0]])
    assert step.g.spikes > 0
    assert np.allclose(got, expected, atol=1e-10)


def test_unknown_lag_only_moves_the_drive(tmp_path):
    from experiments.algorithms.snn_pid_compare.common import prepare_source
    from experiments.algorithms.wingbeat_snn.plant import Plant, HandLaw, cases, gusts
    source = tmp_path/'source'
    prepare_source(source)
    c = cases('development', 1, seconds=1.3)[0]
    plant = Plant(source)
    finals = {}
    for name, controller in (('none', None), ('hand', HandLaw(highpass_s=1.))):
        for lag in (0., 40.):
            plant.run(dict(c, phase_lag_deg=lag), controller, gusts(1., onsets=(1.05,)))
            finals[(name, lag)] = plant.env.sim.data.qpos.copy()
    assert np.array_equal(finals[('none', 0.)], finals[('none', 40.)])
    assert not np.array_equal(finals[('hand', 0.)], finals[('hand', 40.)])


def test_zero_drive_reproduces_the_original_plant_bitwise(tmp_path):
    from experiments.algorithms.snn_pid_compare.common import prepare_source
    from experiments.algorithms.robust_control.environment import ResearchEnv
    from experiments.algorithms.wingbeat_snn.plant import Plant, cases
    source = tmp_path/'source'
    prepare_source(source)
    c = dict(cases('development', 1, seconds=.1)[0], task='hover')
    reference = ResearchEnv(str(source), mode='pid', task='hover')
    reference.mode = 'pid'
    reference.configure(c)
    reference.reset(seed=c['seed'])
    while True:
        _, _, terminated, truncated, _ = reference.step(np.zeros(3))
        if terminated or truncated:
            break
    plant = Plant(source)
    plant.run(c, controller=None)
    assert np.array_equal(reference.sim.data.qpos, plant.env.sim.data.qpos)
    assert np.array_equal(reference.sim.data.qvel, plant.env.sim.data.qvel)
