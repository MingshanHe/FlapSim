"""Scientific interface checks before starting the new experiment."""
import numpy as np
import pytest
import torch
from experiments.algorithms.snn_pid_compare.common import prepare_source
from experiments.algorithms.snn_pid_compare.environment import (
    ControlEnv, case, direct_command, residual_command, LOW, HIGH,
)
from experiments.algorithms.snn_pid_compare.networks import Actor, initial_spec, policy
from experiments.algorithms.topology_hover.environment import HoverEnv


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    return prepare_source(tmp_path_factory.mktemp('pid_source'))


def test_action_authority():
    for base in (LOW, HIGH, (LOW+HIGH)/2):
        np.testing.assert_allclose(residual_command(np.zeros(3), base), base)
        np.testing.assert_allclose(residual_command(-np.ones(3), base), LOW)
        np.testing.assert_allclose(residual_command(np.ones(3), base), HIGH)
    np.testing.assert_allclose(direct_command(-np.ones(3)), LOW)
    np.testing.assert_allclose(direct_command(np.ones(3)), HIGH)


@pytest.mark.parametrize('kind', ['snn', 'ann'])
def test_continuous_export_parity_and_reset(kind):
    torch.set_num_threads(1)
    spec = initial_spec(kind, 8101)
    actor, numpy = Actor(spec), policy(spec)
    hidden = actor.initial_state(1)
    rng = np.random.default_rng(7101)
    with torch.no_grad():
        for index, observation in enumerate(rng.normal(0, .5, (1200, 11))):
            if index == 701:
                hidden = actor.initial_state(1)
                numpy.reset()
            mean, hidden = actor(torch.tensor(observation[None]), hidden)
            np.testing.assert_allclose(numpy.act(observation), mean.tanh()[0].numpy(), atol=2e-9, rtol=2e-8)


def test_snn_temporal_gradients():
    actor = Actor(initial_spec('snn', 8102))
    state = actor.initial_state(2)
    first = torch.ones(2, 11, dtype=torch.float64, requires_grad=True)
    _, state = actor(first, state)
    for _ in range(40):
        output, state = actor(torch.zeros_like(first), state)
    output.square().sum().backward()
    assert first.grad.isfinite().all() and first.grad.abs().sum() > 0
    assert actor.weight.grad.isfinite().all()
    assert actor.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['snn', 'ann'])
def test_zero_readout(kind):
    numpy = policy(initial_spec(kind, 8103, zero_readout=True))
    for _ in range(100):
        np.testing.assert_array_equal(numpy.act(np.ones(11)), np.zeros(3))


def test_teacher_query_is_idempotent(source):
    env = ControlEnv(source, mode='pid')
    env.configure(case(1110000011, .02, 'nominal', 0))
    env.reset()
    a = env.teacher_command()
    state = env.teacher.integral.copy()
    for _ in range(5):
        np.testing.assert_array_equal(env.teacher_command(), a)
    np.testing.assert_array_equal(env.teacher.integral, state)
    assert env.air.steps == 0


def test_pid_physics_matches_original_inner_loop(source):
    old = HoverEnv(source=source, mode='pid')
    new = ControlEnv(source, mode='pid')
    c = case(1110000022, .1, 'stress', 3)
    for env in (old, new):
        env.configure(c)
        env.reset(seed=c['seed'])
    assert new.sim.model.opt.timestep == pytest.approx(.000005)
    assert new.observation_space.shape == (11,)
    for _ in range(10):
        old.step(np.zeros(3))
        for _ in range(5):
            new.step(np.zeros(3))
        np.testing.assert_allclose(old.sim.data.qpos, new.sim.data.qpos, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(old.last_reading, new.last_reading, atol=1e-12, rtol=1e-12)


def test_zero_residual_is_pid_closed_loop(source):
    baseline = ControlEnv(source, mode='pid')
    residual = ControlEnv(source, mode='residual')
    c = case(1110000033, .1, 'stress', 5)
    for env in (baseline, residual):
        env.configure(c)
        env.reset(seed=c['seed'])
    for _ in range(50):
        a = baseline.step(np.zeros(3))
        b = residual.step(np.zeros(3))
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        np.testing.assert_array_equal(baseline.sim.data.qpos, residual.sim.data.qpos)


@pytest.mark.parametrize('kind', ['snn', 'ann'])
def test_neat_import_preserves_actor_and_fixed_tau(tmp_path, kind):
    from experiments.algorithms.snn_pid_compare.evolution import create_population, export
    spec = initial_spec(kind, 8105)
    cfg, population = create_population(tmp_path, spec, 8105, 8)
    first = population.population[min(population.population)]
    imported = export(first, kind)
    a, b = policy(spec), policy(imported)
    for observation in np.random.default_rng(81).normal(0, .5, (200, 11)):
        np.testing.assert_allclose(a.act(observation), b.act(observation), atol=2e-10)
    for _ in range(10):
        first.mutate(cfg.genome_config)
        for node in first.nodes.values():
            assert np.exp(node.log_tau_m) == pytest.approx(.02)
            assert np.exp(node.log_tau_s) == pytest.approx(.01)


def test_gae_terminal_and_truncation_are_different():
    from experiments.algorithms.snn_pid_compare.train_ppo import advantages, GAMMA
    rewards = np.array([[1., 1.]])
    values = np.array([[2., 2.]])
    next_values = np.array([[3., 3.]])
    adv, ret = advantages(rewards, values, next_values, np.array([[True, False]]), np.array([[True, True]]))
    np.testing.assert_allclose(adv, [[-1., -1.+GAMMA*3.]])
    np.testing.assert_allclose(ret, [[1., 1.+GAMMA*3.]])


def test_ppo_neat_budget_alignment():
    from experiments.algorithms.snn_pid_compare.train_neat import allocated_budget
    assert allocated_budget(16, 24) == 3_584_000
    assert allocated_budget(16, 24) % (8*500) == 0


def test_gate_treats_offline_error_as_diagnostic():
    from experiments.algorithms.snn_pid_compare.evaluate import gate
    nominal = dict(summary=dict(successes=16, tilt_rms_deg=3., height_rmse_m=.05))
    result = gate(nominal, dict(summary=dict(successes=4)), nominal, [.2, .1, .3])
    assert result['passed']
    assert not result['offline_diagnostic_pass']


def test_five_seed_exact_comparison_cannot_claim_p_below_point05():
    from experiments.algorithms.snn_pid_compare.analysis import paired_effect
    result = paired_effect([1]*5, [0]*5)
    assert result['sign_flip_p'] == .0625


def test_report_does_not_promote_failed_gate_to_completed_comparison(tmp_path):
    from experiments.algorithms.snn_pid_compare.common import write
    from experiments.algorithms.snn_pid_compare.reporting import generate
    root, destination = tmp_path/'run', tmp_path/'report'
    write(root/'status.json', dict(stage='development_gate_not_met'))
    generate(root, destination, update_index=False)
    report = (destination/'report.md').read_text()
    assert '正式 PPO/NEAT 比较没有启动' in report
    assert '正式独立测试' not in report
    assert read_json(root/'report_delivery.json')['formal_complete'] is False
    assert (destination/'technical_details.md').exists()


def read_json(path):
    import json
    return json.loads(path.read_text())


@pytest.mark.parametrize('pause', [False, True])
def test_formal_scheduler_preserves_jobs_budget_and_pause_boundaries(tmp_path, monkeypatch, pause):
    import threading
    import time
    from pathlib import Path
    from experiments.algorithms.snn_pid_compare import campaign
    protocol = dict(seeds=[5201, 5202, 5203, 5204, 5205], steps=3_584_000,
                    ppo_workers=8, simulation_workers_per_run=8, population=16,
                    generations=24, concurrent_training_runs=2)
    monkeypatch.setattr(campaign, 'freeze', lambda root: protocol)
    initializations = {f'{mode}_{kind}_{seed}': 'initial.json' for mode in ('direct', 'residual')
                       for kind in ('snn', 'ann') for seed in protocol['seeds']}
    monkeypatch.setattr(campaign, 'pretrain', lambda *args: initializations)
    finished = []
    monkeypatch.setattr(campaign, 'finish', lambda *args: finished.append(True))
    lock, barrier = threading.Lock(), threading.Barrier(2)
    jobs = []
    active, peak = 0, 0
    def fake_command(module, arguments, log):
        nonlocal active, peak
        args = list(map(str, arguments))
        with lock:
            active += 1
            peak = max(peak, active)
            jobs.append(Path(log).parent.name)
            number = len(jobs)
        assert args[args.index('--workers')+1] == '8'
        if module == 'train_ppo':
            assert args[args.index('--steps')+1] == '3584000'
        if number <= 2:
            barrier.wait(timeout=5)
        time.sleep(.002)
        if pause:
            (tmp_path/'pause_request.json').touch()
        with lock:
            active -= 1
    monkeypatch.setattr(campaign, 'command', fake_command)
    campaign.run(tmp_path, 16)
    assert peak == 2
    assert len(jobs) == len(set(jobs)) == (2 if pause else 40)
    assert bool(finished) is not pause
    if pause:
        assert read_json(tmp_path/'status.json')['stage'] == 'paused_at_run_boundary'

