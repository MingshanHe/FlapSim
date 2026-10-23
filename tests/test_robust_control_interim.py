"""Protect the interim holdout, paired cohort and sealed checkpoint boundary."""
from collections import Counter
from pathlib import Path

import pytest

from experiments.algorithms.robust_control.campaign import final_cases, job_name
from experiments.algorithms.robust_control.environment import scenarios
from experiments.algorithms.robust_control.interim import (
    FAMILIES, interim_cases, model_definitions, seal_model, report)
from experiments.algorithms.snn_pid_compare.common import read, write, sha


def test_interim_holdout_keeps_severity_and_has_disjoint_rng_seeds():
    cases = interim_cases()
    originals = [c for c in final_cases('hover') if c['test_family'] in FAMILIES]
    assert Counter(c['test_family'] for c in cases) == dict(
        nominal=16, stress=16, long_nominal=4, long_stress=4, ood=8)
    seeds = {c['seed'] for c in cases}
    assert len(seeds) == 48
    forbidden = {c['seed'] for c in final_cases('hover')+final_cases('velocity')}
    for split in ('development', 'validation'):
        for task in ('hover', 'velocity'):
            for condition in ('nominal', 'stress', 'ood'):
                for seconds in (8., 12., 20., 60.):
                    forbidden.update(c['seed'] for c in scenarios(split, 32, seconds, condition, task))
    assert not seeds & forbidden
    assert min(seeds) > 1_590_000_000  # upper bound of v3 training RNG range
    for c, old in zip(cases, originals):
        assert {k: v for k, v in c.items() if k not in ('seed', 'split')} == {
            k: v for k, v in old.items() if k != 'seed'}


def test_core_includes_all_methods_and_first_three_seeds_without_ranking():
    protocol = dict(jobs=[dict(name=job_name('hover', 'residual', kind, method, seed))
                         for kind in ('snn', 'ann') for method in ('neat', 'ppo', 'evo')
                         for seed in range(6201, 6206)])
    models = model_definitions(protocol)
    assert len(models) == 18
    core = [m for m in models if m['role'] == 'core']
    assert len(core) == 13
    assert core[0]['name'] == 'hover_pid'
    for kind in ('snn', 'ann'):
        for method in ('neat', 'ppo'):
            assert all(any(m['name'] == job_name('hover', 'residual', kind, method, seed)
                           for m in core) for seed in (6201, 6202, 6203))
    assert sum(m['role'] == 'fixed_topology' for m in models) == 3


def test_seal_rejects_unfinished_training_and_snapshots_selected_policy(tmp_path):
    root, interim = tmp_path/'original', tmp_path/'interim'
    model = dict(name='example', mode='residual', role='core', job='example')
    with pytest.raises(ValueError, match='unfinished'):
        seal_model(root, interim, model)
    source = root/'training/example'
    write(source/'results.json', dict(status='completed'))
    write(source/'selection.json', dict(generation=8, summary=dict(successes=15)))
    write(source/'best_policy.json', dict(chosen='validation-only'))
    seal_model(root, interim, model)
    sealed = read(interim/'models/example.json')
    assert not sealed['selection_used_interim_tests']
    assert sealed['selection']['generation'] == 8
    write(source/'best_policy.json', dict(chosen='changed later'))
    assert read(sealed['policy_path']) == dict(chosen='validation-only')
    assert sha(sealed['policy_path']) == sealed['policy_sha256']
    write(sealed['policy_path'], dict(chosen='tampered'))
    with pytest.raises(ValueError, match='Changed'):
        seal_model(root, interim, model)


def test_high_overall_success_cannot_hide_failed_long_stress_gate(tmp_path):
    root, interim, destination = tmp_path/'root', tmp_path/'interim', tmp_path/'artifacts'
    plan = dict(models=[dict(name='hover_pid')], robust_gate=dict(
        nominal=15/16, stress=14/16, long_nominal=1., long_stress=.75))
    rows = []
    for c in interim_cases():
        success = c['test_family'] != 'long_stress'
        rows.append(dict(case=c, success=success, tilt_rms_deg=5., height_rmse_m=.04,
                         success_checks=dict(tilt=success, full_duration=True)))
    # 44/48 overall is not sufficient: the predeclared long-stress gate fails.
    write(interim/'evaluation/hover_pid.json', dict(episodes=rows))
    report(root, interim, destination, plan)
    analysis = read(interim/'analysis.json')
    assert analysis['families']['hover_pid']['long_stress']['successes'] == 0
    assert not analysis['verdicts']['hover_pid']['passes_interim_gate']
    text = (destination/'interim_20261012/report.md').read_text()
    assert '| hover_pid | 16 | 16 | 4 | 0 | 8 | 未通过 |' in text
