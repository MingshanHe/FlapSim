"""Deadline-focused interim evidence without changing the sealed v3 experiment.

Drain running jobs, evaluate a paired three-seed cohort on a new holdout, move
predeclared mechanistic controls forward, then resume all remaining v3 jobs.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from ..snn_pid_compare.common import ROOT, read, write, sha, event, heartbeat
from ..snn_pid_compare.analysis import paired_effect
from .campaign import prepare, final_cases, train_one
from .reporting import generate

OWNER = 'interim_20261012'
SEEDS = [6201, 6202, 6203]
FAMILIES = ['nominal', 'stress', 'long_nominal', 'long_stress', 'ood']


def interim_cases():
    cases = [dict(c) for c in final_cases('hover') if c['test_family'] in FAMILIES]
    counters = Counter()
    for c in cases:
        family = c['test_family']
        # A wholly separate RNG namespace; preserve the declared severity,
        # durations, phase coverage and success thresholds of the final test.
        c['seed'] = 1_700_000_000 + 100_000*FAMILIES.index(family) + counters[family]
        c['split'] = 'interim_20261012'
        counters[family] += 1
    return cases


def model_definitions(protocol):
    core = [dict(name='hover_pid', mode='pid', role='core', job=None)]
    for kind, method in [('snn', 'neat'), ('ann', 'neat'), ('snn', 'ppo'), ('ann', 'ppo')]:
        for seed in SEEDS:
            name = f'hover_residual_{kind}_{method}_seed{seed}'
            core.append(dict(name=name, mode='residual', role='core', job=name))
    controls = [dict(name=f'hover_residual_snn_evo_seed{s}', mode='residual',
                     role='fixed_topology', job=f'hover_residual_snn_evo_seed{s}') for s in SEEDS]
    supplements = [dict(name=f'hover_residual_snn_neat_seed{s}', mode='residual',
                        role='additional_snn_neat_seed', job=f'hover_residual_snn_neat_seed{s}')
                   for s in (6204, 6205)]
    known = {j['name'] for j in protocol['jobs']}
    assert all(m['job'] is None or m['job'] in known for m in core+controls+supplements)
    return core+controls+supplements


def eta(protocol, root):
    """Estimate total remaining wall time, with all assumptions recorded."""
    hours = {}
    completed = set()
    for j in protocol['jobs']:
        p = root/'training'/j['name']/'results.json'
        if p.exists():
            r = read(p)
            key = (j['kind'], j['method'])
            hours.setdefault(key, []).append(r['wall_seconds']/3600/j['budget'])
            completed.add(j['name'])
    def estimate_job(j):
        key = (j['kind'], 'neat' if j['method'] == 'evo' else j['method'])
        values = hours[key]
        return sum(values)/len(values)*j['budget']
    remaining = sum(estimate_job(j) for j in protocol['jobs'] if j['name'] not in completed)/2
    return dict(completed=len(completed), total=len(protocol['jobs']),
                remaining_training_hours_approx=remaining,
                assumptions='Two concurrent jobs; completed hover rates scaled by allocated steps; active runs counted in full. Direct-control early failures and velocity costs can change this estimate.',
                final_evaluation_hours_range=[12, 30],
                full_remaining_days_range=[4, 6])


def make_plan(root, interim, protocol):
    path = interim/'plan.json'
    if path.exists():
        if sha(path) != read(interim/'plan_seal.json')['sha256']:
            raise ValueError('Interim plan changed after sealing')
        return read(path)
    jobs = [f'hover_residual_snn_evo_seed{s}' for s in SEEDS]
    # This fills the fourth worker slot in two two-job batches and completes
    # all five predeclared SNN-NEAT seeds without changing the paired cohort.
    jobs.append('hover_residual_snn_neat_seed6205')
    plan = dict(version=1, created_unix=time.time(), owner=OWNER,
                deadline_jst='2026-10-12T14:00:00+09:00',
                original_protocol_sha256=sha(root/'protocol.json'),
                seeds=SEEDS, models=model_definitions(protocol), priority_training=jobs,
                cases=interim_cases(), robust_gate=protocol['robust_gate'],
                training_changes='execution order only; original budgets, code, initializations, selection and training scenarios unchanged',
                cohort_selection='first three predeclared seeds for all four methods, regardless of validation rank; SNN-NEAT seeds 4/5 secondary only',
                test_usage='new interim holdout; never used to select checkpoints or tune gains; original final tests remain unopened',
                limits=['Interim plan was written after observing existing validation results.',
                        'Three paired training seeds are descriptive evidence, not proof of superiority.',
                        'All outcomes, including failures and OOD failures, must be reported.',
                        'Any subsequent adaptation requires a new experiment and another untouched test set.'],
                eta_at_plan=eta(protocol, root))
    write(path, plan)
    write(interim/'plan_seal.json', dict(sha256=sha(path), sealed_unix=time.time()))
    return plan


def seal_model(root, interim, model):
    target = interim/'models'/f'{model["name"]}.json'
    if target.exists():
        sealed = read(target)
        if sealed['policy_path'] and sha(sealed['policy_path']) != sealed['policy_sha256']:
            raise ValueError('Changed interim policy snapshot')
        return
    sealed = dict(**model, sealed_unix=time.time(), policy_path=None, policy_sha256=None,
                  selection_used_interim_tests=False)
    if model['job']:
        source = root/'training'/model['job']
        if not (source/'results.json').exists():
            raise ValueError(f'Cannot evaluate unfinished training: {model["name"]}')
        policy = interim/'policies'/f'{model["name"]}.json'
        policy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/'best_policy.json', policy)
        sealed.update(policy_path=str(policy), policy_sha256=sha(policy),
                      selection=read(source/'selection.json'),
                      results_sha256=sha(source/'results.json'))
    write(target, sealed)


def summarize_families(result):
    families = {}
    for family in FAMILIES:
        rows = [r for r in result['episodes'] if r['case']['test_family'] == family]
        families[family] = dict(successes=sum(r['success'] for r in rows), episodes=len(rows),
            tilt_rms_deg=sum(r['tilt_rms_deg'] for r in rows)/len(rows),
            height_rmse_m=sum(r['height_rmse_m'] for r in rows)/len(rows),
            unmet_criteria=dict(Counter(k for r in rows for k, v in r['success_checks'].items() if not v)))
    return families


def report(root, interim, destination, plan):
    out = destination/'interim_20261012'
    out.mkdir(parents=True, exist_ok=True)
    status = read(interim/'status.json') if (interim/'status.json').exists() else {}
    results = {p.stem: summarize_families(read(p)) for p in (interim/'evaluation').glob('*.json')}
    verdicts = {}
    for name, families in results.items():
        checks = {f: families[f]['successes']/families[f]['episodes'] >= threshold
                  for f, threshold in plan['robust_gate'].items()}
        verdicts[name] = dict(checks=checks, passes_interim_gate=all(checks.values()))
    contrasts = {}
    comparisons = [('snn_neat', 'snn_ppo'), ('ann_neat', 'ann_ppo'),
                   ('snn_neat', 'ann_neat'), ('snn_ppo', 'ann_ppo'), ('snn_neat', 'snn_evo')]
    for left, right in comparisons:
        names = [[f'hover_residual_{arm}_seed{s}' for s in SEEDS] for arm in (left, right)]
        if all(n in results for group in names for n in group):
            contrasts[left+'__minus__'+right] = paired_effect(*[
                [results[n]['stress']['successes']/results[n]['stress']['episodes'] for n in group]
                for group in names])
    if 'hover_pid' in results:
        for arm in ('snn_neat', 'snn_ppo', 'ann_neat', 'ann_ppo'):
            names = [f'hover_residual_{arm}_seed{s}' for s in SEEDS]
            if all(n in results for n in names):
                base = results['hover_pid']['stress']
                contrasts[arm+'__minus__pid'] = paired_effect(
                    [results[n]['stress']['successes']/results[n]['stress']['episodes'] for n in names],
                    [base['successes']/base['episodes']]*len(SEEDS))
    write(interim/'analysis.json', dict(families=results, verdicts=verdicts, paired_contrasts=contrasts))
    timestamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M JST')
    lines = ['# SNN＋NEAT 阶段性独立评估', '', f'更新：{timestamp}。目标：2026-10-12 14:00 JST 前。', '',
             f'状态：`{status.get("stage", "waiting")}`；完整评估 {len(results)}/{len(plan["models"])} 个控制器。', '',
             '先比较 PID 与 SNN/ANN × NEAT/PPO 的前三个预定种子，再加入三个固定拓扑 SNN 进化对照；SNN＋NEAT 的第 4、5 个种子单列为补充证据。', '',
             '每个控制器：16×20 s 常规、16×20 s 扰动、4×60 s 常规、4×60 s 扰动、8×20 s 组合 OOD，共 48 例。所有控制器使用相同案例。阶段测试采用新随机种子，原最终测试集保持未访问。', '',
             '本计划制定前已看过旧验证结果；本次阶段测试不参与训练、调参或检查点选择。计算方式、训练预算和通过门槛未改变。', '',
             '## 实际阶段测试结果', '',
             '| 控制器 | 常规 /16 | 扰动 /16 | 长时常规 /4 | 长时扰动 /4 | OOD /8 | 四项门槛 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for model in plan['models']:
        name = model['name']
        if name not in results:
            continue
        f = results[name]
        cells = ' | '.join(str(f[k]['successes']) for k in FAMILIES)
        gate = '通过' if verdicts[name]['passes_interim_gate'] else '未通过'
        lines.append(f'| {name} | {cells} | {gate} |')
    if not results:
        lines += ['', '尚无完整阶段测试结果。原验证的 16/16 等数字不填入本表。']
    lines += ['', '四项门槛为常规 ≥15/16、扰动 ≥14/16、长时常规 4/4、长时扰动 ≥3/4；OOD 单独报告。只表示在本阶段仿真包络内达标。', '',
              '## 比较与结论边界', '']
    for key, effect in contrasts.items():
        lines.append(f'- `{key}`：三个配对种子的扰动成功率差为 {effect["seed_differences"]}，均值 {100*effect["mean"]:.1f} 个百分点。')
    passing = [n for n, v in verdicts.items() if v['passes_interim_gate']]
    lines += ['', f'目前通过四项门槛的已评估控制器：{", ".join(passing) or "暂无"}。', '',
              '三个训练种子只支持初步比较；精确双侧符号翻转检验最小 p=0.25，不宣称统计显著。不能把 48 个场景当作 48 个独立训练种子，也不能把最好的一个种子代表整种方法。', '',
              'SNN＋NEAT 与 SNN＋固定拓扑进化的配对差异用于检查结构进化的增益；若没有改善，如实报告。阶段测试不评估速度指令，也不回答所有 ANN/SNN 是否都无法替代 PID。', '',
              '完整逐例结果、失败判据、选模记录、策略副本、协议和哈希见本目录 JSON 文件。原 65 组训练会在阶段评估结束后继续；本地保存，不上传 GitHub。', '']
    core_names = [f'hover_residual_snn_neat_seed{s}' for s in SEEDS]
    if all(n in results for n in core_names):
        count = sum(verdicts[n]['passes_interim_gate'] for n in core_names)
        scores = [results[n]['stress']['successes'] for n in core_names]
        lines += ['## 当前证据能支持什么', '',
                  f'SNN＋NEAT 三个主比较种子的扰动通过数为 {scores}（每种子 16 例），其中 {count}/3 通过四项门槛。']
        if count == 3:
            lines += ['这支持“在本阶段仿真包络和给定预算下，PID 内环＋SNN-NEAT 残差可重复获得达标控制器”。与其他方法的优劣仍需结合配对差异、全部五个种子及原最终测试。']
        elif count:
            lines += ['这支持部分种子的策略可行，但尚不足以称该训练方法能稳定产生稳健策略。未达标种子不能省略。']
        else:
            lines += ['当前阶段测试尚不支持已经获得满足预定稳健门槛的 SNN-NEAT 控制器；原验证集的高分不能覆盖这个结果。']
        fixed = contrasts.get('snn_neat__minus__snn_evo')
        if fixed:
            lines += [f'结构进化相对固定拓扑权重进化的扰动成功率平均差为 {100*fixed["mean"]:.1f} 个百分点；'
                      '只有三组配对训练，差异用于描述本实验，不能泛化为结构进化必然有效。']
        lines += ['']
    (out/'report.md').write_text('\n'.join(lines))
    selected = [interim/name for name in ('plan.json', 'plan_seal.json', 'status.json',
                'analysis.json', 'orchestration_source_manifest.json', 'test_access.json')]
    selected += list((interim/'orchestration_source').glob('*.py'))
    selected += list((interim/'models').glob('*.json'))
    selected += list((interim/'policies').glob('*.json'))
    selected += list((interim/'evaluation').glob('*.json'))
    manifest = {}
    for p in selected:
        if not p.exists():
            continue
        relative = p.relative_to(interim)
        target = out/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest[str(relative)] = dict(sha256=sha(target), bytes=target.stat().st_size)
    write(out/'record_manifest.json', manifest)


def change_status(root, interim, destination, plan, stage, owns_queue=False, **details):
    record = dict(stage=stage, unix=time.time(), pid=os.getpid(), final_tests_accessed=False,
                  interim_tests_accessed=(interim/'test_access.json').exists(), **details)
    write(interim/'status.json', record)
    if owns_queue:
        completed = sum((root/'training'/j['name']/'results.json').exists() for j in read(root/'protocol.json')['jobs'])
        event(root, stage, completed=completed, total=65, interim_root=str(interim),
              **{k: v for k, v in record.items() if k not in ('stage', 'unix', 'pid')})
    report(root, interim, destination, plan)


def evaluate_models(root, interim, destination, plan, models):
    # Seal every member of a cohort before inspecting any of its test results.
    for model in models:
        seal_model(root, interim, model)
    runner = interim/'orchestration_source/interim_evaluate.py'
    for model in models:
        if sha(interim/'plan.json') != read(interim/'plan_seal.json')['sha256']:
            raise ValueError('Interim plan changed during execution')
        if not (interim/'test_access.json').exists():
            write(interim/'test_access.json', dict(unix=time.time(), first_model=model['name'],
                  plan_sha256=sha(interim/'plan.json'), original_final_tests_accessed=False))
        change_status(root, interim, destination, plan, 'interim_evaluation', owns_queue=True,
                      active_model=model['name'], runs=[],
                      evaluation_progress_path=str(interim/'evaluation_progress.json'))
        args = [sys.executable, '-u', str(runner), '--root', str(root), '--interim', str(interim),
                '--model', model['name'], '--plan-sha256', sha(interim/'plan.json'), '--workers', '16']
        with (interim/'evaluation.log').open('a') as log:
            subprocess.run(args, cwd=read(root/'protocol.json')['frozen_source'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report(root, interim, destination, plan)


def commit_interim(protocol, interim, destination):
    """Save local evidence; a Git problem must not strand the training queue."""
    try:
        repository = Path(protocol['repository'])
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repository, text=True).strip()
        if branch != protocol['branch']:
            write(interim/'local_delivery.json', dict(committed=False, reason='branch changed', pushed=False))
            return
        path = str(destination.relative_to(repository))
        subprocess.run(['git', 'add', '--', path], cwd=repository, check=True)
        changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', path], cwd=repository).returncode
        if changed:
            subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@localhost',
                            'commit', '-m', 'Record held-out interim SNN NEAT comparisons',
                            '--only', '--', path], cwd=repository, check=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
        write(interim/'local_delivery.json', dict(committed=True, commit=commit, branch=branch, pushed=False))
    except Exception:
        write(interim/'local_delivery.json', dict(committed=False, pushed=False, traceback=traceback.format_exc()))


def execute(root, interim, destination, protocol, plan):
    with (root/'campaign.lock').open('a') as lock:
        change_status(root, interim, destination, plan, 'waiting_for_active_runs_to_finish')
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(10)
                write(interim/'waiting_heartbeat.json', dict(unix=time.time(), pid=os.getpid()))
        pause = root/'pause_request.json'
        if not pause.exists() or read(pause).get('owner') != OWNER:
            raise ValueError('The owned queue handoff request is missing or was replaced')
        prepare(root, destination)  # verify all original computation/asset hashes
        if sha(root/'protocol.json') != plan['original_protocol_sha256']:
            raise ValueError('Original protocol changed')
        with heartbeat(root):
            change_status(root, interim, destination, plan, 'interim_handoff_complete', owns_queue=True)
            generate(root, destination)
            core = [m for m in plan['models'] if m['role'] == 'core']
            evaluate_models(root, interim, destination, plan, core)
            pending = [j for name in plan['priority_training'] for j in protocol['jobs']
                       if j['name'] == name and not (root/'training'/name/'results.json').exists()]
            with ThreadPoolExecutor(2) as pool:
                futures = {pool.submit(train_one, root, protocol, j): j for j in pending}
                change_status(root, interim, destination, plan, 'interim_priority_training', owns_queue=True,
                              priority_training=[j['name'] for j in pending])
                for future in as_completed(futures):
                    future.result()
                    generate(root, destination)
                    report(root, interim, destination, plan)
            evaluate_models(root, interim, destination, plan, [m for m in plan['models'] if m['role'] != 'core'])
            change_status(root, interim, destination, plan, 'interim_completed', owns_queue=True,
                          deadline_met=time.time() <= datetime.fromisoformat(plan['deadline_jst']).timestamp(),
                          original_queue_will_resume=True)
            generate(root, destination)
            commit_interim(protocol, interim, destination)
            if read(pause).get('owner') != OWNER:
                raise ValueError('New pause request must be honored; refusing automatic resume')
            pause.unlink()
        fcntl.flock(lock, fcntl.LOCK_UN)
    # The original launcher skips completed priority jobs; nothing is discarded.
    args = [sys.executable, '-u', '-m', 'experiments.algorithms.robust_control.campaign',
            '--root', str(root), '--destination', str(destination)]
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    root, destination = a.root.resolve(), a.destination.resolve()
    interim = root/'interim_20261012'
    interim.mkdir(parents=True, exist_ok=True)
    protocol = prepare(root, destination)
    with (interim/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = make_plan(root, interim, protocol)
        sources = interim/'orchestration_source'
        sources.mkdir(exist_ok=True)
        for name in ('interim.py', 'interim_evaluate.py'):
            target = sources/name
            if not target.exists():
                shutil.copyfile(Path(__file__).with_name(name), target)
            elif sha(target) != sha(Path(__file__).with_name(name)):
                raise ValueError('Orchestration code changed; record an explicit amendment before resuming')
        write(interim/'orchestration_source_manifest.json', {p.name: sha(p) for p in sources.glob('*.py')})
        change_status(root, interim, destination, plan, 'prepared')
        if a.prepare_only:
            return
        try:
            execute(root, interim, destination, protocol, plan)
        except Exception:
            change_status(root, interim, destination, plan, 'interim_or_resume_failed', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
