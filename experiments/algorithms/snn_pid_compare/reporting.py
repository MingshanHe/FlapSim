"""Read-only analysis of available results and local-only report delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime
from .common import ROOT, read, write, sha


def copy_records(root, destination):
    patterns = [
        'development_protocol.json', 'formal_protocol.json', 'runtime_plan.json',
        'preflight.json', 'training_smoke.json', 'teacher_validation.json',
        'precision_integrity.json', 'implementation_hashes_runtime.json',
        'development_results.json', 'gate_passed.json', 'status.json', 'events.jsonl',
        'analysis.json', 'paired_contrasts.json', 'costs.json', 'final_cases.json',
        'initializations.json', 'learning_curves.png', 'learning_curves.json',
        'source/provenance.json', 'requirements.lock.txt', 'conda-explicit.txt',
        'data/*/manifest.json', 'development/round*/*/gate.json',
        'development/round*/*/closed_loop_epoch*.json',
        'development/round*/*/long_validation.json', 'development/round*/*/stress_validation.json',
        'development/round*/*/arguments.json', 'development/round*/*/training.jsonl',
        'development/round*/*/best_policy.json', 'development/round*/*/status.json',
        'training/*/results.json', 'training/*/selection.json', 'training/*/best_policy.json',
        'training/*/history.jsonl', 'training/*/validation_*.json',
        'initialization/*/zero_residual.json', 'initialization/*/round*/best_policy.json',
        'initialization/*/round*/gate.json', 'initialization/*/round*/training.jsonl',
        'evaluation/*.json',
    ]
    hashes = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            relative = path.relative_to(root)
            target = destination/'records'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = path.read_bytes()
            target.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    write(destination/'record_manifest.json', dict(run_root=str(root), generated_unix=time.time(),
          copied_sha256=hashes,
          large_artifacts='NPZ trajectories, PyTorch checkpoints and full NEAT generation archives remain in run_root; dataset manifests contain individual trajectory hashes.'))


def development_plot(root, destination):
    paths = sorted(root.glob('development/round*/*/training.jsonl'))
    if not paths:
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rounds = sorted({p.parent.parent.name for p in paths})
    fig, axes = plt.subplots(2, len(rounds), figsize=(5*len(rounds), 7), squeeze=False, constrained_layout=True)
    for column, round_name in enumerate(rounds):
        for row, kind in enumerate(('snn', 'ann')):
            ax = axes[row, column]
            for path in paths:
                if path.parent.parent.name != round_name or not path.parent.name.startswith(kind):
                    continue
                records = {}
                for line in path.read_text().splitlines():
                    try:
                        item = json.loads(line)
                        records[item['epoch']] = item
                    except json.JSONDecodeError:
                        continue
                history = [records[k] for k in sorted(records)]
                ax.plot([r['epoch'] for r in history], [r['loss'] for r in history], label=path.parent.name)
            ax.set(title=f'{kind.upper()} / {round_name}', xlabel='Completed epoch', ylabel='Imitation loss')
            ax.legend(fontsize=8)
    fig.savefig(destination/'imitation_learning.png', dpi=150)
    plt.close(fig)
    return True


def generate(root, destination=None, update_index=True):
    root = Path(root).resolve()
    destination = Path(destination) if destination else ROOT/'artifacts/snn_pid_compare_20261010'
    destination.mkdir(parents=True, exist_ok=True)
    status = read(root/'status.json')
    stage = status['stage']
    results = list(root.glob('training/*/results.json'))
    formal_complete = stage == 'completed' and len(results) == 40 and (root/'analysis.json').exists()
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    lines = ['# PID 模仿与 SNN ANN 控制实验报告', '', f'更新：{now}。', '']
    if formal_complete:
        lines += ['**八组、每组五种子的正式训练与独立评估已完成。** 下表比较当前协议下的实际结果。', '']
    elif stage == 'development_gate_not_met':
        lines += ['**模仿阶段及限定的 DAgger 已结束，但六个开发模型未全部达到闭环门槛。正式 PPO/NEAT 比较没有启动。**',
                  '', '这说明当前网络、数据和训练预算还不足以稳定复现 PID；不能据此判断 PPO 与 NEAT、或 SNN 与 ANN 谁更好。', '']
    elif stage in ('failed', 'interrupted'):
        lines += [f'**流程在 `{stage}` 状态结束，结果不完整。** 已完成的检查点和指标保留，尚不能给出八组方法的最终排名。', '']
    else:
        lines += [f'**训练仍在进行，这是进度快照。** 当前阶段 `{stage}`，正式运行完成 {len(results)}/40；尚无完整比较结论。', '']
    lines += ['计算采用原来的逐时间步 Python 与 PyTorch 自动求导、float64 和 5 µs 物理步长。只增加独立任务并行，不使用自定义加速内核。', '',
              '[详细技术说明](technical_details.md) · [原始指标](records/) · [文件校验索引](record_manifest.json)', '',
              '## PID 模仿结果', '']
    if (root/'teacher_validation.json').exists():
        teacher = read(root/'teacher_validation.json')['summary']
        lines += [f'固定 PID 在共同常规验证中通过 {teacher["successes"]}/{teacher["episodes"]}；倾角 RMS {teacher["tilt_rms_deg"]:.3f}°，高度 RMS {100*teacher["height_rmse_m"]:.3f} cm。', '']
    gates = sorted(root.glob('development/round*/*/gate.json'))
    if gates:
        lines += ['以下是参与选模的验证结果，不是独立测试。第 0 轮为初始 PID 模仿，后续轮为共享 DAgger 数据聚合。', '',
                  '| 轮次 | 模型 | 选中轮数 | 常规成功 | 长期成功 | 扰动成功 | 倾角 RMS ° | 高度 RMS cm | 门槛 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---|']
        for path in gates:
            g = read(path)
            n = g['nominal']
            lines.append(f'| {path.parent.parent.name} | {path.parent.name} | {g["selected_checkpoint"]} | '
                         f'{n["successes"]}/16 | {g["long"]["successes"]}/4 | {g["stress"]["successes"]}/16 | '
                         f'{n["tilt_rms_deg"]:.3f} | {100*n["height_rmse_m"]:.3f} | {"通过" if g["passed"] else "未通过"} |')
        lines += ['', '离线动作误差、各项门槛和每回合终止原因见 `records/development/`。离线损失下降不等于闭环稳定；早期失败的 RMS 只覆盖失败前的短片段。', '']
    else:
        lines += ['闭环选模结果尚未全部生成。当前训练进度如下，轮数和损失不代表已经学会控制。', '',
                  '| 模型 | 完成轮数 | 训练损失 |', '|---|---:|---:|']
        for path in sorted(root.glob('development/round*/*/status.json')):
            s = read(path)
            lines.append(f'| {path.parent.parent.name}/{path.parent.name} | {s["epoch"]} | {s["loss"]:.6f} |')
        lines.append('')
    if development_plot(root, destination):
        lines += ['![模仿训练损失](imitation_learning.png)', '']
    if formal_complete:
        summary = read(root/'analysis.json')
        protocol = read(root/'formal_protocol.json')
        lines += ['## 正式独立测试', '',
                  '| 方法 | 常规成功率 | 扰动成功率 | 60 s 常规 | 60 s 扰动 |', '|---|---:|---:|---:|---:|']
        baseline_groups = [('pid', [summary['pid']])]
        baseline_groups += [(f'bc_{kind}', [summary[f'bc_{kind}_{s}'] for s in protocol['seeds']]) for kind in ('snn', 'ann')]
        for key, values in baseline_groups:
            cells = [sum(v[f]['success_rate'] for v in values)/len(values) for f in ('nominal', 'stress', 'long_nominal', 'long_stress')]
            lines.append('| '+key+' | '+' | '.join(f'{100*v:.1f}%' for v in cells)+' |')
        for mode in ('direct', 'residual'):
            for kind in ('snn', 'ann'):
                for method in ('ppo', 'neat'):
                    key = f'{mode}_{kind}_{method}'
                    values = [summary[f'{key}_seed{s}'] for s in protocol['seeds']]
                    cells = [sum(v[f]['success_rate'] for v in values)/len(values) for f in ('nominal', 'stress', 'long_nominal', 'long_stress')]
                    lines.append('| '+key+' | '+' | '.join(f'{100*v:.1f}%' for v in cells)+' |')
        lines += ['', '均值按五个训练种子计算；配对差值见 `records/paired_contrasts.json`，逐种子/单因素测试见 `records/analysis.json`。',
                  '成本与结构规模见 `records/costs.json`。五个种子的比较为探索性证据，不能把测试回合数量当成训练样本数量。', '',
                  '![正式学习曲线](records/learning_curves.png)', '']
    lines += ['## 并行效率与执行状态', '']
    runtime = read(root/'runtime_plan.json') if (root/'runtime_plan.json').exists() else {}
    measured = runtime.get('simulation_benchmark', {})
    if '8' in measured and '16' in measured:
        a, b = measured['8'], measured['16']
        lines += [f'同一批 32 个 2 s PID 场景：8 进程用时 {a["wall_seconds"]:.2f} s，16 进程 {b["wall_seconds"]:.2f} s，'
                  f'本次吞吐提高 {b["steps_per_second"]/a["steps_per_second"]:.2f} 倍。这是含启动开销的单批运行基准，不是整场训练的加速倍数或算法优势。', '']
    lines += ['开发阶段六个模型并行；物理评估最多十六进程；正式训练最多两个运行并行。保持每个训练进程一条计算线程，避免线程争抢。实际占用随任务阶段变化。', '',
              f'运行目录：`{root}`。状态、日志、模型和数据分别保存；当前生成本报告时完成正式运行 {len(results)}/40。', '',
              '## 同输入历史证据', '',
              '[旧 2201 阶段报告](../snn_hover_20261009/interim_2201/report.md)：11 维输入、PID 小幅残差、100 Hz 策略与 8 s 验证。五个演化组通过 8/8，两个 PPO 组为 5/8，且 PPO 选中的是训练前初始化。只有一个比较种子，不能据此认定算法普遍优劣。', '',
              '[旧五种子 topology_hover 正式报告](../topology_hover_20261008/report.md)：同为 11 维输入；其独立 20 s 扰动严格成功率为 NEAT 37.5%、循环 PPO 15.0%、前馈 PPO 22.5%。这是旧残差权限与训练协议下的结果，只作背景，不能与新实验合并统计。', '',
              '`ppo_v2` 因输入不匹配未使用。旧实验仍保留 8/35 次完成结果，剩余 27 次未恢复。', '',
              '## 如何理解本轮结论', '',
              '先判断网络能否独立维持飞行，再比较训练路线。残差组包含一个持续工作的 PID，直接组没有，因此残差获胜并不单独证明其网络更强。', '',
              '所有结果仅适用于当前仿真、预算和场景。yaw 与水平位置未闭环控制；没有实机能耗或实机悬停结论。', '']
    text = '\n'.join(lines)
    (destination/'report.md').write_text(text)
    relative_report = os.path.relpath(destination/'report.md', root)
    (root/'report.md').write_text('# PID 模仿与 SNN ANN 控制实验报告\n\n'
                                f'当前状态：`{stage}`。详见[本轮报告及技术细节]({relative_report})。\n')
    shutil.copyfile(Path(__file__).with_name('TECHNICAL_DETAILS.md'), destination/'technical_details.md')
    copy_records(root, destination)
    if update_index:
        index = ROOT/'report.md'
        previous = index.read_text()
        start, end = '<!-- snn-pid-v21:start -->', '<!-- snn-pid-v21:end -->'
        if start in previous and end in previous:
            a, b = previous.index(start), previous.index(end)+len(end)
            previous = (previous[:a]+previous[b:]).lstrip('\n')
        relative = destination.relative_to(ROOT)
        entry = (f'{start}\n## PID 模仿与 SNN ANN 对照（2026-10-10）\n\n'
                 f'当前状态：`{stage}`；正式运行完成 {len(results)}/40。'
                 f'[本轮 report.md]({relative}/report.md) · [详细技术说明]({relative}/technical_details.md)。\n\n'
                 f'使用原逐时间步计算；结果只提交本地新分支，由用户自行上传 GitHub。\n{end}\n\n')
        index.write_text(entry+previous)
    write(root/'report_delivery.json', dict(generated_unix=time.time(), artifact_path=str(destination),
          campaign_stage=stage, formal_complete=formal_complete, formal_runs=len(results), publication='local_only'))
    return destination


def commit_report(root, destination, branch):
    """Only the requested report paths can enter this local completion commit."""
    def git(*arguments):
        return subprocess.run(['git', *arguments], cwd=ROOT, text=True, capture_output=True, check=True)
    receipt = dict(unix=time.time(), requested_branch=branch, push_attempted=False)
    try:
        current = git('branch', '--show-current').stdout.strip()
        if current != branch:
            receipt.update(status='not_committed', reason=f'Current branch is {current}; refusing to change user branch')
        else:
            paths = ['report.md', str(Path(destination).relative_to(ROOT))]
            git('add', '--', *paths)
            diff = git('diff', '--cached', '--name-only', '--', *paths).stdout.strip()
            if diff:
                git('-c', 'user.name=Codex', '-c', 'user.email=codex@localhost',
                    'commit', '--only', '-m', 'Record PID imitation campaign outcome and technical report', '--', *paths)
            receipt.update(status='committed' if diff else 'unchanged', commit=git('rev-parse', 'HEAD').stdout.strip())
    except subprocess.CalledProcessError as exc:
        receipt.update(status='not_committed', reason=exc.stderr[-2000:])
    write(Path(root)/'local_publication.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit-local', action='store_true')
    parser.add_argument('--branch', default='research/snn-pid-v21-performance')
    args = parser.parse_args()
    destination = generate(args.root)
    if args.commit_local:
        commit_report(args.root, destination, args.branch)
    print(destination/'report.md')


if __name__ == '__main__':
    main()
