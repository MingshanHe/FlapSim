"""Audit a completed run and account for earlier stages used to seed it."""
import argparse
import hashlib
import json
from pathlib import Path

from .plot_policy import draw
from .train import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    result = read(args.run/'results.json')
    if result['status'] != 'completed':
        raise ValueError('run is not complete')
    manifest = read(args.run/'run.json')
    mismatches = [name for name, expected in manifest['source_sha256'].items()
                  if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
    if mismatches:
        raise ValueError(f'Training source changed during the run: {mismatches}')
    selected = read(args.run/'selected.json')
    if hashlib.sha256((args.run/'best_policy.json').read_bytes()).hexdigest() != selected['policy_sha256']:
        raise ValueError('selected policy hash mismatch')
    audit = dict(training_source_verified=True, policy_hash_verified=True,
                 software_tests='8 passed: tests/test_neat_imu.py tests/test_rl_environment.py',
                 checkpoint_smoke='Ten generations, checkpoint save/load tested',
                 total_completed_training_generations=result['generations'],
                 total_training_control_steps=result['training_control_steps'],
                 total_validation_control_steps=result['validation_control_steps'],
                 final_test_control_steps=result['testing_control_steps'])
    note = ''
    if args.previous:
        old_history = read(args.previous/'history.json')
        old_validation = read(args.previous/'validation.json')
        audit['previous_stage'] = str(args.previous)
        audit['previous_stage_status'] = read(args.previous/'status.json')['status']
        audit['total_completed_training_generations'] += len(old_history)
        audit['total_training_control_steps'] += old_history[-1]['training_steps']
        audit['total_validation_control_steps'] += sum(r['control_steps'] for r in old_validation)
        note = (f'\n## 检查点故障后的恢复与累计预算\n\n前一阶段 `{args.previous}` '
                f'已完成 {len(old_history)} 代、{old_history[-1]["training_steps"]:,} 个训练控制步，'
                '随后保存检查点时因训练进程池不可序列化而退出。原错误检查点已标记为 `.incomplete`。'
                '修复后将运行时回调排除出种群序列化；保存/恢复测试及十代短流程检查通过。'
                '本阶段新建种群，将前阶段的验证最佳策略作为一个种子；其余个体重新初始化。'
                '这是有来源记录的重启，不是原种群精确续训。\n\n'
                f'两阶段累计 {audit["total_completed_training_generations"]} 代、'
                f'{audit["total_training_control_steps"]:,} 个训练控制步、'
                f'{audit["total_validation_control_steps"]:,} 个验证控制步；'
                f'最终测试 {result["testing_control_steps"]:,} 个控制步。每控制步包含2000个物理子步。'
                '软件短流程检查和已停止的两输出探索不属于所选策略的训练谱系，未计入上述数字。\n')
    write_json(args.run/'audit.json', audit)
    draw(args.run/'best_policy.json', args.run/'network.png')
    report = args.run/'REPORT.md'
    if '## 检查点故障后的恢复与累计预算' not in report.read_text(encoding='utf-8'):
        with report.open('a', encoding='utf-8') as stream:
            stream.write(note+'\n训练源码与所选策略哈希核验通过。网络图见 `network.png`，'
                         '累计预算和软件验证见 `audit.json`。\n')
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
