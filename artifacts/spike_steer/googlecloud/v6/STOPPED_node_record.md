# Plan v6 ES：GoogleCloud 节点手动停止记录

- 停止时间：2026-10-20 03:05 UTC（用户要求）
- 停止方式：结束 tmux 会话 v6_es，杀掉所有 descend8 finetune 进程
- 机器：32 vCPU AMD EPYC 7B13，Ubuntu 26.04.1，g++ 15.2（精确性检查 MISMATCH，按修正 1 只做 ES 搜索）
- 代码：~/flapsim，提交 3fd14d9

## 断点

| 种子 | checkpoint.npz 的 gen | history.jsonl 行数 | 最后完成的代 | 最后中心目标值 |
|---|---|---|---|---|
| 1 | 28 | 28 | 27 | 2.1800 |
| 2 | 28 | 28 | 27 | 1.5597 |
| 3 | 28 | 28 | 27 | 1.7981 |
| 5 | 28 | 28 | 27 | 1.6749 |

每个种子都已完成第 0–27 代。停止时正在跑的第 28 代没有完成，结果会丢弃；续跑时从第 28 代开始重算。
已保存的中心：runs/spike_steer/v6/finetune_s*/spec_val_g{-1,4,9,14,19,24}.json

## 续跑

    bash ~/v6_es.sh        # 或：tmux new-session -d -s v6_es "bash ~/v6_es.sh 2>&1 | tee -a ~/v6_es.log"

descend8 finetune 会读取 checkpoint.npz，日志里会打印 "resumed at generation 28"。
