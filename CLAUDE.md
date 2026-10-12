# CLAUDE.md（分支 research/wingbeat-synchronous-snn）

本分支研究单电机扑翼机器人的"翅拍同步施力（split-cycle）"抗阵风控制：只用 IMU 锁定翅拍相位，并比较脉冲网络（SNN）、NEAT 与经典控制。论文写作之前的正式实验在其他电脑上运行。

**开始工作前先读** [docs/wingbeat_paper/HANDOFF.md](docs/wingbeat_paper/HANDOFF.md)，其中有完整背景、用户做过的决定和全部结果。再按需读：
- [docs/wingbeat_paper/RUNBOOK.md](docs/wingbeat_paper/RUNBOOK.md)：正式实验怎么跑；
- [docs/wingbeat_paper/paper_outline.md](docs/wingbeat_paper/paper_outline.md)：论文大纲和各主张的判据。

v3（`AI_HANDOFF.md`）和 v4 是另外的研究线，与本分支无关。

## 常用命令

```bash
export PYTHON=$(which python)                                   # 需要 MuJoCo 3.8.0、torch 2.6.0、neat-python 2.0.0
bash experiments/algorithms/wingbeat_snn/paper/check_env.sh     # 环境检查和 9 个单元测试
QUICK=1 bash experiments/algorithms/wingbeat_snn/paper/run_all.sh   # 约 10 分钟的端到端冒烟测试
tmux new-session -d -s wingbeat_paper "PYTHON=$PYTHON bash experiments/algorithms/wingbeat_snn/paper/run_all.sh 2>&1 | tee -a runs/wingbeat_paper.log"
python -m experiments.algorithms.wingbeat_snn.paper_stats --input artifacts/wingbeat_paper --output artifacts/wingbeat_paper/report
```

## 约定

- 用户使用中文交流：先给结论，再给依据；需要用户做决定的地方明确提问。
- 所有结论都必须与"同预算重新调参的经典控制律"比较；否定结果同样如实报告。
- 长任务放在 tmux 里运行，不要修改正在执行的 shell 脚本。等待时用后台等待任务加完成通知，不要用长时间 sleep 轮询。
- 推送 GitHub、删除数据等对外或不可逆的操作，先征得用户同意。
- `runs/` 不进 git；需要保留的结果放在 `artifacts/`。
