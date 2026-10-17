# CLAUDE.md（分支 research/original-env-snn-neat）

本分支在**原 v3 环境接口内**寻找以 SNN 和 NEAT 为核心的论文创新点。当前在验证"脉冲定时转向"：进化出的脉冲回路在特定翅拍相位发出电机脉冲，类似昆虫的转向肌。

原接口指：
- 传感：IMU 与测距高度；
- 执行：云台 α/β 在 ±0.3 rad 内，电机转速在 3115–3807 rad/s 内；每 2 ms 更新一次，经过 8.2 ms 平滑。

**开始工作前先读** [docs/original_env_snn_neat/HANDOFF.md](docs/original_env_snn_neat/HANDOFF.md)，其中有：
- 当前在跑的任务；
- 用户的决定；
- 全部结果；
- 下一步决策树。

再按需读：
- [artifacts/neuro_gates/gates_report.md](artifacts/neuro_gates/gates_report.md)：原环境的预备检验；
- [artifacts/spike_steer/](artifacts/spike_steer/)：脉冲定时转向 v1、v2、v3 的结果，以及备选方案工作流的输出。

旧的翅拍同步线（`docs/wingbeat_paper/`，分支 `research/wingbeat-synchronous-snn`）改动了执行器接口，已不在范围内。v3 线（`AI_HANDOFF.md`）和 v4 线是另外的研究线。

## 常用命令

```bash
PY=/home/robot/miniconda3/envs/mujoco/bin/python      # MuJoCo 3.8.0、torch 2.6.0、neat-python 2.0.0
$PY -m experiments.algorithms.spike_steer.pilot_v3 selftest                 # 不跑仿真的自测
$PY -m experiments.algorithms.spike_steer.pilot_v3 all --quick --output runs/spike_steer/pilot_v3_quick   # 约 6 分钟的冒烟测试
tmux new-session -d -s spike_steer_v3 "bash runs/run_steer_v3.sh 2>&1 | tee -a runs/spike_steer_v3.log"   # 正式 v3，约 2 小时
$PY -m pytest tests/test_wingbeat_snn.py -q                                  # 10 个单元测试
```

## 约定

- 用户使用中文交流：先给结论，再给依据；需要用户做决定的地方明确提问。
- 不改传感器、执行器、更新率或限幅。可以新增测试场景，但要说明。
- 所有结论都必须与"同预算重新调参的经典控制"比较；否定结果同样如实报告。判据在看结果之前写下。
- 长任务放在 tmux 里运行。不要修改正在执行的脚本，也不要修改之后的阶段还会重新导入的 Python 模块。等待时用后台等待任务加完成通知，不要用长时间 sleep 轮询。
- 推送 GitHub、删除数据等对外或不可逆的操作，先征得用户同意。WSL 里的 git 没有凭据，推送要用 Windows 端的 git，见 HANDOFF 第 5 节。
- `runs/` 不进 git；需要保留的结果放在 `artifacts/`。
