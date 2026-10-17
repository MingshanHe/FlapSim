# 投稿前实验清单（RUNBOOK）：在另一台电脑上运行

本文件列出论文 [大纲](paper_outline.md) 所需的全部正式实验：环境搭建、命令、预计耗时、输出位置、验收标准，以及结果如何回传。

**所有脚本都可以中断后续跑**：每一步的输出存在就跳过。2026-10-12 已在原机器（20 个逻辑 CPU、28 GB 内存、WSL2 Ubuntu）上用 QUICK 模式端到端跑通。

## 0. 总览

| 编号 | 内容 | 对应主张 | 脚本 | 20 线程下的预计耗时 | 主要输出（`artifacts/wingbeat_paper/` 下） |
|---|---|---|---|---:|---|
| E0 | 环境检查和单元测试 | — | `paper/check_env.sh` | 2 min | 终端显示 `environment OK` |
| E0q | QUICK 端到端冒烟测试（极小规模） | — | `QUICK=1 paper/run_all.sh` | 10 min（原机器实测 8 min） | `runs/wingbeat_paper/quick/out/` |
| E1 | 自由飞行开环权限，16 个配对案例 | C1 | `paper/run_core.sh` | 10 min | `e1/p0_e1.json` |
| E2 | 阵风生存曲线：PID、B2+、调参 B2+；9 档阵风 × 每档 32 例 | C1 | 同上 | 30 min | `e2_gust_curves.json` |
| E3 | 未知滞后曲线：13 个滞后 × 2 档阵风 × 每格 16 例；3 个手工控制律 | C2 | 同上 | 40 min | `e3_lag_curve.json` |
| E4 | 单因素鲁棒性：13 个因素 × 2 档 × 每格 16 例 | C1、C2 | 同上 | 30 min | `e4_robustness.json` |
| E5 | 消融：去掉高通、陷波；理想相位；调参锁相环；4 档 × 每档 16 例 | C1、C2 | 同上 | 15 min | `e5_ablation.json` |
| E8 | 相位估计：录制 80 次飞行；GRU、梯度 SNN、NEAT SNN 各 5 个种子；锁相环调参；离线和闭环测试 | C2 | `paper/run_phase_seeds.sh` | 3.5 h | `e8_models/`、`e8_phase_closed_loop.json` |
| E6 | 学习型残差（3 个组别 × 5 个种子）对比调参 B2+ | C3 | `paper/run_residual_seeds.sh` | 11 h | `e6_policies/`、`e6_residual_seeds.json` |
| E7 | 带滞后训练的残差（3 个组别 × 5 个种子）对比鲁棒调参 | C3 | `paper/run_h5_seeds.sh` | 10.5 h | `e7_policies/`、`e7_h5_seeds.json` |
| S | 统计表格和图 | 全部 | `paper_stats.py`（`run_all.sh` 最后一步） | 1 min | `report/tables.md`、`report/fig_*.png` |

**合计约 27 小时**（20 个逻辑 CPU）。耗时大致与 CPU 数成反比。

E1–E5、E8、E6、E7 四组之间**互不依赖**，可以分到几台电脑上同时跑，每台运行对应的脚本即可。

上表中的脚本都在 `experiments/algorithms/wingbeat_snn/` 下。

## 1. 准备

### 1.1 硬件和系统

- Linux 或 Windows 上的 WSL2（Ubuntu 22.04/24.04）。原生 C++ 物理步进只支持 Linux。
- 系统工具：`g++`（`sudo apt install build-essential`）、`git`、`tmux`。
- CPU 越多越好：每个仿真进程占一个核，评估阶段默认用（CPU 数 − 2）个进程。不需要 GPU。
- 内存 ≥ 16 GB；磁盘 ≥ 5 GB 空闲（`runs/` 下会有约 1–2 GB 的中间结果）。

### 1.2 获取代码

研究分支 `research/wingbeat-synchronous-snn` 目前**只在原机器本地**，比 GitHub 上的 `research/pid-residual-velocity-v3` 多 10 个左右提交。二选一：

```bash
# 方案 A：在原机器推送分支（需要你确认后执行），新机器 clone
git push origin research/wingbeat-synchronous-snn
git clone https://github.com/MingshanHe/FlapSim.git && cd FlapSim
git checkout research/wingbeat-synchronous-snn

# 方案 B：不推送，用 git bundle 拷贝（文件很小）
#   原机器：
git bundle create wingbeat.bundle origin/research/pid-residual-velocity-v3..research/wingbeat-synchronous-snn
#   新机器（先 clone GitHub 仓库，再从 bundle 取分支）：
git clone https://github.com/MingshanHe/FlapSim.git && cd FlapSim
git fetch origin research/pid-residual-velocity-v3
git fetch /path/to/wingbeat.bundle research/wingbeat-synchronous-snn:research/wingbeat-synchronous-snn
git checkout research/wingbeat-synchronous-snn
```

### 1.3 Python 环境（与原机器的版本一致）

```bash
conda create -n flapsim python=3.11 -y && conda activate flapsim
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r experiments/algorithms/snn_pid_compare/requirements.txt   # mujoco 3.8.0, numpy 2.2.4, neat-python 2.0.0 ...
pip install -e .
```

版本必须一致，尤其是 MuJoCo 3.8.0：录制的数据、调参结果和种子都依赖它，换版本会让数值变化。

### 1.4 环境检查（E0）和冒烟测试（E0q）

```bash
export PYTHON=$(which python)
bash experiments/algorithms/wingbeat_snn/paper/check_env.sh        # 应显示 9 passed 和 environment OK
QUICK=1 bash experiments/algorithms/wingbeat_snn/paper/run_all.sh   # 每个阶段都用极小规模跑一遍，约 10 分钟；结果数字没有意义，只检查流程
cat runs/wingbeat_paper/quick/out/report/tables.md                   # 能生成表格就说明流程完整
```

第一次运行会用 `g++` 把原生步进编译到 `~/.cache/flapsim/`，约 1 分钟。

## 2. 正式运行

```bash
export PYTHON=$(which python)
# 全部按推荐顺序执行（约 27 h），放在 tmux 里，断开 SSH 或关闭终端也不会中断
tmux new-session -d -s wingbeat_paper \
  "PYTHON=$PYTHON bash experiments/algorithms/wingbeat_snn/paper/run_all.sh 2>&1 | tee -a runs/wingbeat_paper.log"

# 或者在不同机器上分别运行
bash experiments/algorithms/wingbeat_snn/paper/run_core.sh            # E1–E5，约 2 h
bash experiments/algorithms/wingbeat_snn/paper/run_phase_seeds.sh     # E8，约 3.5 h
bash experiments/algorithms/wingbeat_snn/paper/run_residual_seeds.sh  # E6，约 11 h
bash experiments/algorithms/wingbeat_snn/paper/run_h5_seeds.sh        # E7，约 10.5 h
```

**可调的环境变量**：
- `WORKERS`：评估用的进程数，默认 CPU 数 − 2。
- `ARM_WORKERS`：每个训练组别用的进程数，默认 `WORKERS/3`。
- `SEEDS`：默认 `"1 2 3 4 5"`。若只想补种子，例如 `SEEDS="6 7"`，已完成的种子会自动跳过。

**查看进度**：

```bash
tmux attach -t wingbeat_paper                 # 查看（Ctrl-b d 退出，不会中断运行）
tail -f runs/wingbeat_paper.log               # 每个步骤的开始、完成或失败
ls runs/wingbeat_paper/*.log                   # 每个步骤自己的详细日志
tail -n 3 runs/wingbeat_paper/residual/*_seed*/history.jsonl   # NEAT 每代的进度
```

**中断后继续**：重新执行同一条命令即可。已完成的步骤自动跳过；正在进行的 NEAT 运行没有中间检查点，会从头重跑那一个种子。

**注意**：
- 正在执行的 `.sh` 不要修改。bash 是边执行边读脚本的，改动会让正在跑的流程执行错位的命令。
- 在 WSL 里启动长任务一定要用 `tmux`。用 `wsl.exe bash -c "cmd &"` 启动的进程，会在调用的 shell 退出时被一起杀掉。
- 每个仿真进程是单线程（`OMP_NUM_THREADS=1`），不要再额外开多线程。

## 3. 各实验的目的与验收标准

| 编号 | 回答的问题 | 支持主张的结果 | 需要修改主张的情况 |
|---|---|---|---|
| E1 | split-cycle 在自由飞行中有多大权限、代价多少 | 权限 ≥ 云台残差的 3 倍，电流 ≤ 6 A 的 60% | 权限 < 3 倍：降低 C1 的表述 |
| E2 | 可承受阵风提高多少 | 调参 B2+ 在 1.5 m/s 的生存率 ≥ 0.9；PID 在 0.5 m/s 的 Wilson 上限 < 0.9 | 按实际曲线改写倍数 |
| E3 | 对施力时刻误差有多敏感 | 能确定容许的滞后范围；鲁棒调参版本的范围更宽 | 只报告曲线 |
| E4 | 结论是否依赖某个模型参数 | 调参 B2+ 在每个因素下都显著优于 PID；横向阵风下两者相同（作为对照） | 某个因素下优势消失：作为局限写入讨论 |
| E5 | 每个设计部件的作用 | 去掉陷波 → 高度误差变大；去掉高通 → 前后漂移变大；理想相位只带来小幅改善 | 按结果改写第 5 节 |
| E8 | 学习型相位估计是否稳定优于调参锁相环 | 5 个种子的梯度 SNN 相位 RMS 都低于调参锁相环；闭环在 1.5 m/s 下俯仰峰值更小 | SNN 未全胜：C2 只保留 GRU，SNN 写作"部分种子" |
| E6 | 进化残差是否不如调参手工控制律（否定结果是否稳健） | 各组别对调参 B2+ 没有一致优势（符号检验 p > 0.05） | 某组别 5/5 个种子都更好：改写 C3，作为正面结果 |
| E7 | 带滞后训练能否带来适应 | 网络与鲁棒调参版本相当 | 网络在 ±60° 下一致更好：作为新发现写入 |

**统计方法**（`paper_stats.py` 自动计算）：
- 生存率用 Wilson 95% 区间；
- 连续指标按案例配对，用 bootstrap 95% 区间；
- 学习型方法以训练种子为单位，用精确符号检验。5 个种子全部同向时 p = 0.0625；若需要 p < 0.05，至少要 6 个种子（`SEEDS="1 2 3 4 5 6"`）。

## 4. 结果回传

完成后，把 `artifacts/wingbeat_paper/` 整个目录带回来，约 10–20 MB，包含结果 JSON、选中的策略、训练历史和 `report/`。`runs/` 下的中间文件不需要。

```bash
# 方案 A：在新机器上提交到同一分支（需要推送权限）
git add artifacts/wingbeat_paper && git commit -m "Paper campaign results" && git push
# 方案 B：打包后拷回原机器
tar czf wingbeat_paper_results.tgz artifacts/wingbeat_paper runs/wingbeat_paper.log
```

拿到结果后，我会核对 `report/tables.md` 中的每一项判据，更新大纲里的数字，并开始撰写正文。

## 5. 非计算的待办事项

- [ ] 画 F1 概念图（翅拍相位、调制波形、昆虫类比）
- [ ] 逐条核实文献：Tu & Dickinson 1996、Lindsay 等 2017、Pringle 1948、Dickinson 1999、Sane 等 2007、Doman/Oppenheimer/Sigthorsson split-cycle 系列、Jeffress 1948、Hammouamri 等 2024、Stagsted 2020、Paredes-Vallés 2024。补充扑翼机器人转向与抗扰相关工作（RoboBee、DelFly Nimble 等）
- [ ] 系统检索"进化或脉冲网络学习 split-cycle 时序控制"是否已有先例
- [ ] 台架验证（强烈建议）：用六维力传感器测真实机构在按相位调制电压下的 ΔFx 和 ΔMy；用真实 IMU 测锁相比例
- [ ] 确定投稿目标和截止日期
- [ ] 决定是否公开代码和数据（`artifacts/` 已包含复现所需的全部小文件）

## 6. 已在原机器完成、无需重跑的内容

| 内容 | 位置 |
|---|---|
| 固定支座预研（split-cycle 权限、翼根转动信号） | `artifacts/wingbeat_feasibility_20261012/` |
| P0 自由飞行预研（E1–E4）和 IMU 锁相检查 | 同上，见 `P0_report.md` |
| P1、H5、相位估计的试跑（每个模型 1 个种子） | `artifacts/wingbeat_snn_p1/`、`artifacts/wingbeat_snn_h5/`、`artifacts/wingbeat_snn_phase/` |
| 手工控制律调参结果（正式实验直接使用） | `artifacts/wingbeat_snn_p1/retune.json`、`artifacts/wingbeat_snn_h5/retune_robust.json`、`artifacts/wingbeat_snn_phase/long_run/pll_tuned.json` |
