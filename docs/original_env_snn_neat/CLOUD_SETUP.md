# 云端（AWS EC2）跑仿真的部署清单

负载特征（2026-10-13 实测）：
- 纯 CPU、单线程物理（MuJoCo 3.8，5 µs 步长），按案例并行（`ProcessPoolExecutor`，`--workers N`）。**不用 GPU，不渲染。**
- 每个 worker 常驻内存约 **120 MB**；内存不是瓶颈。
- 本机（i5-14600KF，18 个 worker）约每分钟 40 个 6 s 案例；一个 8 s 案例约占一个核 45 s。
- `flapsim/rl/native_step.cpp`、`native_fast_step.cpp` 在首次导入时用 `g++ -O3 -std=c++17` 即时编译（ctypes 加载），只要有 g++ 就行，x86 和 ARM 都可以。

## 机型

| 选择 | 机型 | 核 | 适用 |
|---|---|---|---|
| 首选 | **c7a.16xlarge**（AMD Genoa，64 vCPU = 64 物理核，无超线程，最高 3.7 GHz） | 64 | 单线程物理最合适；一次 v3 规模的运行约 35–40 分钟 |
| 省钱 | c7a.8xlarge | 32 | 约本机 2 倍吞吐 |
| 更省 | c7g.16xlarge / c8g.16xlarge（Graviton，ARM，每 vCPU 一个物理核） | 64 | 代码可移植（mujoco、torch 有 aarch64 wheel；C++ 即时编译），但浮点结果与 x86 **不逐位相同**，不要用来复现旧数字 |
| 大批量 | hpc7a.96xlarge（192 物理核）或多台 c7a 加 Spot | 192+ | 一次跑 5–10 个种子时用 |

不选：c7i-flex（持续满载会被限频）、带 GPU 的 g/p 系列（用不上）、按 vCPU 数挑 Intel c7i（它的 vCPU 是超线程，物理核只有一半）。

Spot 便宜 60–70%，但 NEAT 训练阶段（约 1 h）没有代内检查点，被回收就从头来；`upset.py` 和 v3/v4 的其它阶段按阶段 JSON 续跑。长 NEAT 运行用按需，短而多的扫描用 Spot。

## 区域

批处理对延迟不敏感：选 us-east-1 / us-east-2 / us-west-2（价格比东京低 15–20%，c7a 库存也更多）。

## 安装（Ubuntu 22.04/24.04 AMI）

```bash
sudo apt-get update && sudo apt-get install -y build-essential git tmux htop
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o mc.sh && bash mc.sh -b
~/miniconda3/bin/conda create -y -n mujoco python=3.11
~/miniconda3/envs/mujoco/bin/pip install mujoco==3.8.0 numpy==2.2.4 scipy==1.15.2 gymnasium==1.1.1 \
    neat-python==2.0.0 cmaes==0.11.1 pytest
~/miniconda3/envs/mujoco/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # CPU 版即可
git clone <私有仓库> ~/FlapSim-repo/FlapSim && cd ~/FlapSim-repo/FlapSim && git checkout research/original-env-snn-neat
~/miniconda3/envs/mujoco/bin/pip install -e .
```
（ARM 机型把 Miniconda 换成 `Linux-aarch64`；其余相同。）

GitHub 凭据：在实例上用部署密钥或细粒度 token，不要把个人凭据放上去。

## 验证（约 5 分钟）

```bash
PY=~/miniconda3/envs/mujoco/bin/python
$PY -m pytest tests/test_wingbeat_snn.py -q                           # 10 个单元测试
$PY -m experiments.algorithms.spike_steer.pilot_v3 selftest
$PY -m experiments.algorithms.spike_steer.pilot_v4 selftest
$PY -m experiments.algorithms.spike_steer.upset selftest
```
再跑一个确定性的对照数字：旧 16 例测试集上 law_tuned 的 J 应为 **5.86**（`artifacts/spike_steer/v3/test.json`）。**同一台机器上逐位可复现；跨机器不行**（2026-10-13 实测：EPYC/Ubuntu 20.04/g++ 9 机器得 5.8628，本机 5.8586，相对差 7e-4——`native_step.cpp` 的即时编译、glibc 的 libm、OpenBLAS 内核不同，ULP 差被 5 µs 步长的混沌动力学放大）。所以**所有比较都必须在同一台机器上配对**，流水线已如此设计（测试阶段同机重跑经典对照）；跨机器只看统计等价。

## 运行

- 脚本里的 `--workers 18` 改成 `vCPU 数 − 2`（c7a.16xlarge 用 62）；`OMP_NUM_THREADS=1` 等已在脚本中设好。
- 仍然用 tmux：`tmux new-session -d -s <名字> "bash runs/run_xxx.sh 2>&1 | tee -a runs/xxx.log"`。
- **跑完自动关机**，避免空转计费：在运行脚本末尾加 `sudo shutdown -h now`（实例设为"停止"而非"终止"，EBS 保留），或设 CloudWatch 低 CPU 告警自动停机。
- 结果回传：`runs/` 不进 git；需要保留的放 `artifacts/` 后 `git commit`，或 `aws s3 sync runs/ s3://<bucket>/runs/`。EBS gp3 50 GB 足够。

## 费用估算（us-east-1 按需，2025 年价，开机前在价格页核对）

- c7a.16xlarge 约 $3.3/h：一次 v3 规模的运行（本机 2 h）约 40 分钟 ≈ **$2**；10 个种子的复现 ≈ $20。
- c7a.8xlarge 约 $1.65/h；c7g.16xlarge 约 $2.3/h；hpc7a.96xlarge 约 $7.2/h。
- 每天 8 小时、一个月约 $500（按需）/ 约 $180（Spot）。
