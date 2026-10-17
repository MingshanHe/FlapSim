# NEAT / SNN / PPO 实验：迁移与续跑

这份说明用于接续 `research/neat-snn-hover` 分支的冻结实验。阶段分析见 [report_stage_2201.md](report_stage_2201.md)。**本次按要求在 `snn_neat_seed2202` 完整结束后暂停；不会自动启动下一组、独立测试或最终推送。**

## 1. 接续到哪里

暂停后的完整状态以 [pause_completed.json](artifacts/snn_hover_20261009/pause_completed.json) 为准：

- 已完成 8/35 次：2201 的全部七种方法，加上 `snn_neat_seed2202`。
- 剩余 27 次；下一组是 `snn_fixed_tau_seed2202`，随后按原队列完成 2202–2205。
- 每次预算 1,024,000 步；保留 256,000 / 512,000 / 1,024,000 三个预算检查点。
- 独立测试尚未开始。只有全部 35 次训练结束后才运行冻结的 11,888 回合评估。
- `report_final.md` 尚未生成；不要把阶段报告改名当作最终报告。

这是**完整运行之间的续跑**：新电脑跳过八个 `results.json` 已完成的目录，从下一组新运行开始。不需要恢复 2202 的中途环境或神经状态，也不需要重新训练 2201。

当前代码不承诺从任意被中断的 PPO 回合/NEAT 代中精确续跑。若发现没有 `results.json` 的训练目录，队列会停止并保留它；先查原因，不要删除数据或直接覆盖。

## 2. 获取完整分支

推荐 x86-64 Ubuntu，或 Windows 下的 Ubuntu / WSL2。冻结的原生动力学和传感器实现会编译 Linux `.so`，不能直接照搬到原生 Windows Python。训练使用 CPU；本协议不需要 CUDA 或 GPU。当前参考环境为 Python 3.11.11。

在新电脑的 **Ubuntu 终端**运行。仓库应放在 Linux 文件系统中：

```bash
mkdir -p "$HOME/rl"
git -c core.autocrlf=false clone --branch research/neat-snn-hover \
  https://github.com/MingshanHe/FlapSim.git "$HOME/rl/FlapSim-v3-neat-ppo"
cd "$HOME/rl/FlapSim-v3-neat-ppo"
git config core.autocrlf false
git branch --show-current
test -f artifacts/snn_hover_20261009/pause_completed.json
test -f artifacts/snn_hover_20261009/migration_manifest.json
```

不要只复制模型 JSON：还需要配置、物理模型、网格、相位库、原始训练档案、旧一轮对照策略和冻结源码。建议完整克隆，不使用浅克隆；最终审计需要原基准提交 `3711664acf1e9660e304a6c8a473bb55e0eb102f`。

若已经有这个分支，先确认本地修改已保存，再获取并合并最新提交。上述两个 `test` 都应成功，否则拿到的仍是暂停前的分支版本。

本机使用 Git worktree，目录里的 `.git` 是指向原主仓库的小文件。不要把这个指针文件直接复制到另一台电脑后当作完整仓库。离线迁移可以在本次暂停提交完成后使用 Git bundle：

```bash
# 原电脑，仓库根目录；生成仓库外的文件
git bundle create ../FlapSim-snn-handoff.bundle research/neat-snn-hover
# 传输 bundle 后，在新电脑按实际文件路径克隆
git clone --branch research/neat-snn-hover /path/to/FlapSim-snn-handoff.bundle \
  "$HOME/rl/FlapSim-v3-neat-ppo"
cd "$HOME/rl/FlapSim-v3-neat-ppo"
git remote set-url origin https://github.com/MingshanHe/FlapSim.git
```

## 3. 安装环境

系统依赖用于编译、绘图和离屏视频：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libegl1 libgl1 libosmesa6 \
  fonts-dejavu-core ffmpeg
```

若已有 Conda：

```bash
conda create -n flapsim-snn python=3.11.11 pip -y
conda activate flapsim-snn
```

也可以使用已安装的 Python 3.11.11 创建虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

以下命令在已激活的新环境、仓库根目录中执行。先装 CPU 版 PyTorch，再装已固定主要版本的依赖，最后安装当前仓库：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/algorithms/snn_hover/requirements-portable.txt
python -m pip install --no-deps -e .
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
```

主要版本为 NEAT-Python 2.0.0、MuJoCo 3.8.0、NumPy 2.2.4、PyTorch 2.6.0、Stable-Baselines3 2.6.0。不要直接安装当前机器完整的 `pip freeze`：其中包含与本实验无关的 CUDA/JAX 包和本机路径。

本轮已经在独立的干净虚拟环境中检查安装与已用验证场景回放，记录在 [portable_replay_check.json](artifacts/snn_hover_20261009/development/portable_replay_check.json)。这验证了依赖安装流程，不代表已在你的新电脑完成验证；新电脑仍须执行第 5 节。

## 4. 保留冻结 XML，处理网格绝对路径

历史物理 XML 的 `meshdir` 使用 `/home/robot/rl/FlapSim-v3-neat-ppo/...`。**不修改 XML**，通过仓库目录软链接让原路径指向新位置，保持物理模型哈希不变。如果你的用户名就是 `robot` 且按上述路径克隆，通常无需额外操作。

在仓库根目录运行：

```bash
TASK_ACTUAL_REPO="$(pwd -P)"
TASK_CANONICAL_REPO=/home/robot/rl/FlapSim-v3-neat-ppo
if [ "$TASK_ACTUAL_REPO" != "$TASK_CANONICAL_REPO" ]; then
  if [ -e "$TASK_CANONICAL_REPO" ] || [ -L "$TASK_CANONICAL_REPO" ]; then
    test "$(readlink -f "$TASK_CANONICAL_REPO")" = "$TASK_ACTUAL_REPO" || {
      echo '原路径已被其他目录占用，请先检查；未覆盖任何文件。'
      exit 1
    }
  else
    sudo mkdir -p /home/robot/rl
    sudo ln -s "$TASK_ACTUAL_REPO" "$TASK_CANONICAL_REPO"
  fi
fi
```

原路径若属于另一份仓库，不要覆盖或删除它。可将此次完整克隆放到该规范位置，或在隔离的 Linux 用户环境/容器内接续。程序缓存的本机 `.so` 不用拷贝；新电脑会从相同源码重新编译。

## 5. 续跑前检查

```bash
python -m experiments.algorithms.snn_hover.migration \
  --root artifacts/snn_hover_20261009 --replay-check --render-check \
  --output migration_check_this_machine.json

python -m pytest -q tests/test_snn_hover.py tests/test_snn_analysis.py \
  tests/test_snn_publication.py tests/test_topology_hover.py
```

检查内容包括 67 个冻结核心源码、冻结评估/统计代码、迁移清单、八次已完成训练的文件、物理模型与网格、版本、编译器，以及两条**已用过的2201验证场景**回放。回放指标的绝对误差容限为 `1e-6`，成功/终止标记必须相同；它不访问独立测试集，不参与重新选模。`--render-check` 还检查静态离屏帧、字体和视频编码器，不增加物理模拟步数。

如果新电脑的 EGL 离屏渲染不可用，可在新命令前改为 `export MUJOCO_GL=osmesa` 后重新检查，使用第 3 节安装的 Mesa 软件渲染库；训练物理引擎和策略保持不变。未通过渲染检查时，先排除运行环境问题再安排最终视频。

迁移输出文件不覆盖旧记录；重复检查请换一个文件名。若校验失败，先解决路径、版本或数值差异，再启动新训练。不要更新清单来掩盖差异，也不要为了绕过检查修改冻结代码。不同 CPU/编译器仍可能造成浮点差异，后续新增种子的逐位一致性不是保证。

## 6. 显式恢复剩余训练

保持原 `population=64`、`workers=16` 和预算。PPO 的 16 个环境参与定义采样批次，不能只为加速而随意改 `workers`；性能较弱的电脑可运行得更慢，但不应在同一比较集中修改它。

终端 A（持续运行，使用 tmux 等保持终端会话也可以）：

```bash
cd "$HOME/rl/FlapSim-v3-neat-ppo"
# 激活上面创建的环境
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python -u -m experiments.algorithms.snn_hover.campaign \
  --root artifacts/snn_hover_20261009 --resume
```

`--resume` 会先检查暂停边界，归档暂停记录，保存本次电脑和 Python 信息；跳过已完成的八次运行，启动 `snn_fixed_tau_seed2202`。不带 `--resume` 时，有暂停标记就会拒绝启动。若2202还没结束，它也会拒绝提前续跑。同一目录的第二个队列会被锁阻止。

如果续跑之后还想在当前组结束时再次暂停，可在另一个终端写入边界请求；不要把关闭终端或中断进程当成可精确续跑的回合检查点：

```bash
python - <<'PY'
import json, time
from pathlib import Path
from experiments.algorithms.snn_hover.common import write
root = Path('artifacts/snn_hover_20261009')
status = json.loads((root / 'campaign_status.json').read_text())
assert status['status'] == 'training'
write(root / 'pause_request.json', {
    'status': 'pause_requested',
    'after_run': Path(status['current']).name,
    'training_pid': status['pid'],
    'requested_unix': time.time(),
    'reason': 'User requests pause after the active complete run',
})
PY
```

队列会让当前组结束，写入新的 `pause_completed.json` 后返回。本文附带的迁移清单对应本次8组断点；再次暂停若需再次迁移，应为新的已完成边界另做清单，保留这一版。

终端 B（确认 A 已开始训练后，可选）：

```bash
cd "$HOME/rl/FlapSim-v3-neat-ppo"
# 激活同一环境
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
python -u -m experiments.algorithms.snn_hover.finish \
  --root artifacts/snn_hover_20261009
```

这个有限队列等待全部训练完成，然后按顺序执行独立测试、统计、全种群可视化、轨迹复核、视频、最终报告和完整性检查。它不负责 Git 推送。只启动一个收尾队列；暂停标记存在时会退出，不继续测试。

可选终端 C 打开只读进度页：

```bash
python -m http.server 8767 --bind 127.0.0.1 \
  --directory artifacts/snn_hover_20261009
```

浏览器访问 `http://localhost:8767/dashboard.html`。当前运行日志在 `artifacts/snn_hover_20261009/logs/`，各组 `status.json` 给出分配步和实际步。单次训练默认最多4小时；全队列等待最多72小时。若新电脑确实较慢，可只调整 **campaign.py / finish.py 的墙钟超时上限**并记录变更，不能改变科学预算或冻结核心文件。

## 7. 最终报告与提交

全部完成后，应具有根目录 `report_final.md`，以及实验目录中的 `analysis.json`、`research_result.json`、`verification.json` 和完整媒体。阶段报告继续保留。先阅读结论与失败边界，再执行：

```bash
python -m experiments.algorithms.snn_hover.verify --root artifacts/snn_hover_20261009
git status --short
git diff --check
git add report.md report_final.md artifacts/snn_hover_20261009
git commit -m "Complete NEAT-SNN hover study and final research report"
git push origin HEAD:research/neat-snn-hover
```

这个最后的 push 需要新电脑自己的 GitHub 认证。`publish.py` 是原 Windows/WSL 主机专用辅助程序，包含该机器的 Windows Git 路径；**不要在新电脑直接启动它**。本次原电脑上的自动收尾/推送进程已经停止，不会与新电脑抢跑。

所有方法完成前，不调用正式 `evaluate`，不把未完成的五种子比较写成最终结果。后续若修改研究假设、训练超参数或使用不同传感器，另开实验版本，保留这一轮冻结结果。
