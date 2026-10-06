# 当前工作目录：FlapSim-current

整理日期：2026-10-06。后续其他算法开发统一在此目录进行。

```text
Linux / WSL: /home/robot/rl/FlapSim-current
Windows:     \\wsl.localhost\Ubuntu\home\robot\rl\FlapSim-current
本地分支:    algorithms-dev
代码基线:    GitHub v2 / ad1776774ac443b4642fd7407944bb31de2ae58d
远端:        https://github.com/MingshanHe/FlapSim.git
```

这是从最新 `v2` 独立检出的完整仓库，保留 Git 历史。只带入版本管理中的源代码、
机器人资源、测试和 6.7 MB 的已发布训练结果；没有复制旧 `runs/`、构建目录或缓存。
后续运行生成的 `runs/` 属于新实验的本地输出，Git 已忽略该目录。

## 从这里开始

在 WSL Ubuntu 的终端运行：

```bash
conda activate mujoco
cd /home/robot/rl/FlapSim-current
python -c "import flapsim; print(flapsim.__file__)"
```

输出应指向 `FlapSim-current/flapsim/__init__.py`。本机已将此仓库以 editable 方式安装到
`mujoco` 环境；修改源码会立即生效。更换环境后可以在本仓库根目录执行
`python -m pip install -e '.[rl,test,video]'`。已发布策略使用的确切依赖版本记录在
`experiments/ppo_v2/requirements.txt`。

## 文件放在哪里

| 目录 | 用途 |
| --- | --- |
| `flapsim/` | 共享的仿真、执行器、气动力和控制代码 |
| `flapsim/robot/IST_robot-V2/` | 当前 V2 机器人 URDF、MJCF、网格和模型说明 |
| `flapsim/config/training/ist_v2_flight.json` | V2 自由飞行配置 |
| `flapsim/rl/env.py`、`features.py` | 可复用的 Gymnasium 环境和观测，不限定训练算法 |
| `experiments/algorithms/` | 后续其他算法的实验代码，每个算法一个子目录 |
| `runs/<algorithm>/<run_name>/` | 新实验日志、临时检查点和录像，默认不提交 |
| `artifacts/<algorithm>/` | 验证后准备保留的模型、配置和评估结果 |
| `artifacts/ppo_v2/` | 已验证的 PPO 基线、权重、报告数据和视频 |
| `experiments/ppo_v2/` | 原 PPO 研究记录、诊断工具和迁移分析 |
| `tests/` | 物理一致性、环境接口、检查点和轨迹检查 |

现在只建立新算法的工作入口，尚未选择或启动下一种训练算法。
`flapsim/rl/train.py` 仍是原 PPO 专用入口；新训练器放在新的实验目录中。
共用现有的物理环境与评估标准，可以与已有基线直接比较。

## 基线和检查

```bash
python -m pytest -q tests

# 只回放已保存的控制策略，便于核对仿真环境
python -m flapsim.rl.evaluate --run artifacts/ppo_v2/tracking \
  --trajectories hover --episodes 1 --seconds 8 --plot --output runs/baseline/hover_check
```

PPO 历史基线已通过悬停和圆、八字、椭圆、直线的独立验证。
快速 helix/lissajous 仍有失败案例，具体边界见 [历史实验报告](experiments/ppo_v2/README.md)。
历史日志中的绝对路径保留原实验出处；日常操作以本页的新目录为准。

## 旁边几个旧目录是什么

| 旧目录 | 身份 |
| --- | --- |
| `/home/robot/rl/FlapSim` | 旧 v1 工作目录，包含大量未提交修改和旧实验 |
| `/home/robot/rl/FlapSim-ppo-v2` | 已发布 v2 的原训练目录，保留完整原始训练输出 |
| `/home/robot/rl/FlapSim-rl-tracking-demo` | 旧 master 演示，是与旧 FlapSim 关联的 Git worktree |
| `/home/robot/rl/FlapSim-backups` | 原备份资料 |

这些旧目录保留原位，避免丢失未提交工作或破坏 worktree 关联。
后续开发使用 `FlapSim-current`；查历史实验时再访问旧目录。
