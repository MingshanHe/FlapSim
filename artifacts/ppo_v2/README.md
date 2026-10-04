# 已训练的 IST_robot-V2 PPO 策略

在 `/home/robot/rl/FlapSim-ppo-v2` 中运行；建议使用本机 `mujoco` conda 环境。

| 目录 | 用途 |
| --- | --- |
| `tracking/` | 推荐：同一 PPO 完成悬停、圆、八字、椭圆、往返直线 |
| `hover/` | 独立悬停策略，保留为悬停课程的已验证检查点 |
| `experimental_spatial/` | 快速三维任务的细化实验，尚不能作为稳定交付 |
| `evaluation/` | 独立测试汇总、代表轨迹 CSV、图表、MP4 和失败案例 |
| `training/` | 参数、PPO 学习曲线、检查点身份和训练评估历史 |

每个策略目录中的 `policy.zip` 必须和同目录 `vecnormalize.pkl` 配套使用。
`simulation.json` 和 `run.json` 保存物理配置、任务参数和哈希。Linux/WSL 需要 `g++`，
本次运行依赖 MuJoCo 3.8.0，具体版本见实验目录 requirements.txt。

```bash
conda activate mujoco
cd /home/robot/rl/FlapSim-ppo-v2
python -m flapsim.rl.evaluate --run artifacts/ppo_v2/tracking \
  --trajectories hover circle figure8 ellipse line --episodes 3 --randomize \
  --seconds 16 --reference-physics --plot --output runs/recheck

# 交互查看；需要 WSL 图形显示
python -m flapsim.rl.evaluate --run artifacts/ppo_v2/tracking \
  --trajectories circle figure8 ellipse --episodes 1 --render --output runs/view

# 从交付策略继续训练；不要附加 --from-best，导出目录本身就是冻结检查点
python -m flapsim.rl.train --resume artifacts/ppo_v2/tracking \
  --learning-rate 5e-5 --total-timesteps 131072 --n-envs 16 --output runs/continued
```

核心任务名义半径参数 0.20 m、周期 8 s；评估随机化半径/周期 ±20%、相位和初始状态。
成功要求飞满时限且三维位置 RMSE < 0.10 m。已验证时长为 8 s 和 16 s，
不包括实机、地面起飞、着陆和大扰动恢复。快速 helix/lissajous 的失败率单独记录。

详细结果和 master 迁移分析见 [实验报告](../../experiments/ppo_v2/README.md)。
