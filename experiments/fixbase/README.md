# 固定基座实验

**参数更正：用户确认羽化机械限位为 ±26.1°，MJCF 与 URDF 已同步修正。**
以下原始实验及旧报告中的 ±50° 结果仅作历史对照。
当前单翼 10–15 Hz 重跑命令为 `python experiments/fixbase/rerun_26p1deg.py`，
结果单独保存至 `results_single_wing_26p1deg_10_15hz/`，notebook 为
`IST_robot_single_wing_26p1deg.ipynb`。该主扫描仅修正限位；CAD 几何端点诊断单独记录。

打开 `IST_robot_fixbase.ipynb`，已保存真实执行结果和视频回放。
`base_link` 完全固定；电机力矩驱动原有齿轮、滑块与扑翼机构，羽化始终被动。

安装额外依赖并复现：

```bash
pip install -e '.[experiments]'
MUJOCO_GL=egl python experiments/fixbase/experiment.py
```

完整重建并执行 notebook：

```bash
python experiments/fixbase/build_notebook.py
```

默认物理时间 1.2 s、输入轴指令 500 rad/s、升速 0.1 s、物理步长 5 µs。
负载测量采样 2 kHz，视频以 250 仿真帧/s 捕获、30 fps 编码，约 8.33 倍慢放。
单实例高频积分可能需要几分钟；不要将实际等待时间当作仿真时间。

`results/` 保存以下产物：

- `fixbase.mp4` / `fixbase.png`：双视角视频和首帧。
- `wrench_6d.png`：Fx、Fy、Fz、Mx、My、Mz 在一个 figure 上。
- `fixture_wrench_6d.png`：支架反力，包含机械限位的采样尖峰。
- `joint_motion.png` / `motor_torque.png`：电机及翼关节的实际响应。
- `measurements.csv` / `.npz`：全部采样数据。
- `summary.json`：参数、数值检查和翻转统计。
- `diagnosis*`：原始安装方向/正立、100/500 rad/s 的完整行程诊断。

可用 `python experiments/fixbase/diagnose_feathering.py` 重做诊断。
100 rad/s 输入经过 16:1 减速只对应约 1 Hz 扑翼；原始模型完整周期中也会翻转。
此次修正安装朝向，保留原有 ±50° 限位、零弹簧刚度与被动气动/惯性响应，
没有为了视频效果强制指定羽化角。

## 数据含义

`aero_*`：空气作用于机器人，世界坐标，力矩关于 base_link 原点。
`fixture_*`：机器人作用于支架，包含重量、气动和机构惯性载荷。
`support_*`：相反方向，即支架作用于机器人。
`gravity_*`：当前构型下的重量和重力矩。
`fixture_dynamic_*`：fixture 减去 gravity，并非纯气动力。
`left_aero_*` / `right_aero_*`：各翼关于同一基座原点的气动 wrench。
`*_parent_load_*`：父刚体作用于关节子刚体的载荷，力矩关于各自关节锚点，世界坐标。
`motor_torque_Nm` 是驱动轴力矩；`motor_stator_M*_Nm` 是定子所受内部反力矩，
不能重复加入整机气动力。

每次采样在独立 MjData 副本上刷新力并执行 `mj_rnePostConstraint`，
不会改变正在积分的状态或求解器 warm start。
六维顺序统一 `[Fx,Fy,Fz,Mx,My,Mz]`；CSV 后缀给出 N 或 Nm。
气动力主图和支架载荷图分开，避免限位尖峰压扁气动小信号；notebook 中
`include_weight=True` 可以在支架图中加入总支架载荷。没有裁掉尖峰；
2 kHz 采样不能用于精确预测微秒级限位冲击峰值。
电机和气动系数尚未实机标定，时间窗均值不等于严格整周期均值，结果不证明能够悬停。

## 末端 15 Hz 实验

运行 `python experiments/fixbase/run_15hz.py`，输出到独立的 `results_15hz/`，
并生成已执行的 `IST_robot_15hz.ipynb`。保持电机参数不变，通过试运行调整速度指令，
补偿比例调速的负载误差；以实际完整曲柄周期确认末端频率。
最终运行 2 s，排除前 0.6 s 后，仅对完整周期内的总气动力 Fz 做时间积分平均。
`lift_summary.json` 保存频率、平均升力、统计区间和逐周期结果；
`calibration.json` 保存调速过程。视频、六维载荷图及 CSV/NPZ 同时保留。

## 10–15 Hz 平均升力扫描

运行 `python experiments/fixbase/sweep_frequency.py`，以 1 Hz 间隔测试六个频率。
每点运行 2 s，并校准实际频率至目标的 ±0.01 Hz 内；平均方式与 15 Hz 实验一致。
同参数的 15 Hz 原始记录会经过检查后复用。其余频率独立运行，原始数据和调速记录
保存在 `results_10_15hz/`。`sweep.csv` / `sweep.json` 汇总测量结果和数据来源。
`mean_Fz_vs_frequency.png` / `.svg` 是平均总升力曲线，误差棒为逐周期均值范围，
不代表置信区间。已执行的 `IST_robot_10_15hz.ipynb` 包含数据复核、图和结果表。

## 单翼 10–15 Hz（保留两侧传动）

运行 `python experiments/fixbase/single_wing_sweep.py`。
按单翼实测装配方式，移除右侧 `right_wing` 翼板刚体及其质量、惯量和气动力，
保留左右 `flapping` 刚体、扑翼关节及滑块传动。左翼保持原有被动羽化。
模型副本及拆除质量记录位于 `results_single_wing_10_15hz/model/`，原双翼模型不变。
每个频率重新调速并真实仿真，统计口径同上；结果不是将双翼升力除以二。
`results_single_wing_10_15hz/sweep.csv` 汇总单片左翼平均升力，各频率保留
六维气动力、支架载荷、关节角/速度及电机力矩原始记录和曲线。
图为 `mean_Fz_vs_frequency.png` / `.svg`，记录本为 `IST_robot_single_wing_10_15hz.ipynb`。
当前尚未导入实测数据或根据实测数据调整气动系数。
