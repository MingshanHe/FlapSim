# 实测与仿真对比

系数校准与留出验证：`coefficient_validation.ipynb`、`coefficient_validation/report.md`。
只用 10–12 Hz 拟合输出增益，在 13–15 Hz 验证；区分平均升力、周期波形与六维力矩。
该实验属于输出校准，未将经验系数写回气动模型。

新增 STL 和确认材料后的矩形翼实验：

- `rect_material_validation/report.md`：本次 PET frame/翼根 + 10 μm Kapton、PEEK 传动的结果与实测对比。
- `rect_material_validation/lift_comparison.csv`、`lift_comparison.png`：10–15 Hz 六档均值、三次实测 SD、相对误差及旧模型参照。
- `rect_material_validation/model/IST_robot_rectangular_single_left.xml`：本次单左翼台架模型。
- `rect_material_experiment.py`：模型准备、实测读取、运行、数值检查、报告的入口。
- `rectangular_material_lift_10_15hz.ipynb`：已执行 Notebook，重新积分六档升力并核对模型及原始数据哈希。
- `rect_material_validation/video/rectangular_15hz_realtime.mp4` 与 `rectangular_15hz_slow8x.mp4`：15 Hz 双视角原速/8 倍慢放视频；时间标签是仿真时间。
- 双翼 URDF/MJCF 位于项目 `flapsim/robot/IST_robot/urdf/IST_robot_rectangular.*`。

```bash
python experiments/fixbase_curved_real/rect_material_experiment.py --prepare --run --convergence --report
python experiments/fixbase_curved_real/rect_material_notebook.py
python experiments/fixbase_curved_real/rect_material_video.py --hz 15
```

下方 `rect_validation` 为旧模型与旧质量假设的历史实验结果。

此前的矩形翼/曲面翼验证：

- `rectangular_vs_curved_validation.ipynb`：已执行 Notebook，包含升力、三轴均值、周期波形、频谱和方向拟合。
- `rect_validation/validation_report.md`：结果、模型假设、限制与完整复现命令。
- `rect_validation/lift_comparison.csv`：两种翼 10–15 Hz 汇总。
- `rect_validation/orientation_metrics.csv`：逐次重复的方向拟合/留出误差。
- `rect_validation/geometry/rect/IST_robot_single_left_wing.xml`：独立矩形翼模型。

矩形翼膜为 80 × 26 mm，羽化限位 ±26.1°，单翼且保留两侧传动。
STEP 文件只包含翼架；翼膜位置、膜厚、质量分布和安装偏移的假设见报告。
未按实测力拟合气动系数。方向拟合固定 Z，仅拟合绕 Z 的旋转；结果是候选角，不是已确认安装方向。

`0831_curve_capton`、`0831_rect_capton` 的原始文件未改动。
`analysis` 和 `real_vs_sim_analysis.ipynb` 是此前的曲面翼诊断，其中旧 ±50° 结果有历史参数说明。


Purdue Flappy 原版气动力核对：

- `flappy_audit/report.md`：原版 C++ 与现有实现的同运动轨迹/耦合对照，以及 10–15 Hz 复核。
- `flappy_audit/flappy_comparison.png`、`pitch_and_force.png`：平均升力、羽化与瞬时升力对比。
- `flappy_audit/reference_manifest.json`：用户 `rl/flappy` 源码 commit 和哈希；源码保留 MIT 许可证。
- `flappy_audit_models.py`：实际调用 Purdue Wing.cpp 的固定单左翼适配器。

后续在用户 RA-L 数据目录追到旧 baseline 生成代码：它使用外部规定视频轨迹（羽化约 ±65°、扑翼约 ±100°），并不使用 Flappy 默认关节限位。此前按默认 ±45° 解释旧结果的归因已撤回。
统一实际模型条件后，两套实现的 10–15 Hz 平均升力差异小于 0.21%；与实测的差距仍待真实翼运动/形变数据解释。
±45° 案例只作诊断，没有修改正式 URDF，也没有按升力拟合参数。

```bash
python experiments/fixbase_curved_real/flappy_audit_experiment.py --build --capture --offline --coupled --sweep
python experiments/fixbase_curved_real/flappy_audit_report.py
```

最新轨迹核对见 `flappy_audit/historical_trajectory/report.md`：重现旧 Flappy 六档结果、核对 36 份原始力文件哈希、分解原始/扣零偏均值差异，并分别替换扑翼和羽化轨迹。`trajectory_comparison.png` 为角度/角速度叠加，`passive_pitch_stop_diagnostic.png` 为放宽限位的诊断。

```bash
python experiments/fixbase_curved_real/historical_trajectory_audit.py
```
