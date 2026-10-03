# IST_robot 材料与惯性参数

矩形翼版本现已提供：`urdf/IST_robot_rectangular.urdf` 和配套 `IST_robot_rectangular.xml`。
它使用新增的 `meshes/Rectangular.STL`，按毫米转米、对齐原翼根安装特征，保留 PET 翼根连接件。
新 STL 已包含 10 μm 膜实体，采用 PET/Kapton 分材料积分，单翼总质量约 357.796 mg。
`build_rectangular.py` 可重新生成矩形模型；详细分项在 `rectangular_inertials.json`。
主文件 `IST_robot.urdf/xml` 继续保留此前曲面翼配置；两套模型的基座和传动相同。

仿真使用配套 MJCF 以保留 URDF 无法表达的 Scotch-yoke 闭环。
使用矩形翼时，将 `SimulationConfig.model_path` 指向 `urdf/IST_robot_rectangular.xml`，
并根据翼面几何设置 `FluidConfig.wings`；默认叶素参数不会自动从 STL 提取。

---

2026-09-22 根据实际材料修正 URDF 和 MuJoCo XML 中 14 个运动 link 的质量、质心及完整惯性张量。`base_link`、虚拟 `yoke_follower`、关节、传动比、限位、阻尼、执行器、气动参数和可视网格保持不变。

| 部位 | 材料/工艺 | 采用密度 |
| --- | --- | --- |
| motor_sun、carrier、planet、slider、左右 flapping | PEEK 板材 CNC 铣削，实心 | 1300 kg/m³ |
| 左右 wing 中的实体 frame、翼根 | PET | 1390 kg/m³ |
| 左右翼膜 | Kapton，厚度 **10 μm** | 1420 kg/m³ |

材料分配、CNC 工艺和膜厚由用户确认。密度为典型值，未获得实际采购牌号或称重数据：

- [Victrex PEEK 450G](https://cdn.victrex.com/zh-hans/downloads/datasheets/victrex-peek-450g)：结晶态未增强 PEEK，1.30 g/cm³。
- [Mitsubishi Chemical Ertalyte PET-P](https://www.mcam.com/mam/49643/GEP-Ertalyte%C2%AE%20PET-P_en_US.pdf)：ISO 1183-1 密度 1.39 g/cm³。
- [DuPont Kapton HN](https://www.dupont.com/content/dam/electronics/amer/us/en/electronics/public/documents/en/EI-10206-Kapton-HN-Data-Sheet.pdf)：1.42 g/cm³。10 μm 采用用户提供的厚度，不以数据表所列标准厚度替代。

## 计算方法与假设

原始质量除以 STL 有向体积，得到约 996–1003 kg/m³，说明旧 CAD 参数与统一 **1000 kg/m³** 的参考密度相符。该参考密度是几何核对所得的推断，原文件未标注材料密度。以 `cad_inertials_original.json` 保存原始 CAD 质量、质心、惯性及网格 SHA-256；实体质量和质心处惯性按 `ρ新 / 1000` 换算，均质实体的质心不变。这样保留 CAD 精度，避免对非封闭、多实体接触的 STL 做凸包或自动补洞所造成的体积误差。若后续取得原始 CAD 材料属性，应先核实这项参考密度。

左右翼原始 CAD 质量与实体网格体积相符，零厚度翼膜原先未计质量。每侧在 `x = ±0.35 mm` 平面提取 54 个独立膜三角形，正反面仅计一次；其他实体之间的重合面不作为翼膜。单侧面积为 **1690.635524 mm²**，膜质量为 **24.007024 mg**。

膜质量 `m = ρAt`。对三角形解析积分计算膜质心及二阶矩，并加入厚度方向的 `t²/12` 修正；假设 STL 膜面是厚度中面。最后通过平行轴定理合成 PET frame 和 Kapton 膜的总质心及惯性，不能只对整个 wing 乘一个密度比例。胶层及未单独建模的电机转子惯量没有实测数据，未额外加入。

所有惯性单位均为 kg·m²，且相对于各 link 质心、使用 link 坐标轴。URDF 写入六个惯性分量，MJCF 按 `Ixx Iyy Izz Ixy Ixz Iyz` 写入 `fullinertia`。

## 质量变化

| link | 原质量 mg | 新质量 mg |
| --- | ---: | ---: |
| motor_sun | 7.11127 | 9.24466 |
| carrier_1 | 47.70867 | 62.02128 |
| carrier_2 | 41.47154 | 53.91300 |
| 每个 planet，共 6 个 | 4.41360 | 5.73768 |
| slider | 70.26410 | 91.34333 |
| 每侧 flapping | 59.19064 | 76.94784 |
| left_wing（frame + 膜） | 222.90826 | 333.84950 |
| right_wing（frame + 膜） | 222.79515 | 333.69229 |

逐 link 的质心、惯性张量和翼膜分项质量保存在 `material_inertials.json`。

## 复现

在项目根目录运行（只需项目已有的 NumPy）：

```bash
python flapsim/robot/IST_robot/recalculate_inertials.py
python flapsim/robot/IST_robot/recalculate_inertials.py --write
```

第一条仅预览；第二条从固定原始基准重算并同步两种模型，可重复执行。调整牌号密度或膜厚时编辑 `materials.json`。若 STL 改变，脚本会拒绝沿用旧 CAD 基准，需要重新取得该几何的实体惯性。

重新生成矩形翼模型还需安装网格处理依赖：

```bash
python -m pip install -e '.[model]'
python flapsim/robot/IST_robot/build_rectangular.py
```

该命令更新矩形翼 URDF、MJCF、网格及 `rectangular_inertials.json`。
仅使用已生成的模型进行仿真时，无需安装 `model` 可选依赖。
