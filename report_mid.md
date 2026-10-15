# FlapSim 中期报告：仿生分层控制策略的研究脉络、算法与代码（截至 2026-10-15 10:30 JST）

分支 `research/original-env-snn-neat`。本文把 2026-10-08 至 10-15 的工作按"问题 → 方法 → 证据 → 决定"整理成一条线，便于后续写论文和交接。所有数字都来自仓库里归档的 JSON（`artifacts/spike_steer/`），每个数字旁边给出文件位置。仿真平台与实物升力校准一章由用户自己撰写，本文不重复、不改动，只在用到时引用。

---

## 1. 结论先行

**论文主张（研究计划 v4）**：在一台用台架升力数据校准过的单电机双翼扑翼机器人仿真上，一个仿昆虫分层的控制策略（反射层 PID + 翅拍锁相脉冲"转向肌"层 + 学习得到的下行指令层）在预注册的"应力悬停 + 点头 + 前冲急停 + 大扰动恢复"综合基准上，与同预算最强经典方案配对比较。

**到目前为止的证据状态**（第二轮，门控下行指令层；比值 = 学习方得分 / 经典对手得分，越小越好；盲测 32 例机动 + 16 例悬停 + 48 例翻转）：

| 种子 | 搜索机器 | 选代规则 | 综合 | 悬停 | 点头 | 冲刺 | 翻转 | H1 (≤0.95) | H1′ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CloudPC | 原规则（8 例验证） | **0.940** | 1.000 | 0.874 (15/16 胜) | 0.835 (16/16 胜) | 1.050 | 过 | 过* |
| 1 | CloudPC | 修正 2（16 例 + 零坠落） | **0.940** | 1.000 | 0.869 (16/16 胜) | 0.843 (16/16 胜) | 1.050 | 过 | 过* |
| 3 | HomePC | 原规则 | 1.114 | 1.004 | 0.939 | 1.462（4/16 坠落） | 1.049 | 败 | 败 |
| 3 | HomePC | 修正 2 | 0.958 | 1.000 | 0.910 (14/16 胜) | 0.873 (16/16 胜) | 1.050 | 差 0.008 | 过* |
| 2 | Mac（已作废） | 原规则 | 0.981 | 1.000 | 0.901 | 0.971（1/16 坠落） | 1.054 | 败 | 败 |
| 2 | HomePC（跑中） | — | — | — | — | — | — | — | — |
| 3 | CloudPC（跑中，修正 3） | — | — | — | — | — | — | — | — |

\* 翻转比值精确值 1.050006，比 H1′ 的"每项 ≤ 1.05"高出 6×10⁻⁶，这是修正 1 的构造值，三个种子相同；读法（三位小数通过 / 严格不等式不通过）待用户定。

- **已经成立的**：H3（机制消融，所有归档模型、所有消融 ≥ +21%，多数 ≥ +100%）；H4（同尺寸 ANN 走同一流水线，模仿阶段与 SNN 相当，ES 阶段 ANN 三个种子都没有超过模仿起点）；悬停 1.000 和翻转 1.05 按门控构造成立。
- **未定的**：H1（综合 ≤ 0.95，现要求 ≥ 3/5 种子）取决于正在跑的种子 2、3（CloudPC）以及新加的种子 4、5；H1′（不劣 + 单项显著优）目前 2/2 通过（边界读法除外）。
- **第一轮（无门控、40/20/20/20）的结论**：ES 用悬停换机动，H1、H2 不成立；这一发现直接导致第二轮的门控设计。

---

## 2. 研究脉络（时间线）

| 日期 | 阶段 | 结果 | 决定 |
|---|---|---|---|
| 10-08 | 纯 IMU 端到端 NEAT / PPO 悬停 | 0/192 严格成功 | 没有反射层学不会悬停 |
| 10-10 | PID + SNN 残差（v3 线） | 倾角比 PID 好约 1°，但 PID 自己 kp×1.5 就好 1.1–1.7°；SNN ≈ ANN | 残差路线关闭 |
| 10-11/12 | 原环境预备检验 `neuro_gates` | 下体随翅拍晃动 ±4.4°；云台脉冲先让 IMU 所在的下体反向转（两体反向响应）；任何"快连续反馈"坠机；翅拍定时的电机转速脉冲俯仰权限是云台残差的 4.1 倍 | 快通道必须是事件式、锁相的 → "脉冲转向肌" |
| 10-12/13 | 脉冲定时转向 v1–v5b（NEAT 进化 LIF 回路） | J 比值 1.13–1.44；根因：LIF 时间常数太慢 + 初始化盆地 + 单向平台（`artifacts/spike_steer/v3,v4,v5*/analysis.md`） | NEAT 不是可靠的搜索工具 |
| 10-13 | v5c（经典起点 + 镜像绑定） | J 比值 0.997 / 1.016，首次过预注册线 | 持平，不是赢 |
| 10-13 | 全状态反馈上界（C 门） | 理想全状态反馈只比最优 PID 少 1.1° | 悬停调节没有性能空间 |
| 10-13 | 梯度模仿（原始 IMU → 经典律放电，`imitate.py`） | J 比值 1.044 / 1.046 过线，俯仰持平 | 替代梯度是可靠的搜索工具；学习方改以梯度为主 |
| 10-13 | 机动可行性运行（`maneuver.py`） | 经典调度只 3/8 到 30°，前冲 0.5 m 无一到达 | 机动模块必须有**设定值权限** |
| 10-13 晚 | 用户决定：悬停 + 机动综合基准，接受"混合体最优"；MJX/GPU 关闭（35.6 ms/步），只用 CPU | 研究计划 v4 成文（Claude Docs） |
| 10-14 | S0/S1：基准定型 + 预注册；S2 模仿 ×3；S3 ES 微调；S4 盲测 + 消融；ANN 对照 | 第一轮：H1/H2 败，H3/H4 成立 | 发现 ES 用悬停换机动 |
| 10-14 | 第二轮：门控下行指令层（修正 1 翻正反射），权重 25/25/25/25，H1′ | 种子 1 过 H1；种子 3 冲刺坠落 | 修正 2（验证集 16 例 + 零坠落） |
| 10-15 | Mac 数值作废；种子 3 在 CloudPC 重跑（修正 3）；加种子 4、5（修正 4） | 跑中 | — |

---

## 3. 平台与接口（不变）

- 机器人 IST_robot-V2，15 g（推力单元 7 g + 下体 8 g，IMU 在下体），单电机驱动双翼，翅拍约 26.7 Hz；MuJoCo 多体 + 自研叶素气动；升力按台架数据按翼型校准（用户章节）。
- 接口：传感 11 维 IMU + 测距高度（500 Hz）；执行云台 α/β ±0.3 rad、电机转速 3115–3807 rad/s，每 2 ms 更新、经 8.2 ms 平滑。全部实验不改传感器、执行器、更新率、限幅（`CLAUDE.md`）。
- 两个决定架构的物理事实：翅拍晃动 ±4.4°；两体反向响应（`artifacts/neuro_gates/gates_report.md`）。

---

## 4. 综合基准（S0/S1，`bench.py`）

### 4.1 四个任务与打分（越低越好）

| 任务 | 定义 | 得分 | 盲测集 |
|---|---|---|---|
| T1 应力悬停 | 8 s；±5° 初姿态、电机增益 ±1.5%、周期性力/力矩脉冲、传感器偏置与延迟 | 俯仰 RMS(°) + 0.3×高度 RMSE(cm) | 16 例（`pilot.py`） |
| T2 点头 | t0 = 2.4 s 起，尽快到 \|俯仰\| ≥ 30° 并保持 0.1 s，再尽快回到 < 5° | t30 + t_recover + 0.03×掉高(cm) + 0.5×\|漂移\|(m) + 坠机罚；未达成各计 2.5 s | 16 例（`maneuver_v2.py`） |
| T3 前冲急停 | t0 起沿当时航向前飞 **0.15 m** 并停住，窗口 4 s | t_reach + t_settle + 2×超调 + 0.03×掉高 + 0.5×\|侧偏\| + 坠机罚 | 16 例 |
| T4 大扰动恢复 | t0 时 20 ms 冲量力矩，8 个方向 × 3 个校准水平 × 2 种子（半数有第二击），窗口 2.5 s | 击前 + 击后倾角积分 + 0.2×\|dz\|；坠机 60 + 40×未飞完比例 | 48 例（`upset_v2.py`） |

T3 目标从 0.25 m 改为 0.15 m 是**预注册规则**触发的（`preregistration.json: dash_note`：经典 S1 在 0.25 m 时 0/16 到达，取第 12 大行程 0.152 m → 0.15 m）。T4 校准改为按方向类在 1–40 mN·m 对数网格上读等渗存活曲线（Stage A 外推缺陷的修正，`upset_v2.py` 文档）。

### 4.2 三层架构（所有臂共用）

```
IMU(11) ──► 18 维整流原始 IMU 特征 + 2 路指令(nod, dash) ──► 下行指令层（策略，每 2 ms）
                                                        │  输出：本拍是否放电(2 方向)、俯仰参考 ref∈[-0.7, 0.7] rad
            反射层  ReferencePID(kp×1.5, kd×0.85, ki×1.5)，姿态设定值 = ref ─► 云台 α/β、电机均值转速
            转向肌层 放电 → 该拍电机转速指令拉到上限 HIGH（持续 TWITCH=3 拍）
```
监督信号（任务目标）对所有臂相同：nod 在达成前为 1；dash 在前进 0.15 m 前为 1；hover/upset 恒为 0（扰动不预告）。目标达成由真值判定（导航层，不在 IMU 接口内）。

### 4.3 经典对手（同预算调参）与归一化

`bench classical` 在 8 个调参案例上调出每个任务的最强经典臂，归一化分母 = 该臂在盲测集上的得分（`artifacts/spike_steer/plan_v4/baseline.json`）：

| 任务 | 候选臂 | 最强臂 | 盲测得分（分母） |
|---|---|---|---|
| hover | pid 9.238 / law_tuned 5.751 | law_tuned（PID + PLL 锁相脉冲律，108 点网格） | 5.751 |
| nod | pid 5.176 / law_tuned 5.111 / schedule 0.909 | schedule（设定值调度：go/brake 两段，网格） | 0.909（16/16 达成） |
| dash | pid 8.444 / law_tuned 8.509 / schedule 5.871 | schedule | 5.871（16/16 到达） |
| upset | pid 92.4 (3/48) / law_tuned 84.8 (10/48) / pulse_tuned 75.0 (25/48) | pulse_tuned（翻正调参的脉冲律） | 75.025 |

综合 = Σ 权重 × 比值。第一轮权重 40/20/20/20；第二轮 25/25/25/25（用户在看第二轮结果前定）。

### 4.4 预注册假设（`plan_v4/preregistration.json`，`plan_v4/r2/preregistration_r2.json`）

- H1：综合 ≤ 0.95 且 T1 ≤ 1.05，≥ 2/3 种子（修正 4 后 ≥ 3/5）。
- H1′（第二轮新增）：每项 ≤ 1.05 且 T2/T3 至少一项 ≤ 0.90 且配对符号检验 p < 0.05。
- H2：T2–T4 中 ≥ 2 项 ≤ 0.90。
- H3：钳位陀螺+加速度计、延迟全置零，综合各恶化 ≥ 10%。
- H4：同尺寸 ANN 走同一流水线，与 SNN 相当（不主张 SNN 性能优势）。
- 混合体退路：若 H1 败但某任务 ≤ 0.9，报告逐任务取优的混合体。
- 次要统计：每任务报逐例中位比值、配对胜负数、符号检验 p。

---

## 5. 算法

### 5.1 下行指令网络 `DescendingNet`（`bench.py` NumPy 版 ↔ `descend.py` torch 版，逐拍自测一致）

- 输入 20 维：`features()` 把 8 路原始 IMU 通道（AHRS 横滚/俯仰、陀螺 xyz、加速度计 xyz，`pilot_v6.CHANNELS`，尺度 0.1 rad / 10 rad/s / 25 m/s²）按尺度归一后做正负整流得 16 维，加常数 1、0 得 18 维，再接常数 1、0 和两路指令。**没有 PLL、没有相位编码**，翅拍相位只能从陀螺/加速度计的 26.7 Hz 振荡自己建立。
- 隐层 H = 8 个电流型 LIF 神经元；每条输入连接带 11 档突触延迟（0–20 ms，`IN_DELAYS_TICKS`），权重张量 `W_in[H, 20, 11]`。
- 输出：2 个脉冲神经元（上推/下推放电），每条隐层连接也带 11 档延迟（`W_out[2, H, 11]`）；1 个分级单元（20 ms 漏积分，`W_g[H, 11]`），`ref = 0.7·tanh(g)` 作为俯仰设定值。
- 动力学：1 ms 子步，每 2 ms 拍两步；阈值 1、减法复位；τ_m、τ_s 共享可学（典型 1.0–1.3 ms）。
- 参数向量 = (W_in, b_h, W_out, b_o, W_g, b_g)，`descend.flatten/unflatten` 供 ES 使用。

### 5.2 S2 模仿（`descend.py train`）

- 示教者：`TracedClassical`，即每个任务的经典对手在**训练集**新案例上逐拍记录 [特征(20), 是否放电, 方向, ref]（`descend collect`，`collect_train.npz` / `collect_val.npz`）。
- 损失：放电 → 加权 BCE（正样本权重 `pos_weight`，放电率约 1–2%）；ref → MSE（权重 `lam_ref`）。替代梯度 BPTT，窗口 200 拍，Adam 3e-3，40 轮；验证召回 0.65–0.70。
- 产物 `imitate_spec_s{seed}.json`（权重 + τ + 元数据），可直接进闭环。

### 5.3 S3 策略搜索（`descend.py finetune`）

计划书写的是 PPO；实际用 **OpenAI-ES**（偏差记录在第 9 节）：
- 种群 32（对偶采样 ±ε），σ = 0.03，秩变换效用，Adam 0.02；每代对每个任务抽 `cases_per_task` 个新训练案例（公共随机数），目标 = 综合比值。
- 每 5 代在固定验证集上评估中心，保存 `spec_val_g<gen>.json`；选择 = 验证综合最低的中心（`spec_best.json`）。
- 第一轮：4 任务目标，40/20/20/20，每任务 2 例/代，30–40 代。第二轮：只 nod + dash，每任务 4 例/代，60 代。

**选择规则缺陷与修复**（提交 5a005d3）：最初按 (飞完数, −综合) 字典序选，种子 1 因 13 对 12 飞完拒绝了所有代，`ft_s1 == imit_s1`；改为只按验证综合选，并保留每个验证过的中心。

### 5.4 第二轮：指令门控的下行指令层（`descend_gated.py`，`GatedLearned`）

第一轮 ES 把悬停从 1.05 推到 1.37–1.50 换取点头/冲刺收益。第二轮按昆虫下行神经元"门控反射"的方式约束学习层：
- 学习层的放电和 ref **只在指令（nod/dash）为 1 时生效，加 1.0 s 交还尾巴**（刹停）；其余时间运行反射 PID + 悬停脉冲律 `base_law = law_tuned`，与经典悬停控制器逐拍相同 → 悬停比值 1.000 按构造。
- **修正 1 翻正反射**（c19391d，看第二轮结果前）：学习层不在岗且 AHRS 倾角 > 25°（回合内锁存）时，反射脉冲律切换到翻正调参的 `pulse_tuned`。实测阈值 15° 会在悬停误触发（悬停 1.14），25° 给翻转 1.050 / 悬停 1.000，35° 1.095，45°/60° 1.13。
- ES 目标只有 nod + dash；起点 = 第一轮同种子的模仿模型 + 门控字段。
- `install()` 以猴补丁方式把 `bench.Learned`、`bench.DescendingNet`、`bench.pool` 换成门控/ANN 分发版（fork 与 spawn 都安全），不改 `bench.py`。

### 5.5 修正 2：验证集扩大与零坠落选择（`reselect_r2.py`）

种子 3 在 8 例验证上从未坠落，盲测冲刺却 4/16 翻倒。修正 2：验证 8 例/任务（16 例，前 4 例即原验证集），候选 = 已保存的全部验证中心，选"零坠落中无验证综合最低"。验证集不参与 ES 更新，所以对已保存中心重选 **等价于** 用新规则重跑搜索，每种子 30 分钟而非 8 小时。两种规则的结果都报告。

### 5.6 H4 对照：同尺寸 ANN（`descend_ann.py`）

tanh 隐层 + sigmoid 输出（> 0.5 视为放电）+ 同样的 11 档延迟与 20 ms 漏积分 ref；参数名、形状与 SNN 相同，模仿、ES、盲测、消融全走同一流水线。

### 5.7 消融（`bench.evaluate_spec --variants`）

| 变体 | 做法 | 回答的问题 |
|---|---|---|
| clamp_fast | 只保留 AHRS 姿态输入（陀螺、加速度计钳零） | 网络是否依赖快通道建立相位 |
| nodelay_all | 所有延迟档合并到 0 延迟 | 突触延迟是否有功能 |
| noref | 分级单元置零（无设定值权限） | 设定值权限是否必要 |
| nopulse | 输出偏置 −50（永不放电） | 脉冲转向肌是否必要 |

### 5.8 统计

每任务按案例配对（同一种子、同一扰动）：均值比值、逐例中位比值、胜负数、符号检验 p；综合按种子报告。跨机器数值不逐位一致（同一模型同一案例，CloudPC 与 HomePC 的验证综合可差 0.1 以上，贴着倾角终止线的策略尤甚），所以**所有选代与盲测在 CloudPC 完成，搜索可在任何机器**（修正 3）。

---

## 6. 代码地图

| 路径 | 作用 |
|---|---|
| `experiments/algorithms/spike_steer/bench.py` | 基准：任务、打分、经典臂与调参、`DescendingNet`、`Learned`、盲测与消融、配对统计 |
| `.../spike_steer/maneuver_v2.py` | T2/T3 任务定义、设定值权限的经典调度（S1 网格 / S2 CEM / S3 特权刹停）、S0 门 |
| `.../spike_steer/upset_v2.py`（+ `upset.py`） | T4 校准（等渗存活曲线）、经典翻正臂、48 例盲测集 |
| `.../spike_steer/descend.py` | `DescendingSNN`（torch 孪生）、`collect`/`train`（模仿）、`finetune`（ES）、`flatten/unflatten` |
| `.../spike_steer/descend_gated.py` | 第二轮：`GatedLearned`、`prepare`/`finetune`/`evaluate`、预注册写入 |
| `.../spike_steer/descend_ann.py` | H4：`DescendingANN` 同尺寸速率网络及分发 |
| `.../spike_steer/reselect_r2.py` | 修正 2：16 例 + 零坠落重选 + 盲测 |
| `.../spike_steer/render_r2.py` | 同屏回放视频（学习方 vs 对手，同一盲测例，记录态逐帧，OSMesa/EGL） |
| `.../spike_steer/pilot.py`, `pilot_v3.py`, `pilot_v6.py` | 旧线：v1–v6 NEAT 脉冲转向（经典律 law_tuned / PLL 在这里） |
| `.../spike_steer/imitate.py` | 10-13 梯度模仿基线（证明"原始 IMU → 锁相放电"的解存在） |
| `.../neuro_gates/` | 原环境预备检验、翅拍定时权限测量 |
| `experiments/algorithms/robust_control/environment.py` | `ResearchEnv`、`ReferencePID`（原接口） |
| `runs/chain_v4*.sh`, `runs/chain_r2*.sh`, `runs/run_*.sh` | CloudPC 的 tmux 流水线脚本（`runs/` 不进 git） |
| `artifacts/spike_steer/v4/handoff_s3/run_*.sh` | 跨机器交接脚本（HomePC/MACPC 用；`v4/` 目录其余内容是旧的 NEAT τ 进化线，勿混） |
| `artifacts/spike_steer/plan_v4/` | 本计划全部归档：`baseline.json`、`preregistration.json`、`dash_target.json`、`upset_v2/`、`maneuver_v2_*.json`、`imitate_*`、`evaluate_*.json`、`finetune_s*/`、`r2/` |

典型命令（CloudPC，`PY=/root/miniconda3/envs/mujoco/bin/python`，始终 `OMP_NUM_THREADS=1`）：
```bash
$PY -m experiments.algorithms.spike_steer.bench classical --output runs/spike_steer/bench --workers 30
$PY -m experiments.algorithms.spike_steer.descend collect --output runs/spike_steer/bench --workers 30
$PY -m experiments.algorithms.spike_steer.descend train   --output runs/spike_steer/bench --seed 1 --epochs 40
$PY -m experiments.algorithms.spike_steer.bench evaluate  --output runs/spike_steer/bench --spec runs/spike_steer/bench/imitate_spec_s1.json --tag imit_s1 --workers 30
$PY -m experiments.algorithms.spike_steer.descend_gated prepare  --output runs/spike_steer/bench_r2 --base-law law_tuned
$PY -m experiments.algorithms.spike_steer.descend_gated finetune --output runs/spike_steer/bench_r2 --seed 1 --generations 60 --workers 30
$PY -m experiments.algorithms.spike_steer.descend_gated evaluate --output runs/spike_steer/bench_r2 --seed 1 --variants intact,clamp_fast,nodelay_all,noref,nopulse
$PY -m experiments.algorithms.spike_steer.reselect_r2 --seed 1 --centres artifacts/spike_steer/plan_v4/r2/finetune_s1 --tag r2_s1 --workers 30
MUJOCO_GL=osmesa $PY -m experiments.algorithms.spike_steer.render_r2 --spec artifacts/spike_steer/plan_v4/r2/finetune_s1/spec_best.json --tag r2_s1 --tasks nod,dash,hover,upset --case 0
```

---

## 7. 结果

### 7.1 第一轮（无门控，权重 40/20/20/20；`plan_v4/evaluate_*.json`）

| 模型 | 综合 | 悬停 | 点头 | 冲刺 | 翻转（飞完/48） |
|---|---|---|---|---|---|
| imit_s1（模仿） | 1.128 | 1.049 | 1.504 | 1.008 | 1.031 (21) |
| imit_s2 | 1.009 | 1.067 | 0.907 | 0.997 | 1.009 (18) |
| imit_s3 | 1.133 | 1.139 | 1.386 | 0.998 | 1.003 (21) |
| ft_s1_last（ES 末代中心） | 1.168 | 1.367 | 1.134 | **0.880** (16/16 胜, p=3e-5) | 1.095 (15) |
| ft_s3（验证最优，第 4 代） | 1.367 | 1.447 | 1.924 | 0.904 | 1.111 (15) |
| ft_s3_last | 1.265 | 1.503 | 1.249 | 0.901 | 1.171 (8) |
| ft_s2（ES 从未超过起点 = imit_s2） | 1.009 | 1.067 | 0.907 | 0.997 | 1.009 (18) |
| 混合体（逐任务取优） | 0.976–0.981 | | | | |

判读：H1、H2 不成立；ES 换悬停买机动，翻转从未改善；点头均值被个别未达成例主导（逐例中位 0.95–1.04）。

### 7.2 第二轮（门控，25/25/25/25；`plan_v4/r2/`）：见第 1 节表。验证轨迹（8 例，点头+冲刺均值）：种子 1 从 0.909 降到 0.795（第 54 代）；种子 3（HomePC）从 1.362 降到 0.805（第 49 代）；种子 2（Mac）从 0.913 降到 0.851（第 44 代）。

修正 2 对种子 3 的 13 个中心的筛选（`r2/reselect_r2_s3.json`，CloudPC 数值）：第 29、34、59 代在新增 8 例上坠落被淘汰；选第 54 代（16 例 0.821，冲刺峰值倾角 40.5°），盲测 0.958，冲刺 16/16。种子 1（`r2/reselect_r2_s1.json`）：原选第 54 代在新增例中坠落 1 次，改选第 44 代，盲测 0.940（与第 54 代的 0.9396 几乎相同）。

### 7.3 消融（H3；综合比值，`intact` → 变体）

| 模型 | intact | clamp_fast | nodelay_all | noref | nopulse |
|---|---|---|---|---|---|
| r2_s1（修正 2，第 44 代） | 0.940 | 3.54 | 3.63 | 2.32 | 1.72 |
| r2_s3（修正 2，第 54 代） | 0.958 | 4.68 | 1.27 | 2.32 | 2.42 |
| ft_s1_last（第一轮） | 1.168 | 2.09 | 2.53 | 2.23 | 1.58 |
| ft_ann_s3（ANN） | 1.038 | 1.65 | 1.46 | 2.08 | 1.60 |

表中所有变体 ≥ +23%，全部归档模型中最小的是 ft_s3_last 的 nodelay_all +21%，H3（≥ 10%）成立。`noref` 在每个模型上都把点头推到 5.7 左右（无设定值权限就做不成点头，与 10-13 的可行性诊断一致）。

### 7.4 H4：ANN 对照

| 模型 | 模仿（HomePC 训练 / CloudPC 训练） | ES 后 |
|---|---|---|
| ANN s1 | 1.015 / 1.144 | = 模仿（Mac ES，作废） |
| ANN s2 | 1.045 / 1.060 | 1.060（= CloudPC 模仿，ES 未超起点） |
| ANN s3 | 1.038 / 1.103 | 1.038（= 模仿，ES 未超起点） |
| SNN s1/s2/s3 模仿 | 1.128 / 1.009 / 1.133 | 见 7.1、7.2 |

模仿阶段 ANN ≈ SNN（H4 成立，预期之内）；ES 阶段三个 ANN 种子都没有一代超过模仿起点，SNN 第二轮则稳定改善。torch 模仿训练跨机器不可复现（同一种子两台机器 1.015 对 1.144），论文表只用一台机器的数字，另一套存为 `*_cloudpc.json`。

### 7.5 回放视频（`render_r2.py`，盲测 case 0，未入库）

同一扰动下：种子 1 冲刺比对手快 21% 并稳住；种子 3（原选代）前倾过大翻倒。16 例统计：种子 1 脉冲 111 次/占空比 3.4%/ref 0.12 rad/峰值倾角 37°/掉高 2.6 cm；种子 3 为 60 次/2.0%/0.16 rad/**51°**/7.1 cm；对手 27 次/34°。两种策略到达时间相同（0.78 s，对手 1.56 s），区别在"多打脉冲"还是"多低头"，后者离倾角终止线太近。

---

## 8. 发现与诊断

1. **无门控的 ES 用悬停换机动**：归一化综合里点头的收益远大于悬停的损失，ES 理性地牺牲悬停。门控把"学习只发生在机动窗口"写进结构，而不是靠权重。
2. **冲刺的脆弱性来自目标函数没有余量项**：验证集 8 例看不出"贴线飞"，样本到 16 例就能暴露。这是流程缺陷，不是种子运气，种子 1 的后期中心同样有。
3. **跨机器浮点差异会被近坠落策略放大**：同一模型同一案例，HomePC 0.805 对 CloudPC 0.918；Mac 更甚。结论是统计等价而非逐位复现，搜索与选代必须分机器说明。
4. **ANN 在 ES 阶段不提升而 SNN 提升**：目前只是观察（3 个 ANN 种子、2 个 SNN 种子），可作为讨论点，不作主张。
5. **翻转任务学习层没有贡献**：第一轮从未改善；第二轮按构造等于经典翻正律（1.05）。论文里 T4 的角色是"不劣"，不是"更好"。

---

## 9. 与计划的偏差（全部记录在预注册修正里）

| 偏差 | 原因 | 影响 |
|---|---|---|
| S3 用 OpenAI-ES 而非 PPO | 模仿起点好、参数少（2035 个），ES 对脉冲网络无需替代梯度，实施快 | 搜索预算以回合数记入预注册 |
| 选择规则修复（5a005d3） | 字典序规则拒绝所有代 | 种子 1 第一轮结果用 `ft_s1_last` 补报 |
| 修正 1 翻正反射 | 单一 base_law 下翻转 1.13 是构造罚分 | 翻转 1.05 |
| 修正 2 验证 16 例 + 零坠落 | 种子 3 冲刺坠落 | 两种规则并列报告；种子 3 的探索性检查先于修正写入，已披露 |
| 修正 3 种子 3 在 CloudPC 重跑 | 搜索与选代应同机 | 主结果取 CloudPC 搜索的运行，HomePC 运行并列 |
| 修正 4 加种子 4、5 | 结论不系于一个种子 | H1/H1′/H2 改 ≥ 3/5 |
| Mac 结果作废 | 数值差异过大（翻转 1.054 对 1.050） | 种子 2 在 HomePC 重跑 |
| LabPC（16 核）与 LabPC-1（Windows）退出 | 机器问题 / 无运行环境 | 无 |
| `artifacts/spike_steer/v4/` 目录冲突 | 与旧 NEAT τ 线同名 | 本计划归档改到 `plan_v4/`，交接脚本仍在 `v4/handoff_s3/`，待收尾时迁移 |

---

## 10. 正在跑的任务与下一步

| 任务 | 机器 | 预计（JST） |
|---|---|---|
| 第二轮种子 2 ES（Linux 重跑） | HomePC（18 worker） | 17:00 出盲测 → CloudPC 重选 + 盲测 |
| 第二轮种子 3 ES（修正 3） | CloudPC（30 worker） | 19:00 出盲测 → 自动重选 + 盲测 |
| 种子 4、5 模仿（40 轮） | CloudPC | 10:30 完成，门控 spec 推送 |
| 第二轮种子 4 ES | CloudPC，种子 3 之后 | 次日 05:00 |
| 第二轮种子 5 ES | HomePC，种子 2 之后 | 次日 02:00 → CloudPC 重选 |

之后：五个种子按修正 2–4 判 H1/H1′/H2；写最终结果报告（含两种选代规则、机器复现、消融、ANN 对照、逐例统计）；收尾提交（`git mv v4/handoff_s3 → plan_v4/handoff`，更新 HANDOFF.md）。

**待用户决定**：翻转 1.050006 的边界读法；经典对手是否允许轨迹优化（目前只用调度网格/CEM）；目标期刊（计划书建议 Bioinspiration & Biomimetics）；是否做升力/电机增益的敏感性分析（计划书第 3 节可选项）。

---

## 11. 诚实条款（将写入论文局限性）

纯仿真；升力按单翼定基座数据校准，力矩和逐拍波形未验证；锁相脉冲的最佳相位在实物上可能不同；学习方预算（模仿 40 轮 + ES 60 代 × 32 × 8 回合）大于经典调参预算（网格/CEM 数百回合）；跨机器只有统计等价；修正 2 的规则在看到种子 3 盲测后提出（参数未按结果调整，已披露）；T4 的学习层无贡献。
