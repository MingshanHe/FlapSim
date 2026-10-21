## T1 上限：按案例取所有学习型控制器的最小分数（oracle ensemble，极度乐观）

若连"每个案例都挑最好的那个控制器"都到不了 0.95，则单个控制器更到不了。

| plan | arms pooled | controllers | oracle mean ratio | evade | redirect | dash | crashes |
|---|---|---|---|---|---|---|---|
| v8 | snn_es | 5 | 0.793 | 0.700 | 0.780 | 0.900 | 0 |
| v8 | snn_es+snn_bc+snn9_frozen | 10 | 0.789 | 0.700 | 0.772 | 0.895 | 0 |
| v8 | snn_es+snn_bc+snn9_frozen+classical | 11 | 0.785 | 0.700 | 0.766 | 0.889 | 0 |
| v9 | snn_es | 5 | 0.890 | 0.879 | 0.867 | 0.923 | 0 |
| v9 | snn_es+snn_bc+snn9_frozen | 10 | 0.878 | 0.866 | 0.849 | 0.918 | 0 |
| v9 | snn_es+snn_bc+snn9_frozen+classical | 11 | 0.874 | 0.863 | 0.843 | 0.916 | 0 |
| v10 | snn_es | 5 | 0.936 | 0.940 | 0.909 | 0.959 | 0 |
| v10 | snn_es+snn_bc+snn9_frozen | 14 | 0.868 | 0.866 | 0.826 | 0.913 | 0 |
| v10 | snn_es+snn_bc+snn9_frozen+classical | 15 | 0.866 | 0.866 | 0.823 | 0.909 | 0 |

### T1b 每任务取最好 seed（允许不同任务用不同控制器；无 per-case 挑选）

| plan | arm | best-per-task mean | evade | redirect | dash |
|---|---|---|---|---|---|
| v9 | snn_es | 0.996 | 0.991 | 1.000 | 0.995 |
| v10 | snn_es | 1.020 | 1.026 | 1.013 | 1.022 |
| v10 | snn9_frozen | 0.988 | 0.978 | 1.007 | 0.980 |

## T2 覆盖缺陷的配对检验：(SNN − classical)/median，按 trained / untrained cell，逐 seed（v7full+v8+v9+v10 共 20 个 ES seed-run）

### redirect

| plan | seed | ES−cls trained | ES−cls untrained | ES: untrained−trained | BC−cls trained | BC−cls untrained | BC: untrained−trained |
|---|---|---|---|---|---|---|---|
| v7full | 1 | -0.182 | -0.087 | 0.095 | 0.084 | 0.071 | -0.012 |
| v7full | 2 | -0.325 | -0.143 | 0.183 | 0.050 | 0.168 | 0.118 |
| v7full | 3 | -0.212 | -0.144 | 0.069 | 0.058 | 0.116 | 0.058 |
| v7full | 4 | -0.282 | -0.061 | 0.222 | 0.175 | 0.335 | 0.161 |
| v7full | 5 | -0.270 | -0.257 | 0.013 | 0.123 | 0.360 | 0.237 |
| v8 | 1 | -0.316 | -0.083 | 0.232 | -0.008 | 0.159 | 0.167 |
| v8 | 2 | -0.460 | -0.104 | 0.356 | -0.052 | 0.164 | 0.216 |
| v8 | 3 | -0.308 | -0.055 | 0.253 | 0.034 | 0.142 | 0.108 |
| v8 | 4 | -0.402 | 0.425 | 0.827 | 0.158 | 0.356 | 0.197 |
| v8 | 5 | -0.238 | 0.072 | 0.310 | 0.068 | 0.397 | 0.329 |
| v9 | 1 | -0.182 | -0.148 | 0.033 | 0.079 | 0.058 | -0.021 |
| v9 | 2 | -0.301 | 0.285 | 0.586 | 0.016 | 0.109 | 0.093 |
| v9 | 3 | -0.157 | -0.209 | -0.052 | 0.062 | 0.123 | 0.061 |
| v9 | 4 | -0.217 | -0.096 | 0.121 | 0.147 | 0.353 | 0.206 |
| v9 | 5 | 0.111 | -0.203 | -0.314 | 0.093 | 0.421 | 0.328 |
| v10 | 1 | -0.318 | 0.113 | 0.432 | 0.137 | 0.166 | 0.029 |
| v10 | 2 | -0.110 | 0.119 | 0.229 | 0.031 | 0.165 | 0.134 |
| v10 | 3 | -0.015 | 0.133 | 0.148 | nan | nan | nan |
| v10 | 4 | -0.365 | 0.297 | 0.662 | 0.144 | 0.283 | 0.139 |
| v10 | 5 | -0.330 | 0.062 | 0.391 | 0.132 | 0.251 | 0.120 |

ES: untrained−trained 中位数 +0.225，均值 +0.240，>0 的 seed-run 18/20，Wilcoxon p=0.001
BC（对照，看过全部方向）: untrained−trained 中位数 +0.134，均值 +0.140，>0 的 17/19
ES 不对称 − BC 不对称（可归因于 ES 覆盖的净效应）: 中位数 +0.066，均值 +0.104，>0 的 14/19，Wilcoxon p=0.077
对任务 ratio 的含义：若把 untrained cell 修到和 trained 一样，redirect ratio 改善 ≈ 0.052（untrained 占一半案例）

### dash

| plan | seed | ES−cls trained | ES−cls untrained | ES: untrained−trained | BC−cls trained | BC−cls untrained | BC: untrained−trained |
|---|---|---|---|---|---|---|---|
| v7full | 1 | 0.043 | 0.042 | -0.001 | -0.019 | 0.002 | 0.021 |
| v7full | 2 | 0.105 | -0.013 | -0.118 | 0.057 | -0.010 | -0.067 |
| v7full | 3 | -0.048 | -0.061 | -0.013 | 0.051 | -0.039 | -0.089 |
| v7full | 4 | 0.041 | 0.031 | -0.010 | 0.086 | 0.012 | -0.075 |
| v7full | 5 | 0.047 | -0.084 | -0.131 | 0.239 | -0.000 | -0.240 |
| v8 | 1 | -0.024 | 0.044 | 0.067 | 0.040 | 0.014 | -0.026 |
| v8 | 2 | 0.086 | 0.322 | 0.236 | 0.060 | -0.012 | -0.072 |
| v8 | 3 | 0.189 | -0.070 | -0.259 | 0.051 | -0.041 | -0.092 |
| v8 | 4 | 0.123 | 0.030 | -0.093 | 0.102 | 0.031 | -0.071 |
| v8 | 5 | 0.016 | 0.138 | 0.122 | 0.180 | -0.011 | -0.191 |
| v9 | 1 | -0.078 | 0.158 | 0.236 | 0.024 | 0.028 | 0.004 |
| v9 | 2 | 0.038 | -0.020 | -0.058 | 0.039 | 0.003 | -0.036 |
| v9 | 3 | 0.080 | 0.179 | 0.099 | 0.022 | -0.020 | -0.042 |
| v9 | 4 | 0.120 | 0.103 | -0.017 | 0.092 | 0.016 | -0.075 |
| v9 | 5 | -0.045 | -0.007 | 0.038 | 0.158 | 0.019 | -0.139 |
| v10 | 1 | 0.360 | 0.342 | -0.019 | 0.157 | 0.078 | -0.079 |
| v10 | 2 | 0.088 | 0.082 | -0.005 | 0.082 | -0.041 | -0.124 |
| v10 | 3 | 0.077 | -0.011 | -0.088 | nan | nan | nan |
| v10 | 4 | 0.159 | 0.027 | -0.132 | 0.114 | -0.004 | -0.117 |
| v10 | 5 | 0.023 | -0.005 | -0.028 | 0.145 | 0.017 | -0.128 |

ES: untrained−trained 中位数 -0.015，均值 -0.009，>0 的 seed-run 6/20，Wilcoxon p=0.411
BC（对照，看过全部方向）: untrained−trained 中位数 -0.075，均值 -0.086，>0 的 2/19
ES 不对称 − BC 不对称（可归因于 ES 覆盖的净效应）: 中位数 +0.077，均值 +0.082，>0 的 13/19，Wilcoxon p=0.009
对任务 ratio 的含义：若把 untrained cell 修到和 trained 一样，dash ratio 改善 ≈ 0.041（untrained 占一半案例）

## T3 坠机集中在哪些案例 / 方向（所有 plan、所有学习型臂）

| plan | task | dir | type | n crashes (all learned arms/seeds) |
|---|---|---|---|---|
| v10 | dash | 180 | tilt | 4 |
| v7full | dash | 0 | height_escape | 2 |
| v8 | dash | 270 | height_escape | 2 |
| v8 | evade | 180 | tilt | 2 |
| v8 | redirect | 270→180 | tilt | 2 |
| v10 | evade | 180 | tilt | 2 |
| v10 | evade | 0 | tilt | 2 |
| v8 | dash | 180 | tilt | 1 |
| v9 | redirect | 180→90 | tilt | 1 |
| v10 | redirect | 0→90 | tilt | 1 |
| v9 | evade | 180 | tilt | 1 |
| v8 | redirect | 0→270 | height_escape | 1 |
| v8 | redirect | 270→0 | height_escape | 1 |
| v8 | dash | 0 | height_escape | 1 |
| v9 | redirect | 180→270 | tilt | 1 |
| v10 | redirect | 180→90 | tilt | 1 |

同一 case seed 被多个控制器撞的情况（case seed → 撞它的 (arm, seed) 列表）：

- v7full dash case 2098300022 dir {'0'} (classical: ok, tilt 32°): ['snn_es2', 'snn_es5']
- v8 dash case 2098500041 dir {'270'} (classical: ok, tilt 29°): ['snn_es2', 'snn_es5']

经典对手的坠机案例，以及学习型臂在同一案例上的表现：

- v7full evade 0° case 2098300022 (tilt): learned arms crashed 0/10; learned tilt peaks [35, 36, 37, 37, 38, 41, 43, 45, 49, 51]
- v8 evade 0° case 2098500004 (tilt): learned arms crashed 0/10; learned tilt peaks [33, 37, 38, 40, 40, 43, 43, 44, 46, 48]
- v8 evade 0° case 2098500018 (tilt): learned arms crashed 0/10; learned tilt peaks [37, 37, 38, 39, 40, 40, 41, 41, 44, 45]
- v8 evade 0° case 2098500032 (tilt): learned arms crashed 0/10; learned tilt peaks [35, 39, 39, 39, 39, 40, 41, 42, 43, 47]
- v8 evade 180° case 2098500041 (tilt): learned arms crashed 1/10; learned tilt peaks [40, 42, 42, 43, 44, 44, 45, 46, 47, 57]
- v8 evade 180° case 2098500052 (tilt): learned arms crashed 0/10; learned tilt peaks [32, 34, 34, 35, 36, 36, 40, 42, 43, 47]
- v10 evade 0° case 2098900000 (tilt): learned arms crashed 1/14; learned tilt peaks [31, 35, 43, 43, 43, 44, 45, 46, 46, 46, 46, 48, 52, 59]
- v10 evade 180° case 2098900027 (tilt): learned arms crashed 1/14; learned tilt peaks [41, 42, 44, 44, 45, 46, 46, 47, 47, 48, 48, 48, 62, 67]
- v10 evade 180° case 2098900059 (tilt): learned arms crashed 0/14; learned tilt peaks [36, 37, 38, 39, 40, 42, 42, 43, 44, 44, 45, 46, 46, 48]

## T4 接近坠机的频率（tilt peak 阈值），比坠机计数更稳定的风险代理

| plan | arm | episodes | mean tilt peak | p90 | p99 | >50° | >60° | >70° | crashes |
|---|---|---|---|---|---|---|---|---|---|
| v9 | classical | 192 | 37.041 | 45.120 | 48.612 | 0 | 0 | 0 | 0 |
| v9 | snn_bc | 960 | 28.839 | 35.919 | 40.063 | 0 | 0 | 0 | 0 |
| v9 | snn_es | 960 | 37.092 | 46.707 | 54.644 | 47 | 3 | 3 | 3 |
| v10 | classical | 192 | 36.276 | 44.506 | 50.474 | 2 | 1 | 0 | 3 |
| v10 | snn_bc | 768 | 32.345 | 39.793 | 46.037 | 1 | 0 | 0 | 1 |
| v10 | snn_es | 960 | 36.886 | 45.435 | 55.157 | 31 | 6 | 4 | 6 |
| v10 | snn9_frozen | 960 | 36.928 | 46.950 | 55.317 | 50 | 4 | 2 | 3 |

按任务/方向的 >60° 次数（v9+v10, snn_es + snn9_frozen）：

| task | dir | SNN >60° / n | classical >60° / n |
|---|---|---|---|
| dash | 180 | 4/120 | 0/16 |
| evade | 180 | 4/240 | 1/32 |
| redirect | 0 | 1/240 | 0/32 |
| redirect | 180 | 3/240 | 0/32 |
| redirect | 90 | 1/240 | 0/32 |

## T5 ES 训练批 objective 的噪声（稳健统计）：train-batch 中心 vs 选择集，Spearman；以及无坠机代的差

| plan | seed | gens | Spearman all | crash-free gens | Spearman crash-free | gap mean (crash-free) | gap sd (crash-free) | first gen centre crashed on train batch | first gen crashed on selection | mean crashed candidates/gen |
|---|---|---|---|---|---|---|---|---|---|---|
| v9 | 1 | 20 | 0.263 | 13 | 0.615 | 0.012 | 0.076 | 16 | 12 | n/a |
| v9 | 2 | 20 | 0.347 | 15 | 0.525 | 0.018 | 0.084 | 17 | 9 | n/a |
| v9 | 3 | 20 | 0.364 | 20 | 0.364 | -0.024 | 0.070 | None | None | n/a |
| v9 | 4 | 20 | -0.080 | 7 | 0.929 | -0.032 | 0.099 | 7 | 6 | n/a |
| v9 | 5 | 20 | 0.035 | 7 | 0.464 | -0.043 | 0.067 | 8 | 5 | n/a |
| v10 | 1 | 20 | 0.117 | 8 | 0.167 | 0.042 | 0.079 | 17 | 1 | n/a |
| v10 | 2 | 20 | -0.152 | 12 | 0.000 | -0.045 | 0.104 | None | 5 | n/a |
| v10 | 3 | 20 | 0.382 | 13 | 0.582 | 0.002 | 0.077 | 4 | 10 | n/a |
| v10 | 4 | 20 | -0.195 | 3 | nan | -0.061 | 0.045 | 6 | 2 | n/a |
| v10 | 5 | 20 | 0.030 | 13 | 0.099 | 0.033 | 0.092 | 10 | 7 | n/a |

说明：训练批上中心 objective 与选择集 objective 在无坠机代上的 SD(gap) 即 12 例 objective 的噪声水平（约 ±0.05–0.1），与各代之间的真实差异同量级。

## T6 高度项的余量：evade / redirect 的 alt_dev（cm）分布，经典 vs SNN（v9+v10）

| task | arm | flown episodes | mean alt_dev cm | p10 | median | p90 | min |
|---|---|---|---|---|---|---|---|
| evade | classical | 125 | 7.283 | 3.622 | 6.862 | 11.721 | 1.714 |
| evade | snn_bc | 575 | 6.442 | 2.641 | 5.319 | 12.182 | 1.344 |
| evade | snn_es | 637 | 7.219 | 3.398 | 6.476 | 11.324 | 1.213 |
| evade | snn9_frozen | 319 | 7.082 | 3.215 | 6.424 | 10.616 | 1.016 |
| redirect | classical | 128 | 10.942 | 7.314 | 10.184 | 16.209 | 4.349 |
| redirect | snn_bc | 576 | 9.770 | 4.518 | 8.692 | 17.577 | 2.816 |
| redirect | snn_es | 638 | 12.034 | 7.227 | 11.071 | 17.961 | 2.011 |
| redirect | snn9_frozen | 318 | 11.327 | 5.645 | 10.950 | 17.369 | 1.681 |
| dash | classical | 128 | 10.091 | 4.549 | 8.550 | 19.429 | 3.594 |
| dash | snn_bc | 576 | 8.486 | 3.988 | 6.660 | 16.791 | 2.507 |
| dash | snn_es | 636 | 9.715 | 4.898 | 8.438 | 16.618 | 1.910 |
| dash | snn9_frozen | 320 | 8.889 | 4.130 | 8.109 | 14.418 | 1.622 |

## T7 一个 seed 要过 0.95 需要什么：以 v9 经典 rows 为底，各任务可动项全部取"所有臂同案例观测到的最好值"

| plan | task | classical self-ratio | ratio with every component at best observed value (any arm, same case) |
|---|---|---|---|
| v9 | evade | 1.054 | 0.781 |
| v9 | redirect | 1.184 | 0.656 |
| v9 | dash | 1.022 | 0.829 |

| plan | task | classical self-ratio | ratio with every component at best observed value (any arm, same case) |
|---|---|---|---|
| v10 | evade | 1.659 | 0.708 |
| v10 | redirect | 1.147 | 0.628 |
| v10 | dash | 1.013 | 0.793 |
