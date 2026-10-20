| arm | seed | machine | manoeuvre_mean | evade_ratio | evade_pitch | evade_roll | evade_wins | redirect_ratio | redirect_pitch | redirect_roll | redirect_wins | dash_ratio | dash_pitch | dash_roll | dash_diag | dash_wins | crashes | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical | - | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 0 | - |
| mlp_ppo | 1 | HomePC | 1.016 | 0.943 | 0.961 | 0.925 | 14/16 | 1.062 | 1.021 | 1.103 | 8/16 | 1.044 | 0.975 | 1.138 | 1.032 | 5/16 | 0 | False |
| mlp_ppo | 2 | HomePC | 1.107 | 1.110 | 1.205 | 1.016 | 4/32 | 1.168 | 1.115 | 1.220 | 5/32 | 1.043 | 1.067 | 1.043 | 1.031 | 6/32 | 0 | False |
| mlp_bc | 1 | HomePC | 1.108 | 1.101 | 1.180 | 1.022 | 5/16 | 1.176 | 1.136 | 1.217 | 6/16 | 1.046 | 1.078 | 1.055 | 1.025 | 6/16 | 0 | False |
| mlp_bc | 2 | HomePC | 1.107 | 1.110 | 1.205 | 1.016 | 4/32 | 1.168 | 1.115 | 1.220 | 5/32 | 1.043 | 1.067 | 1.043 | 1.031 | 6/32 | 0 | False |

| arm | n seeds | manoeuvre mean (mean ± sd) | evade | redirect | dash | seeds passing |
|---|---|---|---|---|---|---|
| mlp_ppo | 2 | 1.062 ± 0.064 | 1.027 | 1.115 | 1.043 | 0/2 |
| mlp_bc | 2 | 1.107 ± 0.001 | 1.106 | 1.172 | 1.044 | 0/2 |

比值 = 该臂得分 / 该方向类 classical 对手中位数（越低越好，1.0 = 与 classical 相同）；manoeuvre mean = 三任务比值平均；pass：manoeuvre mean <= 0.95 且盲测零坠落（沿用 v6 的 M2 口径，v7 为快速验证，未预注册）。盲测集 bench9 BASES['test'] = 2_098_100_000，每任务 16 例。snn_* 由 descend9 产出，mlp_* 由 v7ppo 产出。
