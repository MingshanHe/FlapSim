| method | seed | selected_generation | manoeuvre_mean | nod_ratio | nod_pitch | nod_roll | nod_wins | dash_ratio | dash_pitch | dash_roll | dash_diag | dash_wins | crashes | M2_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classical (opponent) | - | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 0 | - |
| PPO ref (v6ppo) | 1 | -1 | 1.223 | 1.402 | 0.915 | 1.889 | 10/32 | 1.043 | 1.011 | 1.113 | 1.024 | 5/32 | 0 | False |

| method | n seeds | manoeuvre mean (mean ± sd) | nod | dash | seeds passing M2 |
|---|---|---|---|---|---|
| PPO ref (v6ppo) | 1 | 1.223 ± 0.000 | 1.402 | 1.043 | 0/1 |

比值 = 学到的控制器得分 / 该方向类 classical 对手中位数（越低越好，1.0 = 和 classical 一样）；M2：manoeuvre mean <= 0.95 且盲测 64 例零坠落。盲测集 TEST_BASE 2097100000，nod 32（4 向 × 8）+ dash 32（8 向 × 4）。
