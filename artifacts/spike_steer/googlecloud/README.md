# GoogleCloud 节点结果（近似）

机器：32 vCPU AMD EPYC 7B13，Ubuntu 26.04.1，g++ 15.2。精确性检查与参考机器（g++ 13.3）MISMATCH，按 plan v11 修正 1 只作 ES 搜索 / 探索性结果，标注"GoogleCloud，近似"。
代码：research/original-env-snn-neat @ 74a55d6（v11、v11 上限臂）；更早的实验见各目录日志。
来源：节点上 `~/flapsim/runs/spike_steer/` 原样复制（去掉 `view_*` 符号链接目录和 `collect_*.npz` 副本，后者与 `plan_v7/` 相同）。顶层 `*.sh` 是运行脚本。

| 目录 | 内容 | 状态 | 盲测（mean，坠毁） |
|---|---|---|---|
| `v11/finetune_s{6..10}_v11` 等 | v11 探索性 SNN 种子 6–10，descend12，16 代，reselect 64/任务，盲测 2_099_100_000 × 64/任务 | 完成 | s6 0.997 / s7 1.093 / s8 1.058 / s9 1.013 / s10 1.016，均 0 |
| `v11/cls_es_s{3,4}` 等 | v11 经典上限臂种子 3、4，descend_cls，同一盲测集 | 完成 | s3 0.999 / s4 1.018，均 0 |
| `v9/` | v9 SNN 种子 6–10，descend10 careful（lr 0.01，sigma 0.02，20 代），盲测 2_098_700_000 × 64/任务 | 完成 | s6 1.018(0) / s7 1.149(2) / s8 1.222(4) / s9 1.036(1) / s10 1.074(0) |
| `v7/` | v7 careful 与 tilt45，种子 1、2 | 完成 | careful s1 1.097 / s2 1.104(1)；tilt45 s1 1.079 / s2 0.997(0) |
| `v7ext/` | v7full 扩展种子 6–10：模仿 + ES | 第 8 代处手动停止（0–7 代完成），有 checkpoint | — |
| `v6/` | v6 ES 种子 1/2/3/5，descend8 | 第 28 代处手动停止（0–27 代完成），见 `v6/STOPPED*.md` | — |
