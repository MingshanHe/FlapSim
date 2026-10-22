# 论文工作交接（CloudPC-2 → HomePC，2026-10-22）

CloudPC-2 即将长期关机。所有代码、数据归档、文稿和图都在 GitHub 分支 `research/original-env-snn-neat` 上（`git fetch && git checkout research/original-env-snn-neat`）。本文件说明论文相关的全部状态，HomePC 上的会话读完即可接手。

## 1. 交付物在哪里（全部在 `docs/paper/`）

| 内容 | 路径 |
|---|---|
| 文稿（Markdown 原稿，是 LaTeX 的生成源） | `draft_method_results.md` |
| LaTeX 源码 + 图 | `latex/main.tex`、`latex/figs/`、`latex/README.md`；`md2tex.py` 从 Markdown 重新生成 |
| 编译好的 PDF | `draft_method_results.pdf` |
| 图（PNG + 可编辑 SVG） | `figs/fig_m1–m5`（方法图）、`figs/fig_r0–r6`（结果图，含 r0b、r5b）；生成脚本 `make_method_figs.py`、`make_results_figs.py` |
| 可编辑 PPT（架构 / 流程 / 任务三张原生图形页 + 每张图一页） | `flapsim_paper_figures.pptx`；生成脚本 `make_pptx.py` |
| 全部数据 CSV | `data/*.csv`（汇总表）和 `data/episodes/*.csv`（每一局盲测）；生成脚本 `export_csv.py` |
| 三轮审稿记录 | `reviews/review_method_round1.md`、`review_full_round2.md`、`review_round3.md` |
| kick 对比视频 | `artifacts/spike_steer/plan_v12/videos/v12_kick_s4_crashdiff.mp4` |

本机的 Claude 会话完整记录（JSONL，gzip）另行发给用户，不在仓库里。

## 2. 论文现在的定位（用户已定）

- 脉冲（spike-timed pulses）是学习层自己的通道；经典对照家族是 **PID + 开环设定点调度，机动期间不带脉冲**。预注册对手就是这个，保持为所有判据的对照。
- "经典律在机动期间保留脉冲"（`ClassicalUnsuppressed`，plan v14）只作为 **机理探针** 出现在 3.3，不是竞争者；但要如实写它比学习层更抗摔（主冲击 5 对 22–40）。
- 主张："相对不带机动脉冲的经典调度，脉冲下行层在无扰动机动上持平、在机动中受扰时明显更抗摔；其来源是机动期间保持活跃的脉冲通道，学习层自己发现了它。"
- 没过的预注册判据（M4/M5/M6，比值 ≤ 0.95）照实报告，并说明它对所有训练过的控制器都不可达（classical 自身 1.089）。

## 3. 数据来源索引（每个数字的出处）

- 无扰动 v11：`artifacts/spike_steer/plan_v11/summary.json`（种子 1–3 行见 `runs/spike_steer/v11/`，种子 4–5 见 `plan_v11/homepc/`）；上限臂 `plan_v11/cls_es/`、`plan_v11/homepc/cls_es_s2/`；审计 `plan_v11/audit/audit_report.md`。
- 鲁棒性 16 组：`runs/spike_steer/v12/<test>/summary.json`、`runs/spike_steer/v13/<test>/summary.json`（已归档到 `artifacts/spike_steer/plan_v12/`、`plan_v13/`）；汇总表 `plan_v14/crash_table_with_unsuppressed.md`。
- 机理：`plan_v13/mechanism_summary.json`；消融、探针、模仿起点 2–5、SNN 对 classical+ES：`plan_v14/summary_d1_d4.json`、`plan_v14/summary_unsuppressed.json`、`plan_v14/rows/`。
- 预注册与修订：`plan_v11/preregistration_v11.json`（+ amendment_1）、`plan_v12/preregistration_v12.json`（+ amendment_2_post_hoc、amendment_3）、`plan_v13/preregistration_v13.json`（+ amendment_1/2/3）、`plan_v14/preregistration_v14.json`（+ amendment_1）。
- 早期轮次：`plan_v7/`、`plan_v8/summary.json`、`plan_v9/summary.json`、`plan_v10/summary.json`、`report_v10.md`、`plan_v11/report_v11_v12.md`。
- GoogleCloud 探索性种子 6–10 和上限臂种子 3–4：只有消息里的汇总数，逐局 rows 不在本地（GoogleCloud 无推送凭据）。

## 4. 如何在 HomePC 上重新生成

```bash
# 环境：仓库自带 .venv（python 3.11），或按 scripts/setup_v6_node.sh 重建
PYTHONPATH=. .venv/bin/python docs/paper/make_method_figs.py      # 方法图 PNG+SVG
PYTHONPATH=. .venv/bin/python docs/paper/make_results_figs.py     # 结果图 PNG+SVG
PYTHONPATH=. .venv/bin/python docs/paper/export_csv.py            # CSV
.venv/bin/python docs/paper/make_pptx.py                          # PPT（需要 python-pptx）
.venv/bin/python docs/paper/md2tex.py && cd docs/paper/latex && tectonic main.tex   # 或 pdflatex main.tex 两遍
```
tectonic 是单文件可执行程序（https://github.com/tectonic-typesetting/tectonic/releases，0.15.0），首次编译会联网下载宏包。

注意：`runs/` 不进 git。结果图脚本从 `runs/spike_steer/v12`、`v13`、`v14`、`v11` 读逐局 rows；这些在 `artifacts/spike_steer/plan_v1*/` 下都有归档副本（`rows_classical.json`、`evaluate_*.json`、`plan_v14/rows/`）。HomePC 上若 `runs/` 缺失，把 `artifacts/.../plan_v12/<test>/`、`plan_v13/<test>/` 复制到 `runs/spike_steer/v12|v13/<test>/`，`plan_v14/rows/*.json` 复制到 `runs/spike_steer/v14/`，`plan_v11/` 下的 CloudPC-2 文件（`snn/s1–3`、`cls_es/s1`、`test_opponent_cloudpc2.json` → `runs/spike_steer/v11/test_opponent.json`）复制到 `runs/spike_steer/v11/`。CSV 里已经包含同样的数据，必要时可直接用 CSV 改图。

## 5. 用户待办 / 可能的后续

- 补 Introduction、Related work、Conclusion；按目标会议模板排版（图目前在 PDF 末尾"Figures"一节）。
- 3.3 末段与 3.4 第一条的措辞（手工锁相律比学习层实现得更好）写得直白，可按作者口吻调整，但数据不要删。
- 若要继续实验：对"保留脉冲的经典律"做同预算 ES 重调（确定真正的经典上限）；追查持续湍流 ×1.5 下 SNN 占优的机理（唯一 SNN 赢过探针的场景）。
- 看板（claude.ai artifact "FlapSim Run Board"）由本机会话维护，关机后不再更新。
