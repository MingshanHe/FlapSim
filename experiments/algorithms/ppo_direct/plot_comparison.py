"""总比较图：学习曲线（等算力）+ 第五轮盲测集上 classical / SNN（ES 搜索）/ 三种 PPO 的同场评分。

    python -m experiments.algorithms.ppo_direct.plot_comparison --output runs/ppo_direct/media/comparison.png

左：三条 PPO 学习曲线（flat 端到端、分层 residual、门控 gated），同 18 进程、同步数、同网络同超参。
右：第五轮盲测集（hover 16、nod 16、dash 16、upset 48）上每个任务的比值（分数 / classical 对手，< 1 更好）和复合分；
    classical 四个臂、SNN 第五轮 5 个种子（均值 ± 范围）、三种 PPO；柱上标坠落数。
底部文字：每种方法消耗的仿真回合数（ES：60 代 × 33 候选 × 8 回合 + 验证；PPO：步数 × 5 拍 / 回合拍数）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

R5 = Path("artifacts/spike_steer/plan_v4/r5/claude_ea")
TASKS = ("hover", "nod", "dash", "upset")


def curve(run: Path, tag: str):
    ea = EventAccumulator(str(run))
    ea.Reload()
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def ratios_from_eval(path: Path):
    v = json.loads(path.read_text())["variants"]["intact"]
    return dict(composite=v["composite"], ratios={t: v["table"][t]["ratio"] for t in TASKS},
                crashes={t: v["table"][t]["episodes"] - v["table"][t]["flown"] for t in TASKS})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tb", type=Path, default=Path("runs/ppo_direct/tb"))
    p.add_argument("--flat-run", default="quick_1")
    p.add_argument("--hier-run", default="hier_400k_1")
    p.add_argument("--gated-run", default="gated_400k_1")
    p.add_argument("--flat-eval", type=Path, default=Path("runs/ppo_direct/evaluate_ppo_flat_400k.json"))
    p.add_argument("--hier-eval", type=Path, default=Path("runs/ppo_direct/evaluate_ppo_hier_400k.json"))
    p.add_argument("--gated-eval", type=Path, default=Path("runs/ppo_direct/evaluate_ppo_gated_400k.json"))
    p.add_argument("--output", type=Path, default=Path("runs/ppo_direct/media/comparison.png"))
    a = p.parse_args()

    base = json.loads((R5 / "baseline.json").read_text())
    norm = base["normalisers"]
    opp = base["opponent"]
    # classical arms (ratio vs the task opponent; the opponent itself is 1.0 by definition)
    classical = {}
    for name in ("pid", "law_tuned", "schedule", "pulse_tuned"):
        classical[name] = dict(ratios={t: (base["table"][t][name]["score"] / norm[t]) if name in base["table"][t] else None for t in TASKS},
                               crashes={t: (base["table"][t][name]["episodes"] - base["table"][t][name]["flown"]) if name in base["table"][t] else None for t in TASKS})
    snn = [ratios_from_eval(R5 / f"evaluate_r5_s{s}.json") for s in range(1, 6) if (R5 / f"evaluate_r5_s{s}.json").exists()]
    ppo = {}
    for label, path in (("flat PPO 400k", a.flat_eval), ("hierarchical PPO 400k", a.hier_eval), ("gated PPO 400k", a.gated_eval)):
        if path.exists():
            ppo[label] = ratios_from_eval(path)

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1., 1.25], hspace=.45, wspace=.3)
    # ---- row 1: learning curves
    runs = [("flat PPO (alpha, beta, omega end-to-end)", a.flat_run, "#C44E52"), ("hierarchical (PID + PPO set-points, always in charge)", a.hier_run, "#4C72B0"),
            ("gated (PID + pulse law; PPO in charge only on command)", a.gated_run, "#55A868")]
    for j, (tag, title) in enumerate((("rollout/ep_len_mean", "mean episode length (decisions of 10 ms)"), ("rollout/ep_rew_mean", "mean episode reward"),
                                     ("rollout/success_rate", "hover-at-end success rate"))):
        ax = fig.add_subplot(gs[0, j])
        for name, run, color in runs:
            d = a.tb / run
            if d.exists():
                x, y = curve(d, tag)
                ax.plot(x / 1e3, y, color=color, lw=2, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PPO steps (x1000), 18 envs, same hyper-parameters")
        ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
    # ---- row 2: blind-set bars
    ax = fig.add_subplot(gs[1, :])
    entries = []   # (label, ratios dict, crashes dict, color, err)
    entries.append(("classical PID only", classical["pid"]["ratios"], classical["pid"]["crashes"], "#999999", None))
    best_classical = dict(ratios={t: 1.0 for t in TASKS}, crashes={t: classical[opp[t]]["crashes"][t] for t in TASKS})
    entries.append(("classical opponents\n(law_tuned / schedule / pulse_tuned)", best_classical["ratios"], best_classical["crashes"], "#444444", None))
    if snn:
        mean = {t: float(np.mean([s["ratios"][t] for s in snn])) for t in TASKS}
        err = {t: (mean[t] - min(s["ratios"][t] for s in snn), max(s["ratios"][t] for s in snn) - mean[t]) for t in TASKS}
        crashes = {t: f"{min(s['crashes'][t] for s in snn)}-{max(s['crashes'][t] for s in snn)}" for t in TASKS}
        mean["composite"] = float(np.mean([s["composite"] for s in snn]))
        err["composite"] = (mean["composite"] - min(s["composite"] for s in snn), max(s["composite"] for s in snn) - mean["composite"])
        entries.append((f"SNN round 5, ES search\n({len(snn)} seeds, mean and range)", mean, crashes, "#8172B2", err))
    colors = {"flat PPO 400k": "#C44E52", "hierarchical PPO 400k": "#4C72B0", "gated PPO 400k": "#55A868"}
    for label, r in ppo.items():
        rr = dict(r["ratios"], composite=r["composite"])
        entries.append((label, rr, r["crashes"], colors[label], None))
    cats = list(TASKS) + ["composite"]
    width = .8 / len(entries)
    xs = np.arange(len(cats))
    for k, (label, rr, cr, color, err) in enumerate(entries):
        vals = [rr.get(c) for c in cats]
        pos = xs + (k - (len(entries) - 1) / 2) * width
        shown = [min(v, 6.) if v is not None else 0. for v in vals]
        yerr = None
        if err:
            yerr = np.array([[err[c][0] if c in err else 0. for c in cats], [err[c][1] if c in err else 0. for c in cats]])
        ax.bar(pos, shown, width, color=color, label=label, yerr=yerr, capsize=3, error_kw=dict(lw=1))
        for x, v, c in zip(pos, vals, cats):
            if v is None:
                continue
            txt = f"{v:.2f}" + ("" if c == "composite" else f"\n{cr.get(c)} crash" if cr.get(c) not in (None, 0, "0-0") else "")
            if v > 6.:      # 被截断的柱：标签写在柱内
                ax.text(x, 5.9, txt.replace("\n", "  "), ha="center", va="top", fontsize=7, rotation=90, color="white")
            else:
                ax.text(x, v + .05, txt, ha="center", va="bottom", fontsize=7, rotation=90 if v > 2 else 0)
    ax.axhline(1., color="k", lw=.8, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n({'16' if c in ('hover', 'nod', 'dash') else '48' if c == 'upset' else '25/25/25/25'} {'cases' if c != 'composite' else 'weights'})" for c in cats])
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("score / classical opponent  (lower is better; 1 = opponent)")
    ax.set_title("Round-5 blind set of plan v4, identical opponents, normalisers and scoring for every arm (bars clipped at 6; crashes annotated)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=.3)
    # ---- budget note
    ppo_eps = 400_000 * 5 / 2875    # 400k decisions x 5 ticks / mean(nod 2500, dash 3250) ticks per episode
    es_eps = 60 * 33 * 8 + 13 * 16
    fig.text(.5, .005, f"Simulation budget per arm: ES-searched SNN ~{es_eps:,} episodes per seed (60 generations x 33 candidates x 8 episodes + validation) on 32 cores; "
                       f"each PPO arm ~{ppo_eps:,.0f} episodes (400k decisions, 18 processes, ~25 min on 20 cores) -- the SNN used ~{es_eps / ppo_eps:.0f}x more simulation.",
             ha="center", fontsize=9, color="#333333")
    fig.suptitle("Full flapping-wing model, original interface (IMU + height in; gimbal alpha/beta +-0.3 rad, motor 3115-3807 rad/s out)", fontsize=12)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150, bbox_inches="tight")
    print("wrote", a.output)
    for label, rr, cr, _, _ in entries:
        print(f"{label.replace(chr(10), ' '):55s} " + "  ".join(f"{c} {rr.get(c, float('nan')) if rr.get(c) is None else rr[c]:.3f}" if isinstance(rr.get(c), float) else f"{c} -" for c in cats))


if __name__ == "__main__":
    main()
