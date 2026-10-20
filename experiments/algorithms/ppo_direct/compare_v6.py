"""plan v6 对比数据：把 PPO 参考臂（v6ppo）和 SNN+ES 各种子（descend8 的 evaluate_s<seed>.json）放到同一张表里。

两边的盲测结果文件结构相同（selected / report / rows），report 里是 v6 预注册规定的指标：
  manoeuvre_mean、nod/dash 比值（对手中位数 = 1.0）、按方向类（pitch/roll/diag）的比值与配对胜场、坠落数。
用法（仓库根目录）：
  python -m experiments.algorithms.ppo_direct.compare_v6 \
      --snn-dir artifacts/spike_steer/plan_v6 --ppo-dir artifacts/spike_steer/ppo_direct/v6ppo \
      --out artifacts/spike_steer/ppo_direct/v6ppo/comparison_v6
输出：<out>.md（表格）、<out>.csv、<out>.json、<out>.png（分种子柱状图）。SNN 文件缺失的种子留空，之后补齐再跑一次即可。
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

CLASSES = {"nod": ["pitch", "roll"], "dash": ["pitch", "roll", "diag"]}
M2_RATIO = 0.95   # 预注册 M2：每种子 manoeuvre mean <= 0.95 且 0 坠落


def row_from(method, seed, path):
    """从一个 evaluate json 提炼成一行扁平数据。"""
    d = json.loads(Path(path).read_text())
    rep, sel = d["report"], d.get("selected") or {}
    r = dict(method=method, seed=seed, file=str(path), selected_generation=sel.get("generation"),
             selection_objective=sel.get("objective"), manoeuvre_mean=rep["manoeuvre_mean"])
    for t in ("nod", "dash"):
        r[f"{t}_ratio"] = rep[t]["ratio"]
        r[f"{t}_wins"] = f'{rep[t]["wins"]}/{rep[t]["n"]}'
        r[f"{t}_crashes"] = rep[t]["crashes"]
        r[f"{t}_opp_crashes"] = rep[t].get("opponent_crashes")
        for c in CLASSES[t]:
            bc = rep[t]["by_class"].get(c)
            r[f"{t}_{c}"] = bc["ratio"] if bc else None
            r[f"{t}_{c}_wins"] = f'{bc["wins"]}/{bc["n"]}' if bc else None
    r["crashes"] = rep["nod"]["crashes"] + rep["dash"]["crashes"]
    r["M2_pass"] = bool(r["manoeuvre_mean"] <= M2_RATIO and r["crashes"] == 0)
    return r


def collect(snn_dir, ppo_dir, seeds):
    rows = []
    for s in seeds:
        p = Path(snn_dir) / f"evaluate_s{s}.json"
        if p.exists():
            rows.append(row_from("SNN+ES (v6)", s, p))
        p = Path(ppo_dir) / f"s{s}" / "evaluate.json"
        if p.exists():
            rows.append(row_from("PPO ref (v6ppo)", s, p))
    return rows


def fmt(x):
    if x is None: return ""
    if isinstance(x, float): return f"{x:.3f}"
    return str(x)


def write_md(rows, out):
    cols = ["method", "seed", "selected_generation", "manoeuvre_mean", "nod_ratio", "nod_pitch", "nod_roll", "nod_wins",
            "dash_ratio", "dash_pitch", "dash_roll", "dash_diag", "dash_wins", "crashes", "M2_pass"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines.append("| classical (opponent) | - | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | - | "
                 + fmt(rows[0]["nod_opp_crashes"] + rows[0]["dash_opp_crashes"] if rows and rows[0]["nod_opp_crashes"] is not None else "")
                 + " | - |")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    # 按方法汇总
    lines += ["", "| method | n seeds | manoeuvre mean (mean ± sd) | nod | dash | seeds passing M2 |", "|---|---|---|---|---|---|"]
    for m in dict.fromkeys(r["method"] for r in rows):
        g = [r for r in rows if r["method"] == m]
        mm = np.array([r["manoeuvre_mean"] for r in g])
        lines.append(f"| {m} | {len(g)} | {mm.mean():.3f} ± {mm.std(ddof=1) if len(g) > 1 else 0:.3f} | "
                     f"{np.mean([r['nod_ratio'] for r in g]):.3f} | {np.mean([r['dash_ratio'] for r in g]):.3f} | "
                     f"{sum(r['M2_pass'] for r in g)}/{len(g)} |")
    lines += ["", "比值 = 学到的控制器得分 / 该方向类 classical 对手中位数（越低越好，1.0 = 和 classical 一样）；"
              f"M2：manoeuvre mean <= {M2_RATIO} 且盲测 64 例零坠落。盲测集 TEST_BASE 2097100000，nod 32（4 向 × 8）+ dash 32（8 向 × 4）。"]
    Path(out + ".md").write_text("\n".join(lines) + "\n")


def plot(rows, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(r["method"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    metrics = [("manoeuvre_mean", "manoeuvre mean"), ("nod_ratio", "nod"), ("dash_ratio", "dash")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    w = 0.8 / max(1, len(methods))
    for ax, (key, title) in zip(axes, metrics):
        for i, m in enumerate(methods):
            vals = [next((r[key] for r in rows if r["method"] == m and r["seed"] == s), np.nan) for s in seeds]
            xs = np.arange(len(seeds)) + (i - (len(methods) - 1) / 2) * w
            bars = ax.bar(xs, vals, w, label=m)
            for x, v, s in zip(xs, vals, seeds):
                r = next((r for r in rows if r["method"] == m and r["seed"] == s), None)
                if r and r["crashes"]:
                    ax.text(x, v + 0.02, f'{r["crashes"]}✗', ha="center", fontsize=7, color="red")
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label="classical = 1.0" if key == "manoeuvre_mean" else None)
        if key == "manoeuvre_mean":
            ax.axhline(M2_RATIO, color="green", lw=0.8, ls=":", label=f"M2 bar {M2_RATIO}")
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds]); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score ratio vs classical (lower is better)")
    axes[0].legend(fontsize=8)
    fig.suptitle("plan v6 blind set (32 nod + 32 dash): SNN+ES vs PPO reference; red = crashes")
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snn-dir", default="artifacts/spike_steer/plan_v6")
    p.add_argument("--ppo-dir", default="artifacts/spike_steer/ppo_direct/v6ppo")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--out", default="artifacts/spike_steer/ppo_direct/v6ppo/comparison_v6")
    a = p.parse_args()
    rows = collect(a.snn_dir, a.ppo_dir, a.seeds)
    if not rows:
        raise SystemExit("no evaluate files found")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_md(rows, a.out)
    plot(rows, a.out)
    print(Path(a.out + ".md").read_text())


if __name__ == "__main__":
    main()
