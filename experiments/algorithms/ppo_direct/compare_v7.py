"""plan v7 对比表：把所有臂（snn_es / snn_bc 来自 descend9，mlp_ppo / mlp_bc 来自 v7ppo）的 evaluate_<arm>_s<seed>.json 放到一张表里。

文件结构两边相同：{arm, seed, machine, note, report, rows}，report = descend9.report（evade / redirect / dash 的比值、按方向类的比值与
配对胜场、坠落；manoeuvre_mean = 三任务比值平均；classical 对手 = 1.0）。
用法（仓库根目录）：
  python -m experiments.algorithms.ppo_direct.compare_v7 --dirs artifacts/spike_steer/plan_v7 artifacts/spike_steer/ppo_direct/v7ppo \
      --out artifacts/spike_steer/ppo_direct/v7ppo/comparison_v7
输出：<out>.md、<out>.csv、<out>.json、<out>.png。缺的臂 / 种子留空，补齐后再跑一次即可。
"""
import argparse, csv, json, re
from pathlib import Path

import numpy as np

TASKS = ("evade", "redirect", "dash")
CLASSES = {"evade": ["pitch", "roll"], "redirect": ["pitch", "roll"], "dash": ["pitch", "roll", "diag"]}
ARM_ORDER = ["snn_es", "snn_bc", "mlp_ppo", "mlp_bc"]
M_RATIO = 0.95


def row_from(path):
    d = json.loads(Path(path).read_text())
    rep = d["report"]
    r = dict(arm=d["arm"], seed=d["seed"], machine=d.get("machine"), note=d.get("note"), file=str(path), manoeuvre_mean=rep["manoeuvre_mean"])
    for t in TASKS:
        n = rep[t].get("n") or sum(b["n"] for b in rep[t]["by_class"].values())
        r[f"{t}_ratio"] = rep[t]["ratio"]
        r[f"{t}_wins"] = f'{rep[t]["wins"]}/{n}'
        r[f"{t}_crashes"] = rep[t]["crashes"]
        for c in CLASSES[t]:
            bc = rep[t]["by_class"].get(c)
            r[f"{t}_{c}"] = bc["ratio"] if bc else None
    r["crashes"] = sum(rep[t]["crashes"] for t in TASKS)
    r["opp_crashes"] = sum(rep[t].get("opponent_crashes", 0) for t in TASKS)
    r["pass"] = bool(r["manoeuvre_mean"] <= M_RATIO and r["crashes"] == 0)
    return r


def collect(dirs):
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("evaluate_*_s*.json")):
            if re.match(r"evaluate_[a-z_]+_s\d+\.json", p.name):
                rows.append(row_from(p))
    rows.sort(key=lambda r: (ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99, r["seed"]))
    return rows


def fmt(x):
    if x is None: return ""
    if isinstance(x, float): return f"{x:.3f}"
    return str(x)


def write_md(rows, out, note):
    cols = ["arm", "seed", "machine", "manoeuvre_mean", "evade_ratio", "evade_pitch", "evade_roll", "evade_wins", "redirect_ratio", "redirect_pitch",
            "redirect_roll", "redirect_wins", "dash_ratio", "dash_pitch", "dash_roll", "dash_diag", "dash_wins", "crashes", "pass"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines.append("| classical | - | - | 1.000 | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 1.000 | 1.000 | - | "
                 + fmt(rows[0]["opp_crashes"] if rows else "") + " | - |")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["", "| arm | n seeds | manoeuvre mean (mean ± sd) | evade | redirect | dash | seeds passing |", "|---|---|---|---|---|---|---|"]
    for m in dict.fromkeys(r["arm"] for r in rows):
        g = [r for r in rows if r["arm"] == m]
        mm = np.array([r["manoeuvre_mean"] for r in g])
        lines.append(f"| {m} | {len(g)} | {mm.mean():.3f} ± {mm.std(ddof=1) if len(g) > 1 else 0:.3f} | "
                     + " | ".join(f"{np.mean([r[f'{t}_ratio'] for r in g]):.3f}" for t in TASKS) + f" | {sum(r['pass'] for r in g)}/{len(g)} |")
    lines += ["", "比值 = 该臂得分 / 该方向类 classical 对手中位数（越低越好，1.0 = 与 classical 相同）；manoeuvre mean = 三任务比值平均；"
              f"pass：manoeuvre mean <= {M_RATIO} 且盲测零坠落（{note}）。snn_* 由 descend9 产出，mlp_* 由 v7ppo 产出。"]
    Path(out + ".md").write_text("\n".join(lines) + "\n")


def plot(rows, out, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    seeds = sorted({r["seed"] for r in rows})
    metrics = [("manoeuvre_mean", "manoeuvre mean")] + [(f"{t}_ratio", t) for t in TASKS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8), sharey=True)
    w = 0.8 / max(1, len(arms))
    for ax, (key, title) in zip(axes, metrics):
        for i, m in enumerate(arms):
            vals = [next((r[key] for r in rows if r["arm"] == m and r["seed"] == s), np.nan) for s in seeds]
            xs = np.arange(len(seeds)) + (i - (len(arms) - 1) / 2) * w
            ax.bar(xs, vals, w, label=m)
            for x, v, s in zip(xs, vals, seeds):
                r = next((r for r in rows if r["arm"] == m and r["seed"] == s), None)
                if r and r["crashes"] and not np.isnan(v):
                    ax.text(x, v + 0.02, f'{r["crashes"]}✗', ha="center", fontsize=7, color="red")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        if key == "manoeuvre_mean":
            ax.axhline(M_RATIO, color="green", lw=0.8, ls=":")
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"seed {s}" for s in seeds]); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score ratio vs classical (lower is better)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"plan v7 blind set, {title}: dashed = classical 1.0, dotted = 0.95; red = crashes")
    fig.tight_layout(); fig.savefig(out + ".png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=["artifacts/spike_steer/plan_v7", "artifacts/spike_steer/ppo_direct/v7ppo"])
    p.add_argument("--out", default="artifacts/spike_steer/ppo_direct/v7ppo/comparison_v7")
    p.add_argument("--note", default="v7 快速验证，未预注册；盲测集 bench9 BASES['test'] = 2_098_100_000，每任务 16 例")
    p.add_argument("--title", default="quick validation (16 per task)")
    a = p.parse_args()
    rows = collect(a.dirs)
    if not rows:
        raise SystemExit("no evaluate_<arm>_s<seed>.json found in " + ", ".join(a.dirs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(rows, indent=1))
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_md(rows, a.out, a.note)
    plot(rows, a.out, a.title)
    print(Path(a.out + ".md").read_text())


if __name__ == "__main__":
    main()
