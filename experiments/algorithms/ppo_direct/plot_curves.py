"""等算力对比图：端到端 (flat) PPO vs 分层 (reflex PID + PPO 上层)，同一完整扑翼模型、同网络、同超参、同进程数、同步数。

    python -m experiments.algorithms.ppo_direct.plot_curves --flat runs/ppo_direct/tb/quick_1 --hier runs/ppo_direct/tb/hier_400k_1 --output runs/ppo_direct/media/flat_vs_hier.png

三个面板：平均回合长度（决策数，1 决策 = 10 ms）、平均回合奖励、悬停成功率，横轴训练步数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = [("rollout/ep_len_mean", "mean episode length (decisions, 10 ms each)"),
        ("rollout/ep_rew_mean", "mean episode reward"),
        ("rollout/success_rate", "hover-at-end success rate")]


def load(run: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(run))
    ea.Reload()
    out = {}
    for tag, _ in TAGS:
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flat", type=Path, required=True, help="端到端 PPO 的 TensorBoard 目录")
    p.add_argument("--hier", type=Path, required=True, help="PID + PPO 上层的 TensorBoard 目录")
    p.add_argument("--output", type=Path, default=Path("runs/ppo_direct/media/flat_vs_hier.png"))
    p.add_argument("--max-steps", type=int, default=None, help="只画到这个步数（等算力截断）")
    a = p.parse_args()
    runs = {"flat PPO: alpha, beta, omega end-to-end": (load(a.flat), "#C44E52"),
            "hierarchical: reflex PID + PPO set-points / motor": (load(a.hier), "#4C72B0")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (tag, label) in zip(axes, TAGS):
        for name, (data, color) in runs.items():
            x, y = data[tag]
            if a.max_steps:
                keep = x <= a.max_steps
                x, y = x[keep], y[keep]
            ax.plot(x / 1e3, y, color=color, lw=2, label=name)
        ax.set_xlabel("training steps (x1000)")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=.3)
    axes[0].axhline(500, color="#888888", ls="--", lw=1)
    axes[0].text(axes[0].get_xlim()[1], 505, "nod episode = 500 decisions", ha="right", va="bottom", fontsize=8, color="#666666")
    axes[2].set_ylim(-.02, 1.02)
    axes[0].legend(loc="lower right", fontsize=9)
    steps = max(int(d[0][TAGS[0][0]][0][-1]) for d, _ in runs.values())
    fig.suptitle(f"Full flapping-wing model, same budget for both: 18 parallel envs, {steps / 1e3:.0f}k PPO steps, MLP 64x64, identical hyper-parameters, "
                 "IMU + height inputs, nod / dash / hover tasks", fontsize=10)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150)
    print("wrote", a.output)
    for name, (data, _) in runs.items():
        x, y = data["rollout/ep_rew_mean"]
        l = data["rollout/ep_len_mean"][1]
        s = data["rollout/success_rate"][1]
        print(f"{name}: steps {int(x[-1])}  reward {y[0]:.0f} -> {y[-1]:.0f}  ep_len {l[0]:.0f} -> {l[-1]:.0f}  success {s[0]:.2f} -> {s[-1]:.2f}")


if __name__ == "__main__":
    main()
