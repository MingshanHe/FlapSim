"""把 PPO 直接控制策略放到第五轮的盲测集上，按 bench 的方式评分，和 classical 对手、SNN 各种子同场比较。

    python -m experiments.algorithms.ppo_direct.evaluate --model runs/ppo_direct/models/full.zip --tag ppo_full --workers 18

用例：hover 16、nod 16、dash 16（第五轮的新盲测集 2_096_600_000）、upset 48，与 plan_v4/r5/claude_ea/baseline.json 相同；
对手行、归一化、权重也从那个 baseline 取。输出 runs/ppo_direct/evaluate_<tag>.json（与 bench.evaluate_spec 同格式）
和一行 "ppo_<tag> composite ..."。--compare 再把第五轮 SNN 各种子的 evaluate_r5_s*.json 并排打印。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from ..spike_steer import bench as B
from .env import BASELINE_R5, CALIBRATE, DirectManeuverEnv, prepare_source

OUT = Path("runs/ppo_direct")
R5_DIR = Path("artifacts/spike_steer/plan_v4/r5/claude_ea")
_ENV = None
_POLICY = None


def _init(model_path: str, control: str):
    global _ENV, _POLICY
    from stable_baselines3 import PPO
    _ENV = DirectManeuverEnv(task="hover", control=control)
    _POLICY = PPO.load(model_path, device="cpu")


def _run(args):
    task, case, seed = args
    env = _ENV
    options = dict(task="hover" if task in ("hover", "upset") else task, case=case, forward=True)
    obs, _ = env.reset(seed=seed, options=options)
    done, info = False, {}
    while not done:
        action, _ = _POLICY.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    row = dict(info["metrics"], controller="ppo")
    row["ref_abs_mean"] = float(np.mean(np.abs(np.array(env.rows)[:, 9]))) if env.rows else 0.
    row["task"] = task
    return row


def evaluate(model: Path, tag: str, workers: int, n: int | None = None, control: str = "residual") -> dict:
    prepare_source()
    base = json.loads(BASELINE_R5.read_text())
    levels = json.loads(CALIBRATE.read_text())["levels"]
    tasks = list(base["opponent"])
    from ..spike_steer import descend_r5 as R5
    R5.install()
    jobs = []
    for t in tasks:
        for i, c in enumerate(B.cases(t, "test", levels, n=n)):
            jobs.append((t, c, int(c["seed"]) % (2 ** 31)))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(str(model), control)) as ex:
        rows = list(ex.map(_run, jobs, chunksize=1))
    scores, table, rows_all = {}, {}, {}
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        rows_all[t] = rs
        table[t] = B.summary(rs)
        opp = [r for r in base["test_rows"][t] if r["controller"] == base["opponent"][t]]
        table[t]["ratio"] = table[t]["score"] / base["normalisers"][t]
        table[t]["paired_vs_opponent"] = B.paired(rs, opp)
        scores[t] = table[t]["score"]
    comp = B.composite(scores, base["normalisers"], base["prereg"]["weights"])
    hybrid = {t: min(1., table[t]["ratio"]) for t in tasks}
    report = dict(tag=tag, variants=dict(intact=dict(table=table, composite=comp, ratios={t: table[t]["ratio"] for t in tasks},
                                                      hybrid_composite=B.composite(hybrid, {t: 1. for t in tasks}, base["prereg"]["weights"]), rows=rows_all)),
                  spec=str(model), baseline=str(BASELINE_R5))
    print(f"{tag} intact       composite {comp:.3f} " + " ".join(f"{t}={table[t]['ratio']:.3f}({table[t]['flown']}/{table[t]['episodes']})" for t in tasks), flush=True)
    for t in tasks:
        pv = table[t]["paired_vs_opponent"]
        extra = ""
        if t == "nod":
            extra = f" t30 hits {table[t]['t30_hits']}/16 t_recover mean {table[t]['t_recover']}"
        if t == "dash":
            extra = f" t_reach hits {table[t]['t_reach_hits']}/16 t_settle hits {table[t]['t_settle_hits']}/16"
        print(f"   {t:5s} score {table[t]['score']:.3f} (opponent {base['normalisers'][t]:.3f})  wins {pv['wins']} losses {pv['losses']} sign_p {pv['sign_p']:.3g}{extra}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"evaluate_{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print("wrote", OUT / f"evaluate_{tag}.json", flush=True)
    return report


def compare(report: dict) -> None:
    base = json.loads(BASELINE_R5.read_text())
    tasks = list(base["opponent"])
    print("\n=== same blind set, same scoring (lower is better; ratio = score / classical opponent) ===")
    print(f"{'controller':22s} composite  " + "  ".join(f"{t:>12s}" for t in tasks))
    for name in ("pid", "law_tuned", "schedule", "pulse_tuned"):
        vals = []
        for t in tasks:
            s = base["table"][t].get(name)
            vals.append(f"{s['score'] / base['normalisers'][t]:.3f}({s['flown']}/{s['episodes']})" if s else "      -     ")
        print(f"classical {name:12s} {'-':>9s}  " + "  ".join(f"{v:>12s}" for v in vals))
    for s in range(1, 6):
        f = R5_DIR / f"evaluate_r5_s{s}.json"
        if f.exists():
            v = json.loads(f.read_text())["variants"]["intact"]
            print(f"SNN round-5 seed {s:<5d} {v['composite']:9.3f}  " + "  ".join(f"{v['table'][t]['ratio']:.3f}({v['table'][t]['flown']}/{v['table'][t]['episodes']})".rjust(12) for t in tasks))
    v = report["variants"]["intact"]
    print(f"{'PPO direct (' + report['tag'] + ')':22s} {v['composite']:9.3f}  " + "  ".join(f"{v['table'][t]['ratio']:.3f}({v['table'][t]['flown']}/{v['table'][t]['episodes']})".rjust(12) for t in tasks))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tag", default="ppo")
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--n", type=int, default=None, help="每任务只跑前 n 例（冒烟）")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--control", choices=["direct", "residual", "gated"], default="residual")
    a = p.parse_args()
    report = evaluate(a.model, a.tag, a.workers, a.n, a.control)
    if a.compare:
        compare(report)


if __name__ == "__main__":
    main()
