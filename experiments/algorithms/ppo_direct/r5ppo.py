"""第五轮流程，把"ES 搜索 SNN 权重"换成"PPO 训练同接口的 MLP"，其余环节照搬，便于和 SNN / ES 直接比。

第五轮（descend_r5）的环节 -> 这里的对应：
  架构   descend_r4.RefractoryLearned：门控学习层（指令期 + 1 s 尾巴在岗），输入 20 维特征 (bench.features：16 个整流
         IMU 特征 + 常数 1、0 + 指令 2)，输出电机脉冲（3 拍 3807 rad/s，37 ms 不应期）和俯仰参考 (±0.7 rad)；
         底层 = 反射 PID + law_tuned 脉冲律，AHRS 倾角 > 25° 切 pulse_tuned。
         -> PpoLayer：同一门控、同一底层、同一输出语义；学习层换成 MLP 64x64，每 10 ms 决策一次（SNN 每 2 ms），
            输入 = 当前、10 ms 前、20 ms 前三帧的 18 维特征 + 指令 2 = 56 维（SNN 有 0–20 ms 延迟组，给 MLP 同等时间窗）。
  起点   S2 模仿：descend.py train 在 collect_train.npz（classical 对手的逐拍脉冲 / 俯仰参考）上做 BCE + MSE。
         -> bc 阶段：同一份数据，按 10 ms 窗口聚合（窗内有脉冲起始 -> +1，否则 -1；参考取窗均值），MSE 回归到 MLP，
            再把权重装进 PPO 策略（= spec_val_g-1）。
  搜索   OpenAI-ES 60 代 × 33 候选 × (8 nod + 4 dash) 回合，目标 = nod:dash 2:1 的分数比值均值 + 倾角惩罚 (>60° × 0.3)，
         用例 offset 100_000 + gen*100；每 5 代验证一次（4 nod + 4 dash，offset 60_000）并存中心 spec_val_g*。
         -> finetune 阶段：PPO，回合奖励只在回合末给 -(比值 + 倾角惩罚) × 任务权重（nod 2 / dash 1），没有别的塑形；
            用例从同一池子 (offset 100_000–106_000) 抽；预算按回合数给（--budget-episodes，默认 = ES 的 23 760），
            每 1/12 预算存一个检查点 ckpt_g0..g11（+ bc = g-1，共 13 个中心，与 ES 的 13 个一致）。
  选中心 reselect_r3 规则：64 nod + 64 dash（offset 60_000），零坠落且真实倾角峰 < 70°，取含惩罚复合分最低者；
         -> reselect 阶段：同一批用例、同一规则、同一 JSON 字段。
  盲测   第五轮盲测集 + 消融 clamp_fast / nodelay_all / noref / nopulse；
         -> evaluate 阶段：同一盲测集、同一基准；消融 clamp_fast（陀螺/加速度特征置零）、noref、nopulse（nodelay 不适用）。
  种子   1–5。

    python -m experiments.algorithms.ppo_direct.r5ppo selftest
    python -m experiments.algorithms.ppo_direct.r5ppo bc       --seed 1
    python -m experiments.algorithms.ppo_direct.r5ppo finetune --seed 1 --budget-episodes 23760 --n-envs 18
    python -m experiments.algorithms.ppo_direct.r5ppo reselect --seed 1 --workers 18
    python -m experiments.algorithms.ppo_direct.r5ppo evaluate --seed 1 --workers 18
输出都在 runs/ppo_direct/r5ppo/s<seed>/。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from gymnasium import spaces

from ..neuro_gates import gates
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import CENTER, HALF, HIGH, LOW
from ..spike_steer import bench as B
from ..spike_steer import descend as Dn
from ..spike_steer import descend_r3 as R3
from ..spike_steer import descend_r4 as R4
from ..spike_steer import descend_r5 as R5
from ..spike_steer import pilot_v6 as Y
from .env import BASELINE_R5, CALIBRATE, DirectManeuverEnv, TICKS_PER_DECISION, prepare_source

OUT = Path("runs/ppo_direct/r5ppo")
COLLECT = Path(os.environ.get("COLLECT_NPZ", "artifacts/spike_steer/ppo_direct/collect_train.npz"))   # 第一轮 collect 的 classical 轨迹（HomePC 2026-10-14）
TRAIN_BASELINE = Path("artifacts/spike_steer/plan_v4/r5/train/baseline.json")
N_FEAT = 18                                   # bench.features 的前 18 维（不含指令）
FRAMES = (0, 5, 10)                            # 当前、10 ms 前、20 ms 前（拍数）
N_OBS = N_FEAT * len(FRAMES) + 2
TASK_W = dict(nod=2., dash=1.)                 # 第五轮 nod:dash = 2:1
ES_EPISODES = 60 * 33 * 12                     # ES 每种子的搜索回合数（不含验证）
MEAN_DECISIONS = (2500 + 3250) / 2 / TICKS_PER_DECISION
N_CENTRES = 12                                 # 检查点数（+ bc 起点 = 13 个中心）
FAST = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]   # clamp_fast：陀螺 / 加速度的整流特征


# ----------------------------------------------------------------------------------------- 环境
class PpoLayerEnv(DirectManeuverEnv):
    """RefractoryLearned 的门控学习层换成 PPO；只跑 nod / dash；回合奖励 = 第五轮 ES 目标。"""

    def __init__(self, *, variant: str = "intact", budget_offset_span: int = 6000, **kwargs) -> None:
        super().__init__(task="mixed", task_weights=(2., 1., 0.), control="gated", **kwargs)
        self.variant = variant
        self.budget_offset_span = int(budget_offset_span)
        self.train_base = json.loads(TRAIN_BASELINE.read_text())
        self.norm = self.train_base["normalisers"]
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.refr = R4.REFRACTORY_S
        self.soft, self.pen = R4.SOFT_TILT_DEG, R4.PENALTY_PER_DEG
        self.feat_hist = collections.deque([np.zeros(N_FEAT, np.float32)], maxlen=max(FRAMES) + 1)
        self.cmd_on, self.cmd, self.ref, self.window_tick = False, np.zeros(3), 0., 0

    def _draw_case(self, task: str) -> dict:
        # 与 ES 同一池子：offset 100_000 + gen*100 (+ i)，gen < 60
        i = 100_000 + int(self.np_random.integers(0, self.budget_offset_span))
        from ..topology_hover.environment import case as make_case
        return make_case(B.BASES["man_train"] + i, B.SECONDS[task], "stress", i % 8)

    def _features(self) -> np.ndarray:
        x = B.features(self.env.last_reading, self.cmd[:2] if self.cmd_on else np.zeros(2))
        return np.asarray(x[:N_FEAT], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        hist = self.feat_hist
        frames = [hist[-1 - k] if len(hist) > k else hist[0] for k in FRAMES]
        feat = np.concatenate(frames)
        if self.variant == "clamp_fast":
            mask = np.ones(N_FEAT)
            mask[FAST] = 0.
            feat = np.concatenate([f * mask for f in frames])
        cmd = self.cmd[:2] if self.cmd_on else np.zeros(2)
        return np.concatenate((feat, cmd)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.feat_hist = collections.deque([np.zeros(N_FEAT, np.float32)], maxlen=max(FRAMES) + 1)
        obs, info = super().reset(seed=seed, options=options)
        self.feat_hist = collections.deque([self._features()], maxlen=max(FRAMES) + 1)
        self.twitch = 0
        self.last_onset = None
        self.onsets = self.cancelled = self.pulses = 0
        self.ref = 0.
        return self._observation(), info

    def _tick(self, action):
        env = self.env
        t = env.air.steps * DT
        now = float(env.sim.data.time)
        law_pulse, _ = self.steer.step(env, now, t)
        up_pulse, _ = self.steer_up.step(env, now, t)
        on = bool(self.cmd_on)
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        self.in_charge = on or (self.off_t is not None and t - self.off_t < self.tail_s)
        tilt_ahrs = math.hypot(float(env.last_reading[0]), float(env.last_reading[1]))
        if not self.upset and not self.in_charge and tilt_ahrs > self.upset_tilt:
            self.upset = True
        gate_pulse = bool(up_pulse if self.upset else law_pulse)
        fire = False
        if self.in_charge and t >= gates.HANDOVER_T and action is not None:
            # 学习层：窗口第一拍决定是否起始一个 3 拍脉冲（受 37 ms 不应期约束），俯仰参考保持一窗
            if self.window_tick == 0:
                self.ref = 0. if self.variant == "noref" else B.REF_MAX * float(np.clip(action[1], -1., 1.))
                want = float(action[0]) > 0. and self.variant != "nopulse"
                if want:
                    if self.last_onset is not None and t - self.last_onset < self.refr:
                        self.cancelled += 1
                    else:
                        self.last_onset = t
                        self.onsets += 1
                        self.twitch = B.TWITCH
            env.teacher.rpy_reference = np.array([0., self.ref])
            command = env.teacher_command()
            if self.twitch > 0:
                self.twitch -= 1
                command[2] = HIGH[2]
                fire = True
        else:
            self.twitch = 0
            env.teacher.rpy_reference = np.zeros(2)
            command = env.teacher_command()
            if gate_pulse and t >= gates.HANDOVER_T:
                command[2] = HIGH[2]
                fire = True
        if fire:
            self.pulses += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH) - CENTER) / HALF)
        self.oracle.update()
        pos, vel, r = env.air.env._state()
        tnow = env.air.steps * DT
        self.rows.append([tnow, float(self.oracle.queue[-1][1]), *pos, *vel, math.acos(min(1., max(-1., float(r[2, 2])))),
                          self.ref if self.in_charge else 0., float(fire), *self.cmd[:2]])
        self.feat_hist.append(self._features())
        self.window_tick += 1
        return term, trunc, out

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
        env = self.env
        truth = self._truth()
        t = env.air.steps * DT
        if self.pos0 is None and t - self.t0 >= -1e-9:
            self.pos0, self.z0 = truth["position"].copy(), float(truth["position"][2])
            yaw = self.rotation_yaw(truth["rotation"])
            if self.dash_angle is None:
                self.dash_angle = float(yaw)
            self.heading = np.array([math.cos(self.dash_angle), math.sin(self.dash_angle)])
            self.cmd_on = True
        self.cmd = self._command_vector(truth)
        self.window_tick = 0
        terminated = truncated = False
        out = {}
        for _ in range(TICKS_PER_DECISION):
            terminated, truncated, out = self._tick(action)
            if terminated or truncated:
                break
        truth = self._truth()
        t = env.air.steps * DT
        if self.pos0 is not None and self.cmd_on:
            # 目标检测与 bench 相同（nod: |pitch| >= 30° 持续 0.1 s；dash: 前进 DASH_M）
            pitch_deg = math.degrees(truth["pitch"])
            if self.task == "nod":
                self.hold = self.hold + 1 if abs(pitch_deg) >= 30. else 0
                if self.hold * TICKS_PER_DECISION * DT >= B.HOLD_S - 1e-9:
                    self.cmd_on, self.goal_done, self.goal_t = False, True, t - self.t0
            else:
                d = truth["position"][:2] - self.pos0[:2]
                if float(d @ self.heading) >= self.dash_m:
                    self.cmd_on, self.goal_done, self.goal_t = False, True, t - self.t0
        self.cmd = self._command_vector(truth)
        reward = 0.
        info = dict(task=self.task, t=t, cmd_on=self.cmd_on, goal_done=self.goal_done, pitch_deg=math.degrees(truth["pitch"]))
        if terminated or truncated:
            self.ep = out["episode"]
            m = self.metrics()
            m.update(pulses=self.pulses, pulse_duty=self.pulses / max(1, env.air.steps), onsets=self.onsets, cancelled=self.cancelled,
                     ref_abs_mean=float(np.mean(np.abs(np.array(self.rows)[:, 9]))) if self.rows else 0.)
            pen = max(0., (m.get("tilt_peak_deg") or 0.) - self.soft) * self.pen
            objective = m["score"] / self.norm[self.task] + pen           # 第五轮 ES 的单回合目标
            reward = -TASK_W[self.task] * objective
            m["objective"] = objective
            info["metrics"] = m
            info["is_success"] = bool(m["flown"] and self.goal_done)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def rotation_yaw(r):
        return math.atan2(r[1, 0], r[0, 0])


# ----------------------------------------------------------------------------------------- 工具
def seed_dir(seed: int) -> Path:
    d = OUT / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_policy_model(env, seed: int):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
               ent_coef=0.0, policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.5}, verbose=1, seed=seed, device="cpu",
               tensorboard_log=str(OUT / "tb"))


# ----------------------------------------------------------------------------------------- bc：模仿热启动
def bc_dataset():
    d = np.load(COLLECT)
    X, L, T = d["X"], d["lengths"], d["tasks"]
    obs, tgt = [], []
    for e in range(X.shape[0]):
        if T[e] not in ("nod", "dash"):
            continue
        x = X[e, :L[e]]
        feats, pulse, ref = x[:, :N_FEAT], x[:, B.N_IN], x[:, B.N_IN + 2]
        cmd = x[:, 18:20]
        n = len(x)
        for s in range(max(FRAMES), n - TICKS_PER_DECISION, TICKS_PER_DECISION):
            frames = [feats[s - k] for k in FRAMES]
            o = np.concatenate(frames + [cmd[s]])
            win = slice(s, s + TICKS_PER_DECISION)
            onset = bool(pulse[win].max() > 0 and (s == 0 or pulse[s - 1] == 0 or pulse[win].sum() < TICKS_PER_DECISION))
            tgt.append([1. if onset else -1., float(ref[win].mean()) / B.REF_MAX])
            obs.append(o)
    return np.asarray(obs, np.float32), np.asarray(tgt, np.float32)


def bc(a):
    import torch
    torch.set_num_threads(2)
    obs, tgt = bc_dataset()
    print(f"bc data: {len(obs)} windows from nod/dash episodes, pulse-window rate {np.mean(tgt[:, 0] > 0):.3f}, |ref| mean {np.abs(tgt[:, 1]).mean():.3f}", flush=True)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(obs))
    n_val = len(obs) // 10
    va, tr = idx[:n_val], idx[n_val:]
    env = PpoLayerEnv()
    model = make_policy_model(env, a.seed)
    pol = model.policy
    net = torch.nn.Sequential(pol.mlp_extractor.policy_net, pol.action_net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.tensor(obs), torch.tensor(tgt)
    w = torch.tensor([a.pulse_weight, 1.])
    best, best_state = None, None
    for epoch in range(a.epochs):
        rng.shuffle(tr)
        losses = []
        for b in range(0, len(tr), 256):
            i = torch.tensor(tr[b:b + 256])
            pred = net(Xt[i])
            loss = (w * (pred - Yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            pv = net(Xt[torch.tensor(va)])
            vloss = float((w * (pv - Yt[torch.tensor(va)]) ** 2).mean())
            acc = float(((pv[:, 0] > 0) == (Yt[torch.tensor(va), 0] > 0)).float().mean())
            ref_rmse = float(torch.sqrt(((pv[:, 1] - Yt[torch.tensor(va), 1]) ** 2).mean()))
        print(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, pulse_acc=acc, ref_rmse=ref_rmse)), flush=True)
        if best is None or vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    d = seed_dir(a.seed)
    model.save(d / "ckpt_g-1")
    print(f"wrote {d / 'ckpt_g-1.zip'} (best val loss {best:.4f})", flush=True)


# ----------------------------------------------------------------------------------------- finetune：PPO
def finetune(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    d = seed_dir(a.seed)
    prepare_source()
    steps = int(a.budget_episodes * MEAN_DECISIONS)
    every = steps // N_CENTRES

    def make(rank):
        def _m():
            e = PpoLayerEnv(seed_offset=rank)
            e.reset(seed=a.seed * 1000 + rank)
            return Monitor(e, info_keywords=("is_success",))
        return _m
    env = SubprocVecEnv([make(i) for i in range(a.n_envs)], start_method="fork")
    init = a.init_model or (d / "ckpt_g-1.zip")
    model = PPO.load(init, env=env, device="cpu", tensorboard_log=str(OUT / "tb"), seed=a.seed)

    class Archive(BaseCallback):
        def __init__(self):
            super().__init__()
            self.k = 0
            self.history = []

        def _on_step(self):
            if self.num_timesteps >= (self.k + 1) * every and self.k < N_CENTRES:
                self.model.save(d / f"ckpt_g{self.k}")
                row = dict(centre=self.k, timesteps=int(self.num_timesteps), wall_s=round(time.monotonic() - t0))
                self.history.append(row)
                (d / "history.jsonl").open("a").write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                self.k += 1
            return True
    t0 = time.monotonic()
    print(json.dumps(dict(seed=a.seed, budget_episodes=a.budget_episodes, timesteps=steps, checkpoint_every=every, n_envs=a.n_envs, init=str(init))), flush=True)
    model.learn(total_timesteps=steps, callback=Archive(), tb_log_name=f"r5ppo_s{a.seed}", reset_num_timesteps=True)
    model.save(d / f"ckpt_g{N_CENTRES - 1}")       # 最后一个中心 = 终点（如步数没整除）
    env.close()
    print(f"finetune done: {steps} steps, wall {round(time.monotonic() - t0)} s", flush=True)


# ----------------------------------------------------------------------------------------- 并行回放（选中心 / 盲测）
_ENV = None
_POL = {}


def _init(variant: str):
    global _ENV
    _ENV = PpoLayerEnv(variant=variant)


def _load(path: str):
    from stable_baselines3 import PPO
    if path not in _POL:
        _POL[path] = PPO.load(path, device="cpu")
    return _POL[path]


def _run(args):
    model_path, task, case, seed = args
    pol = _load(model_path)
    env = _ENV
    obs, _ = env.reset(seed=seed, options=dict(task=task, case=case, forward=True))
    done, info = False, {}
    while not done:
        act, _ = pol.predict(obs, deterministic=True)
        obs, _, te, tr, info = env.step(act)
        done = te or tr
    return dict(info["metrics"], controller=Path(model_path).stem, task=task)


def reselect(a):
    d = seed_dir(a.seed)
    R5.install()
    R3.SOFT["value"] = R4.SOFT_TILT_DEG
    tasks = ["nod", "dash"]
    n_val = a.n_val or R5.N_VAL_SELECT
    val = {t: B.cases(t, "train", None, n=n_val, offset=Dn.COLLECT_OFFSET["val"] + 10_000) for t in tasks}
    norm = json.loads(TRAIN_BASELINE.read_text())["normalisers"]
    centres = [-1] + list(range(N_CENTRES))
    paths = {g: d / f"ckpt_g{g}.zip" for g in centres if (d / f"ckpt_g{g}.zip").exists()}
    jobs = [(str(p), t, c, int(c["seed"]) % (2 ** 31)) for g, p in paths.items() for t in tasks for c in val[t]]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=("intact",)) as ex:
        rows = list(ex.map(_run, jobs, chunksize=1))
    out_rows = []
    for g, p in paths.items():
        rs = [r for r in rows if r["controller"] == p.stem]
        comp, _ = R3.composite_r3(rs, tasks, norm, R5.WEIGHTS)
        comp0 = B.composite({t: float(np.mean([r["score"] for r in rs if r["task"] == t])) for t in tasks}, norm, R5.WEIGHTS)
        entry = dict(generation=g, penalised16=comp, composite16=comp0, crashes=sum(not r["flown"] for r in rs),
                     over_hard=sum((r.get("tilt_peak_deg") or 0.) >= R4.HARD_TILT_DEG for r in rs),
                     tilt_peak_max=float(max(r.get("tilt_peak_deg") or 0. for r in rs)), spec=str(p),
                     ratios={t: float(np.mean([r["score"] for r in rs if r["task"] == t])) / norm[t] for t in tasks})
        out_rows.append(entry)
        print(json.dumps(entry), flush=True)
    clean = [e for e in out_rows if e["crashes"] == 0 and e["over_hard"] == 0]
    nocrash = [e for e in out_rows if e["crashes"] == 0]
    if clean:
        pick, fallback = min(clean, key=lambda e: e["penalised16"]), None
    elif nocrash:
        pick, fallback = min(nocrash, key=lambda e: e["penalised16"]), "no centre below the hard ceiling: zero-crash centres only"
    else:
        pick, fallback = min(out_rows, key=lambda e: e["penalised16"]), "every centre crashed at least once: all centres"
    report = dict(seed=a.seed, n_val=n_val, rule=f"lowest penalised composite (soft {R4.SOFT_TILT_DEG} deg x{R4.PENALTY_PER_DEG}, nod:dash 2:1) on 128 cases among "
                  f"centres with zero crashes and tilt peak < {R4.HARD_TILT_DEG} deg", selected=pick, fallback=fallback, centres=out_rows)
    (d / "reselect.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(selected=pick["generation"], penalised=pick["penalised16"], fallback=fallback)), flush=True)


def evaluate(a):
    d = seed_dir(a.seed)
    R5.install()
    base = json.loads(BASELINE_R5.read_text())
    levels = json.loads(CALIBRATE.read_text())["levels"]
    tasks = list(base["opponent"])
    sel = json.loads((d / "reselect.json").read_text())["selected"]["spec"] if a.spec is None else a.spec
    report = dict(tag=f"r5ppo_s{a.seed}", variants={}, spec=str(sel), baseline=str(BASELINE_R5))
    for variant in a.variants.split(","):
        jobs = [(str(sel), t, c, int(c["seed"]) % (2 ** 31)) for t in tasks for c in B.cases(t, "test", levels, n=a.n)]
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=(variant,)) as ex:
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
        report["variants"][variant] = dict(table=table, composite=comp, ratios={t: table[t]["ratio"] for t in tasks},
                                           hybrid_composite=B.composite(hybrid, {t: 1. for t in tasks}, base["prereg"]["weights"]), rows=rows_all)
        print(f"r5ppo_s{a.seed} {variant:12s} composite {comp:.3f} " + " ".join(f"{t}={table[t]['ratio']:.3f}({table[t]['flown']}/{table[t]['episodes']})" for t in tasks), flush=True)
    (d / "evaluate.json").write_text(json.dumps(report, indent=1, default=float))
    print("wrote", d / "evaluate.json", flush=True)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    env = PpoLayerEnv()
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    obs, info = env.reset(seed=3, options=dict(task="nod"))
    assert obs.shape == (N_OBS,)
    # 零动作（不脉冲、参考 0）= 纯 classical 底层：nod 应飞完、不做到 30°
    done = False
    while not done:
        obs, r, te, tr, info = env.step(np.array([-1., 0.], np.float32))
        done = te or tr
    m = info["metrics"]
    assert m["flown"] and m["t30"] is None, m
    assert r < 0 and abs(info["metrics"]["objective"] * 2 + r) < 1e-9
    # 固定 +0.86 参考（0.6 rad）应做到 nod
    obs, info = env.reset(seed=3, options=dict(task="nod"))
    done = False
    while not done:
        obs, r, te, tr, info = env.step(np.array([-1., 1.], np.float32))
        done = te or tr
    m2 = info["metrics"]
    assert (m2["peak_pitch_deg"] or 0.) > 25., m2          # 满参考 0.7 rad 应把俯仰甩到 25° 以上（是否满足 30°/0.1 s 因用例而异）
    o, t = bc_dataset()
    assert o.shape[1] == N_OBS and t.shape[1] == 2 and 0 < np.mean(t[:, 0] > 0) < .5
    print(f"selftest ok (gated PPO layer: zero action nod score {m['score']:.3f}, +0.7 rad reference peak pitch {m2['peak_pitch_deg']:.1f} deg t30 {m2['t30']} score {m2['score']:.3f}; "
          f"bc windows {len(o)}, pulse-window rate {np.mean(t[:, 0] > 0):.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["selftest", "bc", "finetune", "reselect", "evaluate"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pulse-weight", type=float, default=3.)
    p.add_argument("--budget-episodes", type=int, default=ES_EPISODES)
    p.add_argument("--n-envs", type=int, default=18)
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--spec", default=None)
    p.add_argument("--variants", default="intact,clamp_fast,noref,nopulse")
    p.add_argument("--n-val", type=int, default=None, help="选中心每任务用例数（冒烟用；默认 64）")
    p.add_argument("--n", type=int, default=None, help="盲测每任务只跑前 n 例（冒烟用）")
    a = p.parse_args()
    if a.stage == "selftest":
        return selftest()
    dict(bc=bc, finetune=finetune, reselect=reselect, evaluate=evaluate)[a.stage](a)


if __name__ == "__main__":
    main()
