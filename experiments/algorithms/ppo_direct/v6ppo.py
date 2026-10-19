"""plan v6（nod 4 方向、dash 8 方向）流程，把"ES 搜索 SNN"换成"PPO 训练同接口的 MLP"，其余环节照搬 descend8 / bench8。

对应关系（descend8 -> 这里）：
  学习层  Learned8：门控（指令期 + 1 s 尾巴在岗）、扶正反射 25°、不应期 37 ms；输入 bench8.features8 的 25 维
          （16 个整流 IMU 特征 + 偏置 + 8 个整流方向指令通道），输出 up/down 脉冲 + (roll, pitch) 设定点 ±0.7 rad。
          -> PpoLayer8：同门控、同底层（反射 PID + law_tuned 脉冲律 / pulse_tuned）；学习层 = MLP 64x64，每 10 ms 决策，
             输入 = 当前 / 10 ms 前 / 20 ms 前三帧的 16 维 IMU 特征 + 偏置 + 8 维指令 = 57 维；
             输出 3 维：脉冲（>0 则在窗口第一拍起始 3 拍脉冲，受不应期约束）、roll 设定点、pitch 设定点。
  起点    stage_collect（classical 对手逐拍轨迹）+ stage_train（BCE + 5·MSE，40 轮）
          -> bc：同一份 collect_train.npz，按 10 ms 窗口聚合后 MSE 回归到 MLP，装进 PPO 策略（= 第 −1 号中心）。
  搜索    OpenAI-ES 60 代 × 33 候选 × (8 nod + 8 dash)，目标 = (2/3, 1/3) 加权的 (分数 / 该方向类归一化) + 倾角惩罚 (>60° × 0.3)，
          用例 cases8(train, offset 100_000 + gen*100)；每 5 代存中心（13 个）。
          -> finetune：PPO，回合末奖励 = −权重 × 同一目标，没有其他塑形；用例从同一池子抽；预算按回合数（--budget-episodes，
             默认 7 920 = ES 的 1/4），每 1/12 预算存检查点（+ bc = 13 个中心）。
  选中心  stage_reselect：64 nod + 64 dash（offset 60_000），零坠落且真实倾角峰 < 70°，目标最低。-> 同用例、同规则。
  盲测    stage_evaluate：TEST_BASE 2_097_100_000，32 nod + 32 dash，对手 = 按方向类的 2-D schedule；每例比值 = 分数 / 该类
          对手中位数；manoeuvre_mean = (nod + dash)/2。-> 同用例、同对手、同统计。hover / upset 按构造不变，不重跑（同 v6）。
  种子    1–5。修正案 1：ES 可在任意机器跑，选中心和盲测只在 CloudPC-2 上跑——PPO 臂是否沿用由用户决定。

    python -m experiments.algorithms.ppo_direct.v6ppo selftest
    python -m experiments.algorithms.ppo_direct.v6ppo bc       --seed 1
    python -m experiments.algorithms.ppo_direct.v6ppo finetune --seed 1 --budget-episodes 7920 --n-envs 18
    python -m experiments.algorithms.ppo_direct.v6ppo reselect --seed 1 --workers 18
    python -m experiments.algorithms.ppo_direct.v6ppo evaluate --seed 1 --workers 18
输入：artifacts/spike_steer/plan_v6/{baseline.json, signs.json}；模仿数据 runs/spike_steer/v6/collect_train.npz（descend8 collect）。
输出：runs/ppo_direct/v6ppo/s<seed>/。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..neuro_gates import gates
from ..robust_control.environment import ReferencePID
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import CENTER, HALF, HIGH, LOW
from ..spike_steer import bench as B
from ..spike_steer import bench8 as E
from ..spike_steer import descend8 as D8
from ..spike_steer import pilot_v3 as V
from ..topology_hover.environment import case as make_case

OUT = Path("runs/ppo_direct/v6ppo")
PLAN = Path("artifacts/spike_steer/plan_v6")
COLLECT = Path(os.environ.get("COLLECT_NPZ", "runs/spike_steer/v6/collect_train.npz"))
TICKS = 5                                      # 10 ms 决策
FRAMES = (0, 5, 10)
N_IMU, N_CMD = 16, 8
N_OBS = N_IMU * len(FRAMES) + 1 + N_CMD        # 57
ES_EPISODES = 60 * 33 * 16                     # 31 680
MEAN_DECISIONS = (2500 + 3250) / 2 / TICKS
N_CENTRES = 12


def load_plan():
    E.load_signs()
    B.DASH_M["value"] = .15
    return json.loads((PLAN / "baseline.json").read_text())


# ----------------------------------------------------------------------------------------- 环境
class PpoLayer8Env(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, *, variant: str = "intact", task_weights=(2., 1.)) -> None:
        super().__init__()
        self.base = load_plan()
        if not (E.SOURCE / "robot_upper7_lower8.xml").exists():
            B.prepare_source(E.SOURCE)
        B.worker_init(str(E.SOURCE), self.base["dash_m"])
        self.env = B._ENV
        self.norm = self.base["normalisers"]
        self.variant = variant
        w = np.asarray(task_weights, float)
        self.task_p = w / w.sum()
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.feat_hist = collections.deque([np.zeros(N_IMU, np.float32)], maxlen=max(FRAMES) + 1)
        self.cmd = np.zeros(4)

    # ---- 用例：与 ES 同一池子 ----------------------------------------------------
    def _draw_case(self, task: str) -> dict:
        gen, i = int(self.np_random.integers(0, 60)), int(self.np_random.integers(0, 8))
        dirs = E.TASK_DIRS[task]
        c = make_case(E.BASES["train"] + D8.OFF["es"] + gen * 100 + i, B.SECONDS[task], "stress", i % 8)
        c["dir_deg"] = float(dirs[(i + i // 8) % len(dirs)])
        return c

    # ---- 观测 ----------------------------------------------------------------------
    def _feat(self) -> np.ndarray:
        return np.asarray(E.features8(self.env.last_reading, self.cmd)[:N_IMU], np.float32)

    def _observation(self) -> np.ndarray:
        h = self.feat_hist
        frames = [h[-1 - k] if len(h) > k else h[0] for k in FRAMES]
        c = np.asarray(self.cmd, float)
        cmd8 = np.r_[np.maximum(c, 0.), np.maximum(-c, 0.)]
        return np.concatenate(frames + [np.array([1.], np.float32), cmd8.astype(np.float32)]).astype(np.float32)

    # ---- 回合 -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        task = options.get("task") or str(self.np_random.choice(["nod", "dash"], p=self.task_p))
        self.task = task
        self.case = options.get("case") or self._draw_case(task)
        env, c = self.env, self.case
        env.configure(c)
        env.reset(seed=c["seed"])
        env.upset_axes = {}
        p, (sp, sd) = env.p, B.PID_GAINS
        env.teacher = ReferencePID(dict(p, kp=[sp * k for k in p["kp"]], kd=[sd * k for k in p["kd"]], ki=[sp * k for k in p["ki"]]), env.air.last_rpy)
        env._teacher_step = -1
        self.oracle = gates.Oracle(env, "derocked")
        self.steer = V.SteerV3(None, self.base["law"], False)
        self.steer.reset()
        self.steer_up = V.SteerV3(None, self.base["upset_law"], False)
        self.steer_up.reset()
        self.tilt_thr, self.tail, self.refr = math.radians(D8.UPSET_TILT_DEG), D8.TAIL_S, D8.REFRACTORY_S
        self.upset, self.off_t, self.was_on, self.last_onset, self.in_charge = False, None, False, None, False
        self.onsets = self.cancelled = self.pulses = 0
        self.twitch = 0
        self.ref = np.zeros(2)
        self.phi = math.radians(c["dir_deg"])
        self.u_att = E.att_dir(c["dir_deg"])
        self.pos0 = self.fwd = self.left = None
        self.goal_done, self.goal_t, self.big = False, None, 0
        self.rows = []
        self.cmd = np.zeros(4)
        self.feat_hist = collections.deque([self._feat()], maxlen=max(FRAMES) + 1)
        return self._observation(), dict(task=task, seed=c["seed"], dir_deg=c["dir_deg"])

    def _tick(self, action, first: bool):
        env = self.env
        t, now = env.air.steps * DT, float(env.sim.data.time)
        rel = t - B.T0
        if self.pos0 is None and t >= B.T0:
            pos, _, _ = env.air.env._state()
            self.pos0 = pos.copy()
            yaw = float(env.last_reading[2])
            self.fwd = np.array([math.cos(yaw), math.sin(yaw)])
            self.left = np.array([-self.fwd[1], self.fwd[0]])
        self.cmd = np.zeros(4)
        if rel >= 0 and not self.goal_done:
            k = 0 if self.task == "nod" else 2
            self.cmd[k], self.cmd[k + 1] = math.cos(self.phi), math.sin(self.phi)
        # ---- Learned8.act 的门控 / 底层 ----
        law_pulse, _ = self.steer.step(env, now, t)
        up_pulse, _ = self.steer_up.step(env, now, t)
        on = bool(np.any(self.cmd))
        if self.was_on and not on:
            self.off_t = t
        self.was_on = on
        self.in_charge = on or (self.off_t is not None and t - self.off_t < self.tail)
        if not self.upset and not self.in_charge and math.hypot(float(env.last_reading[0]), float(env.last_reading[1])) > self.tilt_thr:
            self.upset = True
        if self.upset:
            law_pulse = up_pulse
        fire = False
        if self.in_charge and t >= gates.HANDOVER_T:
            if first:                                   # 学习层每 10 ms 决策一次
                self.ref = B.REF_MAX * np.clip(np.asarray(action[1:3], float), -1., 1.)
                if self.variant == "noref":
                    self.ref = np.zeros(2)
                if float(action[0]) > 0. and self.variant != "nopulse":
                    if self.last_onset is not None and t - self.last_onset < self.refr:
                        self.cancelled += 1
                    else:
                        self.last_onset, self.onsets, self.twitch = t, self.onsets + 1, B.TWITCH
            if self.twitch > 0:
                self.twitch -= 1
                fire = True
            ref = self.ref
        else:
            self.twitch = 0
            ref = np.zeros(2)
            fire = bool(law_pulse) and t >= gates.HANDOVER_T
        env.teacher.rpy_reference = np.clip(ref, -B.REF_MAX, B.REF_MAX)
        command = env.teacher_command()
        if fire:
            command[2] = HIGH[2]
            self.pulses += 1
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH) - CENTER) / HALF)
        self.oracle.update()
        pos, vel, R = env.air.env._state()
        roll_true, pitch_true = (float(v) for v in self.oracle.queue[-1][:2])
        tnow = env.air.steps * DT
        along = float(np.degrees(np.array([roll_true, pitch_true]) @ self.u_att))
        self.rows.append([tnow, pitch_true, roll_true, *pos, *vel, math.acos(min(1., max(-1., float(R[2, 2])))), *ref, float(fire), along])
        if self.task == "nod" and not self.goal_done and rel >= 0:
            self.big = self.big + 1 if along >= E.NOD_DEG else 0
            if self.big >= int(round(B.HOLD_S / DT)):
                self.goal_done, self.goal_t = True, tnow - B.T0
        if self.task == "dash" and not self.goal_done and self.pos0 is not None:
            dvec = math.cos(self.phi) * self.fwd + math.sin(self.phi) * self.left
            if float((pos[:2] - self.pos0[:2]) @ dvec) >= B.DASH_M["value"]:
                self.goal_done, self.goal_t = True, tnow - B.T0
        self.feat_hist.append(self._feat())
        return term, trunc, out

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1., 1.)
        term = trunc = False
        out = {}
        for k in range(TICKS):
            term, trunc, out = self._tick(action, first=(k == 0))
            if term or trunc:
                break
        reward = 0.
        info = dict(task=self.task, in_charge=self.in_charge, goal_done=self.goal_done)
        if term or trunc:
            res = E.score8(self.task, self.case, out["episode"], np.array(self.rows), self.pos0, self.fwd, self.left, self.pulses, self.goal_t)
            res.update(onsets=self.onsets, cancelled=self.cancelled)
            cl = D8.CLASS[int(self.case["dir_deg"])]
            pen = max(0., (res["tilt_peak_deg"] or 0.) - D8.SOFT_TILT_DEG) * D8.PENALTY_PER_DEG
            res["objective"] = res["score"] / self.norm[self.task][cl] + pen
            reward = -D8.WEIGHTS[self.task] * 3. * res["objective"]        # (2/3, 1/3) × 3 = (2, 1)
            info["metrics"] = res
            info["is_success"] = bool(res["flown"] and self.goal_done)
        return self._observation(), float(reward), bool(term), bool(trunc), info


# ----------------------------------------------------------------------------------------- 工具 / bc
def seed_dir(seed: int) -> Path:
    d = OUT / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_policy_model(env, seed: int):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
               ent_coef=0.0, policy_kwargs={"net_arch": [64, 64], "log_std_init": -1.5}, verbose=1, seed=seed, device="cpu", tensorboard_log=str(OUT / "tb"))


def bc_dataset():
    d = np.load(COLLECT)
    X, L = d["X"], d["lengths"]
    obs, tgt = [], []
    for e in range(X.shape[0]):
        x = X[e, :L[e]]
        imu, bias, cmd8 = x[:, :N_IMU], x[:, N_IMU:N_IMU + 1], x[:, N_IMU + 1:D8.N_IN]
        pulse, ref = x[:, D8.N_IN], x[:, D8.N_IN + 2:D8.N_IN + 4]
        for s in range(max(FRAMES), len(x) - TICKS, TICKS):
            frames = [imu[s - k] for k in FRAMES]
            win = slice(s, s + TICKS)
            onset = bool(pulse[win].max() > 0 and pulse[s - 1] == 0)
            obs.append(np.concatenate(frames + [bias[s], cmd8[s]]))
            tgt.append(np.r_[1. if onset else -1., ref[win].mean(0) / B.REF_MAX])
    return np.asarray(obs, np.float32), np.asarray(tgt, np.float32)


def bc(a):
    import torch
    torch.set_num_threads(2)
    obs, tgt = bc_dataset()
    print(f"bc data: {len(obs)} windows, pulse-window rate {np.mean(tgt[:, 0] > 0):.3f}, |ref| mean {np.abs(tgt[:, 1:]).mean(0).round(3).tolist()}", flush=True)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(obs))
    va, tr = idx[:len(obs) // 10], idx[len(obs) // 10:]
    model = make_policy_model(PpoLayer8Env(), a.seed)
    pol = model.policy
    net = torch.nn.Sequential(pol.mlp_extractor.policy_net, pol.action_net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.tensor(obs), torch.tensor(tgt)
    w = torch.tensor([a.pulse_weight, 5., 5.])          # descend8：BCE(pos weight 3) + 5·ref MSE
    best, state = None, None
    for epoch in range(a.epochs):
        rng.shuffle(tr)
        losses = []
        for b in range(0, len(tr), 256):
            i = torch.tensor(tr[b:b + 256])
            loss = (w * (net(Xt[i]) - Yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        with torch.no_grad():
            iv = torch.tensor(va)
            pv = net(Xt[iv])
            vloss = float((w * (pv - Yt[iv]) ** 2).mean())
            acc = float(((pv[:, 0] > 0) == (Yt[iv, 0] > 0)).float().mean())
            rmse = torch.sqrt(((pv[:, 1:] - Yt[iv, 1:]) ** 2).mean(0)).tolist()
        print(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=vloss, pulse_acc=acc, ref_rmse=[round(r, 4) for r in rmse])), flush=True)
        if best is None or vloss < best:
            best, state = vloss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    d = seed_dir(a.seed)
    model.save(d / "ckpt_g-1")
    print(f"wrote {d / 'ckpt_g-1.zip'} (best val loss {best:.4f})", flush=True)


# ----------------------------------------------------------------------------------------- finetune
def finetune(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    d = seed_dir(a.seed)
    steps = int(a.budget_episodes * MEAN_DECISIONS)
    every = steps // N_CENTRES

    def make(rank):
        def _m():
            e = PpoLayer8Env()
            e.reset(seed=a.seed * 1000 + rank)
            return Monitor(e, info_keywords=("is_success",))
        return _m
    env = SubprocVecEnv([make(i) for i in range(a.n_envs)], start_method="fork")
    init = a.init_model or (d / "ckpt_g-1.zip")
    model = PPO.load(init, env=env, device="cpu", tensorboard_log=str(OUT / "tb"), seed=a.seed)
    t0 = time.monotonic()

    class Archive(BaseCallback):
        def __init__(self):
            super().__init__()
            self.k = 0

        def _on_step(self):
            if self.k < N_CENTRES and self.num_timesteps >= (self.k + 1) * every:
                self.model.save(d / f"ckpt_g{self.k}")
                row = dict(centre=self.k, timesteps=int(self.num_timesteps), wall_s=round(time.monotonic() - t0))
                (d / "history.jsonl").open("a").write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                self.k += 1
            return True
    print(json.dumps(dict(seed=a.seed, budget_episodes=a.budget_episodes, timesteps=steps, checkpoint_every=every, n_envs=a.n_envs, init=str(init))), flush=True)
    model.learn(total_timesteps=steps, callback=Archive(), tb_log_name=f"v6ppo_s{a.seed}", reset_num_timesteps=True)
    model.save(d / f"ckpt_g{N_CENTRES - 1}")
    env.close()
    print(f"finetune done: {steps} steps, wall {round(time.monotonic() - t0)} s", flush=True)


# ----------------------------------------------------------------------------------------- 并行回放
_ENV = None
_POL = {}


def _init(variant):
    global _ENV
    _ENV = PpoLayer8Env(variant=variant)


def _run(args):
    kind, arg, task, case, seed = args
    env = _ENV
    if kind == "classical":                      # 对手：bench8.job 同款
        return dict(E.job(("opp", task, case, "classical", arg)), task=task)
    from stable_baselines3 import PPO
    if arg not in _POL:
        _POL[arg] = PPO.load(arg, device="cpu")
    pol = _POL[arg]
    obs, _ = env.reset(seed=seed, options=dict(task=task, case=case))
    done, info = False, {}
    while not done:
        act, _ = pol.predict(obs, deterministic=True)
        obs, _, te, tr, info = env.step(act)
        done = te or tr
    return dict(info["metrics"], controller=Path(arg).stem, task=task)


def reselect(a):
    d = seed_dir(a.seed)
    base = load_plan()
    n_val = a.n_val or D8.N_VAL_SELECT
    cs = {t: E.cases8(t, E.BASES["train"], n_val, offset=D8.OFF["select"]) for t in D8.TASKS}
    paths = [d / f"ckpt_g{g}.zip" for g in [-1] + list(range(N_CENTRES)) if (d / f"ckpt_g{g}.zip").exists()]
    jobs = [("ppo", str(p), t, c, int(c["seed"]) % (2 ** 31)) for p in paths for t in D8.TASKS for c in cs[t]]
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=("intact",)) as ex:
        rows = list(ex.map(_run, jobs, chunksize=1))
    res = []
    for p in paths:
        rs = [r for r in rows if r["controller"] == p.stem]
        obj, rt = D8.objective(rs, base["normalisers"])
        e = dict(generation=int(p.stem.split("g")[-1]), objective=obj, ratios=rt, crashes=sum(not r["flown"] for r in rs),
                 over_hard=sum((r["tilt_peak_deg"] or 0) >= D8.HARD_TILT_DEG for r in rs), tilt_peak_max=max(r["tilt_peak_deg"] or 0 for r in rs), spec=str(p))
        res.append(e)
        print(json.dumps(e), flush=True)
    ok = [e for e in res if e["crashes"] == 0 and e["over_hard"] == 0]
    sel = min(ok, key=lambda e: e["objective"]) if ok else None
    fallback = None
    if sel is None:                              # v6 没有回退规则（selected = null）；这里记录一个回退并标明
        nocrash = [e for e in res if e["crashes"] == 0]
        sel = min(nocrash or res, key=lambda e: e["objective"])
        fallback = "no eligible centre (v6 rule would select none); fallback = " + ("zero-crash centres" if nocrash else "all centres")
    (d / "reselect.json").write_text(json.dumps(dict(seed=a.seed, n_val=n_val, centres=res, selected=sel, fallback=fallback), indent=1))
    print(json.dumps(dict(selected=sel["generation"], objective=sel["objective"], fallback=fallback)), flush=True)


def evaluate(a):
    d = seed_dir(a.seed)
    base = load_plan()
    sel = json.loads((d / "reselect.json").read_text())["selected"] if a.spec is None else dict(spec=a.spec)
    cs = {t: E.cases8(t, D8.TEST_BASE, a.n or D8.N_TEST) for t in D8.TASKS}
    opp_file = OUT / f"test_opponent{'_n' + str(a.n) if a.n else ''}.json"
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork"), initializer=_init, initargs=("intact",)) as ex:
        if opp_file.exists():
            opp_rows = json.loads(opp_file.read_text())
        else:
            opp_rows = list(ex.map(_run, [("classical", dict(law=base["law"], schedule=base["opponent"][t][D8.CLASS[int(c["dir_deg"])]]), t, c, 0)
                                          for t in D8.TASKS for c in cs[t]], chunksize=1))
            opp_file.write_text(json.dumps(opp_rows))
        rows = list(ex.map(_run, [("ppo", sel["spec"], t, c, int(c["seed"]) % (2 ** 31)) for t in D8.TASKS for c in cs[t]], chunksize=1))
    rep = {}
    for t in D8.TASKS:
        lr_ = [r for r in rows if r["task"] == t]
        op = [r for r in opp_rows if r["task"] == t]
        med = {cl: float(np.median([r["score"] for r in op if D8.CLASS[int(r["dir_deg"])] == cl])) for cl in D8.CLASSES[t]}
        by = {}
        for cl in D8.CLASSES[t]:
            l_ = [r for r in lr_ if D8.CLASS[int(r["dir_deg"])] == cl]
            o_ = [r for r in op if D8.CLASS[int(r["dir_deg"])] == cl]
            by[cl] = dict(ratio=float(np.mean([r["score"] for r in l_]) / med[cl]), opponent_median=med[cl], wins=sum(x["score"] < y["score"] for x, y in zip(l_, o_)),
                          n=len(l_), crashes=sum(not r["flown"] for r in l_), opponent_crashes=sum(not r["flown"] for r in o_))
        rep[t] = dict(ratio=float(np.mean([r["score"] / med[D8.CLASS[int(r["dir_deg"])]] for r in lr_])), wins=sum(x["score"] < y["score"] for x, y in zip(lr_, op)),
                      crashes=sum(not r["flown"] for r in lr_), opponent_crashes=sum(not r["flown"] for r in op), by_class=by,
                      goal_hits=sum(r["goal_t"] is not None for r in lr_), n=len(lr_))
    rep["manoeuvre_mean"] = (rep["nod"]["ratio"] + rep["dash"]["ratio"]) / 2
    (d / "evaluate.json").write_text(json.dumps(dict(selected=sel, report=rep, rows=rows, opponent_rows=opp_rows), indent=1, default=float))
    print(json.dumps(rep, indent=1), flush=True)
    print(f"v6ppo_s{a.seed}: manoeuvre mean {rep['manoeuvre_mean']:.3f}  nod {rep['nod']['ratio']:.3f} ({rep['nod']['crashes']} crash, wins {rep['nod']['wins']}/{rep['nod']['n']})  "
          f"dash {rep['dash']['ratio']:.3f} ({rep['dash']['crashes']} crash, wins {rep['dash']['wins']}/{rep['dash']['n']})", flush=True)


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    from stable_baselines3.common.env_checker import check_env
    env = PpoLayer8Env()
    check_env(env, warn=True)
    c = E.cases8("nod", E.BASES["train"], 1, offset=D8.OFF["select"])[0]
    out = {}
    for name, act in (("zero", np.array([-1., 0., 0.], np.float32)), ("full", None)):
        obs, info = env.reset(seed=1, options=dict(task="nod", case=c))
        done = False
        while not done:
            a = act if act is not None else np.r_[-1., E.att_dir(c["dir_deg"]) * 1.0].astype(np.float32)   # 满设定点沿指令方向
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        out[name] = info["metrics"]
    z, f = out["zero"], out["full"]
    assert z["flown"] and z["t30"] is None and z["pulses"] >= 0, z
    assert (f["peak_along_deg"] or 0.) > 25., f
    assert abs(r + 2. * f["objective"]) < 1e-9
    print(f"selftest ok (dir {c['dir_deg']:.0f} deg: zero action score {z['score']:.3f}; full set-point peak along {f['peak_along_deg']:.1f} deg, t30 {f['t30']}, score {f['score']:.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["selftest", "bc", "finetune", "reselect", "evaluate"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--pulse-weight", type=float, default=3.)
    p.add_argument("--budget-episodes", type=int, default=ES_EPISODES // 4)
    p.add_argument("--n-envs", type=int, default=18)
    p.add_argument("--init-model", type=Path, default=None)
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--spec", default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    a = p.parse_args()
    if a.stage == "selftest":
        return selftest()
    dict(bc=bc, finetune=finetune, reselect=reselect, evaluate=evaluate)[a.stage](a)


if __name__ == "__main__":
    main()
