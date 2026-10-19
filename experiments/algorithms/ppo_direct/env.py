"""PPO 直接控制的 Gym 环境：完整扑翼模型（research 分支的 bench 环境），接口与 rl_imu_hover 相同。

和 rl_imu_hover/maneuver_env.py 的区别只有物理：这里用 bench.py 的 ResearchEnv（hover9 完整翅拍动力学、
电机模型、云台伺服、带噪声/延迟的 IMU 和测距高度），任务、用例、评分全部沿用 bench.run_episode，
因此结果可以和 plan_v4 第五轮的 SNN 各种子、classical 对手在同一盲测集上直接比。

接口（原接口）：
    传感：reading 11 维 [roll, pitch, yaw, gx, gy, gz, ax, ay, az, z, vz]，500 Hz，z 相对起飞高度；
    执行：云台 α、β ∈ ±0.3 rad，电机 ω ∈ [3115.22, 3807.50] rad/s，每 2 ms 一条指令，仿真器内部 8.2 ms 平滑。
控制权（control 参数）：t < 1.5 s（gates.HANDOVER_T，起飞过渡）都由经典反射 PID 飞；之后
    direct   : PPO 直接给 α、β、ω，完全替代 PID（实测学不起来：接管后 0.6 s 倾翻，40 万步曲线平线）；
    residual : 反射 PID 留在底层（与 SNN 各臂相同），PPO 输出 [roll 设定点, pitch 设定点, 电机修正]——
               设定点 ±REF_MAX = 0.7 rad 写进 ReferencePID.rpy_reference（maneuver_v2 的"设定点权限"接口），
               电机 = residual_command(a[2], PID 的高度回路输出)，a[2] = ±1 到达 ω 上/下限。
               执行器上仍是 α、β、ω 三个量、同限幅；这是和 SNN 第五轮（PID + 学习层脉冲/设定点）对等的接法。
决策：每 10 ms 一次（5 拍），指令保持。
观测（17 维，都缩放到 ~[-1, 1]）：
    reading[0:9] / [.35, .35, π, 10, 10, 10, 9.81, 9.81, 9.81]，z/0.2，vz/1.0，
    当前平滑后的执行量 (air.action − CENTER)/HALF（3 维），指令通道 3 维 [nod 符号, dash 方向 x, dash 方向 y]。
任务：hover（无指令，含 upset 扰动用例）、nod、dash；训练时按权重随机，用例从 bench 的训练种子段抽（永不碰盲测）。
奖励与 rl_imu_hover 相同（真值只做训练信号）：
    悬停/目标达成后：exp(−(4|z| + 1.5 tilt + 0.2|v| + 0.03|ω_body| + 0.01|Δa|²)) (+0.5 稳定)；
    nod 指令期：0.5·exp(−4|z|) + clip(sign·pitch/30°, −1, 1)，达成 +5；
    dash 指令期：0.5·exp(−4|z|) + 10·Δforward + 0.5·clip(v_fwd/0.5, −1, 1) − 0.5|lateral|，达成 +5；坠落 −10。
目标检测和评分照抄 bench.run_episode（nod: |pitch| ≥ 30° 保持 0.1 s；dash: 沿航向前进 DASH_M）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..neuro_gates import gates
from ..robust_control.environment import ReferencePID
from ..snn_pid_compare.common import DT
from ..snn_pid_compare.environment import CENTER, HALF, HIGH, LOW
from ..spike_steer import bench as B
from ..spike_steer import descend_r5 as R5
from ..topology_hover.environment import case as make_case

SOURCE = Path("runs/neuro_gates/scratch/source")
CALIBRATE = Path("artifacts/spike_steer/plan_v4/r3/upset_v2/calibrate.json")
BASELINE_R5 = Path("artifacts/spike_steer/plan_v4/r5/claude_ea/baseline.json")
TICKS_PER_DECISION = 5                    # 10 ms
TRAIN_OFFSET = 200_000                    # 训练用例种子段：ES 用了 100_000+，验证 60_000，诊断 70_000/120_000
OBS_SCALE = np.array([.35, .35, math.pi, 10., 10., 10., 9.81, 9.81, 9.81, .2, 1.])
NOD_DEG, HOLD_S, RECOVER_DEG = 30., B.HOLD_S, 5.
SETTLE_V, SETTLE_S = .1, .3


def rotation_to_rpy(r: np.ndarray) -> np.ndarray:
    return np.array([math.atan2(r[2, 1], r[2, 2]), -math.asin(float(np.clip(r[2, 0], -1., 1.))), math.atan2(r[1, 0], r[0, 0])])


def prepare_source() -> Path:
    if not (SOURCE / "robot_upper7_lower8.xml").exists():
        B.prepare_source(SOURCE)
    return SOURCE


class DirectManeuverEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, *, task: str = "mixed", task_weights=(0.4, 0.4, 0.2), upset_fraction: float = 0.5, dash_m: float | None = None,
                 seed_offset: int = 0, control: str = "residual") -> None:
        super().__init__()
        R5.install()                                   # 第五轮的盲测集、penalised objective（只影响 B.cases 的 test 分支）
        prepare_source()
        base = __import__("json").loads(BASELINE_R5.read_text())
        self.dash_m = float(dash_m if dash_m is not None else (base["prereg"]["dash_m"] or .25))
        B.worker_init(str(SOURCE), self.dash_m)        # 本进程的 ResearchEnv + upset 注入
        self.env = B._ENV
        self.levels = __import__("json").loads(CALIBRATE.read_text())["levels"]
        self.task_mode = task
        w = np.asarray(task_weights, float)
        self.task_weights = w / w.sum()               # nod, dash, hover
        self.upset_fraction = float(upset_fraction)   # hover 回合里带 upset 扰动的比例
        self.seed_offset = int(seed_offset)
        if control not in ("direct", "residual"):
            raise ValueError(control)
        self.control = control
        self.episode_counter = 0
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.case = None

    # ---- 用例 ------------------------------------------------------------------
    def _draw_case(self, task: str) -> dict:
        i = TRAIN_OFFSET + int(self.np_random.integers(0, 80_000))
        if task in ("nod", "dash"):
            return make_case(B.BASES["man_train"] + i, B.SECONDS[task], "stress", i % 8)
        if task == "upset":
            return B.cases("upset", "train", self.levels, n=1, offset=i)[0]
        return B.cases("hover", "train", None, n=1, offset=i)[0]

    # ---- 真值 / 观测 --------------------------------------------------------------
    def _truth(self) -> dict[str, Any]:
        pos, vel, r = self.env.air.env._state()
        pitch = float(self.oracle.queue[-1][1]) if self.oracle.queue else float(rotation_to_rpy(r)[1])
        return dict(position=pos.copy(), velocity=vel.copy(), rotation=r.copy(), pitch=pitch,
                    tilt=math.acos(min(1., max(-1., float(r[2, 2])))), z=float(pos[2] - self.env.air.start[2]))

    def _observation(self) -> np.ndarray:
        reading = np.asarray(self.env.last_reading, float) / OBS_SCALE
        actuator = (np.asarray(self.env.air.action, float) - CENTER) / HALF
        return np.clip(np.concatenate((reading, actuator, self.cmd)), -10., 10.).astype(np.float32)

    def _command_vector(self, truth: dict[str, Any]) -> np.ndarray:
        if not self.cmd_on:
            return np.zeros(3)
        if self.task == "nod":
            return np.array([self.nod_sign, 0., 0.])
        yaw = rotation_to_rpy(truth["rotation"])[2]
        a = self.dash_angle - yaw
        return np.array([0., math.cos(a), math.sin(a)])

    # ---- 回合 --------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        task = options.get("task", self.task_mode)
        if task == "mixed":
            task = str(self.np_random.choice(["nod", "dash", "hover"], p=self.task_weights))
        self.case = options.get("case") or self._draw_case("upset" if (task == "hover" and self.np_random.random() < self.upset_fraction) else task)
        self.task = task
        self.episode_counter += 1
        self.nod_sign = float(options.get("nod_sign", self.np_random.choice([-1., 1.])))
        self.dash_angle = options.get("dash_angle")      # None = 沿 t0 时的航向（bench 定义）；训练时随机
        if self.dash_angle is None and task == "dash" and not options.get("forward", False):
            self.dash_angle = float(self.np_random.uniform(-math.pi, math.pi))

        env = self.env
        env.configure(self.case)
        env.reset(seed=self.case["seed"])
        env.upset_axes = {}
        p, (sp, sd) = env.p, B.PID_GAINS
        env.teacher = ReferencePID(dict(p, kp=[sp * k for k in p["kp"]], kd=[sd * k for k in p["kd"]], ki=[sp * k for k in p["ki"]]), env.air.last_rpy)
        env._teacher_step = -1
        self.oracle = gates.Oracle(env, "derocked")
        self.t0 = self.case["upset"]["t0"] if "upset" in self.case else B.T0
        self.window = B.WINDOW.get(task)
        self.cmd_on, self.goal_done, self.goal_t, self.hold = False, False, None, 0
        self.pos0, self.heading, self.z0 = None, None, None
        self.recover_t, self.settle_t, self.settle_count = None, None, 0
        self.max_forward, self.last_forward, self.min_z = 0., 0., math.inf
        self.rows, self.pulses = [], 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.cmd = np.zeros(3)
        self.terminated_reason = None
        self.ep = None
        obs = self._observation()
        return obs, dict(task=self.task, seed=self.case["seed"], nod_sign=self.nod_sign, dash_angle=self.dash_angle)

    def _tick(self, action: np.ndarray | None) -> tuple[bool, bool, dict]:
        """一个 2 ms 拍：t < 1.5 s 用 PID，否则用 PPO 的指令。返回 (terminated, truncated, out)。"""
        env = self.env
        t = env.air.steps * DT
        if action is None or t < gates.HANDOVER_T:
            env.teacher.rpy_reference = np.zeros(2)
            command = env.teacher_command()
        elif self.control == "direct":
            command = CENTER + HALF * np.clip(action, -1., 1.)
        else:
            env.teacher.rpy_reference = B.REF_MAX * np.clip(action[:2], -1., 1.)       # [roll, pitch] 设定点
            base = env.teacher_command()
            command = base.copy()
            a2 = float(np.clip(action[2], -1., 1.))
            command[2] = base[2] + a2 * ((HIGH[2] - base[2]) if a2 >= 0 else (base[2] - LOW[2]))
        _, _, term, trunc, out = env.step((np.clip(command, LOW, HIGH) - CENTER) / HALF)
        self.oracle.update()
        pos, vel, r = env.air.env._state()
        tnow = env.air.steps * DT
        pitch_true = float(self.oracle.queue[-1][1])
        fire = 0.
        ref = float(env.teacher.rpy_reference[1]) if self.control == "residual" else 0.
        self.rows.append([tnow, pitch_true, *pos, *vel, math.acos(min(1., max(-1., float(r[2, 2])))), ref, fire, *self.cmd[:2]])
        return term, trunc, out

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
        env = self.env
        truth = self._truth()
        t = env.air.steps * DT
        rel = t - self.t0
        if self.task in ("nod", "dash") and self.pos0 is None and rel >= -1e-9:
            self.pos0, self.z0 = truth["position"].copy(), float(truth["position"][2])
            yaw = rotation_to_rpy(truth["rotation"])[2]
            if self.dash_angle is None:
                self.dash_angle = float(yaw)              # bench 定义：沿 t0 时的航向（forward）
            angle = self.dash_angle
            self.heading = np.array([math.cos(angle), math.sin(angle)])
            self.cmd_on = True
        self.cmd = self._command_vector(truth)
        terminated = truncated = False
        out = {}
        for _ in range(TICKS_PER_DECISION):
            terminated, truncated, out = self._tick(action)
            if terminated or truncated:
                break
        truth = self._truth()
        t = env.air.steps * DT
        bonus = 0.
        z = truth["z"]
        v = truth["velocity"]
        tilt = truth["tilt"]
        gyro = np.asarray(env.last_reading[3:6], float)
        hover_reward = math.exp(-(4. * abs(z) + 1.5 * tilt + .2 * float(np.linalg.norm(v)) + .03 * float(np.linalg.norm(gyro)) + .01 * float(np.sum((action - self.last_action) ** 2))))
        if abs(z) < .05 and float(np.linalg.norm(v)) < .1 and tilt < math.radians(5.):
            hover_reward += .5
        reward = hover_reward
        if self.pos0 is not None:
            d = truth["position"][:2] - self.pos0[:2]
            forward, lateral = float(d @ self.heading), float(d @ np.array([-self.heading[1], self.heading[0]]))
            self.max_forward = max(self.max_forward, forward)
            self.min_z = min(self.min_z, float(truth["position"][2]))
            v_forward = float(v[:2] @ self.heading)
            pitch_deg = math.degrees(truth["pitch"])
            if self.task == "nod":
                if self.cmd_on:
                    self.hold = self.hold + 1 if abs(pitch_deg) >= NOD_DEG else 0        # bench：不分符号
                    if self.hold * TICKS_PER_DECISION * DT >= HOLD_S - 1e-9:
                        self.cmd_on, self.goal_done, self.goal_t, bonus = False, True, t - self.t0, 5.
                    reward = .5 * math.exp(-4. * abs(z)) + float(np.clip(self.nod_sign * pitch_deg / NOD_DEG, -1., 1.))
                elif self.goal_done and self.recover_t is None and abs(pitch_deg) < RECOVER_DEG:
                    self.recover_t = t - self.t0 - self.goal_t
            else:
                if self.cmd_on:
                    if forward >= self.dash_m:
                        self.cmd_on, self.goal_done, self.goal_t, bonus = False, True, t - self.t0, 5.
                    reward = (.5 * math.exp(-4. * abs(z)) + 10. * (forward - self.last_forward) + .5 * float(np.clip(v_forward / .5, -1., 1.)) - .5 * abs(lateral))
                elif self.goal_done and self.settle_t is None:
                    self.settle_count = self.settle_count + 1 if abs(v_forward) < SETTLE_V else 0
                    if self.settle_count * TICKS_PER_DECISION * DT >= SETTLE_S - 1e-9:
                        self.settle_t = t - self.t0 - self.goal_t
            self.last_forward = forward
        if terminated:
            reward -= 10.
        reward += bonus
        self.last_action = action
        self.cmd = self._command_vector(truth)
        info = dict(task=self.task, t=t, z=z, tilt_deg=math.degrees(tilt), pitch_deg=math.degrees(truth["pitch"]), cmd_on=self.cmd_on, goal_done=self.goal_done,
                    command=env.command.copy())
        if terminated or truncated:
            self.ep = out["episode"]
            info["metrics"] = self.metrics()
            info["is_success"] = bool(not terminated and abs(z) < .1 and float(np.linalg.norm(v)) < .2 and tilt < math.radians(10.))
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    # ---- bench 同款评分（照抄 bench.run_episode 的 nod/dash 分支；hover/upset 直接用 bench 的公式）-----------
    def metrics(self) -> dict[str, Any]:
        ep, task, c = self.ep, self.task, self.case
        a = np.array(self.rows)
        flown = ep["termination"] == "time_limit"
        res = dict(seed=c["seed"], task=task, termination=ep["termination"], elapsed=ep["elapsed"], flown=flown, pulses=0, pulse_duty=0.,
                   goal_t=self.goal_t, ref_abs_mean=0.)
        if task == "hover":
            if "upset" in c:
                return self._upset_metrics(res, ep)
            late = a[:, 0] >= 2.
            pitch_rms = float(np.degrees(np.sqrt(np.mean(a[late, 1] ** 2)))) if late.any() else float("nan")
            J = pitch_rms + B.P.HEIGHT_WEIGHT * 100 * ep["height_rmse_m"] if flown else 40. - ep["elapsed"]
            res.update(task="hover", score=float(J), pitch_rms_deg=pitch_rms, height_rmse_cm=100 * ep["height_rmse_m"], tilt_rms_deg=ep["tilt_rms_deg"])
            return res
        w = a[(a[:, 0] >= B.T0) & (a[:, 0] < B.T0 + self.window)]
        rel_t = w[:, 0] - B.T0
        pitch = np.degrees(w[:, 1])
        pos0, heading = (self.pos0, self.heading) if self.pos0 is not None else (a[0, 2:5], np.array([1., 0.]))
        fwd = (w[:, 2:4] - pos0[:2]) @ heading
        lat = (w[:, 2:4] - pos0[:2]) @ np.array([-heading[1], heading[0]])
        vfwd = w[:, 5:7] @ heading
        dz = float(100 * max(0., pos0[2] - w[:, 4].min())) if len(w) else 0.
        crash = 0. if flown else 10. + 5. * float(np.clip((B.T0 + self.window - ep["elapsed"]) / self.window, 0., 1.))
        miss = self.window
        hold = int(round(HOLD_S / DT))

        def first(mask, start=0.):
            i = np.flatnonzero(mask & (rel_t >= start))
            return float(rel_t[i[0]]) if len(i) else None
        res.update(height_loss_cm=dz, tilt_peak_deg=float(np.degrees(w[:, 8].max())) if len(w) else None)
        if task == "nod":
            bigm = np.abs(pitch) >= 30.
            sustained = np.array([bigm[i:i + hold].all() if i + hold <= len(bigm) else False for i in range(len(bigm))], dtype=bool)
            t30 = first(sustained)
            trec = first(np.abs(pitch) < 5., (t30 or 0.) + .02) if t30 is not None else None
            drift = float(abs(fwd[-1])) if len(fwd) else 0.
            score = (t30 if t30 is not None else miss) + ((trec - t30) if (t30 is not None and trec is not None) else miss) + .03 * dz + .5 * drift + crash
            res.update(t30=t30, t_recover=None if (t30 is None or trec is None) else trec - t30, drift_m=drift,
                       peak_pitch_deg=float(np.abs(pitch).max()) if len(pitch) else None, score=float(score), direction="up" if self.nod_sign > 0 else "down")
        else:
            target = self.dash_m
            treach = first(fwd >= target)
            over = float(max(0., fwd.max() - (target + B.OVER_M))) if len(fwd) else 0.
            tsettle = None
            if treach is not None:
                slow = np.abs(vfwd) < .1
                for i in np.flatnonzero((rel_t >= treach)):
                    j = i + int(.3 / DT)
                    if j <= len(slow) and slow[i:j].all():
                        tsettle = float(rel_t[i] - treach)
                        break
            short = 0. if treach is not None else 2. * max(0., target - (fwd.max() if len(fwd) else 0.))
            score = (treach if treach is not None else miss + short) + (tsettle if tsettle is not None else miss) + 2. * over + .03 * dz \
                + .5 * float(abs(lat[-1]) if len(lat) else 0.) + crash
            res.update(t_reach=treach, t_settle=tsettle, overshoot_m=over, max_forward_m=float(fwd.max()) if len(fwd) else None,
                       lateral_m=float(abs(lat[-1])) if len(lat) else None, score=float(score),
                       direction_deg=None if self.dash_angle is None else math.degrees(self.dash_angle))
        return res

    def _upset_metrics(self, res: dict, ep: dict) -> dict:
        """照抄 bench.run_episode 的 upset 分支。"""
        env, c = self.env, self.case
        t0, window = c["upset"]["t0"], B.WINDOW["upset"]
        s = np.asarray(env.samples)
        ts, tilt, z = s[:, 0], np.degrees(s[:, 1]), s[:, 2]
        end = t0 + window
        pre, win = (ts >= 2.) & (ts < t0), (ts >= t0) & (ts < end)
        i0 = int(np.argmin(abs(ts - t0)))
        dz = float(100 * np.abs(z[win] - z[i0]).max()) if win.any() else float("nan")
        flown = res["flown"]
        J = float(tilt[pre].sum() * DT + tilt[win].sum() * DT + .2 * dz) if flown else 60. + 40. * float(np.clip((end - ts[-1]) / window, 0., 1.))
        res.update(task="upset", score=J, phi=c["upset"]["phi"], mag=c["upset"]["mag"], double=c["upset"]["double"], cls=B.U.klass(c["upset"]["phi"]),
                   tilt_peak_deg=float(tilt[win].max()) if win.any() else None, dz_cm=dz, recovery_s=None, recovery_censored=window)
        return res

    def close(self) -> None:
        pass
