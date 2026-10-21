# A spiking descending layer for flapping-wing manoeuvres: method and results

*Draft of the Method and Results sections (no Introduction / Related work). Every number is traceable to `artifacts/spike_steer/plan_v*/` in the repository (branch `research/original-env-snn-neat`); method figures are `docs/paper/figs/fig_m*.png`, results figures `docs/paper/figs/fig_r*.png`.*

---

## 2. Method

### 2.1 Platform and interface

**Vehicle and simulator.** The vehicle is the IST_robot-V2 flapping-wing micro air vehicle (15 g, one motor driving two wings at ≈26.7 Hz, i.e. a 37 ms wingbeat; IMU on the lower body). It is simulated in MuJoCo 3.8.0 as a multibody system with blade-element aerodynamics whose lift is calibrated against bench measurements of the real vehicle. The controller runs at 500 Hz (2 ms tick); the plant integrates at 200 kHz (400 physics sub-steps per tick).

**Interface (unchanged throughout).** At every tick the controller receives the 11-dimensional sensor vector `[roll, pitch, yaw (rad), gx, gy, gz (rad/s), ax, ay, az (m/s²), z (m), vz (m/s)]`: attitude and rates from the IMU with Gaussian noise (0.02°, 0.002 rad/s, 0.02 m/s²), `z` from a time-of-flight range finder (50 Hz, 20 ms latency, 2 mm noise) and `vz` from an α-β observer. It commands the two gimbal angles α, β ∈ ±0.3 rad and the motor speed ω ∈ [3115, 3807] rad/s; the commands are updated every 2 ms and pass through an 8.2 ms first-order actuator lag. Measured hover needs ≈3140 rad/s, only ≈25 rad/s above the lower limit, so downward thrust authority is small. The sensor set, actuator set, update rate and limits (`LOW`/`HIGH`, `DT`, the untouched `ControlEnv`) are shared by every controller in this paper; new test scenarios only change the case definitions (Section 2.6).

**Stress condition.** All training, validation and blind cases are "stress" cases: initial attitude error ±5°, motor gain uncertainty ±1.5 %, accelerometer bias 0.01 m/s² and attitude bias 0.05°, one control step (2 ms) of sensor delay, and periodic disturbance pulses: a force of 0.006 N and a torque of 8·10⁻⁶ N·m, switched on for 0.2 s whenever `t mod 2 s ∈ [0.6, 0.8) s` (i.e. at 0.6, 2.6 and 4.6 s) with a random sign per episode, so one pulse lands 0.2–0.4 s into every redirect and dash manoeuvre. Each case is one random seed; eight strata fix the wing state at release (the wingbeat phase at take-off). Episodes last 5 s (evade) or 6.5 s (redirect, dash).

### 2.2 Manoeuvre tasks and scoring

Three manoeuvres are commanded during hover (Fig. M5, `fig_m5_tasks.png`). Directions are given in the heading frame fixed at the command onset `t0` (yaw at `t0`; 0° = forward, 90° = left); the measured attitude-to-motion sign map is (roll, pitch)_ref ∝ (−sin φ, +cos φ). The command is a unit vector (cos φ, sin φ) on one of two channels (evade; dash/redirect) and is presented to the learned layer as eight rectified inputs (± of each component on each channel). The command switches off when the goal is detected on the ground-truth position.

| task | command | goal | window | classes |
|---|---|---|---|---|
| **evade** | displace 0.10 m along φ ∈ {0, 90, 180, 270}°; `t0 ~ U(2.2, 2.8) s` | reach 0.10 m as fast as possible | 1.5 s | pitch (0/180°), roll (90/270°) |
| **redirect** | go along φ₁ ∈ {0, 90, 180, 270}°; when 0.08 m or 1.0 s is reached the command switches to φ₂ = φ₁ ± 90°; `t0 = 2.4 s` | reach 0.10 m along φ₂ after the switch with little overrun along φ₁ | 4 s | by φ₂: pitch, roll |
| **dash** | displace 0.15 m along φ ∈ {0, 45, …, 315}° and stop; `t0 = 2.4 s` | reach, then settle (speed along φ < 0.1 m/s for 0.3 s), little overshoot | 4 s | pitch, roll, diagonal |

**Score (lower is better).** With `W` the window, `short` the shortfall to the goal distance (m), `cross` the lateral position error at the reach time (evade) or at the end of the window (dash), `max|Δz|` the largest altitude deviation (cm) from the altitude at `t0` over the rest of the flight (symmetric; plan v9 onwards), and `crash = 10 + 5·(unflown fraction of the window)` if the episode terminated early (else 0):

- evade: `(t_reach, or 1.5 + 10·short) + 0.5·|cross| + 0.03·max|Δz| + crash`;
- redirect: `t_switch + (t_reach2, or (4 − t_switch) + 10·short) + 2·max(0, overrun − 0.05) + 0.03·max|Δz| + crash`, where `overrun` is the displacement along φ₁ after the switch; a case that never switches scores `4 + 10·0.10 + …`;
- dash: `(t_reach, or 4 + 2·short) + (t_settle, or 4) + 2·max(0, max_along − 0.25) + 0.5·|cross| + 0.03·max|Δz| + crash`.

A **crash** is any early termination: true tilt > 80.2°, |z − z_start| > 1 m, absolute height z < 0.12 m, any body part within 3 cm of the ground, or a numerical failure. Tilt is the true angle between the body axis and the vertical (acos R₃₃ from the simulator state).

**Ratio.** For comparability across tasks and directions, each case score is divided by the median score of the classical opponent over the blind cases of the same direction class (32 cases per class for evade and redirect; 16/16/32 for dash pitch/roll/diagonal); the task ratio is the mean over cases and the *manoeuvre ratio* is the mean over the three tasks. Because the score distributions are right-skewed, this ratio is above 1 even for the opponent itself (1.089 on the plan-v11 blind set, Section 3.1). The main comparisons therefore use the **paired difference** Δ = (arm ratio − opponent's own ratio) on identical cases, with a paired bootstrap 95 % interval over cases within task (4000 resamples), the paired win count, and crash counts with exact Clopper–Pearson 95 % intervals.

### 2.3 Hierarchical controller

The controller has three layers (Fig. M1, `fig_m1_architecture.png`; one 2 ms tick in Fig. M3, `fig_m3_gating.png`).

**Reflex layer — PID.** A PID attitude loop drives the gimbal: `e = ref − [roll, pitch]`, derivative on the measurement through a 4.5 ms low-pass, integral clipped to ±0.3 with anti-windup, `[α, β] = trim + [effort_pitch, effort_roll]`; a PID altitude loop sets the motor speed (gains 900, 500, 80 on z error, vz and ∫z). The attitude gains were re-tuned on stress hover (kp × 1.5, kd × 0.85, ki × 1.5 relative to the original baseline); a full-state-feedback upper bound improves the hover tilt RMS by only 1.1°, so no learned component touches hovering.

**Steering-muscle layer — phase-locked motor pulses.** A motor pulse drives the motor command to its maximum (3807 rad/s) for a few milliseconds; after the 8.2 ms actuator lag it raises the speed by ≈690 rad/s during part of a wingbeat, and the resulting lift asymmetry within the stroke produces a **pitch torque** whose sign depends on the wingbeat phase of the pulse. The classical pulse law (the base law of every arm, active outside manoeuvre commands) is `u = −(k_θ·pitch + k_q·q_slow)` with k_θ = 8.0, k_q = 0.4, `q_slow` a notched pitch rate, and a PLL estimate of the crank phase; it holds the maximum command while `|u| > 0.5` and the phase is within ±22.5° of 135° (u > 0) or 315° (u < 0), i.e. ≈4.6 ms per wingbeat at most. The learned layer's pulses are 3-tick (6 ms) twitches triggered by output spikes (Section 2.3, descending layer).

**Descending layer — the spiking network** (Fig. M2, `fig_m2_snn.png`). A leaky integrate-and-fire (LIF) network with axonal delays:

- input `x_t ∈ ℝ²⁵` per tick: eight IMU channels (roll, pitch, gx, gy, gz, ax, ay, az) scaled by 0.1 rad, 10 rad/s and 25 m/s² respectively and split into rectified + and − channels (16), a constant 1 (bias), and the eight rectified command channels;
- 11 input delay taps `k = 0…10` ticks (0–20 ms): hidden drive `I_h(t) = Σ_k W_in[:, :, k] x_{t−k} + b_h`, `W_in ∈ ℝ^{8×25×11}`;
- 8 hidden LIF neurons updated in two 1 ms sub-steps per tick as unit-gain first-order filters, `c ← a_s c + (1 − a_s) I`, `v ← a_m v + (1 − a_m) c`, `a = exp(−1 ms / τ)`, spike when `v ≥ 1` followed by subtractive reset; τ_m and τ_s are learned in the imitation stage and frozen afterwards (selected centres: τ_m = 1.9–2.1 ms, τ_s = 1.05–1.15 ms);
- hidden → output through 11 axonal delays `d = 0, 2, …, 20 ms` on the hidden spike trains `s`, with a fixed spike scale of 5: `I_o = 5·Σ_d W_out[:, :, d] s_{t−d} + b_o` into two output LIF neurons of the same form (`b_o` initialised to −2), and `I_g = 5·Σ_d W_g[:, :, d] s_{t−d} + b_g` into two graded units `g ← a_g g + (1 − a_g) c_g`, τ_g = 20 ms;
- outputs: a spike of either output neuron triggers one 3-tick motor twitch (two output neurons are two routes to the same twitch; the timing within the wingbeat carries the meaning); the graded units give the **roll and pitch set-points** `ref = 0.7 rad · tanh(g)` handed to the PID;
- 2564 parameters in total (`W_in`, `b_h`, `W_out`, `b_o`, `W_g`, `b_g`); the network has no notion of height or of time since the command (plan v10 tested 12 extra inputs for these and found no gain).

**Command gate and safety rules.** The learned layer is in charge only while a command is on and for a 1 s tail after it ends; outside the gate the base pulse law acts and the set-points are zero, so every arm flies identically tick for tick before `t0` (checked bit-exactly: the prefix cache of Section 2.4 reproduces the plain episode to the last bit, and in the pre-command-hit scenario of Section 3.3 all arms had identical pre-`t0` traces on 192/192 cases). While in charge: pulse onsets are separated by a refractory period of at least 37 ms, and the righting reflex is off. Outside the gate, if the sensed tilt exceeds 25° the base law is replaced for the rest of the episode by a righting pulse law (k_θ = 3.0, k_q = 0.4, threshold 0.5, 90° window). The learned layer can act no earlier than 1.5 s after take-off (before that its set-points are zeroed and its pulses suppressed).

### 2.4 Training

The pipeline (Fig. M4, `fig_m4_pipeline.png`) has three stages; the controller that is tested is the frozen output of stage 3. All stages run in the stress condition on disjoint seed ranges (training, validation and blind sets never share a seed).

**Stage 1 — imitation of the classical schedule.** The classical opponent (Section 2.5) flies 16 training and 4 validation cases per task. From `t = 2 s` on, every tick records the 25-D input, whether the opponent fired a pulse (its base-law pulses are suppressed while a schedule segment is active, so the teacher fires only outside active segments), the sign of its law variable `u` (the nose-up / nose-down label), and its set-points. A torch twin of the network with identical tick arithmetic is trained by back-propagation through time with a sigmoid surrogate gradient (slope 1/0.25) on windows of 200 ticks (stride 100, batches of 32 windows). The loss is `mean[(1 + 2y)·BCE(p, y)] + 5·mean[(ref/0.7 − r)²]`, where `y` is the recorded pulse label, `p = max over the two sub-steps of σ((v_o − 1)/0.25)` is the surrogate firing probability of the output neuron of the labelled sign, and `r` the recorded set-points; Adam 3·10⁻³, gradient norm clipped to 1, 40 epochs, best epoch by validation loss. Five seeds give five imitation starts, shared by all plans from v7 on.

**Stage 2 — OpenAI-ES on the manoeuvre score.** Starting from the imitation weights θ₀ (the six weight/bias tensors; the time constants are frozen), each generation draws 8 antithetic pairs (population 16) of perturbations ε ~ N(0, I), evaluates θ + σε and the centre θ on the generation's batch, and updates θ with rank-based utilities (`u_i ∝ max(0, log(9) − log(rank_i + 1))`, centred), the gradient estimate `Σ u_i ε_i / σ` normalised by Adam moments (0.9, 0.999, bias-corrected), σ = 0.02, learning rate 0.01 · 0.85^g, 16 generations (random seed 1000 + seed). The objective of a candidate is the mean over tasks of the mean over its cases of `score / N_{task,class}`, where `N` is the opponent's class mean on 24 fixed normalisation cases per task, plus a soft tilt penalty of 0.3 per degree of the tilt peak above 60°. Two features distinguish the final protocol (plan v11) from the earlier runs (plans v7–v10):

- *full coverage*: 8 cases per task per generation, which covers all 8 dash directions, all 8 redirect pairs (both turn directions), all 4 evade directions and all 8 release strata every generation (the earlier 4-case batches used fixed direction slots — dash 0/45/90/135° and one turn direction only — and only strata 0–3, a defect found in plan v10);
- *crash-aware ranking*: any candidate with a crashed episode ranks below every crash-free candidate.

A generation costs 17 × 24 = 408 episodes (6 528 per seed). The population shares the pre-command prefix of each case through a bit-exact environment snapshot (`cache8`: a deep copy of the simulator state with the network state rebuilt by replaying the recorded inputs; verified to reproduce the plain episode bit for bit), so only the post-command part is simulated per candidate.

**Stage 3 — selection.** The imitation start and the centre of every odd generation (9 centres) fly a fixed validation set of 64 cases per task (192 cases); centres with any crash or a tilt peak ≥ 70° are excluded and the centre with the lowest objective is selected. The selected centre is frozen; it is the only network that is ever evaluated on a blind set.

### 2.5 Classical opponent and classical upper bound

**Opponent.** The strongest classical scheme with the same actuators, PID and base law: an open-loop 2-D set-point schedule per task and direction class — go set-point magnitude along φ, go duration, brake set-point magnitude (reversed) and brake duration, durations in 40 ms beats — started by the command, restarted on a direction change (redirect) and ended by the command drop; base-law pulses are suppressed while a segment is active. Each schedule was tuned on its own grid and tuning cases (evade and redirect: 24 configurations × 8 cases per task; dash: 108 configurations × 16 cases, inherited from plan v6), best configuration per class by mean score. No learned component is involved. The opponent also provides the class medians of the ratio and the normalisers of the ES objective.

**Upper-bound arm (plan v11, amendment 1).** To test whether the score leaves room above the opponent at all, the 28 schedule parameters (7 classes × 4) were optimised with exactly the ES budget of stage 2 — population 16, 16 generations, the same 24-episode batches with full coverage, crash-aware ranking — with σ = 0.05 and learning rate 0.02 in knob-normalised units (set-points / 0.5 rad, beats / 20), selected on the same 192 validation cases; four seeds on three machines.

### 2.6 Evaluation protocol

Every plan was preregistered: tasks, scoring, arms, blind-set seed base, number of cases and the pass criterion were committed to the repository before any blind episode (e.g. plan v11: commit `47d79a0`, before its first training episode; plan v12: `0726527`; plan v13 amendment 2: `abb78bc`), and amendments were committed before the episodes they cover; the one post-hoc amendment (`plan_v12/amendment_2_post_hoc.json`) demotes the plant-mismatch scenario from a primary to a secondary claim after its result was seen and is reported as such. Blind sets use seed bases never used before; each has 64 cases per task (192 episodes per arm) with the same direction and stratum coverage as the validation set. The classical opponent flies every blind case, so every comparison is paired. The reference machine (CloudPC-2, 32 vCPU) and a second machine (HomePC) produce identical episodes (0 differences in score, pulses and duration over 192 opponent episodes); a third machine (GoogleCloud) does not reproduce the reference bit for bit (floating-point differences amplified by the dynamics) and was used only for exploratory seeds that are reported separately and never counted in a criterion.

**Robustness scenarios (plans v12–v13; frozen controllers, no training).** The selected v11 centres (5 seeds), the v9 centres (5 seeds), the imitation start of seed 1, the opponent and two upper-bound centres were flown, unchanged, on fresh blind sets in which one element of the case is changed (Table R1 in Section 3):

- a 20 ms torque hit about a horizontal axis fixed in the heading frame (the platform's "upset" mechanism), at a random time during the manoeuvre (`t0 + U(0.15, 0.45) s`; also early `+0–0.15 s`, late `+0.5–1.0 s`, before the command `−0.6…−0.3 s`, a double hit 80–200 ms apart, and hits restricted to the roll or the pitch axis); magnitudes 2, 3 and 4.5 mN·m, where 3 mN·m was fixed before any learned-arm episode as the largest level at which the opponent crashes on ≤ 20 % of 24 calibration cases;
- the periodic disturbance pulses of the stress condition scaled ×1.25 and ×1.5 (×2 and ×3 were also run and lie outside every controller's envelope: all arms crash on nearly every case);
- sensor degradation: attitude and gyro noise ×5 (0.10°, 0.010 rad/s); sensor delay 3 steps (6 ms) instead of 1;
- plant mismatch: lower-body mass and inertia +12.5 % (the platform's built-in "ood" condition).

The preregistered criterion per scenario (H1) is: a v11 seed passes when its paired-Δ interval lies below 0 and its crashes do not exceed the opponent's; the scenario "holds" when ≥ 3 of 5 seeds pass. The 3 mN·m mid-manoeuvre hit is the primary scenario; the others were added by amendments before they ran and are reported descriptively with the same rule.

---

## 3. Results

### 3.1 Nominal manoeuvres: parity with the re-tuned classical schedule, and why the ratio criterion cannot be met

**Preregistered criterion.** Plans v7–v11 each asked, per seed, for a manoeuvre ratio ≤ 0.95 with no more crashes than the opponent, in at least 3 of 5 seeds. No plan met it (Fig. R0, `fig_r0_nominal.png`). Plan v11, the final protocol, gives 1.046 / 1.146 / 1.033 / 1.054 / 1.063 for seeds 1–5 (Table R0), with **0 crashes in 960 blind episodes**; the earlier plans gave 0.97–1.07 (v8, the v7 controllers re-tested on 64 cases per task), 1.02–1.13 with 3 crashes (v9) and 1.02–1.25 with 6 crashes (v10). The v7 "full" values of 0.89–1.00 were obtained on 32 cases per task on a blind set where the opponent itself crashed once; re-tested on 64 fresh cases the same five controllers scored 0.97–1.07.

**The criterion is structural, not a controller failure.** Because the ratio normalises each case by the opponent's class *median* and then averages, right-skewed score distributions push it above 1 for any controller, the opponent included: the opponent's own ratio is 1.089 on the v11 blind set (1.086 on v9, 1.273 on v10 where it crashed 3 times, 1.415 on v8 with 5 crashes). The classical upper-bound arm — the same 28-knob schedule optimised with the full ES budget — reaches 1.019 / 1.032 (CloudPC-2 / HomePC, bit-identical machines) and 0.999 / 1.018 (GoogleCloud, approximate) with 0 crashes, i.e. it does not reach 0.95 either although it is significantly better than the hand-tuned opponent (paired Δ −0.070 [−0.118, −0.029] and −0.056 [−0.100, −0.017]). An independent audit of the per-case rows put the ceiling of the metric at 0.87–0.89 even when the best of all ten-plus controllers is taken per case; in the dash task no controller ever meets the settle condition in the pitch and diagonal classes, so a constant 4 s term dominates both numerator and denominator there.

**Table R0 — plan v11 blind set (base 2 099 100 000, 64 cases per task; opponent 0 crashes).**

| arm | seed | machine | mean ratio | paired Δ [95 % CI] | wins /192 | crashes |
|---|---|---|---|---|---|---|
| classical opponent (own ratio) | – | CloudPC-2 | 1.089 | 0 | – | 0 |
| classical + ES (upper bound) | 1 / 2 | CloudPC-2 / HomePC | 1.019 / 1.032 | −0.070 [−0.118, −0.029] / −0.056 [−0.100, −0.017] | 111 / 115 | 0 / 0 |
| SNN + ES | 1 | CloudPC-2 | 1.046 | −0.043 [−0.091, +0.001] | 100 | 0 |
| SNN + ES | 2 | CloudPC-2 | 1.146 | +0.057 [+0.015, +0.096] | 70 | 0 |
| SNN + ES | 3 | CloudPC-2 | 1.033 | −0.056 [−0.108, −0.008] | 110 | 0 |
| SNN + ES | 4 | HomePC | 1.054 | −0.035 [−0.092, +0.018] | 116 | 0 |
| SNN + ES | 5 | HomePC | 1.063 | −0.026 [−0.079, +0.024] | 116 | 0 |
| SNN, imitation only | 1 / 3 / 4 / 5 | – | 1.166 / 1.137 / 1.238 / 1.257 | +0.077 … | 46–63 | 0 |
| SNN + ES, exploratory | 6–10 | GoogleCloud | 0.997 / 1.093 / 1.058 / 1.013 / 1.016 | (approx.; opponent's own ratio 1.097 there) | – | 0 |

Four of the five confirmatory seeds are at or slightly below the opponent (Δ −0.03 … −0.06; one significant), one is worse (seed 2, whose selection returned the imitation start because every ES centre crashed on the validation set); the five exploratory seeds behave the same way (mean 1.035, best 0.997, 0 crashes). ES improves every seed over its imitation start by 0.06–0.20. Within tasks the SNN is at or ahead of the opponent on evade and redirect and behind on dash, where the roll-direction class (90°/270°) remains at a ratio of 1.4–1.6 in every plan: the learned layer reaches the target 20–30 % sooner than the opponent but brakes later (overshoot 0.10–0.17 m vs 0.04 m).

**Training and selection behaviour (Fig. R1, `fig_r1_training.png`).** With full coverage and crash-aware ranking (plan v11) the ES centres stay crash-free on their 24-case batches, reach an objective of 0.90–0.97 by generation 7–9, and the selection objective decreases monotonically for seeds 3–5; in the earlier plans every seed began to crash on the validation set after 2–10 generations and only early centres were admissible. The classical + ES arm follows the same trajectory (0.96–0.98 on its batches, 0.975 on the validation set), which locates the plateau in the task and score rather than in the learned layer. A reference MLP + PPO arm trained on the same tasks (plan v8, 5 seeds) scored 1.109 ± 0.024 with 1 crash in 960 episodes.

### 3.2 Robustness: a closed-loop learned layer survives disturbances that an open-loop schedule does not

All controllers below are frozen; the opponent flies the same cases. Table R1 gives the crash counts (out of 192) on every blind set and the number of v11 seeds that pass H1; Fig. R2 (`fig_r2_crashes.png`) shows the same as rates, Fig. R3 (`fig_r3_paired_delta.png`) the paired Δ of every seed.

**Table R1 — crashes /192 on the blind robustness sets (14 arms; the five v9 centres and the opponent's own ratio are in `plan_v13/crash_table.md` and the summaries).**

| scenario | classical | classical + ES s1 / s2 | SNN + ES v11, seeds 1–5 | SNN imitation s1 | H1 (v11) |
|---|---|---|---|---|---|
| hit 2 mN·m, +0.15–0.45 s | 17 | 16 / 16 | 9 / 4 / 8 / 2 / 6 | 2 | **5/5** |
| **hit 3 mN·m, +0.15–0.45 s (primary)** | 60 | 56 / 58 | 40 / 32 / 34 / 22 / 40 | 27 | **4/5** |
| hit 4.5 mN·m | 86 | 87 / 89 | 95 / 71 / 67 / 46 / 76 | 68 | 1/5 |
| hit 3 mN·m, +0–0.15 s | 56 | 53 / 56 | 42 / 30 / 31 / 22 / 45 | 25 | **4/5** |
| hit 3 mN·m, +0.5–1.0 s | 40 | 39 / 33 | 12 / 5 / 14 / 9 / 16 | 5 | **5/5** |
| hit 3 mN·m, 0.3–0.6 s before the command | 7 | 3 / 6 | 4 / 2 / 1 / 1 / 2 | 0 | **3/5** |
| double hit 3 + 3 mN·m | 84 | 86 / 80 | 73 / 69 / 60 / 44 / 68 | 56 | 2/5 |
| hit about the roll axis only | 0 | 2 / 0 | 17 / 0 / 0 / 0 / 0 | 0 | **4/5** |
| hit about the pitch axis only | 74 | 71 / 71 | 46 / 48 / 47 / 36 / 51 | 43 | **3/5** |
| disturbance pulses ×1.25 | 10 | 5 / 8 | 2 / 0 / 1 / 2 / 0 | 0 | **5/5** |
| disturbance pulses ×1.5 | 40 | 36 / 38 | 7 / 5 / 9 / 8 / 9 | 8 | **5/5** |
| disturbance pulses ×2 / ×3 | 192 / 192 | 192 / 192 | 174–175 / 192 | 174 / 192 | outside envelope |
| IMU noise ×5 | 3 | 1 / 1 | 3 / 0 / 1 / 1 / 1 | 0 | 1/5 |
| sensor delay 3 steps | 9 | 11 / 12 | 2 / 0 / 4 / 4 / 4 | 0 | **4/5** |
| mass +12.5 % (secondary) | 0 | 0 / 0 | 0 / 0 / 0 / 0 / 0 | 0 | 0/5 |

**Primary scenario.** With a 3 mN·m hit landing 0.15–0.45 s into the manoeuvre, the opponent crashes in 60 of 192 cases (31 %), the classical + ES centres in 56–58, the v11 SNN centres in 22–40 and the imitation start in 27. The paired Δ of the five v11 seeds is −0.125 [−0.426, +0.209], −0.282 [−0.448, −0.121], −0.355 [−0.540, −0.186], −0.584 [−0.815, −0.377] and −0.263 [−0.473, −0.024]; **H1 holds (4 of 5)**. The advantage sits in dash (Δ −0.36 … −0.72; 3–9 crashes vs 20) and redirect (12–22 vs 24), not in evade.

**Dose–response (Fig. R4, `fig_r4_dose_response.png`).** At 2 mN·m the SNN crashes 2–9 times against 17 (all five seeds pass, Δ −0.57 … −0.89); at 4.5 mN·m the opponent crashes in 45 % of the cases, the SNN in 24–49 %, and only seed 4 remains significantly better (Δ −0.65 [−0.94, −0.40]; seed 1 is worse, +0.40). As stated in the amendment before the run, the advantage shrinks once the hit exhausts the learned layer's own tilt margin. The classical + ES centres track the opponent at every level (16 / 56–58 / 87–89), so re-optimising the open-loop schedule buys no robustness.

**Timing, axis, repetition.** The advantage is largest for late hits (0.5–1.0 s after the command, during braking and settling, while the learned layer is still in charge through its 1 s tail): 5–16 crashes against 40, all seeds pass. Early hits (0–0.15 s, during the initial lean) are the hardest for both: 56 vs 22–45. A hit before the command (−0.6 … −0.3 s), when every arm flies the same base law, produces bit-identical trajectories up to `t0` for all arms (192/192 cases) and the learned layer then handles the command-during-recovery better (1–4 crashes vs 7). Hits about the roll axis crash nobody (opponent 0; one SNN seed 17) whereas hits about the pitch axis crash the opponent in 74 cases and the SNN in 36–51: the learned layer's advantage lies on the axis its pulses can actuate. A second hit 80–200 ms after the first reduces the margin again (84 vs 44–73, 2 of 5 seeds pass).

**Sustained disturbance.** With the periodic disturbance pulses of the stress condition scaled ×1.5 the opponent crashes 40 times and the classical + ES centres 36–38, the v11 SNN centres 5–9 and the imitation start 8 (Δ −0.87 … −1.03; 5 of 5); at ×1.25, 10 vs 0–2. At ×2 and ×3 every controller is outside its envelope (174–192 of 192 crashes).

**Sensor degradation and plant mismatch.** Five-fold IMU noise does not harm either side (0–3 crashes; Δ −0.12 … −0.24, one seed significant). Tripling the sensor delay harms the opponent more than the learned layer (9 and 11–12 vs 0–4; Δ −0.33 … −0.45, 4 of 5), against the expectation stated before the run. With +12.5 % body mass nothing crashes and the SNN is slightly worse (Δ +0.02 … +0.07, two seeds significantly); this scenario was demoted to a secondary claim after its result was seen (Section 2.6) and is reported as a negative result: the learned layer has no advantage under plant mismatch.

**Older centres and the imitation start.** The plan-v9 centres (trained without full coverage and crash-aware ranking) span the whole range in the primary scenario (26–52 crashes; Δ −0.53 … +0.50), whereas the five v11 centres all lie below the opponent; the imitation-only network is, in most scenarios, as robust as the best ES centre (27 crashes in the primary scenario; 2 at 2 mN·m). Robustness therefore comes with the closed-loop layer itself, and ES fine-tuning trades some of it for speed unless crashes are ranked explicitly.

### 3.3 Mechanism: reaction through the pulse channel, not a smaller tilt margin

Re-flying the 192 primary-scenario cases with traces (Fig. R5, `fig_r5_mechanism.png`; `plan_v13/mechanism_summary.json`) separates two explanations. *Conservatism* would mean the learned layer flies at a smaller tilt when the hit lands; it does not: the mean tilt at the hit is 14.6° for the opponent and 13.9–16.9° for the SNN seeds (seed 4, the most robust, +2.4° [+1.7, +3.0]). At matched tilt-at-hit the SNN crashes far less — 11–17 % against 31 % below 20°, 12–27 % against 31 % at 20–35° — so the difference arises after the hit. *Reaction* is visible in the pulse channel: in the 150 ms after the hit the SNN seeds fire 1.1–3.6 more motor pulses than in the 150 ms before (seed 4: +3.58 [+3.0, +4.2]), the opponent +0.43 (its base-law pulses are suppressed while a schedule segment is active); the tilt peak within 0.5 s after the hit is 2–6° lower for three of the four learned arms. The set-point channel changes more than the opponent's (|Δref| 0.20–0.39 rad vs 0.25 rad over 150 ms) but not consistently against the induced tilt (32–51 % of cases), so the set-points are not the main route. Together with the axis experiment (Section 3.2) this gives a consistent picture: the learned layer reacts to the IMU within one or two wingbeats through pitch-torque pulses, which is exactly the authority it has, and that is where its robustness comes from. Fig. R6 (`fig_r6_kick_frame.png`) shows one case per task in which the opponent crashes and the v11 seed-4 centre completes the manoeuvre.

### 3.4 Limitations

All results are from one simulated vehicle; the aerodynamic model is bench-calibrated but the disturbance and sensor models are idealised. The learned pulses give pitch torque only; roll manoeuvres go through set-points and the PID, and the dash roll class remains the weakest cell. The ratio criterion of the preregistrations turned out to be unreachable for any controller and the comparison had to move to a paired statistic defined in an amendment (before the v11 results, after the v7–v10 ones). The plant-mismatch scenario was demoted after its result was seen. Five of the ten SNN seeds and two of the four upper-bound seeds come from a machine that does not reproduce the reference bit for bit and are reported separately. The robustness scenarios were run on frozen controllers that never saw a hit or scaled disturbance during training; this is deliberate, but it means the scenarios test generalisation rather than a trained capability.
