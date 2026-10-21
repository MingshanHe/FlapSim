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

*(written after the last exploratory seeds finish; figures `fig_r0`–`fig_r6` are final)*
