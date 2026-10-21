# A spiking descending layer for flapping-wing manoeuvres: method and results

*Draft of the Method and Results sections (no Introduction / Related work). Every number is traceable to `artifacts/spike_steer/plan_v*/` in the repository; figure files are in `docs/paper/figs/` and `docs/figs/`.*

---

## 2. Method

### 2.1 Platform and interface

**Vehicle and simulator.** The vehicle is the IST_robot-V2 flapping-wing micro air vehicle: 15 g, a single motor driving two wings at about 26.7 Hz (37 ms wingbeat), with the IMU on the lower body. It is simulated in MuJoCo 3.8.0 as a multibody system with blade-element aerodynamics whose lift is calibrated against bench data. Control and simulation both run at 2 ms (500 Hz).

**Interface (unchanged throughout).** The controller receives an 11-dimensional sensor vector `[roll, pitch, yaw, gx, gy, gz, ax, ay, az, z, vz]` at 500 Hz with noise; `z` comes from a time-of-flight range finder (50 Hz, 20 ms latency) and `vz` from an α-β observer. It commands two gimbal angles α, β ∈ ±0.3 rad and the motor speed ω ∈ [3115, 3807] rad/s, updated every 2 ms and passed through an 8.2 ms first-order actuator lag. Hover needs about 3140 rad/s, only ≈25 rad/s above the lower limit, so downward thrust authority is small. No sensor, actuator, update rate or limit was changed in any experiment; new test scenarios only change the cases.

**Stress condition.** All training, validation and blind cases are "stress" cases: initial attitude error ±5°, motor gain uncertainty ±1.5 %, accelerometer and attitude biases, one control step (2 ms) of sensor delay, attitude/gyro/accelerometer noise, and periodic disturbance force and torque pulses (0.006 N, 8·10⁻⁶ N·m). Each case is one random seed; 8 strata fix the wingbeat phase at the command onset.

### 2.2 Manoeuvre tasks and scoring

Three manoeuvres are commanded at `t0` after a 2.2–2.8 s hover (Fig. M-tasks, `docs/figs/fig10_directions.png`). The command is a unit vector in the heading frame on two "channels" (evade; dash/redirect), presented to the learned layer as 8 rectified inputs.

| task | command | goal | window | direction classes |
|---|---|---|---|---|
| **evade** | displace 0.10 m along a direction | reach 0.10 m as fast as possible | 1.5 s | 4 directions (pitch / roll class) |
| **redirect** | go along φ₁; when 0.08 m or 1.0 s is reached, the command switches to φ₂ = φ₁ ± 90° | reach 0.10 m along φ₂ after the switch, with little overrun along φ₁ | 4 s | 4 × 2 turn directions (pitch / roll) |
| **dash** | displace 0.15 m along one of 8 directions and stop | reach, then settle (|v| < 0.1 m/s for 0.3 s), little overshoot | 4 s | 8 directions (pitch / roll / diagonal) |

The per-case score (lower is better) sums the time to reach the goal (or the window length plus 10 m⁻¹ × shortfall if not reached), task-specific penalties (dash: settle time or the 4 s window if never settled, 2 × overshoot beyond 0.1 m, 0.5 × lateral drift; redirect: 2 × overrun beyond 0.05 m; evade: 0.5 × lateral error), 0.03 per cm of the largest altitude deviation from the altitude at `t0` over the rest of the flight (symmetric, plan v9 onwards), and a crash term of 10 + 5 × (unflown fraction of the window). A **crash** is any early termination: tilt > 80.2°, |Δz| > 1 m, z < 0.12 m, or a numerical failure.

**Ratio.** For comparability across tasks and directions, each case score is divided by the median score of the classical opponent over the blind cases of the same direction class; the task ratio is the mean over cases and the *manoeuvre ratio* is the mean over the three tasks. Section 3.1 shows that this ratio is > 1 for the opponent itself, so the main comparisons use the **paired difference** Δ = (arm ratio − opponent's own ratio) on identical cases, with a paired bootstrap 95 % interval over cases (4000 resamples), the paired win count, and crash counts with exact Clopper–Pearson 95 % intervals.

### 2.3 Hierarchical controller

The controller has three layers (Fig. M-arch, `docs/figs/fig1_architecture.png`; one 2 ms tick in `docs/figs/fig3_gating.png`).

**Reflex layer — PID.** A PID attitude loop tracks roll/pitch set-points with the gimbal and a PID altitude loop sets the motor speed. Gains were re-tuned on stress hover (kp × 1.5, kd × 0.85, ki × 1.5 relative to the original baseline); a full-state-feedback upper bound improves the hover tilt RMS by only 1.1°, so no learned component touches hovering.

**Steering-muscle layer — phase-locked motor pulses.** A pulse sets the motor command to the maximum (3807 rad/s) for three ticks (6 ms). After the 8.2 ms actuator lag it raises the speed by about 690 rad/s for part of a wingbeat; depending on the wingbeat phase the asymmetric lift produces a nose-up or nose-down **pitch torque**. Outside a manoeuvre command, pulses are issued by a classical pulse law (a PLL wingbeat-phase estimator with threshold 0.5 and a 45° firing window, gains k_θ = 8.0, k_q = 0.4; the same law is the base law of every arm).

**Descending layer — the spiking network.** A leaky integrate-and-fire (LIF) network with axonal delays (`descend8.Net8`; Fig. M-snn, `docs/figs/fig2_snn.png`):

- input, 25-D per tick: eight IMU channels (roll, pitch, three gyro rates, three accelerations) each split into rectified +/− channels (16), a bias (1), and the 8 rectified command channels;
- 11 input delay taps of 0–20 ms (2 ms spacing): `W_in ∈ ℝ^{8×25×11}`;
- 8 hidden LIF neurons, two sub-steps per tick, threshold 1, reset by subtraction; the membrane and synaptic time constants are learned during imitation (selected centres: τ_m ≈ 2.1 ms, τ_s ≈ 1.2 ms);
- hidden → output through 11 axonal delays of 0–20 ms: `W_out ∈ ℝ^{2×8×11}` to two spiking output neurons and `W_g ∈ ℝ^{2×8×11}` to two graded units;
- outputs: a spike of either output neuron triggers one motor pulse ("up"/"down" are two routes to the same twitch, the timing carries the meaning); the two graded units (τ_g = 20 ms, tanh × 0.7 rad) are the **roll and pitch set-points** handed to the PID;
- 2564 parameters in total (`W_in`, `b_h`, `W_out`, `b_o`, `W_g`, `b_g`).

**Command gate.** The learned layer is in charge only while a command is on and for a 1 s tail after it ends; outside the gate the classical base law and zero set-points act, so the arms are identical tick for tick before `t0` (verified bit-exactly in Section 3.3). Two safety rules apply while the learned layer is in charge: a refractory period of 37 ms between pulse onsets, and no righting reflex (the 25° righting reflex of the wrapper is active only outside the gate). The gate opens no earlier than 1.5 s after take-off.

### 2.4 Training

The pipeline (Fig. M-pipe, `docs/figs/fig4_pipeline.png`) has three stages; the controller that is tested is the frozen output of stage 3.

**Stage 1 — imitation of the classical schedule (behavioural cloning through surrogate gradients).** The classical opponent (Section 2.5) flies 16 training cases per task; its pulses, their sign, and its set-points are recorded together with the 25-D input at every tick. A torch twin of the network (identical tick arithmetic) is trained by back-propagation through time with a sigmoid surrogate (β = 0.25) on windows of 200 ticks (stride 100, batch 32): loss = mean[(1 + 2y) · BCE(p, y)] + 5 · mean[(ref/0.7 − r)²], Adam 3·10⁻³, gradient norm clipped to 1, 40 epochs, best epoch by validation loss on 4 held-out cases per task. Five seeds give five imitation starts (`snn_bc`).

**Stage 2 — OpenAI-ES on the manoeuvre score.** Starting from the imitation weights θ₀, each generation draws 8 antithetic pairs (population 16) of perturbations ε ~ N(0, I), evaluates θ + σε on a batch of cases, and updates θ with rank-based utilities (Adam-style normalisation of the gradient estimate, σ = 0.02, learning rate 0.01 decayed by 0.85 per generation, 16 generations). The objective of a candidate is the mean of its case scores divided by the classical class means of the training cases (plus a soft tilt penalty of 0.3 per degree above 60°), averaged over tasks. Two features distinguish the final protocol (plan v11) from the earlier runs (plans v7–v10):

- *full coverage*: 8 cases per task per generation, which covers all 8 dash directions, all 8 redirect direction pairs (both turn directions), all 4 evade directions and all 8 wingbeat-phase strata every generation (the earlier 4-case batches covered only half of the directions and one turn direction, a defect found in plan v10);
- *crash-aware ranking*: any candidate with a crashed episode ranks below every crash-free candidate.

A candidate evaluation costs 24 episodes; the population shares the pre-command prefix of each case through a bit-exact environment snapshot (`cache8`), so only the post-command part is simulated per candidate.

**Stage 3 — selection.** The imitation start and the centre of every odd generation (9 centres) fly a fixed validation set of 64 cases per task (192 cases, disjoint from training and blind seeds); centres with any crash or a tilt peak ≥ 70° are excluded and the centre with the lowest objective is selected. The selected centre is frozen; it is the only network that is ever evaluated on a blind set.

### 2.5 Classical opponent and classical upper bound

**Opponent.** The strongest classical scheme with the same actuators and PID: the base pulse law plus an open-loop 2-D set-point schedule per task and direction class (go set-point magnitude, go duration in wingbeats, brake magnitude and duration; gated by the command, restarted on a direction change). Each of the 7 (task, class) schedules was tuned on its own grid (24–180 configurations) on 8 tuning cases; no learned component is involved. It also provides the class medians of the ratio.

**Upper-bound arm (plan v11 amendment 1).** To test whether the score itself leaves room above the opponent, the 28 schedule parameters were optimised with exactly the ES budget of stage 2 (population 16, 16 generations, the same 24-episode batches, crash-aware ranking, σ = 0.05 and learning rate 0.02 in knob-normalised units) and selected on the same 192 validation cases; four seeds on three machines.

### 2.6 Evaluation protocol

Every plan was preregistered: tasks, scoring, arms, blind-set seed base, number of cases and the pass criterion were committed to the repository before any blind episode; amendments were committed before the episodes they cover (one post-hoc amendment, which only re-labels a scenario, is recorded as such). Blind sets use seed bases never used before; each has 64 cases per task (192 episodes per arm) with the same direction and stratum coverage as the validation set. The classical opponent flies every blind case, so every comparison is paired. The reference machine (CloudPC-2, 32 vCPU) and a second machine (HomePC) produce bit-identical episodes (0 differences in 192 opponent episodes); a third machine (GoogleCloud) differs in compiler and is used only for exploratory seeds, labelled as approximate.

**Robustness scenarios (plans v12–v13, frozen controllers, no training).** The selected v11 centres (5 seeds), the v9 centres (5 seeds), the v11 imitation start, the opponent and the upper-bound centres were flown, unchanged, on fresh blind sets in which one element of the case is changed (Table R-tests):

- a 20 ms torque hit about a horizontal axis fixed in the heading frame (the "upset" mechanism of the platform), at a random time during the manoeuvre (t0 + U(0.15, 0.45) s; also early 0–0.15 s, late 0.5–1.0 s, before the command −0.6…−0.3 s, a double hit 80–200 ms apart, and hits restricted to the roll or the pitch axis); magnitudes 2, 3 and 4.5 mN·m, where 3 mN·m was fixed before any learned-arm episode as the largest level at which the opponent crashes on ≤ 20 % of 24 calibration cases;
- sustained turbulence: the stress disturbance pulses scaled ×1.25 and ×1.5 (×2 and ×3 were also run and are outside every controller's envelope: all arms crash on nearly every case);
- sensor degradation: attitude and gyro noise ×5; sensor delay 3 steps instead of 1;
- plant mismatch: lower-body mass +12.5 % (the platform's built-in "ood" condition).

The preregistered criterion per scenario (H1) is: a v11 seed passes when its paired-Δ interval lies below 0 and its crashes do not exceed the opponent's; the scenario "holds" when ≥ 3 of 5 seeds pass. The 3 mN·m mid-manoeuvre hit is the primary scenario; the others were added by amendments before they ran.

---

## 3. Results

*(to be written after the last exploratory seeds finish; figures `fig_r0`–`fig_r6` are final)*
