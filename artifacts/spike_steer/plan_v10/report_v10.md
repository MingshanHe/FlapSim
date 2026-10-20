# Plan v10 report (2026-10-21, one autonomous iteration)

Preregistration: `preregistration_v10.json` (commit b6f2cf2, written before any v10 training episode).

## Result: M5 does not hold (0/5)

Blind set 2_098_900_000, 64 cases per task. The classical opponent crashed 3 times in 192 episodes (all evade, tilt).

| seed | machine | selected gen | v10 SNN+ES | v10 imitation | v9 centre frozen (same cases) |
|---|---|---|---|---|---|
| 1 | CloudPC-2 | 5 | 1.215 (4 crashes, dash-pitch tilt) | 1.167 (0) | 1.065 (0) |
| 2 | CloudPC-2 | 3 | 1.095 (0) | 1.109 (0) | 1.047 (1) |
| 3 | CloudPC-2 | −1 (imitation start) | 1.100 (0) | = SNN+ES | 1.083 (0) |
| 4 | HomePC | 1 | 1.254 (2, evade-pitch tilt) | 1.184 (0) | 1.045 (0) |
| 5 | HomePC | 2 | 1.020 (0) | 1.253 (1) | 1.128 (2) |
| mean | | | **1.137 (6 crashes)** | | **1.074 (3 crashes)** |

Secondary S1 is the paired comparison on identical cases. The 37-input layer (phasic command copies plus height channels) did **not** improve on v9; it is worse by about 0.06. Imitation fitted better: validation loss fell from 0.143–0.158 (v7) to 0.120–0.129. That gain did not survive ES.

## What v9 + v10 together show

1. **Height escapes are solved by the v9 symmetric altitude term.** There were 0 height escapes in 1,920 v9/v10 blind SNN episodes. The remaining crashes are tilt.
2. **Missing inputs are not the bottleneck.** v10 gave the layer time-since-command and height, and the blind ratio got worse.
3. **ES training cases cover only half of the task space.** This is a defect found during this iteration.
   - `bench9.cases9` assigns directions by index, and the per-generation offset changes only the seed.
   - In every generation, ES dash trains on directions 0°, 45°, 90° and 135°. Directions 180–315° are never trained.
   - Redirect trains only one turn handedness: 0→90, 90→180, 180→270 and 270→0.
   - Blind and selection sets cover all directions and both turn handednesses.
4. **The ES objective is too noisy.** Each candidate flies only 12 episodes per generation, with one roll-class dash among them. Gains on the training batch (redirect 0.73–0.86 in v10) do not transfer to the 32-case selection set (redirect at or above 1.2 on seed 4).
5. **ES becomes unstable late.** Every seed on every machine — v9 seeds 1–10 and v10 seeds 1–5 — starts crashing on the selection set after generations 2–10. Selection can therefore only pick early centres, and sometimes the imitation start.
6. **Selection is optimistic.** Selection objectives of 0.96–1.02 became blind ratios of 1.02–1.13. Separately, centres that were crash-free on 96 selection cases still crashed on the blind set: 4 in 64 dash cases for v10 seed 1, 2 in 64 evade cases for v10 seed 4.

## Proposed v11 (needs the user's approval; not run)

Keep the network and scoring of v9 (25 inputs, bench10). Change only the training and selection of the learned layer:

- **Full coverage in every ES generation:** all 8 dash directions, all 8 redirect direction pairs and all 4 evade directions. That is 20 cases per candidate, up from 12, using common random numbers across the population.
- **Crash-aware ES:** a crashed candidate ranks last regardless of score. The learning rate decays as lr·0.85^gen, and the run stops early when the centre crashes twice in a row.
- **Selection on 64 cases per task,** with all directions and both handednesses, and 0 crashes required.
- **A fresh blind set,** 64 cases per task, with the same criterion as M4/M5.
- **Compute:** about 1.7 times v9 per generation. GoogleCloud can take the exploratory seeds.
