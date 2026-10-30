# Cost of running the controllers and pulse phase (HomePC, 2026-10-25)
Scripts: experiments/algorithms/spike_steer/compute_cost.py, energy.py, pulse_phase.py. Figure: docs/paper/figs/fig_r9_cost_phase.png (make_results_figs.fig_cost).
Cases: plan-v15 nominal stress cases, 16 per task (48 episodes per arm); pulse phase also on the v12 primary kick cases.

1. Compute (compute_cost.json), shared PID / phase law / supervisor not counted.
   SNN11 has 2564 parameters (8 LIF hidden, 2 LIF out, 2 graded; 25 inputs x 11 delay taps). PPO actor 57-64-64-3 has 8067.
   SNN dense: 1.12 M MAC/s + 0.35 M AC/s (5.5 µW at 45 nm, Horowitz). Event-driven with zero skipping (37 % of input taps non-zero; hidden
   70-114 Hz per neuron): 0.43 M MAC/s + 0.02-0.04 M AC/s (2.0 µW). PPO, 100 decisions/s: 0.79 M MAC/s (3.7 µW).
   The input FIR layer dominates the SNN cost; the hidden layer is not sparse (70-114 Hz).
2. Energy (energy/energy.json, paired_ci.txt), from the motor's V·I sampled every 0.1 ms. Chunked stepping is bit-identical (selftest), and the sampling error is 0.0075 %.
   Every arm draws a mean of 6.57-6.61 W; the classical arm uses 16.3 J in the 2.5 s window. Paired difference vs classical: SNN −0.8…+0.9 %, PPO −0.4…−0.7 %,
   cls_es within ±0.1 %. Pulses cost no measurable energy. Compute power is about 1e-6 of motor power.
3. Pulse phase (pulse_phase/onsets.json, conditional.txt). 94-98 % of the SNN's own pulse onsets fall within ±45° of the classical windows
   (135° up, 315° down); uniform timing would give 50 %. The net path has no phase gate, so this phase locking is emergent.
   Window choice vs the wingbeat-averaged pitch rate: when q̄ < 0, 135° is chosen 61-75 % of the time (classical law about 55 %); when q̄ > 0, 315° is chosen 43-68 %. This is a modest
   but consistent bias. The instantaneous gyro sign gives about 95 % agreement, but that comes from wingbeat rocking (the raw rate's sign is tied to crank phase); do not report it.
