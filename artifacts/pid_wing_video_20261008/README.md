# Why the wings look stationary in the original PID video

The frozen first nominal flight (seed 310900000) is unchanged. In the 5–6 s
segment, carrier rotation measures **25.00868 cycles/s**, and both wing flap
joint peak counts measure about **25 Hz**. Each flap joint sweeps approximately
**−79.33° to +79.33°**. The wings are moving.

The original video sampled the scene at **25 frames/s**. This nearly matches one
complete wing beat per frame, so successive pictures show almost the same wing
phase. The average phase advance relative to one complete cycle is only about
0.125° per original frame in this segment. This is temporal aliasing (a strobe
effect), not an inactive wing actuator.

[Paired close-up video](wings_sampling_comparison.mp4):

- Left: sample the same simulation at 250 Hz.
- Right: retain the original 25 Hz sampling and hold each sampled image until
  the next original sample.
- Both panels play at **10× slow motion**, so one second of simulation becomes
  ten seconds of playback, followed by a one-second final hold.
- The camera follows the body for this wing-motion diagnostic. It should not be
  used to judge height or horizontal drift; those remain in the original flight
  report and videos.

No interpolated wing motion, altered actuator command or changed physics is
used. All 3,000 flight states and geometric audit states through 6 s match the
frozen recorded trajectory to 1e-12. Raw 500 Hz joint positions are retained in
`joint_trace.npz`; measurements and column names are in `verification.json`.

Replay from the repository root into a fresh output directory:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
python -m experiments.algorithms.pid_airborne.wing_motion \
  --root artifacts/pid_airborne_20261008 \
  --output /tmp/pid-wing-motion-replay
```
