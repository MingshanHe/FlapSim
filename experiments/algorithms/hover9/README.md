# IMU-history residual-control experiment

The sensor source is nine channels at `base_link`: Euler angles, gyro and specific force. A causal virtual-IMU frontend adds estimated relative position and velocity, making **15 actor features**. It assumes ideal high-rate orientation before decimation; it is not a validated hardware IMU model.

Both learners use the same frozen six-input NEAT attitude prior, actuator correction map and history estimator. This is a residual-control comparison. The PID screening did not produce a stable inner loop, so the experiment does not claim successful PID-inner / learned-outer control.

The published report and videos are in `artifacts/hover9_pid_20261008/`. Never rerun setup, protocol generation or PID preparation in that published directory. Training and final testing reject existing result directories.

## Reproduce the learning comparison

Use the repository root and the dependency versions in the published `training_manifest.json`. MuJoCo, neat-python, stable-baselines3, PyTorch, NumPy, Gymnasium and a C++17 compiler are required. Rendering also needs imageio/ffmpeg, DejaVu Sans and an EGL-capable installation. Limit numerical-library threads to one per process.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.algorithms.hover9.setup --root runs/history_repeat
python -m experiments.algorithms.hover9.train --root runs/history_repeat --method neat --seed 941 --workers 8
python -m experiments.algorithms.hover9.train --root runs/history_repeat --method ppo --seed 941 --workers 8
python -m experiments.algorithms.hover9.train --root runs/history_repeat --method neat --seed 942 --workers 8
python -m experiments.algorithms.hover9.train --root runs/history_repeat --method ppo --seed 942 --workers 8
python -m experiments.algorithms.hover9.results --root runs/history_repeat
MUJOCO_GL=egl python -m experiments.algorithms.hover9.results --root runs/history_repeat --video-label neat_seed941
```

Each PPO run reads its completed NEAT counterpart and rounds the actual transition count to a full rollout. Validation selects checkpoints; final testing freezes all four models before using any held-out cases. The common zero-residual prior is also tested. `zero_history` ablation zeros only the six explicit position/velocity estimates; raw acceleration and the NEAT networks' recurrent state remain available.

The report generator for the published full round additionally reads the PID development records, release diagnostic and verification files. A newly trained comparison alone is not a reproduction of those additional development searches.

## Replay a frozen video from another checkout path

```bash
MUJOCO_GL=egl python -m experiments.algorithms.hover9.replay \
  --root artifacts/hover9_pid_20261008 --label neat_seed941 \
  --video /tmp/neat941-history-replay.mp4
```

Replay relocates only the mesh directory in a temporary model, preserves the original files, and checks termination time and position RMSE against the recorded episode. Videos show the first predeclared test case, in 4x slow motion, with a one-second final hold and a fixed-origin trajectory panel.

## Relevant tests

```bash
python -m pytest --import-mode=importlib \
  tests/test_neat_residual.py tests/test_neat_activation.py \
  tests/test_neat_hover_initial.py tests/test_neat_mass_model.py \
  tests/test_neat_ppo_compare.py tests/test_imu_history_probe.py \
  experiments/algorithms/hover9/test_history.py -q
```

The importlib mode avoids the repository's `experiments/algorithms/neat` directory shadowing the installed `neat-python` package during test discovery.
