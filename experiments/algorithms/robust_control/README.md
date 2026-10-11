# Robust control study v3

See [research_proposal_v3.md](../../../research_proposal_v3.md) for the frozen-design rationale and [v2.1 failure analysis](../../../artifacts/snn_pid_compare_20261010/failure_analysis.md) for evidence behind it. This is an independent study; the earlier failed campaign and all historical sources remain unchanged.

Runtime: `runs/robust_control/20261010_v3`. Report: `artifacts/robust_control_20261010/report.md`.

The user-authorized [October 12 interim delivery](../../../artifacts/robust_control_20261010/interim_plan_20261012.md)
adds a separate holdout and brings predeclared fixed-topology controls forward.
It drains active whole runs, compares the first three paired seeds for all four
methods against PID, evaluates controls and additional SNN-NEAT seeds, and then
automatically resumes the original queue. It changes scheduling only, not the
sealed training implementation, budgets, checkpoint selection, or final tests.
The [interim report](../../../artifacts/robust_control_20261010/interim_20261012/report.md)
updates after each model; runtime case progress is in
`runs/robust_control/20261010_v3/interim_20261012/evaluation_progress.json`.

```bash
bash experiments/algorithms/snn_pid_compare/run.sh -m experiments.algorithms.robust_control.interim \
  --root runs/robust_control/20261010_v3 \
  --destination artifacts/robust_control_20261010
```

This launcher requires the owned `pause_request.json` handoff described in the
amendment; do not start an independent original queue during the handoff.

The 65-run queue contains five seeds for each direct/residual ANN/SNN × PPO/NEAT hover arm, five fixed-topology SNN evolution controls, and twenty residual velocity runs. Each hover run allocates 3,584,000 original 500 Hz action steps; velocity allocates 5,120,000, with longer episodes covering the whole command sequence. Allocated and consumed steps are reported separately. This is a multi-day study, not a completed result.

```bash
bash experiments/algorithms/snn_pid_compare/run.sh -m pytest -q tests/test_robust_control.py
bash experiments/algorithms/snn_pid_compare/run.sh -m experiments.algorithms.robust_control.campaign \
  --root runs/robust_control/20261010_v3
```

The same command resumes the queue. Existing completed runs are skipped. NEAT resumes at a generation boundary. PPO explicitly restores its actor, critic, optimizer and RNG while restarting unfinished simulator episodes and neural state; this event is recorded, not represented as exact simulator-state continuation. `pause_request.json` stops admission of new jobs after active whole runs finish. Do not use old campaign pause/resume commands for this run.

Preparation requires twelve successful trainer smoke checks, spanning both tasks and both control modes. The launcher seals both packages, assets and initial policies; every training child runs from `frozen_source/`. Editing the working tree does not change the running experiment. A source hash mismatch in that sealed directory stops a resume.

Original Python/PyTorch float64 temporal computation is retained. Two jobs with eight simulator workers each share the CPU. Final evaluation uses sixteen workers. No GPU or remote machine is assumed. Final tests run only after the full declared queue completes and policy hashes are sealed.

Hover has the unchanged eleven observation meanings. Velocity is a separate sixteen-input task with explicit command inputs and an additional noisy delayed horizontal velocity sensor assumption. See `environment.py`, `protocol.json`, and `artifacts/robust_control_20261010/technical_details.md` for exact clocks, sensors, action limits and metrics.

Diagnostics and direction identification are development data. Earlier velocity attempts are retained in separate directories; they are not independent test results. The final held-out sets must not be used to adjust gains or choose checkpoints.

Reports are generated after each completed run and every completed final-model evaluation, including failures. All work remains on the local branch; no GitHub push is attempted.
