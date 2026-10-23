# PID imitation and matched SNN/ANN control study

The v2.1 development run finished without meeting its imitation gate. The new
authorized research queue is [robust_control v3](../robust_control/README.md),
with live status in `runs/robust_control/20261010_v3/status.json`. Keep this
campaign's failed-gate records as evidence; do not resume its gated formal queue
to launch the new study.

This implements the approved repository-root `research_proposal.md` v2.1 in a
separate 500 Hz environment. Historical `snn_hover` and `topology_hover` sources
remain frozen. `ppo_v2` policies and demonstrations are not used.

## Environment

The current host uses a Conda-format prefix at
`/home/hemingshan/FlapSim/.venv`. It was installed without administrator access
and can be managed by the user's Miniconda:

```bash
source /home/hemingshan/miniconda3/etc/profile.d/conda.sh
conda activate /home/hemingshan/FlapSim/.venv
python -m pip check
```

Python 3.11.11, MuJoCo 3.8.0, NumPy 2.2.4, PyTorch 2.6.0 CPU and NEAT-Python
2.0.0 are installed. The C++ compiler lives in this prefix; no global compiler
or system Python is required. Small recurrent networks currently train on CPU.
The full 5 µs physical simulation also runs on CPU.

At the user's request, all temporal training uses the original per-step Python
and PyTorch autograd implementation in float64. No custom sequence kernel is
enabled. Independent models and simulations run concurrently.

For a new host, use an unused prefix, then install the pinned requirements:

```bash
conda create -p .venv -c conda-forge python=3.11.11 pip gxx_linux-64=13
conda activate ./.venv
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/algorithms/snn_pid_compare/requirements.txt
python -m pip install --no-deps -e .
```

The frozen native builders invoke `g++`. If the Conda compiler is only available
as `x86_64-conda-linux-gnu-g++`, add a `g++` symlink inside the new prefix's `bin`
directory. The current prefix already has this link. `run.sh` sets the prefix
path, one-thread BLAS/OpenMP and headless MuJoCo settings for every child process.

The new run copies PID assets and changes only the XML `compiler.meshdir` to the
current repository's meshes. `source/provenance.json` records both hashes and
the relocation; no old XML or physical parameter is rewritten.

## Active run and progress

Run root: `runs/snn_pid_compare/20261010_v21`.
Persistent tmux session: `flapsim-pid-v21`.

```bash
tmux attach -t flapsim-pid-v21
cat runs/snn_pid_compare/20261010_v21/status.json
tail -f runs/snn_pid_compare/20261010_v21/development.log
tail -f runs/snn_pid_compare/20261010_v21/development/round0/snn_seed5101/train.log
```

Detach tmux with `Ctrl-b`, then `d`. Closing an attached terminal does not stop
the session. Suspending or shutting down the WSL host does interrupt computation.

The launched command is:

```bash
bash experiments/algorithms/snn_pid_compare/run.sh \
  -m experiments.algorithms.snn_pid_compare.supervise \
  --root runs/snn_pid_compare/20261010_v21 --workers 16
```

The run has 32 complete 20-second PID demonstrations (320,000 samples) and eight
separate offline-validation episodes (80,000 samples). Input and teacher labels
are aligned before each action. The teacher is queried once per tick and keeps
independent state. Only successful teacher episodes enter the initial dataset.

The finite sequence is:

1. Train SNN and ANN seeds 5101–5103 concurrently on the same data. Each
   model receives 30 complete dataset passes, batches of eight episodes and
   one-second truncated BPTT windows with state carried between windows.
2. Select among epochs 0/10/20/30 on 16 shared 20-second validation cases. Check
   four 60-second cases, paired PID error differences and 16 stress cases.
   Offline action RMSE is diagnostic; the advancement gate uses closed-loop
   success and paired control errors.
3. If necessary, collect at most two shared DAgger batches, each capped at 12
   student-controlled episodes of 20 seconds, then repeat 30 training passes
   from each model's selected checkpoint. These batches are training data,
   including causal teacher labels on failed student trajectories.
4. If every development model passes, freeze the formal source/protocol, create
   paired initializations for seeds 5201–5205 and execute the eight groups.
   If the bounded development phase still fails, record
   `development_gate_not_met` and stop before formal training.
5. Only after all 40 formal training runs finish, evaluate the frozen policies
   and PID/BC baselines on common final cases, then write `analysis.json`.
   The supervisor produces `artifacts/snn_pid_compare_20261010/report.md` and
   `technical_details.md` after completion or an unmet development gate. On an
   ordinary child-process failure it produces an explicitly incomplete report.
   Historical 2201 results remain a separately labelled reference.

The formal queue runs at most two experiments concurrently, each with eight
PPO environments or NEAT workers. Standalone physical evaluations use up to
sixteen workers. A 32-case PID benchmark measured 498 action steps/s with eight
workers and 785 with sixteen; this single batch includes startup and is not an
end-to-end training speedup. Six imitation jobs cannot independently occupy
sixteen CPU threads; the initial PID baseline is overlapped where possible.

The formal budget candidate is 16 individuals × 24 generations × four cases,
with eight generations each at 2/4/8 seconds: 3,584,000 allocated 500 Hz action
steps per run. PPO receives the same allocation with eight environments and
500-step rollouts/BPTT windows. Early failures reduce NEAT's actual consumed
steps; allocation and actual interaction are reported separately. Formal
configuration is sealed only after development; none of the 40 formal runs
has been represented as already complete.

## Records and recovery

`deployment.json`, `requirements.lock.txt`, `compiler.txt` and `preflight.json`
record the installed environment. `development_protocol.json` fixes development
budgets and thresholds. `events.jsonl` is the stage log; `status.json` is the
latest stage plus a roughly ten-second heartbeat, CPU usage and model progress.
Each model has its own log, checkpoints, metrics and exported
JSON policy.

Imitation resumes at completed epoch checkpoints; NEAT resumes at generation
checkpoints. Completed runs are skipped. PPO checkpoints contain optimizer and
network state, but not a complete live simulator snapshot: the runner refuses
to silently restart an incomplete PPO run. Its directory is preserved for an
explicit recovery decision. This is not a promise of exact mid-episode resume.

The campaign takes a file lock to prevent duplicate launches. A
`pause_request.json` in the new run root stops the formal queue at the next
whole-run boundary. Do not remove old campaign pause markers or invoke its
`--resume` command to control this new study.

## Validation

```bash
bash experiments/algorithms/snn_pid_compare/run.sh \
  -m pytest -q tests/test_snn_pid_compare.py
```

Checks cover physical PID equivalence against the old inner loop, idempotent
teacher queries, actuator authority, zero-residual equivalence, continuous
Torch/NumPy parity, SNN temporal gradients, exact NEAT actor import, fixed time
constants, time-limit bootstrapping and budget alignment. The initial eight
short trainer runs are recorded in `training_smoke.json`; they validate execution
and are not scientific performance measurements.

## Local report delivery

The local branch is `research/snn-pid-v21-performance`. Per the user's updated
instruction, no GitHub push is performed. The supervisor commits only the
generated report paths on this branch after the finite campaign exits. If the
user changes branches, it preserves the files and records why it did not commit.

The report directory contains compact metrics, selected policies and checksums;
large trajectory datasets, optimizer states and population archives remain in
the ignored run directory. The repository-root `report.md` links the new report
while preserving earlier reports. A report made during training is explicitly
a progress snapshot. To regenerate it without training or accessing test cases:

```bash
bash experiments/algorithms/snn_pid_compare/run.sh \
  -m experiments.algorithms.snn_pid_compare.reporting \
  --root runs/snn_pid_compare/20261010_v21
```

See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for equations, state handling,
training settings, gates, evaluation splits and interpretation limits.
