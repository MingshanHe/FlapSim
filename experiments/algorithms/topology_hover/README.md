# Topology evolution with a fixed airborne PID

The experiment protocol and results live in `artifacts/topology_hover_20261008/`.
The original PID/NEAT/PPO artifact directories are inputs, never overwritten.

Run from the repository root in the existing MuJoCo Python environment:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest tests/test_topology_hover.py tests/test_pid_airborne.py tests/test_native_physics.py -q
python -m experiments.algorithms.topology_hover.campaign --root artifacts/topology_hover_20261008 --workers 16
python -m experiments.algorithms.topology_hover.finish --root artifacts/topology_hover_20261008 --workers 16
```

The finite queue skips **completed** runs. An incomplete run is retained and stops
the queue for investigation; it is never silently overwritten or counted as a
completed experiment. `finish` waits for the queue, freezes policy hashes, runs
held-out tests, checks replay metrics, generates plots/videos and updates the
report. It does not commit or push anything.

In the recorded campaign, `early_transfer --seeds 1201 1202 1203` was started
alongside the primary queue after those NEAT parents had completed. It uses
identical PPO arguments and writes its own `parallel_transfer_status.json`;
the primary queue later reuses those completed results. This only changes
scheduling. Do not start it when the primary queue is already approaching the
same transfer jobs. Per-run training wall times are not comparable under this
concurrency; final actor inference timings are measured on an idle host.
After the first six transfers completed, one additional
`early_transfer --seeds 1204 --methods graph_ppo` run was started while the primary
queue was still training the feedforward PPO for seed 1205. The first schedule's
completed record is retained as `early_transfer_1201_1203_schedule.json`.

The saved actors can also be replayed independently. This entry point relocates
mesh paths in a temporary copy, so replay works from another clone path without
rewriting the frozen physics assets:

```sh
python -m experiments.algorithms.topology_hover.replay \
  --policy artifacts/topology_hover_20261008/pilot/neat_seed1201/best_policy.json \
  --output artifacts/manual_neat_replay --condition stress --seconds 20
```

Choose a new output directory for each replay. The command saves `result.json`
and `trajectory.npz`; it does not change the campaign's registered test results.
Omit `--policy` to replay the PID baseline. The source model for the full training
campaign retains the original absolute mesh path; the independent replay and
video paths explicitly handle relocation.

Single experiments can be launched through `evolve.py` or `ppo.py` with a fresh
`--output`, explicit seed and actor-interaction budget. `--smoke` reduces horizons
for pipeline verification; those outputs do not provide learning evidence.

## Interfaces and interpretation

- Physics: unchanged 7/8 g MuJoCo and blade-element model, 5 microseconds.
- Control: fixed PID at 500 Hz; actor at 100 Hz for every learned method.
- Actor inputs: 9 IMU channels plus measured relative height and estimated vertical
  speed in the main 11-D experiment. The pure IMU removal check is an ablation,
  not a claim of a separately trained 9-D control benchmark.
- All non-input graph nodes update synchronously. An edge from any non-input
  node uses the previous actor tick, even if it is drawn from left to right.
- NEAT uses a tanh activation without neat-python's default tanh input gain.
  NumPy and Torch use the same graph equation; sequence equality is tested.
- PPO uses a pre-tanh Gaussian, clipped policy objective, GAE, a shared-design
  feedforward critic, 32-step truncated recurrent backpropagation, and explicit
  time-limit bootstrapping. The tanh Jacobian cancels in the likelihood ratio;
  the Gaussian entropy term is an exploration surrogate.
- The sensor model clips and filters before decimation. Its virtual AHRS still
  starts from simulated orientation. Neither it nor the range sensor is a
  validated hardware model. Exact initial release height is supplied as the
  initial reference; subsequent height feedback is delayed and noisy.
- The CAD collision masks remain disabled. Whole-robot AABB clearance, rather
  than contact forces, defines the geometric clearance condition.

## Evolution records

Each `generation_XXXX.json.gz` stores every evaluated individual, fitness, species,
parents, graph, innovation IDs and post-crossover mutation events. Weight/bias
values are losslessly represented in JSON. Elite genomes retain their identity;
their old mutation event is not counted as a new birth.

To inspect an experiment-generated NEAT checkpoint, use
`load_evolution_checkpoint(path)` from this directory's `checkpoint.py`; it
handles the class names recorded by the module entry point. The checkpoint
contains the population **after reproduction**, ready for its next generation.
The evaluated individuals are in the generation archives. Loading a checkpoint
does not automatically resume or alter this registered campaign.

```sh
python -m experiments.algorithms.topology_hover.visualize artifacts/topology_hover_20261008/pilot/neat_seed1201
python -m experiments.algorithms.topology_hover.report --root artifacts/topology_hover_20261008 --install-root
```

`visuals/explorer.html` contains its data and works offline. It provides arbitrary
individual selection, generation playback, parent navigation, a weight threshold,
disabled-edge display, mutation details and all node biases. Static PNGs, a GIF,
genealogy JSON and full population/weight/bias CSV archives are linked from the
report. The champion movie is explicitly **not** presented as one lineage.
Both-parent ancestry and the largest-changing inherited weight curves supplement
the heatmaps. Flight activation traces describe changing neuron states, while
the learned weights remain fixed during deployment.

`flight/neat_nominal.html` and `flight/neat_stress.html` are offline flight
inspectors. Scrub or play the 100 Hz recorded input, node-state and action stream;
select a node to inspect its weighted current-input and previous-state terms.
Every recorded state and action is recomputed with the saved policy before
export, with an absolute-error bound of 2e-6. Graph line width shows fixed weight
magnitude; node colors show changing state. Neither is a causal importance score.

`dashboard.html` provides an offline overview. For the local live overview, run
`python -m experiments.algorithms.topology_hover.dashboard --root artifacts/topology_hover_20261008 --serve 8766`
and open `http://localhost:8766/dashboard.html`. The service binds only to loopback
and reads experiment status; it has no controller or training mutation endpoints.

Pinned dependencies used here: Python 3.11, neat-python 2.0.0, Torch 2.6.0,
stable-baselines3 2.6.0 (vector environments), Matplotlib 3.10.1; MuJoCo, NumPy,
imageio, Pillow and the existing repository dependencies. Native sensor code
requires g++ and builds into the user cache. Video rendering uses `MUJOCO_GL=egl`.
