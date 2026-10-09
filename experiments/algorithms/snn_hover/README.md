# NEAT-SNN hovering study

The accepted design is in `research_proposal.md`; the immutable execution contract
is `artifacts/snn_hover_20261009/protocol.json`. Do not change frozen training code
while the campaign runs. Historical `topology_hover` code and artifacts are reused
without modification. All paths below are relative to the repository root.

The current experiment includes five evolution arms and two PPO arms, each with
five training seeds. All actors receive the same 11 sensor values and issue the
same three signed residual actions on the same fixed PID. The neural clock is
1 kHz; sensor access and actuator decisions remain 100 Hz. Only temporal actors
perform ten internal steps per decision. The CUBA-LIF hidden neurons are spiking;
their inputs and readouts are continuous, so this is a hybrid SNN.

Budgets are 256,000 and 1,024,000 allocated transitions, rounded down from the
proposal by 2.34375% to avoid truncated evolution generations. Early failures use
fewer actual transitions in evolution; both quantities are reported. PPO consumes
actual transitions, including transitions after episode resets. Never describe
unequal actual counts as equal. Initial policies are eligible for validation
selection and all validation costs are separately counted.

```bash
python -m pytest -q tests/test_snn_hover.py tests/test_topology_hover.py
python -m experiments.algorithms.snn_hover.development --root artifacts/snn_hover_20261009
python -m experiments.algorithms.snn_hover.protocol --root artifacts/snn_hover_20261009
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u -m experiments.algorithms.snn_hover.campaign --root artifacts/snn_hover_20261009
```

`protocol` refuses to replace a frozen protocol. `campaign` skips completed jobs
but refuses to replace incomplete output. Every individual, parent pair, gene,
mutation and episode is archived for every evolution generation. Checkpoints use
Python pickle and should only be loaded from this trusted local campaign.

Development-only PID envelope cases and smoke runs remain in `development/`.
Their seeds are disjoint from training, validation and test namespaces. A timing
or score obtained during a smoke run is not a formal performance claim.

The large-angle adapter changes only reset conditions and external test torque.
It records recovery, saturation and height-sensor angular availability. It does
not replace the previous physical plant, PID, observation, or reward. The unit
tests compare baseline qpos and reward against the frozen environment, verify
exact tilt across eight directions and check no truth-state feedback is added.
