# Reinforcement learning and parameter studies

Install the extra: `pip install "flapsim[rl]"` (Gymnasium; add
`stable-baselines3` for the PPO scripts).

## `SimulationEnv`

`flapsim.envs.SimulationEnv` wraps any simulation as a Gymnasium environment.
You list the signals that form the observation and the commands that the
action drives; everything else stays in plugins.

```python
from flapsim.envs import SimulationEnv, hover_reward, hover_terminated

env = SimulationEnv(
    make_sim,  # () -> Simulation (sensors included)
    observations=[
        "imu/euler",
        "imu/gyro",
        {"signal": "imu/accel", "scale": 0.1},
        "tof/range",
        "tof/range_rate",
    ],
    actions=[
        {"command": "gimbal/pitch_joint", "low": -0.3, "high": 0.3, "tau": 0.0082},
        {"command": "gimbal/roll_joint", "low": -0.3, "high": 0.3, "tau": 0.0082},
        {"command": "stroke/frequency_hz", "low": 36.0, "high": 44.0, "initial": 40.0},
    ],
    decision_dt=0.01,
    episode_seconds=6.0,
    reward=hover_reward(0.5),
    terminated=hover_terminated(),
    on_reset=randomize,
)
```

* Actions live in `[-1, 1]` and map affinely onto `[low, high]`; `tau` adds a
  first-order lag between the command and the actuator (actuator dynamics or
  a communication filter).
* `include_previous_action` appends the last action to the observation.
* `reward(env, obs, action)` may return `(value, terms)`; the terms land in
  `info["reward_terms"]`. `hover_reward` and `hover_terminated` reproduce the
  IMU-hover training signal (true-state reward, sensor-only observation).
* `on_reset(env)` randomises the initial state or plugin parameters using
  `env.rng`.

The environment passes `gymnasium.utils.env_checker.check_env`.

## The IMU + TOF hover example

`examples/rl/imu_hover_env.py` rebuilds the sensor-only hover task on the
floating-base cycle-averaged model: 14-dimensional observation (attitude,
rates, specific force, range, range rate, previous action), three actions
(gimbal pitch, gimbal roll, stroke frequency within +/- 10 % of hover).

```bash
python examples/rl/train_ppo.py --timesteps 500000 --n-envs 4 --output models/imu_hover_ppo
python examples/rl/evaluate.py --model models/imu_hover_ppo.zip --episodes 10
python examples/rl/evaluate.py --baseline --episodes 3          # zero-action reference
```

Trained policies and logs are not committed (`models/*.zip`, `runs/` are
ignored).

## Parameter studies

`flapsim.study.parameter_study(evaluate, grid, workers=n)` evaluates a
module-level function over the Cartesian product of parameter lists in
parallel processes, captures per-case errors, and returns a `StudyResult`
with `rows()`, `best(metric)` and `to_csv()`. See
`examples/wing_length_study.py` (lift of the flat-plate flapper versus span
and frequency) and `flapsim sweep` for the cycle-averaged identification.
