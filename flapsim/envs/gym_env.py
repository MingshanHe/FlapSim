"""A Gymnasium environment over any FlapSim :class:`~flapsim.core.Simulation`.

Observations are named signals (``imu/gyro``, ``tof/range``, ...) scaled and
concatenated; actions in ``[-1, 1]`` are mapped affinely onto plugin
commands (``cmd/gimbal/pitch_joint``, ``cmd/stroke/frequency_hz``, ...) with
optional first-order smoothing.  Reward and termination are plain callables
so tasks stay outside the core.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "flapsim.envs needs gymnasium: pip install 'flapsim[rl]'"
    ) from exc


@dataclass
class Observation:
    """One block of the observation vector taken from a signal."""

    signal: str
    size: int | None = None
    scale: Any = 1.0
    offset: Any = 0.0
    clip: float | None = None

    @classmethod
    def parse(cls, spec: str | Mapping[str, Any] | Observation) -> Observation:
        if isinstance(spec, Observation):
            return spec
        if isinstance(spec, str):
            return cls(spec)
        data = dict(spec)
        return cls(
            signal=data.pop("signal"),
            size=data.pop("size", None),
            scale=data.pop("scale", 1.0),
            offset=data.pop("offset", 0.0),
            clip=data.pop("clip", None),
        )

    def read(self, signals) -> np.ndarray:
        value = signals.get(self.signal)
        if value is None:
            if self.size is None:
                raise KeyError(
                    f"observation signal {self.signal!r} is not published yet"
                )
            value = np.zeros(self.size)
        array = np.atleast_1d(np.asarray(value, dtype=np.float64)).reshape(-1)
        if self.size is None:
            self.size = int(array.size)
        elif array.size != self.size:
            raise ValueError(
                f"signal {self.signal!r} has size {array.size}, expected {self.size}"
            )
        out = (array - np.asarray(self.offset, dtype=np.float64)) * np.asarray(
            self.scale, dtype=np.float64
        )
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return out


@dataclass
class Action:
    """One action element mapped onto a plugin command."""

    command: str
    low: float
    high: float
    tau: float = 0.0
    initial: float | None = None

    @classmethod
    def parse(cls, spec: Mapping[str, Any] | Action) -> Action:
        if isinstance(spec, Action):
            return spec
        data = dict(spec)
        return cls(
            command=data["command"],
            low=float(data["low"]),
            high=float(data["high"]),
            tau=float(data.get("tau", 0.0)),
            initial=None if data.get("initial") is None else float(data["initial"]),
        )

    @property
    def signal(self) -> str:
        return (
            self.command if self.command.startswith("cmd/") else f"cmd/{self.command}"
        )

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)

    def to_value(self, normalized: float) -> float:
        a = float(np.clip(normalized, -1.0, 1.0))
        return self.midpoint + 0.5 * (self.high - self.low) * a

    def to_normalized(self, value: float) -> float:
        half = 0.5 * (self.high - self.low)
        return (
            0.0
            if half == 0.0
            else float(np.clip((value - self.midpoint) / half, -1.0, 1.0))
        )


class SimulationEnv(gym.Env):
    """Gymnasium wrapper: decisions every ``decision_dt`` seconds.

    ``make_sim`` builds the simulation once; episodes call ``sim.reset()``.
    ``reward(env, obs, action)`` returns a float or ``(float, info_dict)``;
    ``terminated(env)`` returns a bool; ``on_reset(env)`` may randomise the
    state.  With ``include_previous_action`` the previous normalised action is
    appended to the observation.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        make_sim: Callable[[], Any],
        *,
        observations: Sequence[str | Mapping[str, Any] | Observation],
        actions: Sequence[Mapping[str, Any] | Action],
        decision_dt: float = 0.01,
        episode_seconds: float = 6.0,
        reward: Callable[..., Any] | None = None,
        terminated: Callable[..., bool] | None = None,
        on_reset: Callable[..., None] | None = None,
        include_previous_action: bool = True,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (480, 360),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sim = make_sim()
        self.observations = [Observation.parse(o) for o in observations]
        self.actions = [Action.parse(a) for a in actions]
        if not self.actions:
            raise ValueError("at least one action is required")
        self.decision_dt = float(decision_dt)
        self.episode_seconds = float(episode_seconds)
        self.steps_per_decision = max(
            1, int(round(self.decision_dt / self.sim.timestep))
        )
        self.max_decisions = int(math.ceil(self.episode_seconds / self.decision_dt))
        self.reward_fn = reward
        self.terminated_fn = terminated
        self.on_reset = on_reset
        self.include_previous_action = bool(include_previous_action)
        self.render_mode = render_mode
        self.render_size = render_size
        self._renderer = None
        self.rng = np.random.default_rng(seed)
        self.previous_action = np.zeros(len(self.actions), dtype=np.float64)
        self.command_values = np.array(
            [a.initial if a.initial is not None else a.midpoint for a in self.actions],
            dtype=np.float64,
        )
        self.elapsed_decisions = 0
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(len(self.actions),), dtype=np.float32
        )
        # publish initial commands and step once so that every signal exists
        self._publish_commands()
        self.sim.step()
        sample = self._observe()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=sample.shape, dtype=np.float32
        )

    # -- helpers ------------------------------------------------------------

    @property
    def time(self) -> float:
        return self.sim.time

    def _publish_commands(self) -> None:
        for action, value in zip(self.actions, self.command_values, strict=True):
            self.sim.signals.publish(action.signal, float(value))

    def _observe(self) -> np.ndarray:
        parts = [o.read(self.sim.signals) for o in self.observations]
        if self.include_previous_action:
            parts.append(self.previous_action)
        return np.concatenate(parts).astype(np.float32)

    def command_from_action(self, action: np.ndarray) -> np.ndarray:
        return np.array(
            [a.to_value(v) for a, v in zip(self.actions, action, strict=True)],
            dtype=np.float64,
        )

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.sim.reset()
        self.elapsed_decisions = 0
        self.previous_action[:] = 0.0
        self.command_values = np.array(
            [a.initial if a.initial is not None else a.midpoint for a in self.actions],
            dtype=np.float64,
        )
        if self.on_reset is not None:
            self.on_reset(self)
        self._publish_commands()
        self.sim.step()
        return self._observe(), {"time": self.sim.time}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (len(self.actions),):
            raise ValueError(f"action must have shape {(len(self.actions),)}")
        action = np.clip(action, -1.0, 1.0)
        targets = self.command_from_action(action)
        dt = self.sim.timestep
        for _ in range(self.steps_per_decision):
            for index, spec in enumerate(self.actions):
                if spec.tau > 0.0:
                    blend = 1.0 - math.exp(-dt / spec.tau)
                    self.command_values[index] += blend * (
                        targets[index] - self.command_values[index]
                    )
                else:
                    self.command_values[index] = targets[index]
            self._publish_commands()
            self.sim.step()
        obs = self._observe()
        self.elapsed_decisions += 1
        terminated = (
            bool(self.terminated_fn(self)) if self.terminated_fn is not None else False
        )
        truncated = self.elapsed_decisions >= self.max_decisions
        info: dict[str, Any] = {"time": self.sim.time}
        reward = 0.0
        if self.reward_fn is not None:
            result = self.reward_fn(self, obs, action)
            if isinstance(result, tuple):
                reward, terms = result
                info["reward_terms"] = terms
            else:
                reward = float(result)
        self.previous_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import mujoco

        if self._renderer is None:
            width, height = self.render_size
            model = self.sim.model
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        try:
            self._renderer.update_scene(self.sim.data, camera="camera")
        except (KeyError, ValueError):
            self._renderer.update_scene(self.sim.data)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["Action", "Observation", "SimulationEnv"]
