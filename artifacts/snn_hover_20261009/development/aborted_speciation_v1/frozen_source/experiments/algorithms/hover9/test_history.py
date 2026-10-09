"""Sensor and control contracts that protect this experiment's interpretation."""
from pathlib import Path
import json
import mujoco
import numpy as np
from experiments.algorithms.hover9.plant import Plant
from experiments.algorithms.hover9.imu import attach
from experiments.algorithms.hover9.environment import HistoryEnv
from experiments.algorithms.neat_imu.activation import cases

ROOT=Path(__file__).resolve().parents[3]/'artifacts/hover9_pid_20261008'


def test_sensor_instrumentation_preserves_physics():
    original=Plant(ROOT);instrumented=Plant(ROOT)
    attach(instrumented.sim)
    try:
        case=cases([302000100],.08)[0]
        for env in (original,instrumented):env.configure_case(case);env.reset(seed=case['seed'])
        for _ in range(8):
            action=np.array([.04,-.03,.05])
            original.step(action);instrumented.step(action)
            np.testing.assert_allclose(original.sim.data.qpos,instrumented.sim.data.qpos,atol=1e-10,rtol=0)
            np.testing.assert_allclose(original.sim.data.qvel,instrumented.sim.data.qvel,atol=1e-9,rtol=0)
    finally:original.close();instrumented.close()


def test_history_has_no_absolute_position_and_resets():
    env=HistoryEnv(ROOT)
    try:
        case=cases([302000101],.08)[0];env.configure_case(case)
        first,_=env.reset(seed=case['seed'])
        assert first.shape==(15,) and np.isfinite(first).all()
        np.testing.assert_array_equal(first[9:],np.zeros(6))
        before=env.features(env.raw6)
        env.sim.data.qpos[env.free_qpos:env.free_qpos+2]+=[2.,-3.]
        mujoco.mj_forward(env.sim.model,env.sim.data)
        after=env.features(env.raw6)
        np.testing.assert_array_equal(before,after)
        env.reset(seed=case['seed']);env.step(np.zeros(3))
        assert np.linalg.norm(np.asarray(env.frontend.imu.position))>0
        reset,_=env.reset(seed=case['seed']);np.testing.assert_array_equal(reset,first)
    finally:env.close()


def test_residual_covers_full_range_and_observer_is_short_horizon_accurate():
    env=HistoryEnv(ROOT)
    try:
        case=cases([302000102],.1)[0];env.configure_case(case)
        for sign in (-1,1):
            env.reset(seed=case['seed']);_,_,_,_,info=env.step(np.full(3,sign,dtype=float))
            np.testing.assert_array_equal(info['physical_action'],np.full(3,sign))
        env.reset(seed=case['seed'])
        for _ in range(10):obs,_,_,_,info=env.step(np.zeros(3))
        assert info['observer_error_m']<.02
        env.ablation='zero_history';obs=env.features(env.raw6)
        np.testing.assert_array_equal(obs[9:],np.zeros(6))
    finally:env.close()
