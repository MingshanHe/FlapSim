from dataclasses import replace
import mujoco
import numpy as np
import pytest

from flapsim import Simulator, load_config
from flapsim.rl.env import PRESET
from flapsim.rl.native import accelerate
from flapsim.rl.native_fast import accelerate_fast


@pytest.mark.parametrize('fixed', [True, False])
@pytest.mark.parametrize('split', [True, False])
def test_native_matches_numpy(fixed, split):
    config = load_config(PRESET)
    config = replace(config, scene=replace(config.scene, fixed_base=fixed))
    slow, fast = Simulator(config), accelerate(Simulator(config))
    if split:
        accelerate_fast(fast)
    for sim in (slow, fast):
        sim.set_speed(2800.)
        sim.set_gimbal(pitch=.025, roll=-.015)
        sim.external_body_forces[1, 0] = .001
    for _ in range(5):
        slow.step(200)
        fast.step(200)
        np.testing.assert_allclose(fast.data.qpos, slow.data.qpos, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(fast.data.qvel, slow.data.qvel, atol=2e-6, rtol=2e-7)
    assert not any(w.number for w in fast.data.warning)
    fast.reset()
    slow.reset()
    np.testing.assert_array_equal(fast.data.qpos, slow.data.qpos)


def test_powered_full_wing_cycle_matches_python():
    config = load_config(PRESET)
    config = replace(config, scene=replace(config.scene, fixed_base=True))
    fast = accelerate_fast(accelerate(Simulator(config)))
    fast.set_speed(3400.)
    fast.step(40000)
    slow = Simulator(config)
    slow.set_speed(3400.)
    mujoco.mj_copyData(slow.data, slow.model, fast.data)
    # Covers more than one 26 Hz flap after reaching the operating regime.
    slow.step(10000)
    fast.step(10000)
    np.testing.assert_allclose(fast.data.qpos, slow.data.qpos, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(fast.data.qvel, slow.data.qvel, atol=1e-5, rtol=1e-7)
