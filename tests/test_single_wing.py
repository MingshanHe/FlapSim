"""Physical model removal and recorder compatibility for the single-wing bench."""
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'fixbase'))
from single_wing_sweep import build_single_wing_model, single_config
from experiment import ExperimentConfig, run_experiment
from flapsim import Simulator


def test_remove_plate_retains_both_drives(tmp_path):
    original = Simulator()
    single = Simulator(single_config(build_single_wing_model(tmp_path / 'model')))
    a, b = original.model, single.model
    assert b.nq == a.nq-1
    assert b.nbody == a.nbody-1
    assert mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, 'right_wing') == -1
    for side in ('left', 'right'):
        assert b.body(f'{side}_flapping').id > 0
        assert b.equality(f'{side}_flapping_joint_coupling').id >= 0
    np.testing.assert_allclose(a.body_mass.sum()-b.body_mass.sum(), a.body('right_wing').mass[0], atol=1e-12)
    assert [w.body for w in single.config.fluid.wings] == ['left_wing']


def test_single_wing_recording_and_drive_motion(tmp_path):
    config = single_config(build_single_wing_model(tmp_path / 'model'))
    data, summary = run_experiment(tmp_path / 'run', ExperimentConfig(duration=.3),
                                   sim_config=config, video=False)
    assert summary['wing_bodies'] == ['left_wing']
    assert 'right_feathering_joint_angle_rad' not in data
    assert np.ptp(data['right_flapping_joint_angle_rad']) > 1
    np.testing.assert_allclose(data['aero_Fz_N'], data['left_aero_Fz_N'], atol=1e-12)
    assert summary['max_base_pose_error'] == 0
    assert not any(summary['numerical_warnings'])
