"""Overlays, video export and recorder exports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from flapsim.core import Simulation
from flapsim.io import ForceArrows, record_video
from flapsim.robot import builtin_robots

ROOT = Path(__file__).resolve().parents[1]
SIM_YAML = builtin_robots()["SingleThrustFlapper"].parent / "sim.yaml"


def test_force_arrows_add_geoms_to_a_scene():
    sim = Simulation.from_config(SIM_YAML)
    sim.advance(0.01)
    scene = mujoco.MjvScene(sim.model, maxgeom=50)
    ForceArrows(total=True)(sim, scene)
    assert scene.ngeom == 2 * len(sim.robot.wings) + 1
    kinds = {int(scene.geoms[i].type) for i in range(scene.ngeom)}
    assert {
        int(mujoco.mjtGeom.mjGEOM_ARROW),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
    } <= kinds

    class FakeViewer:
        user_scn = mujoco.MjvScene(sim.model, maxgeom=10)

    ForceArrows()(sim, FakeViewer())
    assert FakeViewer.user_scn.ngeom == 2 * len(sim.robot.wings)


def _offscreen_available() -> bool:
    try:
        sim = Simulation("SingleThrustFlapper", check=False)
        renderer = mujoco.Renderer(sim.model, height=16, width=16)
        renderer.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _offscreen_available(), reason="no offscreen OpenGL backend")
def test_record_gif_and_png_frames(tmp_path):
    sim = Simulation.from_config(SIM_YAML)
    gif = record_video(
        sim,
        tmp_path / "clip.gif",
        duration=0.004,
        fps=500.0,
        width=64,
        height=48,
        overlay=ForceArrows(),
    )
    assert gif.is_file() and gif.stat().st_size > 0
    frames_dir = record_video(
        sim, tmp_path / "frames", duration=0.002, fps=1000.0, width=32, height=24
    )
    assert len(list(frames_dir.glob("frame_*.png"))) >= 2


def test_recorder_hdf5_export(tmp_path):
    pytest.importorskip("h5py")
    sim = Simulation.from_config(SIM_YAML)
    recorder = sim.add(
        {"type": "recorder/memory", "signals": ["aero/F_world"], "every": 1e-3}
    )
    sim.advance(0.005)
    path = recorder.to_hdf5(tmp_path / "out.h5")
    assert path.is_file()


@pytest.mark.parametrize(
    "script, args",
    [
        ("examples/hover_pid.py", ["--duration", "1.5"]),
        (
            "examples/sensors.py",
            ["--duration", "0.05", "--output", "{tmp}/sensors.csv"],
        ),
        ("examples/lift_sweep.py", ["--frequencies", "30", "--cycles", "2"]),
        ("examples/custom_aero_model.py", []),
        (
            "examples/wing_length_study.py",
            ["--spans", "0.05", "0.07", "--frequencies", "30", "--workers", "2"],
        ),
    ],
)
def test_example_scripts_run(tmp_path, script, args):
    args = [a.format(tmp=tmp_path) for a in args]
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONPATH=str(ROOT)
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.isfinite(len(result.stdout))
