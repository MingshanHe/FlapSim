"""Compatibility entry point; new applications should use Simulator(config)."""
from dataclasses import replace
from pathlib import Path
from .simulator import Simulator
from .config import SimulationConfig, load_config


class SimCore(Simulator):
    def __init__(self, robot='IST_robot', *, fix=True):
        preset = Path(__file__).parent / 'config' / 'robots' / f'{robot}.json'
        config = load_config(preset) if preset.is_file() else SimulationConfig(robot=robot)
        super().__init__(replace(config, scene=replace(config.scene, fixed_base=fix)))
