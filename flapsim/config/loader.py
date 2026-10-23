"""Load one JSON file, with model paths relative to that file."""
import json
from pathlib import Path
from dataclasses import asdict
from .settings import (SimulationConfig, SceneConfig, ActuatorConfig,
                       ControllerConfig, FluidConfig, WingConfig, GimbalConfig)


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path).expanduser().resolve()
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, dict):
        raise ValueError('configuration root must be an object')
    for key, cls in (('scene', SceneConfig), ('actuator', ActuatorConfig),
                     ('controller', ControllerConfig), ('fluid', FluidConfig)):
        section = values.get(key, {})
        if key == 'fluid' and 'wings' in section:
            section['wings'] = tuple(WingConfig(**w) for w in section['wings'])
        values[key] = cls(**section)
    if values.get('gimbal') is not None:
        values['gimbal'] = GimbalConfig(**values['gimbal'])
    if values.get('model_path') is not None:
        values['model_path'] = (path.parent / values['model_path']).resolve()
    return SimulationConfig(**values)


def save_config(config: SimulationConfig, path: str | Path):
    values = asdict(config)
    if config.model_path is not None:
        values['model_path'] = str(config.path)
    Path(path).write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
