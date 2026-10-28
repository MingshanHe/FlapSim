"""FlapSim: open simulation platform for flapping-wing robots on MuJoCo."""

from importlib.metadata import PackageNotFoundError, version

from .robot import RobotManifest, RobotModel
from .robot import load as load_robot
from .sim_core import SimCore

try:
    __version__ = version("flapsim")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.2.0.dev0"

__all__ = ["RobotManifest", "RobotModel", "SimCore", "__version__", "load_robot"]
