"""GPU stepping; install flapsim[gpu] first on Linux/WSL."""
from flapsim import Simulator, SimulationConfig

sim = Simulator(SimulationConfig(backend='mjx', device='gpu'))
sim.step(200)
print(sim.backend.device, sim.snapshot().time)
