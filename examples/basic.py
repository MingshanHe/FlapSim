"""Run from an installed package: python examples/basic.py."""
from flapsim import Simulator

sim = Simulator()
sim.set_speed(100.0)
sim.run(viewer=False, duration=.01)
data = sim.snapshot()
print('time:', data.time)
print('motor angle:', data.joint('motor_joint').qpos[0])
