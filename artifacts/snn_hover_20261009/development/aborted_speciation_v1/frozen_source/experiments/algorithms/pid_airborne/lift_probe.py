"""Free-flight impulse measurement: no fixed-root support or sampled-force alias."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco,numpy as np
from .core import Airborne,DT
from ..neat_imu.train import write_json

ENV=None
def init(root):
    global ENV
    ENV=Airborne(root)
def measure(task):
    speed,seed=task
    p=dict(kp=[0,0],kd=[0,0],motor=speed,motor_limits=[2000,4500])
    ENV.reset(seed,p,0.);d=ENV.sim.data;m=ENV.sim.model;body=ENV.env.body
    mujoco.mj_subtreeVel(m,d);v0=d.subtree_linvel[body].copy()
    initial_phase=float(d.joint('carrier_2_joint').qpos[0]);speeds=[];tilts=[]
    for _ in range(100):
        ENV.advance(ENV.control(p),p)
        speeds.append(float(d.joint('motor_joint').qvel[0]));tilts.append(np.arccos(np.clip(d.xmat[body,8],-1,1)))
    mujoco.mj_subtreeVel(m,d)
    force=m.body_subtreemass[body]*((d.subtree_linvel[body]-v0)/.2-m.opt.gravity)
    return dict(speed_command=speed,seed=seed,phase=ENV.env.phase_index,
        mean_force_N=force.tolist(),mean_vertical_gf=float(force[2]/.00980665),
        actual_motor_rad_s=float(np.mean(speeds)),flapping_hz=(float(d.joint('carrier_2_joint').qpos[0])-initial_phase)/(.2*2*np.pi),
        tilt_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(tilts))))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(measure,[(speed,seed) for speed in [2400,2700,2900,3100,3300,3461.36,3700,4000] for seed in [310100000,310100001]]))
    write_json(a.root/'free_lift_probe.json',dict(scope='First 0.2 s release transient; full COM impulse includes exact simulated net forces. Not a steady trim.',rows=rows))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
