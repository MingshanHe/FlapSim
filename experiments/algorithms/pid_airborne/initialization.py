"""Explicit alternative release states, changed only once at episode reset."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco,numpy as np
from .core import Airborne
from ..neat_imu.train import write_json

class InitializedAirborne(Airborne):
    def reset(self,seed,p,degrees=2.,**kwargs):
        super().reset(seed,p,degrees,**kwargs)
        mode=p.get('release','original')
        if mode=='original':return
        e=self.env;m=self.sim.model;d=self.sim.data
        for name,value in zip(['pitch_joint','roll_joint'],e.reference_angles):
            d.joint(name).qpos[0]=value;d.joint(name).qvel[0]=0.
        if mode=='rest':d.qvel[:]=0.
        elif mode=='aligned':
            d.qvel[e.free_dof+3:e.free_dof+6]=0.
            mujoco.mj_forward(m,d)
            a=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,a,e.body)
            d.qvel[e.free_dof+3:e.free_dof+6]=-np.linalg.solve(a[:,e.free_dof+3:e.free_dof+6],a@d.qvel)
        else:raise ValueError(mode)
        qpos=d.qpos.copy();qvel=d.qvel.copy();self.sim.reset(qpos=qpos,qvel=qvel)
        self.sim.set_speed(p.get('motor',e.center[2]))
        self.sim.set_gimbal(pitch=float(e.reference_angles[0]),roll=float(e.reference_angles[1]))
        self.frontend.reset_imu(attitude_tau=p.get('sensor_tau',.02),tau=.002,
            bias=np.array(p.get('acceleration_bias',[0,0,0])))
        self.last_rpy=np.array(self.frontend.imu.attitude)
        if mode=='rest' and p.get('motor_start_s',0):
            self.action[2]=0.
            self.sim.set_speed(0.)

    def advance(self,command,p):
        if p.get('motor_start_s',0):
            command=command.copy()
            command[2]*=min(1.,(self.steps+1)*.002/p['motor_start_s'])
        return super().advance(command,p)

ENV=None
def init_worker(root):
    global ENV
    ENV=InitializedAirborne(root)
def evaluate(task):
    p,seeds,seconds,degrees=task
    rows=[ENV.rollout(p,s,seconds,degrees) for s in seeds]
    scores=np.array([r['score'] for r in rows])
    return dict(parameters=p,score=float(.7*scores.mean()+.3*scores.min()),episodes=rows)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    best=json.loads((a.root/'best.json').read_text())['parameters'];specs=[]
    for release in ['original','aligned','rest']:
        for p in [dict(kp=[0,0],kd=[0,0]),dict(kp=[.3,.3],kd=[.01,.01]),best]:
            specs.append(dict(p,release=release))
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100001],4.,1.) for p in specs]))
    write_json(a.root/'release_screen.json',rows)
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
