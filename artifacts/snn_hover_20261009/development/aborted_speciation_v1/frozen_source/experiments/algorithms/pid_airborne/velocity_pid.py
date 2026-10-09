"""Attitude PID with IMU odometry compensation for velocity-dependent air torque."""
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .core import Airborne,DT
from ..neat_imu.train import write_json

class VelocityPID(Airborne):
    def reset(self,seed,p,degrees=2.,**kwargs):
        super().reset(seed,p,degrees,**kwargs)
        self.velocity=np.zeros(3)
    def control(self,p):
        command=super().control(p)
        self.velocity+=(1.-np.exp(-DT/p.get('velocity_tau',.03)))*(np.array(self.frontend.imu.velocity)-self.velocity)
        yaw=float(self.frontend.imu.attitude[2]);c,s=np.cos(yaw),np.sin(yaw)
        vx,vy=self.velocity[:2];vx,vy=c*vx+s*vy,-s*vx+c*vy
        command[:2]+=np.array(p.get('velocity_feedforward',[1.,1.]))*np.array([vx,-vy])
        command[:2]=np.clip(command[:2],-.3,.3)
        return command

ENV=None
def init_worker(root):
    global ENV
    ENV=VelocityPID(root)
def evaluate(task):
    p,seeds,seconds,degrees=task
    rows=[ENV.rollout(p,s,seconds,degrees) for s in seeds]
    scores=np.array([r['score'] for r in rows])
    return dict(parameters=p,score=float(.7*scores.mean()+.3*scores.min()),episodes=rows)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();started=time.monotonic();specs=[]
    for kp,kd,ff,tau in itertools.product([.1,.3,.6],[-.01,.005,.02],[.5,1.,1.5],[.003,.02]):
        specs.append(dict(kp=[.3,kp],kd=[.02,kd],velocity_feedforward=[ff,ff],sensor_tau=tau,
                          motor=3270.,height_gains=[300,150,20],action_tau=.02))
    with ProcessPoolExecutor(max_workers=6,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100001],3.,1.) for p in specs]))
    rows.sort(key=lambda r:r['score'],reverse=True)
    write_json(a.root/'velocity_screen.json',rows)
    print(json.dumps(dict(wall_seconds=time.monotonic()-started,best=rows[:3])),flush=True)
if __name__=='__main__':main()
