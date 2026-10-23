"""Diagnostic baseline flight trials before residual PPO training."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from flapsim.rl.env import TrackingConfig, TrajectoryTrackingEnv


def trial(gains=(2., .5, 1., .2), speed=3500., seconds=3., output=None):
    env = TrajectoryTrackingEnv(TrackingConfig(trajectories=('hover',), native_physics=True,
                     initial_speed=speed, episode_seconds=seconds, randomize=False, action_tau=.01))
    env.reset(seed=1)
    m,d=env.sim.model,env.sim.data
    am=np.zeros((3,m.nv))
    mujoco.mj_angmomMat(m,d,am,env.body)
    av=env.free_dof+3
    d.qvel[av:av+3] -= np.linalg.solve(am[:,av:av+3],am@d.qvel)
    mujoco.mj_forward(m,d)
    angle_filter = np.zeros(2)
    rate_filter = np.zeros(2)
    rows = []
    for _ in range(env.horizon):
        p,v,r = env._state()
        angles = np.array([np.arctan2(r[2,1],r[2,2]), np.arcsin(np.clip(-r[2,0],-1,1))])
        mujoco.mj_angmomMat(m,d,am,env.body)
        rate = np.linalg.solve(am[:,av:av+3],am@d.qvel)
        angle_filter += .2*(angles-angle_filter)
        rate_filter += .2*(rate[:2]-rate_filter)
        desired = np.clip(np.array([1.0*p[1]+1.2*v[1], -1.0*p[0]-1.2*v[0]])/9.81,-.15,.15)
        error = angle_filter-desired
        roll = gains[0]*error[0]+gains[1]*rate_filter[0]
        pitch = -gains[2]*error[1]-gains[3]*rate_filter[1]
        command = np.clip(speed - 1200*(p[2]-.8)-700*v[2], 0, 5000)
        a0 = (command-speed)/(5000-speed if command>=speed else speed)
        action = np.array([a0,pitch/.3,roll/.3])
        obs,rew,term,trunc,info = env.step(action)
        rows.append([info['elapsed'],*p,*v,*angles,*rate,command,pitch,roll])
        if term or trunc:
            break
    result = dict(gains=list(gains),speed=speed,seconds=info['elapsed'],rmse=info['tracking_rmse'],
                  reason=info['termination_reason'],position=info['position'].tolist(),
                  angles=angles.tolist())
    print(json.dumps(result),flush=True)
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(output,np.asarray(rows),delimiter=',',header='t,x,y,z,vx,vy,vz,roll,pitch,wx,wy,wz,speed,pitch_cmd,roll_cmd')
    env.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gains',type=float,nargs=4,default=[2.,.5,1.,.2])
    p.add_argument('--speed',type=float,default=3500)
    p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--output',default='runs/baseline.csv')
    a=p.parse_args()
    trial(a.gains,a.speed,a.seconds,a.output)


if __name__=='__main__':
    main()
