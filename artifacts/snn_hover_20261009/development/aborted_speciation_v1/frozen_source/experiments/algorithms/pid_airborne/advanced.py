"""Notch filtered cascade PID candidates, using only the causal IMU frontend."""
import numpy as np
from .core import Airborne,DT


class Notch:
    def __init__(self, frequency, q=2.):
        w=2*np.pi*frequency*DT;alpha=np.sin(w)/(2*q)
        self.b=np.array([1.,-2*np.cos(w),1.])/(1+alpha)
        self.a=np.array([1.,-2*np.cos(w)/(1+alpha),(1-alpha)/(1+alpha)])
        self.x=np.zeros((2,2));self.y=np.zeros((2,2));self.ready=False
    def filter(self,x):
        if not self.ready:self.x[:]=x;self.y[:]=x;self.ready=True
        y=self.b[0]*x+self.b[1]*self.x[0]+self.b[2]*self.x[1]-self.a[1]*self.y[0]-self.a[2]*self.y[1]
        self.x[1]=self.x[0];self.x[0]=x;self.y[1]=self.y[0];self.y[0]=y
        return y


class Cascade(Airborne):
    def reset(self,seed,p,degrees=2.,**kwargs):
        super().reset(seed,p,degrees,**kwargs)
        frequency=26.7477*p.get('motor',3461.36)/3461.36
        self.notches=[Notch(frequency*k,p.get('notch_q',2.)) for k in (1,2)]
        self.rate_notches=[Notch(frequency*k,p.get('notch_q',2.)) for k in (1,2)]
        self.previous_notched=self.last_rpy[:2].copy()
        self.filtered_notched=self.previous_notched.copy()
        self.filtered_notched_rate=np.zeros(2)

    def control(self,p):
        s=self.frontend.imu
        angle=np.array(s.attitude)[:2]
        rate=np.array(s.gyro)[:2]
        for a,b in zip(self.notches,self.rate_notches):
            angle=a.filter(angle);rate=b.filter(rate)
        if p.get('derivative','difference')=='difference':
            rate=(angle-self.previous_notched)/DT
        self.previous_notched=angle.copy()
        blend=1-np.exp(-DT/p.get('post_tau',.002))
        self.filtered_notched+=blend*(angle-self.filtered_notched)
        self.filtered_notched_rate+=blend*(rate-self.filtered_notched_rate)
        # Velocity is relative inertial odometry, initialized to rest.
        yaw=float(s.attitude[2]);c,sn=np.cos(yaw),np.sin(yaw)
        vx,vy=np.array(s.velocity)[:2]
        vx,vy=c*vx+sn*vy,-sn*vx+c*vy
        target=np.array(p.get('reference',[0.,0.]))+np.array(p.get('velocity_gain',[0.,0.]))*np.array([vy,-vx])
        target=np.clip(target,-.2,.2)
        error=target-self.filtered_notched
        self.integral=np.clip(self.integral+error*DT,-.3,.3)
        effort=np.array(p['kp'])*error-np.array(p['kd'])*self.filtered_notched_rate+np.array(p.get('ki',[0.,0.]))*self.integral
        angles=np.array(p.get('trim',self.env.reference_angles))+effort[::-1]
        z,vz=self.frontend.imu.position[2],self.frontend.imu.velocity[2]
        self.z_integral=float(np.clip(self.z_integral-z*DT,-.5,.5))
        kph,kdh,kih=p.get('height_gains',[200.,100.,20.])
        motor=p.get('motor',3461.36)-kph*z-kdh*vz+kih*self.z_integral
        return np.r_[np.clip(angles,-.3,.3),np.clip(motor,3115.224079,3807.496097)]


ENV=None
def init_worker(root):
    global ENV
    ENV=Cascade(root)
def evaluate(task):
    p,seeds,seconds,degrees=task
    rows=[ENV.rollout(p,s,seconds,degrees) for s in seeds]
    scores=np.array([r['score'] for r in rows])
    return dict(parameters=p,score=float(.7*scores.mean()+.3*scores.min()),episodes=rows)
