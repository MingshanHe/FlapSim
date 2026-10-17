"""Direct physical actuator control with IMU feedback and explicit flight metrics."""
import math
import numpy as np
import mujoco
from ..hover9.plant import Plant
from ..hover9.imu import attach
from ..neat_imu.activation import cases

DT = .002


class Airborne:
    def __init__(self, root):
        self.env = Plant(root)
        self.sim = self.env.sim
        self.frontend = attach(self.sim)
        self.original_gain = self.sim.model.actuator_gainprm.copy()
        self.original_bias = self.sim.model.actuator_biasprm.copy()
        self.bank_qpos = self.env.phase_qpos.copy()
        self.bank_qvel = self.env.phase_qvel.copy()

    def reset(self, seed, p, degrees=2., phase=None, stress=False):
        e = self.env
        c = cases([seed], 8., stress)[0]
        c['degrees'] = degrees
        e.configure_case(c)
        e.phase_qpos[:] = self.bank_qpos
        e.phase_qvel[:] = self.bank_qvel
        if phase is not None:
            # Override the bank before reset, never overwrite state during flight.
            e.phase_qpos[:] = self.bank_qpos[phase]
            e.phase_qvel[:] = self.bank_qvel[phase]
        e.reset(seed=seed)
        self.frontend.reset_imu(attitude_tau=p.get('sensor_tau', .02), tau=.002,
            bias=np.array(p.get('acceleration_bias', [0.,0.,0.])))
        model = self.sim.model
        model.actuator_gainprm[:] = self.original_gain
        model.actuator_biasprm[:] = self.original_bias
        for i in range(2):
            model.actuator_gainprm[i, 0] = p.get('servo_kp', .5)
            model.actuator_biasprm[i, 1] = -p.get('servo_kp', .5)
            model.actuator_biasprm[i, 2] = -p.get('servo_kd', .005)
        self.start = self.sim.data.xpos[e.body].copy()
        self.action = np.r_[e.reference_angles, p.get('motor', e.center[2])]
        self.last_rpy = np.array(self.frontend.imu.attitude)
        self.rate = np.zeros(2)
        self.integral = np.zeros(2)
        self.z_integral = 0.
        self.predicted_gimbal = e.reference_angles.copy()
        self.steps = 0

    def control(self, p):
        s = self.frontend.imu
        rpy = np.array(s.attitude)
        # Filtered attitude differences avoid using aliased instantaneous gyro.
        derivative = np.arctan2(np.sin(rpy[:2]-self.last_rpy[:2]),
                                np.cos(rpy[:2]-self.last_rpy[:2])) / DT
        self.last_rpy = rpy.copy()
        derivative_blend = 1.-np.exp(-DT/p.get('derivative_tau', .01))
        self.rate += derivative_blend*(derivative-self.rate)
        rate = self.rate if p.get('derivative', 'difference') == 'difference' else np.array(s.gyro)[:2]
        error = np.array(p.get('reference', [0.,0.]))-rpy[:2]
        kappa = np.array(p.get('kappa', [0.,0.]))
        error -= kappa*(self.predicted_gimbal-self.env.reference_angles)[::-1]
        kp, kd, ki = (np.array(p.get(name, default)) for name,default in
                      [('kp',[.2,.2]),('kd',[.01,.01]),('ki',[0.,0.])])
        candidate = np.clip(self.integral+error*DT, -.3, .3)
        effort = kp*error-kd*rate+ki*candidate
        trim = np.array(p.get('trim', self.env.reference_angles))
        angles = trim+effort[::-1]
        self.integral = np.where(np.abs(angles[::-1]) < .3, candidate, self.integral)
        if p.get('height_source','imu') == 'truth':
            position,velocity,_ = self.env._state()
            z,vz = position[2]-self.start[2],velocity[2]
        else:
            z,vz = s.position[2],s.velocity[2]
        self.z_integral = float(np.clip(self.z_integral-z*DT,-.5,.5))
        zkp,zkd,zki = p.get('height_gains', [0.,0.,0.])
        motor = p.get('motor', self.env.center[2])-zkp*z-zkd*vz+zki*self.z_integral
        # All commands remain inside the original physical speed and angle limits.
        motor = np.clip(motor, *p.get('motor_limits', [3115.224079,3807.496097]))
        return np.r_[np.clip(angles,-.3,.3),motor]

    def advance(self, command, p):
        tau = p.get('action_tau', .02)
        blend = 1. if tau == 0 else 1.-np.exp(-DT/tau)
        self.action += blend*(command-self.action)
        self.predicted_gimbal += (1.-np.exp(-DT/.02))*(self.action[:2]-self.predicted_gimbal)
        self.sim.set_gimbal(pitch=float(self.action[0]), roll=float(self.action[1]))
        self.sim.set_speed(float(self.action[2]))
        self.sim.step(round(DT/self.sim.model.opt.timestep))
        self.steps += 1

    def rollout(self, p, seed, seconds=3., degrees=2., trace=False):
        self.reset(seed,p,degrees)
        rows=[]; score=0.; min_z=10.; tilt_square=0.; z_square=0.; peak_tilt=0.
        saturation=0; reason='time_limit'
        for i in range(round(seconds/DT)):
            cmd=self.control(p); self.advance(cmd,p)
            position,velocity,rotation=self.env._state()
            tilt=math.acos(float(np.clip(rotation[2,2],-1,1)))
            z=position[2]-self.start[2]
            min_z=min(min_z,float(position[2]));peak_tilt=max(peak_tilt,tilt)
            tilt_square+=tilt**2; z_square+=z*z
            saturation+=int(np.any(np.abs(cmd[:2])>=.2999))
            # Airborne tilt/height objective has no yaw or XY tracking term.
            score+=DT*(.3+math.exp(-(tilt/.35)**2)+.5*math.exp(-(z/.2)**2))
            if trace:
                rows.append([self.steps*DT,*position,*velocity,*self.env.angles(),
                    *self.action,*np.array(self.frontend.imu.position),*np.array(self.frontend.imu.attitude),
                    *[float(self.sim.data.joint(n).qpos[0]) for n in ['pitch_joint','roll_joint']]])
            if not np.isfinite(self.sim.data.qpos).all() or any(w.number for w in self.sim.data.warning):
                reason='numerical';break
            if position[2]<.12:reason='low_height';break
            if abs(z)>1.:reason='height_escape';break
            if tilt>1.4:reason='tilt';break
        return dict(seed=seed,phase=self.env.phase_index,elapsed=self.steps*DT,score=score,
            termination=reason,tilt_rms_deg=math.degrees(math.sqrt(tilt_square/self.steps)),
            max_tilt_deg=math.degrees(peak_tilt),height_rmse_m=math.sqrt(z_square/self.steps),
            min_base_height_m=min_z,final_position=position.tolist(),final_velocity=velocity.tolist(),
            saturation=saturation/self.steps,trace=rows)


ENV=None
def init_worker(root):
    global ENV
    ENV=Airborne(root)

def evaluate(task):
    p,seeds,seconds,degrees=task
    rows=[ENV.rollout(p,s,seconds,degrees) for s in seeds]
    scores=np.array([r['score'] for r in rows])
    return dict(parameters=p,score=float(.7*scores.mean()+.3*scores.min()),episodes=rows)
