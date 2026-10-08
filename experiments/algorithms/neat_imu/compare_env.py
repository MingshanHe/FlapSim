"""Identical six-input plant, actions, reward and tests for NEAT and PPO."""
import json
import math
from pathlib import Path
import numpy as np
from .activation import cases,signed_map
from .mass_model import MassVariantEnv
from .residual import ResidualPolicy

GAMMA=.995


def training_case(seed,fraction,stress):
    seconds=2. if fraction<.3 else 4. if fraction<.6 else 8.
    case=cases([seed],seconds,stress)[0]
    if fraction<.3:
        case['degrees']=2.
        if stress:
            case.update(angular_rate=.1,noise_degrees=.1,noise_gyro_rad_s=.01,motor_uncertainty=.01)
    return case


class CompareEnv(MassVariantEnv):
    def __init__(self,root,training=False,seed=0,rank=0,workers=1,budget=524288,render_mode=None):
        self.experiment_root=Path(root)
        trim=json.loads((self.experiment_root/'calibration/trim.json').read_text())
        super().__init__(trim,model_path=self.experiment_root/'robot_upper7_lower8.xml',sensor_inputs=6,
                         initialization='momentum',warmup_seconds=0.,render_mode=render_mode)
        bank=np.load(self.experiment_root/'phase_bank.npz')
        self.phase_qpos=bank['qpos']; self.phase_qvel=bank['qvel']
        self.training=training; self.run_seed=seed; self.rank=rank; self.workers=workers
        self.budget=budget; self.train_steps=0; self.episode_index=0
        self.configure_case(cases([0],8.)[0])

    def reset(self,*,seed=None,options=None):
        if self.training:
            rng=np.random.default_rng(np.random.SeedSequence([self.run_seed,self.rank,self.episode_index,9631]))
            seed=int(rng.integers(10000,800000000))
            self.configure_case(training_case(seed,min(1.,self.train_steps*self.workers/self.budget),self.episode_index%2==1))
            self.episode_index+=1
        if seed is None: seed=self.case['seed']
        phase=int(np.random.default_rng(np.random.SeedSequence([seed,5713])).integers(len(self.phase_qpos)))
        self._initial_qpos[:]=self.phase_qpos[phase]; self._initial_qvel[:]=self.phase_qvel[phase]
        self.tilt_squared=0.; self.return_sum=0.; self.discounted_return=0.
        self.saturation_count=0; self.previous_action=np.zeros(3)
        obs,info=super().reset(seed=seed,options=options)
        self.phase_index=phase
        return obs,dict(info,phase_index=phase)

    def step(self,action):
        action=np.clip(np.asarray(action,dtype=float),-1.,1.)
        obs,_,terminated,truncated,info=super().step(action)
        position,velocity,rotation=self._state()
        tilt=math.acos(float(np.clip(rotation[2,2],-1,1)))
        omega=self.sim.data.cvel[self.body,:3]@rotation
        self.tilt_squared+=tilt**2
        self.saturation_count+=int(np.any(np.abs(action)>.98))
        change=float(np.sum((self.filtered_action-self.previous_action)**2))
        self.previous_action=self.filtered_action.copy()
        reward=.01*(1.+2.*math.exp(-(tilt/.35)**2)
                    +2.*math.exp(-(info['position_error']/.2)**2)
                    +math.exp(-(np.linalg.norm(velocity)/.5)**2)
                    -.1*min(float(np.sum((omega/10.)**2)),10.)
                    -.05*change-.005*float(action@action))
        if terminated: reward-=2. if info['termination_reason']=='numerical' else 1.
        self.return_sum+=reward; self.discounted_return+=GAMMA**(self.steps-1)*reward
        self.train_steps+=int(self.training)
        tilt_rmse=float(np.rad2deg(np.sqrt(self.tilt_squared/self.steps)))
        info.update(tilt_rmse_deg=tilt_rmse,return_sum=self.return_sum,
                    discounted_return=self.discounted_return,phase_index=self.phase_index,
                    saturation_fraction=self.saturation_count/self.steps,
                    hover_success=bool(info['is_success'] and tilt_rmse<10. and self.case['seconds']>=8.))
        return obs,float(reward),terminated,truncated,info


class NumpyPPO:
    def __init__(self,payload):
        self.layers=[(np.asarray(l['weight'],dtype=np.float32),np.asarray(l['bias'],dtype=np.float32)) for l in payload['layers']]
    def reset(self,obs): pass
    def act(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        for weight,bias in self.layers: x=np.tanh(weight@x+bias)
        return x


def make_policy(payload):
    return NumpyPPO(payload) if payload.get('format')=='flapsim-ppo-imu6-v1' else ResidualPolicy(payload)


ENV=None


def init_worker(root):
    global ENV
    ENV=CompareEnv(root)


def evaluate(task):
    payload,batch,trace=task
    policy=make_policy(payload); results=[]
    for case in batch:
        ENV.configure_case(case); obs,initial=ENV.reset(seed=case['seed']); policy.reset(obs)
        rows=[]
        while True:
            action=policy.act(obs)
            obs,_,terminated,truncated,info=ENV.step(action)
            if trace:
                rows.append([info['elapsed'],*np.rad2deg(info['rpy']),info['alpha'],info['beta'],
                             *info['position'],info['motor_command']])
            if terminated or truncated: break
        results.append(dict(seed=case['seed'],case=case,steps=ENV.steps,phase=initial['phase_index'],
                            elapsed=info['elapsed'],success=info['hover_success'],termination=info['termination_reason'],
                            tracking_rmse_m=info['tracking_rmse'],tilt_rmse_deg=info['tilt_rmse_deg'],
                            discounted_return=info['discounted_return'],return_sum=info['return_sum'],
                            saturation_fraction=info['saturation_fraction'],trace=rows))
    return results


def summarize(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                mean_elapsed=float(np.mean([r['elapsed'] for r in rows])),
                p10_elapsed=float(np.quantile([r['elapsed'] for r in rows],.1)),
                mean_tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in rows])),
                mean_tilt_rmse_deg=float(np.mean([r['tilt_rmse_deg'] for r in rows])),
                mean_discounted_return=float(np.mean([r['discounted_return'] for r in rows])),
                mean_saturation_fraction=float(np.mean([r['saturation_fraction'] for r in rows])),
                control_steps=sum(r['steps'] for r in rows),
                termination_counts={k:sum(r['termination']==k for r in rows) for k in sorted({r['termination'] for r in rows})})


def selection_key(summary):
    return (summary['successes'],summary['mean_discounted_return'],summary['mean_elapsed'])
