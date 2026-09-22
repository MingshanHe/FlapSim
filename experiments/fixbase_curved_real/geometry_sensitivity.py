"""Diagnostic only: rerun 10/15 Hz with CAD strip widths, without fitting data."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import numpy as np

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/fixbase'))
from flapsim import Simulator
from flapsim.fluid.blade_element import BladeElement
from flapsim.kinematics import point_velocity
from single_wing_sweep import single_config, OUTPUT as SINGLE
from run_15hz import cycle_statistics
from inspect_planform import planform
OUTPUT=FOLDER/'analysis/geometry_sensitivity'


class CADBladeElement(BladeElement):
    def __init__(self,config,mode):
        super().__init__(config)
        self.mode=mode

    def bind(self,model):
        super().bind(model)
        r,c,z,dr=planform(ROOT/'flapsim/robot/IST_robot/meshes/left_wing.stl')
        self.profiles=[]
        for wing,body,root,points,selector in self.elements:
            radius=(np.arange(self.config.strips)+.5)*wing.span/self.config.strips
            width=np.interp(radius,r,c,left=0,right=0)
            center=np.interp(radius,r,z,left=0,right=0)
            local=points.copy()
            if self.mode=='full_planform':
                local += (center-wing.chord_offset)[:,None]*np.asarray(wing.chord_axis)
            area=width*wing.span/self.config.strips
            rotation_integral=(width**4).sum()*wing.span/self.config.strips if self.mode=='full_planform' else wing.span*wing.chord**4
            self.profiles.append((wing,body,root,local,selector,area,rotation_integral))

    def forces(self,data,xp=np):
        result=xp.zeros((self.nbody,6))
        c=self.config
        for wing,body,root,local,selector,area,rotation_integral in self.profiles:
            rotation=data.xmat[body].reshape(3,3)
            points=xp.asarray(local)@rotation.T+data.xpos[body]
            velocity=point_velocity(data,body,root,points,xp)-xp.asarray(c.wind)
            normal=rotation@xp.asarray(wing.normal_axis)
            chord=rotation@xp.asarray(wing.chord_axis)
            span=rotation@xp.asarray(wing.span_axis)
            vn,vc=velocity@normal,velocity@chord
            alpha=xp.arctan2(vn,xp.abs(vc)+1e-12)
            cn=1.8*xp.sin(2*alpha)*xp.cos(alpha)+1.95*xp.sin(alpha)-1.5*xp.cos(2*alpha)*xp.sin(alpha)
            fn=-.5*c.density*xp.asarray(area)*(vn**2+vc**2)*cn
            fc=-.5*c.density*xp.asarray(area)*c.chord_drag*xp.abs(vc)*vc
            force=fn[:,None]*normal+fc[:,None]*chord
            moment=xp.sum(xp.cross(points-data.xipos[body],force),axis=0)
            rate=data.cvel[body,:3]@span
            damping=-.125*c.density*c.rotational_drag*rotation_integral*xp.abs(rate)*rate
            wrench=xp.concatenate((xp.sum(force,axis=0),moment+damping*span))
            result+=xp.asarray(selector)[:,None]*wrench
        return result


def run(target,mode,*,model_path=None,output=OUTPUT):
    config=single_config(model_path or SINGLE/'model/IST_robot_single_left_wing.xml')
    command=2*np.pi*16*target+.4*target**2
    prior=[]
    folder=output/f'{mode}_{target}hz';folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        sim=Simulator(config,fluid=CADBladeElement(config.fluid,mode))
        records=[]
        for i in range(4001):
            if i:
                phase=min((i-1)/2000/.15,1)
                sim.set_speed(command*(.5-.5*np.cos(np.pi*phase)))
                sim.step(100)
            aero=sim.fluid.forces(sim.data)
            records.append([sim.data.time,sim.data.joint('motor_joint').qpos[0],
                            aero[:,2].sum(),sim.data.joint('left_feathering_joint').qpos[0]])
        a=np.asarray(records)
        assert np.isfinite(a).all() and not np.any(sim.data.warning.number)
        data=dict(zip(('time_s','motor_joint_angle_rad','aero_Fz_N','pitch_rad'),a.T))
        stats=cycle_statistics(data,target_hz=target)
        np.savez_compressed(folder/f'run_{attempt+1}.npz',**data)
        prior.append({'command_rad_s':float(command),**stats})
        (folder/'calibration.json').write_text(json.dumps(prior,indent=2))
        f=stats['measured_wing_frequency_hz']
        print(mode,target,attempt+1,round(f,5),round(stats['mean_lift_mN'],3),'mN',flush=True)
        if abs(f-target)<.01:
            result={**stats,'mode':mode,'source':str((folder/f'run_{attempt+1}.npz').relative_to(FOLDER)),
                    'pitch_min_max_deg':np.degrees([a[a[:,0]>.6,3].min(),a[a[:,0]>.6,3].max()]).tolist()}
            (folder/'result.json').write_text(json.dumps(result,indent=2))
            return result
        slope=1/(2*np.pi*16+.8*target)
        if len(prior)>1:
            slope=(f-prior[-2]['measured_wing_frequency_hz'])/(command-prior[-2]['command_rad_s'])
        command+=(target-f)/slope
    raise RuntimeError('Sensitivity frequency calibration did not converge')


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,hz,mode) for hz in (10,15) for mode in ('width_only','full_planform')]
        for f in as_completed(futures):
            results.append(f.result())
            (OUTPUT/'summary.json').write_text(json.dumps(results,indent=2))
