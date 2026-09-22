"""Controlled fixed-base comparisons with Purdue's actual C++ aerodynamic core.

These are experiment adapters, not replacements for general free-flight fluid.
"""
from pathlib import Path
import ctypes
import subprocess
import numpy as np
from rect_simulation import ProfileBladeElement
from flapsim.kinematics import point_velocity

FOLDER = Path(__file__).resolve().parent
OUT = FOLDER/'flappy_audit'
REFERENCE = OUT/'reference'


def build_reference():
    path=REFERENCE/'libwing_audit.so'
    subprocess.run(['g++','-std=c++11','-O2','-shared','-fPIC',str(REFERENCE/'Wing.cpp'),
                    str(REFERENCE/'bridge.cpp'),'-o',str(path)],check=True)
    return path


class NativeWing:
    def __init__(self, geometry, density=1.225):
        self.lib=ctypes.CDLL(str(REFERENCE/'libwing_audit.so'))
        pointer=ctypes.POINTER(ctypes.c_double)
        self.lib.wing_create.argtypes=[pointer,ctypes.c_double]
        self.lib.wing_create.restype=ctypes.c_void_p
        self.lib.wing_eval.argtypes=[ctypes.c_void_p]+[ctypes.c_double]*4+[pointer]
        self.lib.wing_delete.argtypes=[ctypes.c_void_p]
        self.geometry=np.ascontiguousarray(geometry,dtype=np.float64)
        self.handle=self.lib.wing_create(self.geometry.ctypes.data_as(pointer),density)
        self.output=np.empty(6)
        self.pointer=self.output.ctypes.data_as(pointer)

    def evaluate(self, stroke, rate, pitch, pitch_rate):
        self.lib.wing_eval(self.handle,stroke,rate,pitch,pitch_rate,self.pointer)
        return self.output

    def close(self):
        if self.handle:
            self.lib.wing_delete(self.handle)
            self.handle=None


def cn(alpha):
    return 1.8*np.sin(2*alpha)*np.cos(alpha)+1.95*np.sin(alpha)-1.5*np.cos(2*alpha)*np.sin(alpha)


def dcp(alpha):
    return .46-.332*np.cos(alpha)-.037*np.cos(3*alpha)-.013*np.cos(5*alpha)


class StripAuditFluid(ProfileBladeElement):
    """Change one choice at a time, keeping the original strip discretization."""
    def __init__(self, config, profile, *, pressure=False, exclude_pitch=False, chord_drag=True):
        super().__init__(config,profile)
        self.pressure=pressure
        self.exclude_pitch=exclude_pitch
        self.chord_drag=chord_drag

    def bind(self, model):
        super().bind(model)
        self.widths=[entry[-2]/(entry[0].span/self.config.strips) for entry in self.profiles]

    def forces(self,data,xp=np):
        result=np.zeros((self.nbody,6))
        for entry,width in zip(self.profiles,self.widths):
            wing,body,root,local,selector,area,rotation_integral=entry
            R=data.xmat[body].reshape(3,3)
            n,c,s=(R@np.asarray(a) for a in (wing.normal_axis,wing.chord_axis,wing.span_axis))
            points=local@R.T+data.xpos[body]
            velocity=point_velocity(data,body,root,points)-np.asarray(self.config.wind)
            rate=data.cvel[body,:3]@s
            if self.exclude_pitch:
                velocity-=np.cross(rate*s,points-data.xpos[body])
            vn,vc=velocity@n,velocity@c
            alpha=np.arctan2(vn,np.abs(vc)+1e-12)
            fn=-.5*self.config.density*area*(vn*vn+vc*vc)*cn(alpha)
            fc=-.5*self.config.density*area*self.config.chord_drag*np.abs(vc)*vc if self.chord_drag else np.zeros_like(vc)
            force=fn[:,None]*n+fc[:,None]*c
            application=points
            if self.pressure:
                # Preserve which chord edge is upstream, unlike folded CN angle.
                unfolded=np.arctan2(vn,-vc)
                application=points+((dcp(unfolded)-.5)*width)[:,None]*c
            moment=np.cross(application-data.xipos[body],force).sum(0)
            damping=-.125*self.config.density*self.config.rotational_drag*rotation_integral*abs(rate)*rate
            result+=selector[:,None]*np.r_[force.sum(0),moment+damping*s]
        return result


class PurdueBenchFluid(ProfileBladeElement):
    """Use original Wing.cpp force/CoP/damping with this bench's frame mapping.

    Purdue local (x,y,z)=IST local (-x,+y,-z), stroke=-IST stroke,
    pitch=IST pitch. The wing-root's 7 mm distance from the stroke axis is
    included in span moments. Offset of the membrane LE is added to CoP.
    """
    def __init__(self, config, profile, *, density=None):
        super().__init__(config,profile)
        self.native_density=config.density if density is None else density

    def bind(self,model):
        super().bind(model)
        if len(self.profiles)!=1 or self.profiles[0][0].body!='left_wing':
            raise ValueError('PurdueBenchFluid is restricted to the fixed single-left-wing audit')
        self.body=model.body('left_wing').id
        self.stroke_q=int(model.joint('left_flapping_joint').qposadr[0])
        self.stroke_v=int(model.joint('left_flapping_joint').dofadr[0])
        self.pitch_q=int(model.joint('left_feathering_joint').qposadr[0])
        self.pitch_v=int(model.joint('left_feathering_joint').dofadr[0])
        p=np.load(self.profile)
        r,c,z=p['radius'],p['width'],p['center']
        dr=r[1]-r[0]
        self.root_offset=.00699999999933
        radial=r+self.root_offset
        length=radial[-1]+dr/2
        mean=c.sum()*dr/length
        moments=[float(np.sum(c*(radial/length)**k)*dr/(length*mean)) for k in (3,2,1,0)]
        cp_moments=[float(np.sum(c*c*(radial/length)**k)*dr/(length*mean**2)) for k in (2,1,0)]
        rd=float(np.sum(c**4)*dr/(length*mean**4))
        self.geometry=[length,mean,*moments,*cp_moments,rd]
        self.native=NativeWing(self.geometry,self.native_density)
        self.leading=float(np.sum(c*radial**2*(z-c/2))/np.sum(c*radial**2))
        self.area_r2=float(np.sum(c*radial**2)*dr)

    def forces(self,data,xp=np):
        fn,r_cp,z_cp,ma,mrd,alpha=self.native.evaluate(
            -data.qpos[self.stroke_q],-data.qvel[self.stroke_v],
            data.qpos[self.pitch_q],data.qvel[self.pitch_v])
        R=data.xmat[self.body].reshape(3,3)
        force=-fn*R[:,0]
        point=data.xpos[self.body]+R@np.array([0.,r_cp-self.root_offset,self.leading+z_cp])
        moment=np.cross(point-data.xipos[self.body],force)+mrd*R[:,1]
        result=np.zeros((self.nbody,6))
        result[self.body]=np.r_[force,moment]
        return result
