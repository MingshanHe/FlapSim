"""Test powered hover-reference initial conditions without stabilizing forces.

Zero centroidal momentum is an initial-condition experiment, not a proof that
the periodic hovering orbit has been found. The free body is never reset later.
"""
import numpy as np
import mujoco
from flapsim.kinematics import euler_matrix
from .activation import SignedHoverEnv
from .residual import ResidualHoverEnv


class HoverInitialEnv(SignedHoverEnv):
    def __init__(self, trim, initialization='release', **kwargs):
        if initialization not in ('release','trim','momentum'):
            raise ValueError('unknown hover initialization')
        self.initialization=initialization
        self.reference_rpy=np.asarray(trim.get('hover_attitude_rad',[0.,0.,0.]))
        self.reference_angles=np.asarray(trim['command'][1:3])
        super().__init__(trim,**kwargs)
        if initialization!='release':
            reference=ResidualHoverEnv(trim,warmup_seconds=kwargs.get('warmup_seconds',.1),simulation_config=kwargs.get('simulation_config'))
            try:
                for joint in self.internal:
                    self._initial_qpos[joint.qposadr[0]]=reference._initial_qpos[joint.qposadr[0]]
                    self._initial_qvel[joint.dofadr[0]]=reference._initial_qvel[joint.dofadr[0]]
            finally: reference.close()

    def reset(self, *, seed=None, options=None):
        obs,info=super().reset(seed=seed,options=options)
        if self.initialization=='release': return obs,info
        model,data=self.sim.model,self.sim.data
        qpos,qvel=data.qpos.copy(),data.qvel.copy()
        quaternion=np.zeros(4)
        rpy=self.angles()+self.reference_rpy
        mujoco.mju_mat2Quat(quaternion,euler_matrix(rpy).ravel())
        qpos[self.free_qpos+3:self.free_qpos+7]=quaternion
        self.sim.reset(qpos=qpos,qvel=qvel)
        angular=np.zeros((3,model.nv)); linear=np.zeros_like(angular)
        mujoco.mj_angmomMat(model,data,angular,self.body)
        mujoco.mj_jacSubtreeCom(model,data,linear,self.body)
        matrix=np.vstack((model.body_subtreemass[self.body]*linear,angular))
        before=matrix@qvel
        free=slice(self.free_dof,self.free_dof+6)
        internal=qvel.copy(); internal[free]=0.
        correction=np.zeros(6)
        if self.initialization=='momentum':
            correction=-np.linalg.solve(matrix[:,free],matrix@internal)
            qvel[free]+=correction
            self.sim.reset(qpos=qpos,qvel=qvel)
        self.sim.set_speed(self.center[2])
        self.sim.set_gimbal(pitch=self.reference_angles[0],roll=self.reference_angles[1])
        self.filtered_action[:2]=self.reference_angles/.3
        self.applied_action[1:]=self.reference_angles/.3
        clean=self._last_observation=self._observation()
        self.previous_observation=clean.copy()
        mujoco.mj_subtreeVel(model,data)
        after=np.r_[model.body_subtreemass[self.body]*data.subtree_linvel[self.body],data.subtree_angmom[self.body]]
        np.testing.assert_allclose(after,matrix@qvel,atol=1e-12)
        info.update(rpy=rpy,initialization=self.initialization,
                    initial_momentum_before=before.tolist(),initial_momentum_after=after.tolist(),
                    root_velocity_correction=correction.tolist())
        return self.sensor(clean),info
