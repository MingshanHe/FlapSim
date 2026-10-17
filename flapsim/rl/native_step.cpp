// Same DC motor, blade elements and mj_step/mj_forward order as CPUBackend.
// Deliberately no fast-math: numerical equivalence is checked against NumPy.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstring>

struct Wing {
  int body, root, count;
  const double *points, *areas;
  double normal[3], chord[3], span[3], rotational_area;
};
struct Motor {
  int dof;
  double resistance, kt, ke, i0, voltage, current, torque, velocity, gain;
};
static double dot(const double* a, const double* b) {
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static void cross(double* c, const double* a, const double* b) {
  c[0]=a[1]*b[2]-a[2]*b[1]; c[1]=a[2]*b[0]-a[0]*b[2]; c[2]=a[0]*b[1]-a[1]*b[0];
}
static void rotate(double* out, const double* r, const double* p) {
  for(int j=0;j<3;++j) out[j]=dot(r+3*j,p);
}
extern "C" int fs_version() { return mj_version(); }
extern "C" void fs_advance(const mjModel* m, mjData* d, int steps,
    const Motor* motor, double command, int voltage_mode, const Wing* wings, int nwings,
    double density, double chord_drag, double rotational_drag, const double* wind,
    const double* joint_forces, const double* body_forces, const double* controls) {
  std::memcpy(d->ctrl,controls,m->nu*sizeof(double));
  for(int t=0;t<steps;++t) {
    std::memcpy(d->qfrc_applied,joint_forces,m->nv*sizeof(double));
    std::memcpy(d->xfrc_applied,body_forces,m->nbody*6*sizeof(double));
    double v=d->qvel[motor->dof], loss=motor->i0*std::tanh(v);
    double target=std::clamp(command,-motor->velocity,motor->velocity);
    double voltage=voltage_mode ? command : motor->resistance*(motor->gain*(target-v)/motor->kt+loss)+motor->ke*v;
    voltage=std::clamp(voltage,-motor->voltage,motor->voltage);
    double current=std::clamp((voltage-motor->ke*v)/motor->resistance,-motor->current,motor->current);
    d->qfrc_applied[motor->dof]+=std::clamp(motor->kt*(current-loss),-motor->torque,motor->torque);
    for(int w=0;w<nwings;++w) {
      const Wing& wing=wings[w]; const double* r=d->xmat+9*wing.body;
      const double* cv=d->cvel+6*wing.body;
      double normal[3],chord[3],span[3],force_sum[3]={},moment[3]={};
      rotate(normal,r,wing.normal); rotate(chord,r,wing.chord); rotate(span,r,wing.span);
      for(int i=0;i<wing.count;++i) {
        double p[3],arm[3],velocity[3],force[3],moment_i[3];
        rotate(p,r,wing.points+3*i);
        for(int j=0;j<3;++j) { p[j]+=d->xpos[3*wing.body+j]; arm[j]=p[j]-d->subtree_com[3*wing.root+j]; }
        cross(velocity,cv,arm);
        for(int j=0;j<3;++j) velocity[j]+=cv[3+j]-wind[j];
        double vn=dot(velocity,normal),vc=dot(velocity,chord);
        double alpha=std::atan2(vn,std::abs(vc)+1e-12);
        double cn=1.8*std::sin(2*alpha)*std::cos(alpha)+1.95*std::sin(alpha)-1.5*std::cos(2*alpha)*std::sin(alpha);
        double fn=-.5*density*wing.areas[i]*(vn*vn+vc*vc)*cn;
        double fc=-.5*density*wing.areas[i]*chord_drag*std::abs(vc)*vc;
        for(int j=0;j<3;++j) { force[j]=fn*normal[j]+fc*chord[j]; arm[j]=p[j]-d->xipos[3*wing.body+j]; force_sum[j]+=force[j]; }
        cross(moment_i,arm,force);
        for(int j=0;j<3;++j) moment[j]+=moment_i[j];
      }
      double rate=dot(cv,span), damping=-.125*density*rotational_drag*wing.rotational_area*std::abs(rate)*rate;
      for(int j=0;j<3;++j) { d->xfrc_applied[6*wing.body+j]+=force_sum[j]; d->xfrc_applied[6*wing.body+3+j]+=moment[j]+damping*span[j]; }
    }
    mj_step(m,d);
    mj_forward(m,d);
  }
}
