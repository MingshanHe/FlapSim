// Sensor-only instrumentation. Aerodynamic and motor force laws are unchanged.
#include <mujoco/mujoco.h>
#include <cmath>
#include <algorithm>
struct SensorState {
  int site, accel_id, gyro_id, counter, period;
  double accel_limit, gyro_limit, filter_tau, attitude_tau;
  double acc1[3], acc2[3], gyro1[3], gyro2[3];
  double force[3], gyro[3], attitude[3], velocity[3], position[3];
  double bias[3], angle_bias[3];
  long long clipped_acc, clipped_gyro, samples;
};
static thread_local SensorState* current_sensor=nullptr;
static void sensed_step2(const mjModel* m,mjData* d) {
  mj_step2(m,d);
  if (!current_sensor) return;
  auto& s=*current_sensor;
  const double* f=d->sensordata+m->sensor_adr[s.accel_id];
  const double* g=d->sensordata+m->sensor_adr[s.gyro_id];
  const double blend=1-std::exp(-m->opt.timestep/s.filter_tau);
  for(int i=0;i<3;++i) {
    s.clipped_acc+=std::abs(f[i])>s.accel_limit;
    s.clipped_gyro+=std::abs(g[i])>s.gyro_limit;
    const double af=std::clamp(f[i],-s.accel_limit,s.accel_limit)+s.bias[i];
    const double gf=std::clamp(g[i],-s.gyro_limit,s.gyro_limit);
    s.acc1[i]+=blend*(af-s.acc1[i]); s.acc2[i]+=blend*(s.acc1[i]-s.acc2[i]);
    s.gyro1[i]+=blend*(gf-s.gyro1[i]); s.gyro2[i]+=blend*(s.gyro1[i]-s.gyro2[i]);
  }
  if(++s.counter<s.period) return;
  s.counter=0; ++s.samples;
  const double dt=s.period*m->opt.timestep;
  const double* r=d->site_xmat+9*s.site;
  // Explicit virtual AHRS: sampled orientation plus bias, not a simulated AHRS algorithm.
  const double angles[3]={std::atan2(r[7],r[8]),std::asin(std::clamp(-r[6],-1.,1.)),std::atan2(r[3],r[0])};
  const double ab=1-std::exp(-dt/s.attitude_tau);
  double measured[3];
  for(int i=0;i<3;++i) {
    measured[i]=angles[i]+s.angle_bias[i];
    s.attitude[i]+=ab*std::remainder(measured[i]-s.attitude[i],2*M_PI);
    s.force[i]=s.acc2[i]; s.gyro[i]=s.gyro2[i];
  }
  const double cr=std::cos(measured[0]),sr=std::sin(measured[0]);
  const double cp=std::cos(measured[1]),sp=std::sin(measured[1]);
  const double cy=std::cos(measured[2]),sy=std::sin(measured[2]);
  const double R[9]={cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr};
  for(int i=0;i<3;++i) {
    double a=m->opt.gravity[i];
    for(int j=0;j<3;++j) a+=R[i*3+j]*s.force[j];
    s.position[i]+=s.velocity[i]*dt+.5*a*dt*dt; s.velocity[i]+=a*dt;
  }
}
#define fs_advance fs_sensor_inner
#define mj_step sensed_step2
#define mj_forward mj_step1
#include "../../../flapsim/rl/native_step.cpp"
#undef fs_advance
#undef mj_step
#undef mj_forward
extern "C" void fs_advance_sensor(const mjModel* m,mjData* d,int steps,
    const Motor* motor,double command,int voltage_mode,const Wing* wings,int nwings,
    double density,double chord_drag,double rotational_drag,const double* wind,
    const double* joint_forces,const double* body_forces,const double* controls,SensorState* sensor) {
  current_sensor=sensor; mj_step1(m,d);
  fs_sensor_inner(m,d,steps,motor,command,voltage_mode,wings,nwings,density,chord_drag,rotational_drag,wind,joint_forces,body_forces,controls);
  mj_forward(m,d); current_sensor=nullptr;
}
