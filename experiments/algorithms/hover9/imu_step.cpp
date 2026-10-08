// Instrument the original force laws without reading position or velocity truth.
// High-rate virtual IMU preintegration is an optimistic sensor-front-end model.
#include <mujoco/mujoco.h>
#include <cmath>
#include <cstring>
struct ImuState {
  int site, acceleration_sensor, gyro_sensor;
  double acceleration_bias[3], attitude_error[9];
  double acceleration[3], velocity[3], position[3], specific_force[3];
  double gyro[3], attitude[3];
  double tau, attitude_tau;
};
static thread_local ImuState* active_imu=nullptr;
static void instrumented_step2(const mjModel* m,mjData* d) {
  mj_step2(m,d);
  if (!active_imu) return;
  ImuState& s=*active_imu;
  // sensorAcc and site_xmat are both evaluated at the same preintegration state.
  const double* r=d->site_xmat+9*s.site;
  const double* f=d->sensordata+m->sensor_adr[s.acceleration_sensor];
  const double* omega=d->sensordata+m->sensor_adr[s.gyro_sensor];
  const double dt=m->opt.timestep;
  const double blend=1-std::exp(-dt/s.tau);
  const double attitude_blend=1-std::exp(-dt/s.attitude_tau);
  const double measured_angles[3]={std::atan2(r[7],r[8]),std::asin(std::fmax(-1.,std::fmin(1.,-r[6]))),std::atan2(r[3],r[0])};
  double body[3], corrected[3];
  for(int i=0;i<3;++i) body[i]=f[i]+s.acceleration_bias[i];
  for(int i=0;i<3;++i) {
    corrected[i]=0.;
    for(int j=0;j<3;++j) corrected[i]+=s.attitude_error[3*i+j]*body[j];
  }
  for(int i=0;i<3;++i) {
    double a=m->opt.gravity[i];
    for(int j=0;j<3;++j) a+=r[3*i+j]*corrected[j];
    // Causal filtering before decimation, in a common inertial frame.
    s.acceleration[i]+=blend*(a-s.acceleration[i]);
    s.position[i]+=s.velocity[i]*dt+.5*s.acceleration[i]*dt*dt;
    s.velocity[i]+=s.acceleration[i]*dt;
    s.specific_force[i]+=blend*(body[i]-s.specific_force[i]);
    s.gyro[i]+=attitude_blend*(omega[i]-s.gyro[i]);
    s.attitude[i]+=attitude_blend*std::remainder(measured_angles[i]-s.attitude[i],2*3.141592653589793);
  }
}
#define fs_advance fs_imu_inner
#define mj_step instrumented_step2
#define mj_forward mj_step1
#include "../../../flapsim/rl/native_step.cpp"
#undef fs_advance
#undef mj_step
#undef mj_forward
extern "C" void fs_advance_imu(const mjModel* m,mjData* d,int steps,
    const Motor* motor,double command,int voltage_mode,const Wing* wings,int nwings,
    double density,double chord_drag,double rotational_drag,const double* wind,
    const double* joint_forces,const double* body_forces,const double* controls,
    ImuState* imu) {
  active_imu=imu;
  mj_step1(m,d);
  fs_imu_inner(m,d,steps,motor,command,voltage_mode,wings,nwings,
               density,chord_drag,rotational_drag,wind,joint_forces,body_forces,controls);
  mj_forward(m,d);
  active_imu=nullptr;
}
