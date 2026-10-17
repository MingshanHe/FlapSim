// Wingbeat-synchronous motor drive on top of the unchanged sensor/physics kernel.
// Only the motor speed target varies per 5 us substep:
//   target = command + m_sin*sin(phi) + m_cos*cos(phi) + (time < split ? offset_a : offset_b)
//   phi    = phi0 + omega*(time - t0)          (estimated crank phase, extrapolated)
// With all drive terms zero this is the same call sequence as fs_advance_sensor.
#include "../topology_hover/sensor_step.cpp"

extern "C" void fs_advance_phase(const mjModel* m,mjData* d,int steps,
    const Motor* motor,double command,int voltage_mode,const Wing* wings,int nwings,
    double density,double chord_drag,double rotational_drag,const double* wind,
    const double* joint_forces,const double* body_forces,const double* controls,
    SensorState* sensor,const double* drive) {
  // drive = {phi0, omega, t0, m_sin, m_cos, offset_a, offset_b, split}
  current_sensor=sensor; mj_step1(m,d);
  for(int k=0;k<steps;++k) {
    const double phi=drive[0]+drive[1]*(d->time-drive[2]);
    const double target=command+drive[3]*std::sin(phi)+drive[4]*std::cos(phi)
        +(d->time<drive[7] ? drive[5] : drive[6]);
    fs_sensor_inner(m,d,1,motor,target,voltage_mode,wings,nwings,density,chord_drag,rotational_drag,
                    wind,joint_forces,body_forces,controls);
  }
  mj_forward(m,d); current_sensor=nullptr;
}
