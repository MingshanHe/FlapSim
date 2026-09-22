// Thin C ABI around the unmodified Purdue Wing.cpp; no reimplementation.
#include "Wing.hpp"
class AuditWing : public FWMAV::Wing {
 public:
  explicit AuditWing(const double* p) : FWMAV::Wing(
      0, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], 0, 0) {}
  void density(double rho) { air_density_ = rho; }
};
extern "C" {
void* wing_create(const double* geometry, double rho) {
  auto* w = new AuditWing(geometry);
  if (rho > 0) w->density(rho);
  return w;
}
void wing_delete(void* ptr) { delete static_cast<AuditWing*>(ptr); }
void wing_eval(void* ptr, double stroke, double stroke_rate, double pitch,
               double pitch_rate, double* output) {
  auto* w = static_cast<AuditWing*>(ptr);
  w->UpdateStates(0,0,0,0,0,0,0,0,stroke,stroke_rate,0,0,pitch,pitch_rate);
  w->UpdateAeroForce();
  output[0]=w->GetNormalForce(); output[1]=w->GetSpanCoP();
  output[2]=w->GetChordCoP(); output[3]=w->GetM_aero();
  output[4]=w->GetM_rd(); output[5]=w->GetAoA();
}
}
