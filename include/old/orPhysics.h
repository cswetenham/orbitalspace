#pragma once

#include "orStd.h"
#include "orMath.h"

#include "orCore/orSystem.h"

#include <vector>

class PhysicsSystem {
public:
  struct Body
  {
    Body() : m_pos(), m_vel() {}

    orVec3 m_pos;
    orVec3 m_vel;
  };

  struct ParticleBody : public Body
  {
    ParticleBody() : Body(), m_userAcc() {}
    orVec3 m_userAcc;

    // Computed each frame
    orVec3 m_soiParentPos;
    orEphemerisHybrid m_osculatingOrbit;
  };

  DECLARE_SYSTEM_TYPE(ParticleBody, ParticleBodies);

  struct GravBody : public Body
  {
    GravBody() : Body(), m_radius(0), m_mass(0), m_parentBodyId() {}

    double m_radius;
    double m_mass;
    orEphemerisJPL m_ephemeris; // constant
    orbital::Id<GravBody> m_parentBodyId;
  };

  DECLARE_SYSTEM_TYPE(GravBody, GravBodies);

  enum IntegrationMethod {
    IntegrationMethod_ExplicitEuler = 0,
    IntegrationMethod_ImprovedEuler,
    IntegrationMethod_RK4,
    IntegrationMethod_Count
  };

  void update(IntegrationMethod const integrationMethod, double const t, double const dt);
  GravBody const& findSOIGravBody(ParticleBody const& body) const;

private:

void CalcDxDt(
  int numParticles,
  double t,
  Eigen::Array3Xd const& x0, // initial states (pos+vel)
  Eigen::Array3Xd& dxdt0 // output, rate of change in state
);

template< class P, class OA >
void CalcAccel(
  double t,
  int numParticles,
  P const& p, // position for each body
  OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a // output, accelerations
);

template< class PP, class OA >
void CalcParticleAccel(
  double t,
  int numParticles,
  PP const& pp, // position of particle bodies
  OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a // output, accelerations
);

template< class PP, class OA >
void CalcParticleGrav(double t, int numParticles, PP const& pp, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);

template< class OA >
void CalcParticleUserAcc(int numParticles, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);

void CalcGravEphemerisCartesian(double t, std::vector<orEphemerisCartesian>& out);

}; // class PhysicsSystem