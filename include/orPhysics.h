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
  };

  DECLARE_SYSTEM_TYPE(ParticleBody, ParticleBodies);

  struct GravBody : public Body
  {
    GravBody() : Body(), m_radius(0), m_mass(0), m_soiParentBody(0) {}

    double m_radius;
    double m_mass;
    int m_soiParentBody; // TODO want to avoid this later.
  };

  DECLARE_SYSTEM_TYPE(GravBody, GravBodies);

  struct KeplerBody : public Body
  {
    KeplerBody() : Body() {}
  };

  DECLARE_SYSTEM_TYPE(KeplerBody, KeplerBodies);

  enum IntegrationMethod {
    IntegrationMethod_ExplicitEuler = 0,
    IntegrationMethod_ImprovedEuler,
    IntegrationMethod_RK4,
    IntegrationMethod_Count
  };

  void update(IntegrationMethod const integrationMethod, double const dt);
  GravBody const& findSOIGravBody(ParticleBody const& body) const;

private:
  void CalcDxDt(int numParticles, int numGravBodies, Eigen::VectorXd const& mgravs, Eigen::Array3Xd const& x0, Eigen::Array3Xd& dxdt0);
  template< class P, class OA >
  void CalcAccel(int numParticles, int numGravBodies, P const& p, Eigen::VectorXd const& mgravs, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PP, class PG, class OA >
  void CalcParticleAccel(int numParticles, PP const& pp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PP, class PG, class OA >
  void CalcParticleGrav(int numParticles, PP const& pp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class OA >
  void CalcParticleUserAcc(int numParticles, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PG, class OA >
  void CalcGravAccel(int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
}; // class PhysicsSystem