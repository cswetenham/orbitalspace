#pragma once

#include "orStd.h"
#include "orMath.h"

#include "orCore/orSystem.h"

#include <vector>

class PhysicsSystem {
public:
  struct Body
  {
    double m_pos[3];
    double m_vel[3];
  };

  struct ParticleBody : public Body
  {
    double m_userAcc[3];
  };
  
  DECLARE_SYSTEM_TYPE(ParticleBody, ParticleBodies);

  struct GravBody : public Body
  {
    double m_radius;
    double m_mass;
    int m_soiParentBody; // TODO want to avoid this later.
  };
  
  DECLARE_SYSTEM_TYPE(GravBody, GravBodies);

  struct KeplerBody : public Body
  {
    
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
  template< class P, class V, class OA >
  void CalcAccel(int numParticles, int numGravBodies, P const& p, V const& v, Eigen::VectorXd const& mgravs, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleAccel(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleGrav(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PP, class VP, class OA >
  void CalcParticleUserAcc(int numParticles, PP const& pp, VP const& vp, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
  template< class PG, class VG, class OA >
  void CalcGravAccel(int numGravBodies, PG const& pg, VG const& vg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a);
}; // class PhysicsSystem