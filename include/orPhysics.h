#pragma once

#include "orStd.h"
#include "orMath.h"

#include <vector>

class PhysicsSystem {
public:
  struct Body
  {
    Vector3d m_pos;
    Vector3d m_vel;
  };

  struct ParticleBody : public Body
  {
    Vector3d m_userAcc;
  };
  
  int makeParticleBody() { m_particleBodies.push_back(ParticleBody()); return m_particleBodies.size() - 1; }
  ParticleBody& getParticleBody(int i) { return m_particleBodies[i]; }

  struct GravBody : public Body
  {
    double m_radius;
    double m_mass;
  };

  int makeGravBody() { m_gravBodies.push_back(GravBody()); return m_gravBodies.size() - 1; }
  GravBody& getGravBody(int i) { return m_gravBodies[i]; }

  enum IntegrationMethod {
    IntegrationMethod_ExplicitEuler = 0,
    IntegrationMethod_ImprovedEuler,
    IntegrationMethod_RK4,
    IntegrationMethod_Count
  };

  void update(IntegrationMethod const integrationMethod, double const dt);

private:
  void CalcDxDt(int numParticles, int numGravBodies, Eigen::VectorXd const& mgravs, Eigen::Array3Xd const& x0, Eigen::Array3Xd& dxdt0);
  template< class P, class V, class OA >
  void CalcAccel(int numParticles, int numGravBodies, P const& p, V const& v, Eigen::VectorXd const& mgravs, OA& o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleAccel(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA& o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleGrav(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA& o_a);
  template< class PP, class VP, class OA >
  void CalcParticleUserAcc(int numParticles, PP const& pp, VP const& vp, OA& o_a);
  template< class PG, class VG, class OA >
  void CalcGravAccel(int numGravBodies, PG const& pg, VG const& vg, Eigen::VectorXd const& mg, OA& o_a);

  std::vector<ParticleBody> m_particleBodies;
  std::vector<GravBody> m_gravBodies;
}; // class PhysicsSystem