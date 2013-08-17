#include "orPhysics.h"

#include "util.h"
#include "orProfile/perftimer.h"

#include "constants.h"

void PhysicsSystem::update(IntegrationMethod const integrationMethod, double const dt) {

  int numParticles = (int)m_particleBodies.size();
  int numGravs = (int)m_gravBodies.size();
  int stateSize = numParticles + numGravs;

  // State:
  // numParticles * particle positions
  // numGravs * grav body positions
  // numParticles * particle velocities
  // numGravs * grav body velocities
  Eigen::Array3Xd x_0(3, 2*stateSize);
  Eigen::Array3Xd x_1;

  // Load world state into state array

  int curId = 0;
  for (int i = 0; i < numParticles; ++i, ++curId) {
    Body& body = m_particleBodies[i];
    x_0.col(curId) = Vector3d(body.m_pos);
  }

  for (int i = 0; i < numGravs; ++i, ++curId) {
    Body& body = m_gravBodies[i];
    x_0.col(curId) = Vector3d(body.m_pos);
  }

  for (int i = 0; i < numParticles; ++i, ++curId) {
    Body& body = m_particleBodies[i];
    x_0.col(curId) = Vector3d(body.m_vel);
  }

  for (int i = 0; i < numGravs; ++i, ++curId) {
    Body& body = m_gravBodies[i];
    x_0.col(curId) = Vector3d(body.m_vel);
  }

  Eigen::VectorXd mgravs(numGravs);

  for (int i = 0; i < numGravs; ++i, ++curId) {
    GravBody& body = m_gravBodies[i];
    mgravs[i] = body.m_mass;
  }

  switch (integrationMethod) {
    case IntegrationMethod_ExplicitEuler: { // Comically bad
      // Previous code, for reference:

      // Vector3d const a0 = CalcAccel(i, p0, v0);
      // Vector3d const p1 = p0 + v0 * dt;
      // Vector3d const v1 = v0 + a0 * dt;

      Eigen::Array3Xd dxdt_0(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime, */ x_0, dxdt_0);

      x_1 = x_0 + dxdt_0 * dt;

      break;
    }
    case IntegrationMethod_ImprovedEuler: { // Looks perfect at low speeds. Really breaks down at 16k x speed... is there drift at slightly lower speeds than that?

      // TODO is this just the simplest version of RK?

      // Previous code, for reference:

      // Vector3d const a0 = CalcAccel(i, p0, v0); // TODO this is wrong, needs to store the acceleration/thrust last frame
      // Vector3d const pt = p0 + v0 * dt;
      // Vector3d const vt = v0 + a0 * dt;
      // Vector3d const at = CalcAccel(i, pt, vt);
      // Vector3d const p1 = p0 + .5f * (v0 + vt) * dt;
      // Vector3d const v1 = v0 + .5f * (a0 + at) * dt;

      Eigen::Array3Xd dxdt_0(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime, */ x_0, dxdt_0);

      x_1 = x_0 + dxdt_0 * dt;

      Eigen::Array3Xd x_t = x_0 + dxdt_0 * dt;

      Eigen::Array3Xd dxdt_t(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime + dt, */ x_t, dxdt_t);

      x_1 = x_0 + .5 * (dxdt_0 + dxdt_t) * dt;
      break;
    }
    case IntegrationMethod_RK4: { // Stable up to around 65535x...

      Eigen::Array3Xd k_1(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime, */ x_0, k_1);
      Eigen::Array3Xd k_2(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime + .5 * dt, */ x_0 + k_1 * .5 * dt, k_2);
      Eigen::Array3Xd k_3(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime + .5 * dt, */ x_0 + k_2 * .5 * dt, k_3);
      Eigen::Array3Xd k_4(3, 2*stateSize);
      CalcDxDt(numParticles, numGravs, mgravs, /* m_simTime + dt, */ x_0 + k_3 * dt, k_4);

      x_1 = x_0 + ((k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0) * dt;

      break;
    }
    default: {
      orErr("Unknown Integration Method!");
      break;
    }
  }

  // Store world state from array

  curId = 0;
  for (int i = 0; i < numParticles; ++i, ++curId) {
    Body& body = m_particleBodies[i];

    const double* const bodyPos = x_1.col(curId).data();
    
    body.m_pos[0] = bodyPos[0];
    body.m_pos[1] = bodyPos[1];
    body.m_pos[2] = bodyPos[2];
  }

  for (int i = 0; i < numGravs; ++i, ++curId) {
    Body& body = m_gravBodies[i];
    
    const double* const bodyPos = x_1.col(curId).data();
    
    body.m_pos[0] = bodyPos[0];
    body.m_pos[1] = bodyPos[1];
    body.m_pos[2] = bodyPos[2];
  }

  for (int i = 0; i < numParticles; ++i, ++curId) {
    Body& body = m_particleBodies[i];
    
    const double* const bodyVel = x_1.col(curId).data();
    
    body.m_vel[0] = bodyVel[0];
    body.m_vel[1] = bodyVel[1];
    body.m_vel[2] = bodyVel[2];
  }

  for (int i = 0; i < numGravs; ++i, ++curId) {
    Body& body = m_gravBodies[i];
    
    const double* const bodyVel = x_1.col(curId).data();
    
    body.m_vel[0] = bodyVel[0];
    body.m_vel[1] = bodyVel[1];
    body.m_vel[2] = bodyVel[2];
  }
}

PhysicsSystem::GravBody const& PhysicsSystem::findSOIGravBody(ParticleBody const& _body) const {
  // TODO HACK
  // SOI really requires each body to have a "parent body" for the SOI computation.
  // At the moment we hack in the parent for all grav bodies...
  ensure(numGravBodies() > 0);

  Vector3d const bodyPos(_body.m_pos);

  double minDist = DBL_MAX;
  int minDistId = -1;
  
  for (int i = 0; i < numGravBodies(); ++i)
  {
    GravBody const& soiBody = getGravBody(i);
    GravBody const& parentBody = getGravBody(soiBody.m_soiParentBody);

    Vector3d const soiPos(soiBody.m_pos);
    Vector3d const parentPos(parentBody.m_pos);
    
    double soi;
    if (soiBody.m_soiParentBody == i) {
      // If body is own parent, set infinite SOI
      soi = DBL_MAX;
    } else {
      double const orbitRadius = (parentPos - soiPos).norm();

      // Distances from COM of Earth-Moon system
      double const parentOrbitRadius = orbitRadius * soiBody.m_mass / (parentBody.m_mass + soiBody.m_mass);
      double const childOrbitRadius = orbitRadius - parentOrbitRadius;

      soi = childOrbitRadius * pow(soiBody.m_mass / parentBody.m_mass, 2.0/5.0);
    }

    double const soiDistance = (bodyPos - soiPos).norm();

    if (soiDistance < soi && soiDistance < minDist) {
      minDist = soiDistance;
      minDistId = i;
    }
  }
  ensure(minDistId >= 0); // TODO return Id instead, -1 for none?
  return getGravBody(minDistId);
}

void PhysicsSystem::CalcDxDt(int numParticles, int numGravBodies, Eigen::VectorXd const& mgravs, Eigen::Array3Xd const& x0, Eigen::Array3Xd& dxdt0)
{
  // State: positions, velocities
  // DStateDt: velocities, accelerations
  dxdt0.block(0, 0, 3, numParticles + numGravBodies) = x0.block(0, numParticles + numGravBodies, 3, numParticles + numGravBodies);
  CalcAccel(numParticles, numGravBodies, x0.block(0, 0, 3, numParticles + numGravBodies), x0.block(0, numParticles + numGravBodies, 3, numParticles + numGravBodies), mgravs, dxdt0.block(0, numParticles + numGravBodies, 3, numParticles + numGravBodies));
}

template< class P, class V, class OA >
void PhysicsSystem::CalcAccel(int numParticles, int numGravBodies, P const& p, V const& v, Eigen::VectorXd const& mgravs, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  CalcParticleAccel(numParticles, p.block(0, 0, 3, numParticles), v.block(0, 0, 3, numParticles), numGravBodies, p.block(0, numParticles, 3, numGravBodies), mgravs, o_a.block(0, 0, 3, numParticles));
  CalcGravAccel(numGravBodies, p.block(0, numParticles, 3, numGravBodies), v.block(0, numParticles, 3, numGravBodies), mgravs, o_a.block(0, numParticles, 3, numGravBodies));
}

template< class PP, class VP, class PG, class OA >
void PhysicsSystem::CalcParticleAccel(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  CalcParticleGrav(numParticles, pp, vp, numGravBodies, pg, mg, o_a);
  CalcParticleUserAcc(numParticles, pp, vp, o_a);
}

// Calculates acceleration on first body by second body
template< class PP, class VP, class PG, class OA >
void PhysicsSystem::CalcParticleGrav(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  double const G = GRAV_CONSTANT;

  for (int pi = 0; pi < numParticles; ++pi) {
    Vector3d a(0.0, 0.0, 0.0);
    for (int gi = 0; gi < numGravBodies; ++gi) {

      double const M = mg[gi];

      double const mu = M * G;

      // Calc acceleration due to gravity
      Vector3d const r = (pg.col(gi) - pp.col(pi));
      double const r_mag = r.norm();

      Vector3d const r_dir = r / r_mag;

      Vector3d const a_grav = r_dir * mu / (r_mag * r_mag);
      a += a_grav;
    }
    o_a.col(pi) = a;
  }
}

template< class PP, class VP, class OA >
void PhysicsSystem::CalcParticleUserAcc(int numParticles, PP const& pp, VP const& vp, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  for (int pi = 0; pi < numParticles; ++pi) {
    o_a.col(pi) += Eigen::Array<double, 3, 1>(getParticleBody(pi).m_userAcc);
  }
}

template< class PG, class VG, class OA >
void PhysicsSystem::CalcGravAccel(int numGravBodies, PG const& pg, VG const& vg, Eigen::VectorXd const& mg, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  double const G = GRAV_CONSTANT;

  for (int g1i = 0; g1i < numGravBodies; ++g1i) {
    for (int g2i = 0; g2i < numGravBodies; ++g2i) {
      if (g1i == g2i) { continue; }

      double const M = mg[g2i];

      // Calc acceleration due to gravity
      Vector3d const r = (pg.col(g2i) - pg.col(g1i));
      double const r_mag = r.norm();

      Vector3d const r_dir = r / r_mag;

      double const a_mag = (M * G) / (r_mag * r_mag);
      Vector3d const a_grav = a_mag * r_dir;

      o_a.col(g1i) = a_grav;
    }
  }
}

