#include "orPhysics.h"

#include "util.h"
#include "orProfile/perftimer.h"

#include "constants.h"

// TODO new concept is to have some bodies 'on rails' with their position
// computed according to the current mean anomaly (this is the parameter than
// increases at a constant rate with time, the rate is the mean motion 'n')
// 'on rails' bodies could be using JPL planets approximations
// for 3000BC to 3000AD from http://ssd.jpl.nasa.gov/?planet_pos
// or for satellites, using http://ssd.jpl.nasa.gov/?sat_elem (TODO: will need
// a different structure for the data, or reuse other JPL structure; will need
// ability to define the transformation from the ecliptic to the inertial plane
// to apply it to the Moon, and whatever transforms are required for other
// satellites)

// TODO I didn't apply the final part of the JPL computation; values are in the J2000 ecliptic rather than J2000 frame?
// difference is 23.umpt degrees, which isn't the same as the earth's ecliptic inclination of 7.umpt degrees...

// Let's assume we can stuff the parameters for satellites into the existing JPL structure setting most
// So we will have on-rails bodies, with:
// - mass
// - parent body (Sun, Earth, etc)
// - parent frame (ICRF, parent Ecliptic, laplace plane?) (laplace planes seem to need some extra parameters of their own)
// - orbital elements

// And propagated bodies, with:
// - mass
// - parent body (for orbit drawing - the actual parent can change and should be recomputed based on SOI each frame)
// - orbital elements (cartesian)

// For both on-rails and propagated bodies, we can have some bodies which
// contribute to gravity computation and some which don't - currently called
// ParticleBody and GravBody.

// Every frame, we will want to compute the new position + velocity of each body.

// For the RK4 integration for propagated bodies, we want to be able to compute the accelerations from gravity at intermediate times - this should include the calculation of the on-rails bodies (at least the ones that contribute to gravity calcs)

// Some bodies - let's say the ParticleBodies - can also have a thrust from user or AI input / nodes.
// Might eventually just have impulse based calculations instead? Or proper prediction + arcs with finite thrust over time?

// TODO simplification:
// - grav bodies will always be based on JPL data or equivalent
// - particle bodies will always be based on cartesian position + RK4 propagation
// - Once that's implemented, will worry about maneuver nodes
// -- Probably the best way of handling that is: if a time step contains a node,
// split the time step into time before node and time after node. Simulate the
// first part, then apply node modifiers, then simulate the later part. Probably
// best to give the app some priority queue of events, and use that to drive
// calls to the physics system. During any timestep the user thrust should be
// constant. If we want to simulate finite thrust, will have two events for a
// node: start thrust and end thrust. Should be symmetric about the node, if we
// simulate light lag we'll have to take into consideration that thrust can't
// start earlier than the command horizon


void PhysicsSystem::update(IntegrationMethod const integrationMethod, double const t, double const dt) {

  int const numParticles = (int)numParticleBodies();
  int const numGravs = (int)numGravBodies();
  int const stateSize = numParticles;

  // State:
  // numParticles * particle positions
  // numParticles * particle velocities
  Eigen::Array3Xd x_0(3, 2*stateSize);
  Eigen::Array3Xd x_1;

  // Load world state into state array

  int curIdx = 0;
  for (int i = 0; i < numParticles; ++i, ++curIdx) {
    Body& body = m_instancedParticleBodies[i];
    x_0.col(curIdx) = Vector3d(body.m_pos);
  }

  for (int i = 0; i < numParticles; ++i, ++curIdx) {
    Body& body = m_instancedParticleBodies[i];
    x_0.col(curIdx) = Vector3d(body.m_vel);
  }

  switch (integrationMethod) {
    case IntegrationMethod_ExplicitEuler: { // Comically bad
      // Previous code, for reference:

      // Vector3d const a0 = CalcAccel(i, p0, v0);
      // Vector3d const p1 = p0 + v0 * dt;
      // Vector3d const v1 = v0 + a0 * dt;

      Eigen::Array3Xd dxdt_0(3, 2*stateSize);
      CalcDxDt(numParticles, t, x_0, dxdt_0);

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
      CalcDxDt(numParticles, t, x_0, dxdt_0);

      x_1 = x_0 + dxdt_0 * dt;

      Eigen::Array3Xd x_t = x_0 + dxdt_0 * dt;

      Eigen::Array3Xd dxdt_t(3, 2*stateSize);
      CalcDxDt(numParticles, t + dt, x_t, dxdt_t);

      x_1 = x_0 + .5 * (dxdt_0 + dxdt_t) * dt;
      break;
    }
    case IntegrationMethod_RK4: { // Stable up to around 65535x...

      Eigen::Array3Xd k_1(3, 2*stateSize);
      CalcDxDt(numParticles, t,           x_0,                 k_1);
      Eigen::Array3Xd k_2(3, 2*stateSize);
      CalcDxDt(numParticles, t + .5 * dt, x_0 + k_1 * .5 * dt, k_2);
      Eigen::Array3Xd k_3(3, 2*stateSize);
      CalcDxDt(numParticles, t + .5 * dt, x_0 + k_2 * .5 * dt, k_3);
      Eigen::Array3Xd k_4(3, 2*stateSize);
      CalcDxDt(numParticles, t + dt,      x_0 + k_3 * dt,      k_4);

      x_1 = x_0 + ((k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0) * dt;

      break;
    }
    default: {
      orErr("Unknown Integration Method!");
      break;
    }
  }

  // Store world state from array

  curIdx = 0;
  for (int i = 0; i < numParticles; ++i, ++curIdx) {
    Body& body = m_instancedParticleBodies[i];

    const double* const bodyPos = x_1.col(curIdx).data();

    body.m_pos[0] = bodyPos[0];
    body.m_pos[1] = bodyPos[1];
    body.m_pos[2] = bodyPos[2];
  }

  for (int i = 0; i < numParticles; ++i, ++curIdx) {
    Body& body = m_instancedParticleBodies[i];

    const double* const bodyVel = x_1.col(curIdx).data();

    body.m_vel[0] = bodyVel[0];
    body.m_vel[1] = bodyVel[1];
    body.m_vel[2] = bodyVel[2];
  }

  // Update grav body state at end of timestep
  std::vector<orEphemerisCartesian> gravCartesian;
  gravCartesian.resize(numGravs);
  for (int gi = 0; gi < numGravs; ++gi) {
    int const gid = gi + 1;
    GravBody& gravBody = getGravBody(gid);
    ephemerisCartesianFromJPL(gravBody.m_ephemeris, t+dt, gravCartesian[gi]);
    if (gravBody.m_parentBodyId != 0) {
      gravCartesian[gi].pos += gravCartesian[gravBody.m_parentBodyId-1].pos;
      gravCartesian[gi].vel += gravCartesian[gravBody.m_parentBodyId-1].vel;
    }
    gravBody.m_pos = orVec3(gravCartesian[gi].pos);
    gravBody.m_vel = orVec3(gravCartesian[gi].vel);
  }
}

PhysicsSystem::GravBody const& PhysicsSystem::findSOIGravBody(ParticleBody const& _body) const {
  // TODO HACK
  // SOI really requires each body to have a "parent body" for the SOI computation.
  // At the moment we hack in the parent for all grav bodies...
  ensure(numGravBodies() > 0);

  Vector3d const bodyPos(_body.m_pos);

  double minDist = DBL_MAX;
  int minDistId = 0;

  for (int i = 0; i < numGravBodies(); ++i)
  {
    int const curId = i + 1;
    GravBody const& soiBody = getGravBody(curId);
    Vector3d const soiPos(soiBody.m_pos);

    double soi;
    if (soiBody.m_parentBodyId == 0) {
      // If body has no own parent, set infinite SOI
      soi = DBL_MAX;
    } else {
      GravBody const& parentBody = getGravBody(soiBody.m_parentBodyId);
      Vector3d const parentPos(parentBody.m_pos);

      double const orbitRadius = (parentPos - soiPos).norm();

      // Distances from COM of Earth-Moon system
      double const parentOrbitRadius = orbitRadius * soiBody.m_mass / (parentBody.m_mass + soiBody.m_mass);
      double const childOrbitRadius = orbitRadius - parentOrbitRadius;

      soi = childOrbitRadius * pow(soiBody.m_mass / parentBody.m_mass, 2.0/5.0);
    }

    double const soiDistance = (bodyPos - soiPos).norm();

    if (soiDistance < soi && soiDistance < minDist) {
      minDist = soiDistance;
      minDistId = curId;
    }
  }
  ensure(minDistId > 0); // TODO return Id instead, -1 for none?
  return getGravBody(minDistId);
}

void PhysicsSystem::CalcDxDt(
  int numParticles,
  double t,
  Eigen::Array3Xd const& x0, // initial states (pos+vel)
  Eigen::Array3Xd& dxdt0 // output, rate of change in state
) {
  // State: positions, velocities
  // DStateDt: velocities, accelerations
  dxdt0.block(0, 0, 3, numParticles) = x0.block(0, numParticles, 3, numParticles);
  CalcAccel(t, numParticles, x0.block(0, 0, 3, numParticles), dxdt0.block(0, numParticles, 3, numParticles));
}

template< class P, class OA >
void PhysicsSystem::CalcAccel(
  double t,
  int numParticles,
  P const& p, // position for each body
  OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a // output, accelerations
)
{
  CalcParticleAccel(t, numParticles, p.block(0, 0, 3, numParticles), o_a.block(0, 0, 3, numParticles));
}

template< class PP, class OA >
void PhysicsSystem::CalcParticleAccel(
  double t,
  int numParticles,
  PP const& pp, // position of particle bodies
  OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a // output, accelerations
)
{
  CalcParticleGrav(t, numParticles, pp, o_a);
  CalcParticleUserAcc(numParticles, o_a);
}

// Calculates acceleration on first body by second body
// TODO make this make sense again. Vectorise better?
template< class PP, class OA >
void PhysicsSystem::CalcParticleGrav(double t, int numParticles, PP const& pp, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  double const G = GRAV_CONSTANT;

  // TODO pull out this code into its own function
  int numGravs = (int)numGravBodies();
  std::vector<orEphemerisCartesian> gravCartesian;
  gravCartesian.resize(numGravs);
  for (int gi = 0; gi < numGravs; ++gi) {
    int const gid = gi + 1;
    GravBody const& gravBody = getGravBody(gid);
    ephemerisCartesianFromJPL(gravBody.m_ephemeris, t, gravCartesian[gi]);
    if (gravBody.m_parentBodyId != 0) {
      gravCartesian[gi].pos += gravCartesian[gravBody.m_parentBodyId-1].pos;
      gravCartesian[gi].vel += gravCartesian[gravBody.m_parentBodyId-1].vel;
    }
  }

  for (int pi = 0; pi < numParticles; ++pi) {
    Vector3d a(0.0, 0.0, 0.0);
    for (int gi = 0; gi < numGravs; ++gi) {
      int const gid = gi + 1;
      double const M = getGravBody(gid).m_mass;

      double const mu = M * G;

      // Calc acceleration due to gravity
      Vector3d const r = gravCartesian[gi].pos - Vector3d(pp.col(pi));
      double const r_mag = r.norm();

      Vector3d const r_dir = r / r_mag;

      Vector3d const a_grav = r_dir * mu / (r_mag * r_mag);
      a += a_grav;
    }
    o_a.col(pi) = a;
  }
}

template< class OA >
void PhysicsSystem::CalcParticleUserAcc(int numParticles, OA /* would be & but doesn't work with temporary from Eigen's .block() */ o_a)
{
  // TODO need a better way of iterating over instances
  for (int pi = 0; pi < numParticles; ++pi) {
    int pid = pi + 1;
    o_a.col(pi) += Eigen::Array<double, 3, 1>(getParticleBody(pid).m_userAcc.data);
  }
}

