#include "orEntity.h"

#include "orCamera.h"
#include "orRender.h"
#include "orPhysics.h"

#include "constants.h"

// TODO want ability to record past simulation states and extrapolate future ones...
// Ideally will need to support entities being created or destroyed...
// Will want to support AI; we'll want to 

void EntitySystem::update(double const _dt, Vector3d const _origin)
{
  // Update Planets
  for (int i = 0; i < (int)m_planets.size(); ++i) {
    Planet& planet = getPlanet(i);

    PhysicsSystem::Body& body = m_physicsSystem.getGravBody(planet.m_gravBodyId);
      
    RenderSystem::Sphere& sphere = m_renderSystem.getSphere(planet.m_sphereId);
    sphere.m_pos = body.m_pos;

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(planet.m_cameraTargetId);
    camTarget.m_pos = body.m_pos;
  }

  // Update Moons
  for (int i = 0; i < (int)m_moons.size(); ++i) {
    Moon& moon = getMoon(i);

    PhysicsSystem::GravBody& body = m_physicsSystem.getGravBody(moon.m_gravBodyId);

    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(moon.m_orbitId);
    UpdateOrbit(body, m_physicsSystem.getGravBody(body.m_soiParentBody), orbit);

    RenderSystem::Trail& trail = m_renderSystem.getTrail(moon.m_trailId);
    trail.Update(_dt, body.m_pos);

    RenderSystem::Sphere& sphere = m_renderSystem.getSphere(moon.m_sphereId);
    sphere.m_pos = body.m_pos;

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(moon.m_cameraTargetId);
    camTarget.m_pos = body.m_pos;
  }

  // Update ships
  for (int i = 0; i < (int)m_ships.size(); ++i) {
    Ship& ship = getShip(i);

    PhysicsSystem::ParticleBody& body = m_physicsSystem.getParticleBody(ship.m_particleBodyId);
     
    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(ship.m_orbitId);
    UpdateOrbit(body, m_physicsSystem.findSOIGravBody(body), orbit);

    RenderSystem::Trail& trail = m_renderSystem.getTrail(ship.m_trailId);
    trail.m_HACKorigin = _origin;
    trail.Update(_dt, body.m_pos);
    
    RenderSystem::Point& point = m_renderSystem.getPoint(ship.m_pointId);
    point.m_pos = body.m_pos;

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(ship.m_cameraTargetId);
    camTarget.m_pos = body.m_pos;
  }

  // Update POIs

}

void EntitySystem::UpdateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, RenderSystem::Orbit& o_params)
{
  // TODO will want to just forward-project instead, this is broken with >1 body

  // Find body whose sphere of influence we are in
  // This is the one with the smallest sphere of influence

  // Compute Kepler orbit

  double const G = GRAV_CONSTANT;
  double const M = parentBody.m_mass;
    
  double const mu = M * G;

  Vector3d v = body.m_vel - parentBody.m_vel;

  Vector3d r = parentBody.m_pos - body.m_pos;
  double const r_mag = r.norm();

  Vector3d r_dir = r/r_mag;

  double const vr_mag = r_dir.dot(v);
  Vector3d vr = r_dir * vr_mag; // radial velocity
  Vector3d vt = v - vr; // tangent velocity
  double const vt_mag = vt.norm();
  Vector3d t_dir = vt/vt_mag;

  double const p = pow(r_mag * vt_mag, 2) / mu;
  double const v0 = sqrt(mu/p); // todo compute more accurately/efficiently?

  Vector3d ex = ((vt_mag - v0) * r_dir - vr_mag * t_dir) / v0;
  double const e = ex.norm();

  double const ec = (vt_mag / v0) - 1;
  double const es = (vr_mag / v0);
  double const theta = atan2(es, ec);

  Vector3d x_dir = cos(theta) * r_dir - sin(theta) * t_dir;
  Vector3d y_dir = sin(theta) * r_dir + cos(theta) * t_dir;

  o_params.e = e;
  o_params.p = p;
  o_params.theta = theta;
  o_params.x_dir = x_dir;
  o_params.y_dir = y_dir;
  o_params.m_pos = parentBody.m_pos;
}