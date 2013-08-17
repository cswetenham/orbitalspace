#include "orEntity.h"

#include "orCamera.h"
#include "orRender.h"
#include "orPhysics.h"

#include "constants.h"

void EntitySystem::update(double const _dt, double const _origin[3])
{
  // Update Planets
  for (int i = 0; i < (int)m_planets.size(); ++i) {
    Planet& planet = getPlanet(i);

    PhysicsSystem::Body& body = m_physicsSystem.getGravBody(planet.m_gravBodyId);

    RenderSystem::Sphere& sphere = m_renderSystem.getSphere(planet.m_sphereId);
    {
      sphere.m_pos[0] = body.m_pos[0];
      sphere.m_pos[1] = body.m_pos[1];
      sphere.m_pos[2] = body.m_pos[2];
    }
    
    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(planet.m_cameraTargetId);
    {
      camTarget.m_pos[0] = body.m_pos[0];
      camTarget.m_pos[1] = body.m_pos[1];
      camTarget.m_pos[2] = body.m_pos[2];
    }
  }
  
  // Update Moons
  for (int i = 0; i < (int)m_moons.size(); ++i) {
    Moon& moon = getMoon(i);

    PhysicsSystem::GravBody& body = m_physicsSystem.getGravBody(moon.m_gravBodyId);

    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(moon.m_orbitId);
    UpdateOrbit(body, m_physicsSystem.getGravBody(body.m_soiParentBody), orbit);

    RenderSystem::Trail& trail = m_renderSystem.getTrail(moon.m_trailId);
    trail.Update(_dt, Vector3d(body.m_pos));

    RenderSystem::Sphere& sphere = m_renderSystem.getSphere(moon.m_sphereId);
    {
      sphere.m_pos[0] = body.m_pos[0];
      sphere.m_pos[1] = body.m_pos[1];
      sphere.m_pos[2] = body.m_pos[2];
    }

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(moon.m_cameraTargetId);
    {
      camTarget.m_pos[0] = body.m_pos[0];
      camTarget.m_pos[1] = body.m_pos[1];
      camTarget.m_pos[2] = body.m_pos[2];
    }
  }

  // Update ships
  for (int i = 0; i < (int)m_ships.size(); ++i) {
    Ship& ship = m_ships[i];

    PhysicsSystem::ParticleBody& body = m_physicsSystem.getParticleBody(ship.m_particleBodyId);
     
    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(ship.m_orbitId);
    UpdateOrbit(body, m_physicsSystem.findSOIGravBody(body), orbit);

    RenderSystem::Trail& trail = m_renderSystem.getTrail(ship.m_trailId);
    {
      trail.m_HACKorigin[0] = _origin[0];
      trail.m_HACKorigin[1] = _origin[1];
      trail.m_HACKorigin[2] = _origin[2];

      trail.Update(_dt, Vector3d(body.m_pos));
    }
    
    RenderSystem::Point& point = m_renderSystem.getPoint(ship.m_pointId);
    {
      point.m_pos[0] = body.m_pos[0];
      point.m_pos[1] = body.m_pos[1];
      point.m_pos[2] = body.m_pos[2];
    }

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(ship.m_cameraTargetId);
    {
      camTarget.m_pos[0] = body.m_pos[0];
      camTarget.m_pos[1] = body.m_pos[1];
      camTarget.m_pos[2] = body.m_pos[2];
    }
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

  Vector3d const bodyPos(body.m_pos);
  Vector3d const parentPos(parentBody.m_pos);

  Vector3d const bodyVel(body.m_vel);
  Vector3d const parentVel(parentBody.m_vel);

  Vector3d const v = bodyVel - parentVel;

  Vector3d const r = parentPos - bodyPos;
  double const r_mag = r.norm();

  Vector3d const r_dir = r/r_mag;

  double const vr_mag = r_dir.dot(v);
  Vector3d const vr = r_dir * vr_mag; // radial velocity
  Vector3d const vt = v - vr; // tangent velocity
  double const vt_mag = vt.norm();
  Vector3d const t_dir = vt/vt_mag;

  double const p = pow(r_mag * vt_mag, 2) / mu;
  double const v0 = sqrt(mu/p); // todo compute more accurately/efficiently?

  Vector3d const ex = ((vt_mag - v0) * r_dir - vr_mag * t_dir) / v0;
  double const e = ex.norm();

  double const ec = (vt_mag / v0) - 1;
  double const es = (vr_mag / v0);
  double const theta = atan2(es, ec);

  Vector3d const x_dir = cos(theta) * r_dir - sin(theta) * t_dir;
  Vector3d const y_dir = sin(theta) * r_dir + cos(theta) * t_dir;

  const double* const x_dir_data = x_dir.data();
  const double* const y_dir_data = y_dir.data();

  o_params.e = e;
  o_params.p = p;
  o_params.theta = theta;

  o_params.x_dir[0] = x_dir_data[0];
  o_params.x_dir[1] = x_dir_data[1];
  o_params.x_dir[2] = x_dir_data[2];

  o_params.y_dir[0] = y_dir_data[0];
  o_params.y_dir[1] = y_dir_data[1];
  o_params.y_dir[2] = y_dir_data[2];

  o_params.m_pos[0] = parentBody.m_pos[0];
  o_params.m_pos[1] = parentBody.m_pos[1];
  o_params.m_pos[2] = parentBody.m_pos[2];
}