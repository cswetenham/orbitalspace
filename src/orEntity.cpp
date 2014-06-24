#include "orEntity.h"

#include "orCamera.h"
#include "orRender.h"
#include "orPhysics.h"

#include "constants.h"

void EntitySystem::update(double const _dt, const orVec3 _origin)
{
  // Update Bodies
  for (int i = 0; i < (int)m_instancedBodies.size(); ++i) {
    int id = i + 1;
    Body& moon = getBody(id);

    PhysicsSystem::GravBody& body = m_physicsSystem.getGravBody(moon.m_gravBodyId);
    orVec3 offset_pos = orVec3(Vector3d(body.m_pos) - Vector3d(_origin));

    if (moon.m_orbitId && body.m_soiParentBody) {
      RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(moon.m_orbitId);
      UpdateOrbit(body, m_physicsSystem.getGravBody(body.m_soiParentBody), _origin, orbit);
    }

    if (moon.m_trailId) {
      RenderSystem::Trail& trail = m_renderSystem.getTrail(moon.m_trailId);
      trail.Update(_dt, offset_pos);
    }

    {
      RenderSystem::Sphere& sphere = m_renderSystem.getSphere(moon.m_sphereId);
      sphere.m_pos = offset_pos;
    }

    {
      CameraSystem::Target& camTarget = m_cameraSystem.getTarget(moon.m_cameraTargetId);
      camTarget.m_pos = body.m_pos; // Only RenderSystem objects get the origin shift...is that right?
    }

    if (moon.m_label3DId)
    {
      RenderSystem::Label3D& label3d = m_renderSystem.getLabel3D(moon.m_label3DId);
      label3d.m_pos = offset_pos;
    }
  }

  // Update ships
  for (int i = 0; i < (int)m_instancedShips.size(); ++i) {
    int id = i + 1;
    Ship& ship = getShip(id);

    PhysicsSystem::ParticleBody& body = m_physicsSystem.getParticleBody(ship.m_particleBodyId);
    orVec3 offset_pos = orVec3(Vector3d(body.m_pos) - Vector3d(_origin));

    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(ship.m_orbitId);
    UpdateOrbit(body, m_physicsSystem.findSOIGravBody(body), _origin, orbit);

    {
      RenderSystem::Trail& trail = m_renderSystem.getTrail(ship.m_trailId);
      trail.m_HACKorigin = _origin; // TODO hmm
      trail.Update(_dt, offset_pos);
    }

    {
      RenderSystem::Point& point = m_renderSystem.getPoint(ship.m_pointId);
      point.m_pos = offset_pos;
    }

    {
      CameraSystem::Target& camTarget = m_cameraSystem.getTarget(ship.m_cameraTargetId);
      camTarget.m_pos = body.m_pos; // Only RenderSystem objects get the origin shift...is that right?
    }
  }

  // Update POIs
  // TODO?
}

void EntitySystem::UpdateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, Vector3d const& cam_pos, RenderSystem::Orbit& o_params)
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

  o_params.e = e;
  o_params.p = p;
  o_params.theta = theta;

  o_params.x_dir = x_dir;

  o_params.y_dir = y_dir;

  o_params.m_pos = orVec3(Vector3d(parentBody.m_pos) - cam_pos);
}