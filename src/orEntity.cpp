#include "orEntity.h"

#include "orCamera.h"
#include "orRender.h"
#include "orPhysics.h"

#include "constants.h"

void EntitySystem::updateRenderObjects(double const _dt, const orVec3 _origin)
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

#if 0
    if (moon.m_trailId) {
      RenderSystem::Trail& trail = m_renderSystem.getTrail(moon.m_trailId);
      trail.Update(_dt, offset_pos);
    }
#endif

    {
      RenderSystem::Sphere& sphere = m_renderSystem.getSphere(moon.m_sphereId);
      sphere.m_pos = offset_pos;
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

#if 0
    {
      RenderSystem::Trail& trail = m_renderSystem.getTrail(ship.m_trailId);
      trail.m_HACKorigin = _origin; // TODO hmm
      trail.Update(_dt, offset_pos);
    }
#endif

    {
      RenderSystem::Point& point = m_renderSystem.getPoint(ship.m_pointId);
      point.m_pos = offset_pos;
    }
  }

  // Update POIs
  // TODO?
}

void EntitySystem::updateCamTargets(double const _dt, const orVec3 _origin)
{
  // Update Bodies
  for (int i = 0; i < (int)m_instancedBodies.size(); ++i) {
    int id = i + 1;
    Body& moon = getBody(id);

    PhysicsSystem::GravBody& body = m_physicsSystem.getGravBody(moon.m_gravBodyId);
    {
      CameraSystem::Target& camTarget = m_cameraSystem.getTarget(moon.m_cameraTargetId);
      camTarget.m_pos = body.m_pos; // Only RenderSystem objects get the origin shift...is that right?
    }
  }

  // Update ships
  for (int i = 0; i < (int)m_instancedShips.size(); ++i) {
    int id = i + 1;
    Ship& ship = getShip(id);

    PhysicsSystem::ParticleBody& body = m_physicsSystem.getParticleBody(ship.m_particleBodyId);
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

  Vector3d const bodyPos(body.m_pos);
  Vector3d const parentPos(parentBody.m_pos);

  Vector3d const bodyVel(body.m_vel);
  Vector3d const parentVel(parentBody.m_vel);

  Vector3d const v = bodyVel - parentVel;
  Vector3d const r = bodyPos - parentPos;

  orbitParamsFromPosAndVel(r, v, parentBody.m_mass, o_params.m_params);

  // Origin of an orbit is the position of the parent body
  // For all rendering objects we subtract the camera position to reduce error
  // in the render pipeline
  o_params.m_pos = orVec3(Vector3d(parentBody.m_pos) - cam_pos);
}