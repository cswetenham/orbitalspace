#pragma once

#include <vector>

#include "orPhysics.h" // TODO don't like this dependency
#include "orRender.h" // TODO don't like this dependency

class CameraSystem;

class EntitySystem {
public:
  EntitySystem(
    CameraSystem& cameraSystem,
    RenderSystem& renderSystem,
    PhysicsSystem& physicsSystem
  ) :
    m_cameraSystem(cameraSystem),
    m_renderSystem(renderSystem),
    m_physicsSystem(physicsSystem)
  {
  }

  struct Ship {
    Ship() : m_particleBodyId(0), m_pointId(0), m_trailId(0), m_orbitId(0), m_cameraTargetId(0) {}
    int m_particleBodyId;
    int m_pointId;
    int m_trailId;
    int m_orbitId;
    int m_cameraTargetId;
  };
  DECLARE_SYSTEM_TYPE(Ship, Ships);

  // Right now the moon orbits the planet, can get rid of distinction later

  // TODO rename, collides with Physics::Body
  struct Body {
    Body() : m_gravBodyId(0), m_sphereId(0), m_orbitId(0), m_trailId(0), m_cameraTargetId(0), m_label3DId(0) {}
    int m_gravBodyId;
    int m_sphereId;
    int m_orbitId;
    int m_trailId;
    int m_cameraTargetId;
    int m_label3DId;
  };
  DECLARE_SYSTEM_TYPE(Body, Bodies);

  // Point of interest; camera-targetable point.
  struct Poi {
    Poi() : m_pointId(0), m_cameraTargetId(0) {}
    int m_pointId;
    int m_cameraTargetId;
  };
  DECLARE_SYSTEM_TYPE(Poi, Pois);

  void updateCamTargets(double const _dt, const orVec3 _origin);
  void updateRenderObjects(double const _dt, const orVec3 _origin);

private:
  // TODO not happy this lives here
  void UpdateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, orEphemerisHybrid& o_params);

  CameraSystem& m_cameraSystem;
  RenderSystem& m_renderSystem;
  PhysicsSystem& m_physicsSystem;
}; // class EntitySystem