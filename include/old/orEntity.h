#pragma once

#include <vector>

#include "orPhysics.h" // TODO don't like this dependency
#include "orRender.h"
#include "orCamera.h" // TODO don't like this dependency

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
    orbital::Id<PhysicsSystem::ParticleBody> m_particleBodyId;
    orbital::Id<RenderSystem::Point> m_pointId;
    orbital::Id<RenderSystem::Orbit> m_orbitId;
    orbital::Id<CameraSystem::Target> m_cameraTargetId;
  };
  DECLARE_SYSTEM_TYPE(Ship, Ships);

  // Right now the moon orbits the planet, can get rid of distinction later

  // TODO rename, collides with Physics::Body
  struct Body {
    orbital::Id<PhysicsSystem::GravBody> m_gravBodyId;
    orbital::Id<RenderSystem::Sphere>  m_sphereId;
    orbital::Id<RenderSystem::Orbit>   m_orbitId;
    orbital::Id<CameraSystem::Target>  m_cameraTargetId;
    orbital::Id<RenderSystem::Label3D> m_label3DId;
  };
  
  DECLARE_SYSTEM_TYPE(Body, Bodies);

  // Point of interest; camera-targetable point.
  struct Poi {
    orbital::Id<RenderSystem::Point> m_pointId;
    orbital::Id<CameraSystem::Target> m_cameraTargetId;
  };
  DECLARE_SYSTEM_TYPE(Poi, Pois);

  void updateCamTargets(double const _dt, const orVec3 _origin);
  void updateRenderObjects(double const _dt, const orVec3 _origin);

private:
  // TODO not happy this lives here
  void updateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, orEphemerisHybrid& o_params);

  CameraSystem& m_cameraSystem;
  RenderSystem& m_renderSystem;
  PhysicsSystem& m_physicsSystem;
}; // class EntitySystem