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
    int m_particleBodyId;
    int m_pointId;
    int m_trailId;
    int m_orbitId;
    int m_cameraTargetId;
  };

  int numShips() const { return (int)m_ships.size(); }
  int makeShip() { m_ships.push_back(Ship()); return numShips() - 1; }
  Ship&       getShip(int id)       { return m_ships[id]; }
  Ship const& getShip(int id) const { return m_ships[id]; }

  // Right now the moon orbits the planet, can get rid of distinction later

  struct Planet {
    int m_gravBodyId;
    int m_sphereId;
    int m_cameraTargetId;
  };

  int numPlanets() const { return (int)m_planets.size(); }
  int makePlanet() { m_planets.push_back(Planet()); return numPlanets() - 1; }
  Planet&       getPlanet(int id)       { return m_planets[id]; }
  Planet const& getPlanet(int id) const { return m_planets[id]; }

  struct Moon {
    int m_gravBodyId;
    int m_sphereId;
    int m_orbitId;
    int m_trailId;
    int m_cameraTargetId;
    int m_labelId;
  };

  int numMoons() const { return (int)m_moons.size(); }
  int makeMoon() { m_moons.push_back(Moon()); return numMoons() - 1; }
  Moon&       getMoon(int id)       { return m_moons[id]; }
  Moon const& getMoon(int id) const { return m_moons[id]; }

  // Point of interest; camera-targetable point.
  struct Poi {
    int m_pointId;
    int m_cameraTargetId;
  };

  int numPois() const { return (int)m_pois.size(); }
  int makePoi() { m_pois.push_back(Poi()); return numPois() - 1; }
  Poi&       getPoi(int id)       { return m_pois[id]; }
  Poi const& getPoi(int id) const { return m_pois[id]; }

  void update(double const _dt, Vector3d const _origin);

private:
  // TODO not happy this lives here
  void UpdateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, RenderSystem::Orbit& o_params);

  CameraSystem& m_cameraSystem;
  RenderSystem& m_renderSystem;
  PhysicsSystem& m_physicsSystem;

  std::vector<Planet> m_planets;
  std::vector<Moon> m_moons;
  std::vector<Ship> m_ships;
  std::vector<Poi> m_pois;
}; // class EntitySystem