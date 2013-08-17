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
    m_numShips(0),

    m_cameraSystem(cameraSystem),
    m_renderSystem(renderSystem),
    m_physicsSystem(physicsSystem)
  {
    initShips();
  }

// TODO all this could go into a templated container
public:
  struct Ship {
    int m_particleBodyId;
    int m_pointId;
    int m_trailId;
    int m_orbitId;
    int m_cameraTargetId;
  };

  struct ShipId { int value; };

  void initShips() {
    // Set up free list
    for (int i = 0; i < MAX_SHIPS; ++i) {
      m_shipIdxById[i] = i + 1;
    }
    m_shipIdxById[MAX_SHIPS] = 0;
  }

  ShipId makeShip() {
    // TODO s/handle/id/g

    // TODO error if m_numShips == MAX_SHIPS

    // Claim the new space
    int const newShipIdx = m_numShips;
    m_numShips++;

    // Zero out the new object, for sanity
    memset( &m_ships[newShipIdx], 0, sizeof(Ship) );
    
    // Claim the first free handle
    // TODO error if free list empty (newShipId == 0)
    int const newShipId = m_shipIdxById[0];
    m_shipIdxById[0] = m_shipIdxById[newShipId];
    
    // Set up the indirection
    m_shipIdxById[newShipId] = newShipIdx;
    m_shipIdByIdx[newShipIdx] = newShipId;

    // Id is just the raw index right now
    // Would be safer to have unused bits in the Id (indices > MAX_SHIPS)
    // serve as a counter incremented on allocation
    // (To catch stale Ids that alias live ones)
    ShipId result = { newShipId };
    
    return result;
  }

  // TODO need a better system (or just more careful thought) for multithreading
  // At one point was considering giving all objects a "disabled" flag which would mean
  // it would act as nonexistent but not be freed, and a "dead" flag which would mean
  // it was requested to be freed but shouldn't be actually freed until it is safe to do so...
  // But perhaps the "dead" flag would be better implemented outside the system, so long as the
  // user has a way of knowing or enforcing a time when it is safe to free?
  void unmakeShip(ShipId shipId) {
    // To unmake a ship, we move the last ship to the one to be freed
    // and update the indirection for the ship we moved.
    // We also add the old Id to the free list.

    // TODO check handle is valid

    int const oldShipId = shipId.value;
    int const oldShipIdx = m_shipIdxById[oldShipId];

    // Copy last ship over ship to be freed
    int srcShipIdx = m_numShips;
    memcpy( &m_ships[oldShipIdx], &m_ships[srcShipIdx], sizeof(Ship));

    // Copy corresponding id in id array
    m_shipIdByIdx[oldShipIdx] = m_shipIdByIdx[srcShipIdx];
    
    // Update the indirection for the ship we moved
    int srcShipId = m_shipIdByIdx[srcShipIdx];
    m_shipIdxById[srcShipId] = oldShipIdx;

    // Free the old ship in the free list
    m_shipIdxById[oldShipId] = m_shipIdxById[0];
    m_shipIdxById[0] = oldShipId;
  }

  // TODO how to handle case of invalid id?
  // Either hard crash, or return a NULL pointer...
  Ship&       getShip(ShipId id)       { return m_ships[m_shipIdxById[id.value]]; }
  Ship const& getShip(ShipId id) const { return m_ships[m_shipIdxById[id.value]]; }

private:
  enum { MAX_SHIPS = 128 };
  // All ships in m_ships from index 0 to index m_numShips - 1 are in use; the other are free.
  int m_numShips;
  Ship  m_ships[MAX_SHIPS];
  
  // new structure:
  // Each handle/id contains a value that is an index into this array.
  // Handle value 0 is always invalid.
  // Handles can be deallocated in any order, so need a way to find a free one.
  // Array entry 0 points to the first free handle; that one points to the next; etc.
  // If a handle in the free list points to 0 it is the last one.
  // Manipulations in the free list only touch entry 0 and one other entry.
  // Most annoying part: array must have size MAX_SHIPS+1.
  int m_shipIdxById[MAX_SHIPS+1];
  // This array maps the reverse direction, needed when freeing an object.
  int m_shipIdByIdx[MAX_SHIPS];

public:
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
    int m_label3DId;
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

  void update(double const _dt, double const _origin[3]);

private:
  // TODO not happy this lives here
  void UpdateOrbit(PhysicsSystem::Body const& body, PhysicsSystem::GravBody const& parentBody, RenderSystem::Orbit& o_params);

  CameraSystem& m_cameraSystem;
  RenderSystem& m_renderSystem;
  PhysicsSystem& m_physicsSystem;

  std::vector<Planet> m_planets;
  std::vector<Moon> m_moons;
  
  std::vector<Poi> m_pois;
}; // class EntitySystem