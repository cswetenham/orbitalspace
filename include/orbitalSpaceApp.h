/* 
 * File:   ssrm1app.h
 * Author: s1149322
 *
 * Created on 12 December 2011, 17:11
 */

#ifndef ORBITALSPACEAPP_H
#define	ORBITALSPACEAPP_H

#include "app.h"
#include "rnd.h"
#include "util.h"

#include "orStd.h"
#include "orMath.h"

#include "orRender.h"

#include <vector>

#include <SFML/Audio/Music.hpp>


// TODO Renderables, etc should go into their own systems.
// Code outside the system should refer to the instances only by opaque Id - not iterate over the collection (when needed this should happen internally in the system)

class OrbitalSpaceApp :
  public App
{
public:
  OrbitalSpaceApp();
  virtual ~OrbitalSpaceApp();
  
  // From App
public:
  virtual void Run();

protected:
  virtual void InitRender();
  virtual void ShutdownRender();

  virtual void InitState();
  virtual void ShutdownState();

  virtual void HandleEvent(sf::Event const& _event);
  virtual void UpdateState(double const _dt);
    
  virtual void RenderState();

private:
  struct Body;
  struct GravBody;  
  GravBody const& FindSOIGravBody(Vector3d const& p);
  void UpdateOrbit(Body const& body, RenderSystem::Orbit& o_params);
  void LookAt(Vector3d pos, Vector3d target, Vector3d up);

  void CalcDxDt(int numParticles, int numGravBodies, Eigen::VectorXd const& mgravs, double m_simTime, Eigen::Array3Xd const& x0, Eigen::Array3Xd& dxdt0);
  template< class P, class V, class OA >
  void CalcAccel(int numParticles, int numGravBodies, P const& p, V const& v, Eigen::VectorXd const& mgravs, OA& o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleAccel(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA& o_a);
  template< class PP, class VP, class PG, class OA >
  void CalcParticleGrav(int numParticles, PP const& pp, VP const& vp, int numGravBodies, PG const& pg, Eigen::VectorXd const& mg, OA& o_a);
  
  template< class PG, class VG, class OA >
  void CalcGravAccel(int numGravBodies, PG const& pg, VG const& vg, Eigen::VectorXd const& mg, OA& o_a);

  Vector3d CalcThrust(Vector3d p, Vector3d v);

private:
  Rnd64 m_rnd;
  
  double m_simTime;

  struct Config
  {
    int width;
    int height;
  };
  
  Config m_config;

  // Simulation options
  bool m_paused;
  bool m_singleStep;
  // Rendering options
  bool m_wireframe;
  bool m_camOrig;

  double m_camDist;
  double m_camTheta;
  double m_camPhi;
  
  //// Physics ////

  double m_timeScale;

  struct Body
  {
    Vector3d m_pos;
    Vector3d m_vel;
  };

  struct ParticleBody : public Body
  {
  };
  std::vector<ParticleBody> m_particleBodies;

  int makeParticleBody() { m_particleBodies.push_back(ParticleBody()); return m_particleBodies.size() - 1; }
  ParticleBody& getParticleBody(int i) { return m_particleBodies[i]; }

  struct GravBody : public Body
  {
    double m_radius;
    double m_mass;
  };
  std::vector<GravBody> m_gravBodies;

  int makeGravBody() { m_gravBodies.push_back(GravBody()); return m_gravBodies.size() - 1; }
  GravBody& getGravBody(int i) { return m_gravBodies[i]; }

  // TODO
  class PhysicsSystem {};
  PhysicsSystem m_physicsSystem;

  //// Rendering ////
  
  RenderSystem m_renderSystem;
  
  int m_comPointId;
  int m_lagrangePointIds[5];
  
  //// Entities ////

  struct ShipEntity {
    int m_particleBodyId;
    int m_pointId;
    int m_trailId;
    int m_orbitId;
  };
  std::vector<ShipEntity> m_shipEntities;

  int makeShip() { m_shipEntities.push_back(ShipEntity()); return m_shipEntities.size() - 1; }
  ShipEntity& getShip(int i) { return m_shipEntities[i]; }

  int m_playerShipId;
  int m_suspectShipId;

  // Right now the moon orbits the planet, can get rid of distinction later

  struct PlanetEntity {
    int m_gravBodyId;
    int m_sphereId;
  };
  std::vector<PlanetEntity> m_planetEntities;

  int makePlanet() { m_planetEntities.push_back(PlanetEntity()); return m_planetEntities.size() - 1; }
  PlanetEntity& getPlanet(int i) { return m_planetEntities[i]; }
    
  struct MoonEntity {
    int m_gravBodyId;
    int m_sphereId;
    int m_orbitId;
    int m_trailId;
  };
  std::vector<MoonEntity> m_moonEntities;

  int makeMoon() { m_moonEntities.push_back(MoonEntity()); return m_moonEntities.size() - 1; }
  MoonEntity& getMoon(int i) { return m_moonEntities[i]; }

  class EntitySystem {};
  EntitySystem m_entitySystem;

  int m_earthPlanetId;
  int m_moonMoonId;

  // Camera

  Body* m_camTarget;
  size_t m_camTargetId;
  std::vector<std::string> m_camTargetNames;

  enum CameraMode {
    CameraMode_FirstPerson = 0,
    CameraMode_ThirdPerson = 1
  };
  CameraMode m_camMode;

  enum InputMode {
    InputMode_Default = 0,
    InputMode_RotateCamera = 1
  };
  InputMode m_inputMode;

  sf::Vector2i m_savedMousePos;
  
  enum Thrusters
  {
    ThrustFwd = 1 << 0,
    ThrustBack = 1 << 1,
    ThrustLeft = 1 << 2,
    ThrustRight = 1 << 3,
    ThrustUp = 1 << 4,
    ThrustDown = 1 << 5
  };

  uint32_t m_thrusters;

  enum IntegrationMethod {
    IntegrationMethod_ExplicitEuler = 0,
    // IntegrationMethod_ImplicitEuler,
    IntegrationMethod_ImprovedEuler,
    // IntegrationMethod_WeirdVerlet,
    // IntegrationMethod_VelocityVerlet,
    IntegrationMethod_RK4,
    IntegrationMethod_Count
  };
  IntegrationMethod m_integrationMethod;
  
  // TODO make into a palette array.
  // TODO Convert to HSV so can modify the hue to make new palettes.
  enum {PALETTE_SIZE = 5};
  Vector3f m_colG[PALETTE_SIZE];
  Vector3f m_colR[PALETTE_SIZE];
  Vector3f m_colB[PALETTE_SIZE];
  
  Vector3d m_light;

  bool m_hasFocus;

  sf::Music m_music;
};

#endif	/* ORBITALSPACEAPP_H */

