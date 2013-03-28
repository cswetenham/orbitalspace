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
#include "orGfx.h"

#include <SFML/Audio/Music.hpp>

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
  struct RenderableOrbit;
  void UpdateOrbit(Body const& body, RenderableOrbit& o_params);
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

  void DrawCircle(double const radius, int const steps);
  void DrawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks);

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

  // TODO?
  class PhysicsWorld {};
  PhysicsWorld m_physicsWorld;

  //// Rendering ////

  struct RenderablePoint {
    Vector3d m_pos;

    Vector3f m_col;
  };
  std::vector<RenderablePoint> m_renderablePoints;

  int m_comPointIdx;

  int makePoint() { m_renderablePoints.push_back(RenderablePoint()); return m_renderablePoints.size() - 1; }
  RenderablePoint& getPoint(int i) { return m_renderablePoints[i]; }

  struct RenderableSphere {
    double m_radius;
    Vector3d m_pos;

    Vector3f m_col;
  };
  std::vector<RenderableSphere> m_renderableSpheres;

  int makeSphere() { m_renderableSpheres.push_back(RenderableSphere()); return m_renderableSpheres.size() - 1; }
  RenderableSphere& getSphere(int i) { return m_renderableSpheres[i]; }

  struct RenderableOrbit
  {
    double p;
    double e;
    double theta;
    Vector3d x_dir;
    Vector3d y_dir;
    Vector3d m_pos;

    Vector3f m_col;
  };
  std::vector<RenderableOrbit> m_renderableOrbits;

  int makeOrbit() { m_renderableOrbits.push_back(RenderableOrbit()); return m_renderableOrbits.size() - 1; }
  RenderableOrbit& getOrbit(int i) { return m_renderableOrbits[i]; }

  struct RenderableTrail
  {
    // TODO make into method on app instead
    RenderableTrail(double const _duration);
    void Update(double const _dt, Vector3d _pos);
    void Render() const;

    // TODO this stores a fixed number of frames, not the best approach
    enum { NUM_TRAIL_PTS = 1000 };
    double m_duration;
    double m_timeSinceUpdate;

    int m_headIdx;
    int m_tailIdx;
    Vector3d m_trailPts[NUM_TRAIL_PTS];
    double m_trailDuration[NUM_TRAIL_PTS];

    Vector3f m_colOld;
    Vector3f m_colNew;
  };
  std::vector<RenderableTrail> m_renderableTrails;

  int makeTrail() { m_renderableTrails.push_back(RenderableTrail(3.0)); return m_renderableTrails.size() - 1; }
  RenderableTrail& getTrail(int i) { return m_renderableTrails[i]; }

  //// Entities ////

  struct ShipEntity {
    int m_particleBodyIdx;
    int m_pointIdx;
    int m_trailIdx;
    int m_orbitIdx;
  };
  std::vector<ShipEntity> m_shipEntities;

  int makeShip() { m_shipEntities.push_back(ShipEntity()); return m_shipEntities.size() - 1; }
  ShipEntity& getShip(int i) { return m_shipEntities[i]; }

  int m_playerShipId;
  int m_suspectShipId;

  // Right now the moon orbits the planet, can get rid of distinction later

  struct PlanetEntity {
    int m_gravBodyIdx;
    int m_sphereIdx;
  };
  std::vector<PlanetEntity> m_planetEntities;

  int makePlanet() { m_planetEntities.push_back(PlanetEntity()); return m_planetEntities.size() - 1; }
  PlanetEntity& getPlanet(int i) { return m_planetEntities[i]; }
    
  int m_earthPlanetId;

  struct MoonEntity {
    int m_gravBodyIdx;
    int m_sphereIdx;
    int m_orbitIdx;
    int m_trailIdx;
  };
  std::vector<MoonEntity> m_moonEntities;

  int makeMoon() { m_moonEntities.push_back(MoonEntity()); return m_moonEntities.size() - 1; }
  MoonEntity& getMoon(int i) { return m_moonEntities[i]; }

  int m_moonMoonId;

  // Camera

  Body* m_camTarget;
  size_t m_camTargetIdx;
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

