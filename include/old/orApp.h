/*
 * File:   ssrm1app.h
 * Author: s1149322
 *
 * Created on 12 December 2011, 17:11
 */

#ifndef ORBITALSPACEAPP_H
#define	ORBITALSPACEAPP_H

// STL
#include <vector>

// Boost

#include "boost_begin.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "boost_end.h"

// Eigen

#include <Eigen/Geometry>

// Old style libs

#include "util.h"
#include "rnd.h"
#include "timer.h"

// New style libs

#include "orStd.h"
#include "orMath.h"
#include "orGfx.h"

// Systems

#include "orRender.h"
#include "orPhysics.h"
#include "orCamera.h"
#include "orEntity.h"

// TODO forward decl for SDL_GLContext?

#include "SDL_video.h"

// Code outside the system should refer to the instances only by opaque Id - not iterate over the collection (when needed this should happen internally in the system)

struct SDL_Window;
union SDL_Event;

class orApp
{
public:
  struct Config {
    int windowWidth;
    int windowHeight;
    int renderWidth;
    int renderHeight;
  };

  // TODO Eigen::Vectors are SSE magic. Want those and also some plain old data types.

  orApp(Config const& config);
  ~orApp();

public:
  void Run();

private:
  void Init();
    void InitRender();
    void InitState();

  void Shutdown();
    void ShutdownState();
    void ShutdownRender();

  void RunOneStep();

  void PollEvents();
    void HandleEvent(SDL_Event const& _event);

  void HandleInput();

  void UpdateState();
    void UpdateState_Bodies(double const dt);
    void UpdateState_CamTargets(double const dt);
    void UpdateState_RenderObjects(double const dt);

  void RenderState();

private:
  // TODO change all methods to start with lowercase
  Vector3d CalcPlayerThrust(PhysicsSystem::ParticleBody const& playerBody);
  orbital::Id<EntitySystem::Body> spawnBody(
    std::string const& name,
    double const radius,
    double const mass,
    orEphemerisJPL const& ephemeris_jpl,
    orbital::Id<PhysicsSystem::GravBody> const parent_grav_body_id
  );

  Eigen::Matrix4d calcScreenMatrix() const;
  Eigen::Matrix4d calcProjMatrix() const;
  Eigen::Matrix4d calcCamMatrix() const;

  orVec2 getRenderMousePos() const;
  orRay3 getMouseRay() const;

  void updateOrbitHighlight();

private:
  // TODO finish implementing title screen, etc
  enum AppScreen { Screen_Title, Screen_Level } m_appScreen;

  class TitleScreen {
    void Init();
    void Shutdown();
  } m_titleScreen;

  class MainLevel {
    void Init();
    void Shutdown();
  } m_mainLevel;

  Timer::PerfTime m_lastFrameDuration;
  bool m_running;

  Rnd64 m_rnd;

  double m_simTime;

  Config m_config;

  // Simulation options
  bool m_paused;
  bool m_singleStep;

  //// Camera ////

  CameraSystem m_cameraSystem;

  enum CameraMode {
    CameraMode_FirstPerson = 0,
    CameraMode_ThirdPerson = 1
  };
  CameraMode m_camMode;

  orbital::Id<CameraSystem::Camera> m_cameraId;
  orbital::Id<CameraSystem::Target> m_cameraTargetId;

  struct OrbitalCamParams {
    OrbitalCamParams(double _dist) : dist(_dist), theta(0), phi(0) {}
    double dist;
    double theta;
    double phi;
  };

  OrbitalCamParams m_camParams;

  Vector3d CamPosFromCamParams(OrbitalCamParams const& params);

  //// Rendering ////

  RenderSystem m_renderSystem;

  orbital::Id<RenderSystem::Label2D> m_uiTextTopLabel2DId;
  orbital::Id<RenderSystem::Label2D> m_uiTextBottomLabel2DId;

  enum {PALETTE_SIZE = 5};

  // Lazy
  RenderSystem::Colour m_colG[PALETTE_SIZE];
  RenderSystem::Colour m_colR[PALETTE_SIZE];
  RenderSystem::Colour m_colB[PALETTE_SIZE];

  // TODO make into an id-handle thing
  RenderSystem::FrameBuffer m_frameBuffer;

  // Rendering options
  bool m_wireframe;

  //// Physics ////

  PhysicsSystem m_physicsSystem;

  double m_timeScale;
  PhysicsSystem::IntegrationMethod m_integrationMethod;

  //// Entities ////

  EntitySystem m_entitySystem;

  orbital::Id<EntitySystem::Ship> m_playerShipId;
  orbital::Id<EntitySystem::Ship> m_suspectShipId;

  orbital::Id<EntitySystem::Body> m_sunBodyId;

  orbital::Id<EntitySystem::Body> m_earthBodyId;
  orbital::Id<EntitySystem::Body> m_moonBodyId;

  orbital::Id<EntitySystem::Body> m_mercuryBodyId;
  orbital::Id<EntitySystem::Body> m_venusBodyId;
  orbital::Id<EntitySystem::Body> m_marsBodyId;
  orbital::Id<EntitySystem::Body> m_jupiterBodyId;
  orbital::Id<EntitySystem::Body> m_saturnBodyId;
  orbital::Id<EntitySystem::Body> m_neptuneBodyId;
  orbital::Id<EntitySystem::Body> m_uranusBodyId;
  orbital::Id<EntitySystem::Body> m_plutoBodyId;

#if 0
  int m_comPoiId;
  int m_lagrangePoiIds[5];
#endif

  // Input

  enum InputMode {
    InputMode_Default = 0,
    InputMode_RotateCamera = 1
  };
  InputMode m_inputMode;

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

  bool m_hasFocus;
  SDL_Window* m_window;
  SDL_GLContext m_gl_context;

  int m_lastMouseX;
  int m_lastMouseY;
  orbital::Id<RenderSystem::Point> m_mousePointId;
  orbital::Id<RenderSystem::Label2D> m_mouseLabelId;
  int m_savedMouseX;
  int m_savedMouseY;
  struct Music {}; // TODO replace with appropriate type
  Music* m_music;

  static char const* s_jpl_names[];
  static orEphemerisJPL s_jpl_elements_t0[];
};

#endif	/* ORBITALSPACEAPP_H */

