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

// SFML

#include <SFML/System.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio/Music.hpp>

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

// Code outside the system should refer to the instances only by opaque Id - not iterate over the collection (when needed this should happen internally in the system)

class orApp
{
public:
  struct Config {
    int windowWidth;
    int windowHeight;
    int renderWidth;
    int renderHeight;
  };
  
  orApp(Config const& config);
  ~orApp();

public:
  void Run();

private:
  void Init();
  void Shutdown();

  void InitRender();
  void ShutdownRender();

  void InitState();
  void ShutdownState();

  void RunOneStep();
  
  void PollEvents();
  void HandleEvent(sf::Event const& _event);

  void HandleInput();

  void UpdateState();
  
  void BeginRender();

  void RenderState();
   
  void EndRender();

private:
  Vector3d CalcPlayerThrust(PhysicsSystem::ParticleBody const& playerBody);

private:
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

  // Rendering options
  bool m_wireframe;
  bool m_camOrig;

  double m_camDist;
  double m_camTheta;
  double m_camPhi;

  //// Camera ////

  CameraSystem m_cameraSystem;

  enum CameraMode {
    CameraMode_FirstPerson = 0,
    CameraMode_ThirdPerson = 1
  };
  CameraMode m_camMode;

  int m_cameraId;
  int m_cameraTargetId;

  //// Rendering ////

  RenderSystem m_renderSystem;

  int m_uiTextTopLabel2DId;
  int m_uiTextBottomLabel2DId;

  //// Physics ////

  PhysicsSystem m_physicsSystem;

  double m_timeScale;
  PhysicsSystem::IntegrationMethod m_integrationMethod;

  //// Entities ////

  EntitySystem m_entitySystem;

  int m_playerShipId;
  int m_suspectShipId;

  int m_earthPlanetId;
  int m_moonMoonId;

  int m_comPoiId;
  int m_lagrangePoiIds[5];

  // Input

  enum InputMode {
    InputMode_Default = 0,
    InputMode_RotateCamera = 1
  };
  InputMode m_inputMode;

  // sf::Vectors are just plain old data. Eigen::Vectors are SSE magic.
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

  enum {PALETTE_SIZE = 5};
  
  // Lazy
  sf::Vector3f m_colG[PALETTE_SIZE];
  sf::Vector3f m_colR[PALETTE_SIZE];
  sf::Vector3f m_colB[PALETTE_SIZE];

  double m_lightDir[3];

  bool m_hasFocus;

  uint32_t m_frameBufferId;
  uint32_t m_renderedTextureId;
  uint32_t m_depthRenderBufferId;

  sf::RenderWindow* m_window;
  sf::Music* m_music;
};

#endif	/* ORBITALSPACEAPP_H */

