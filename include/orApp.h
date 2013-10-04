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

  void RenderState();
  
private:
  // App

  // TODO this doesn't feel very DOD but it'll do for now
  class IScreen {
  public:
    virtual ~IScreen() {}
    virtual void Init() {};
    virtual void Shutdown() {};
    virtual void HandleEvent(sf::Event const& _event) {}
    virtual void HandleInput() {};
    virtual void UpdateState() {};
    virtual void RenderState() {};
  };

  class TitleScreen : public IScreen {
  public:
    // virtual void Init();
    // virtual void Shutdown();
    // virtual void HandleEvent(sf::Event const& _event);
    // virtual void HandleInput();
    // virtual void UpdateState();
    // virtual void RenderState();
  private:
  } m_titleScreen;

  class MainScreen : public IScreen {
  public:
    MainScreen() :
      m_thrusters(0),
      m_playerShipId(0),
      m_suspectShipId(0),
      m_earthPlanetId(0),
      m_moonMoonId(0),
      m_comPoiId(0)
    {
      for (int i = 0; i < 5; ++i) {
        m_lagrangePoiIds[i] = 0;
      }
    }
    // virtual void Init();
    // virtual void Shutdown();
    virtual void HandleEvent(sf::Event const& _event);
    // virtual void HandleInput();
    // virtual void UpdateState();
    // virtual void RenderState();
  
  public: // TODO make private
    Vector3d CalcPlayerThrust(PhysicsSystem const& physicsSystem, PhysicsSystem::ParticleBody const& playerBody);

  public: // TODO make private

    //// Input ////
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

    //// Entities ////
    int m_playerShipId;
    int m_suspectShipId;

    int m_earthPlanetId;
    int m_moonMoonId;

    int m_comPoiId;
    int m_lagrangePoiIds[5];
  private:
  } m_mainScreen;
  
  IScreen* m_curScreen;

  bool m_running;
  
  Timer::PerfTime m_lastFrameDuration;
    
  Rnd64 m_rnd;

  Config m_config;

  CameraSystem m_cameraSystem;
  RenderSystem m_renderSystem;
  PhysicsSystem m_physicsSystem;
  EntitySystem m_entitySystem;

  ////  Simulation ////

  double m_simTime;
  bool m_paused;
  bool m_singleStep;

  //// Camera ////
  
  enum CameraMode {
    CameraMode_FirstPerson = 0,
    CameraMode_ThirdPerson = 1
  };
  CameraMode m_camMode;

  int m_cameraId;
  int m_cameraTargetId;

  double m_camDist;
  double m_camTheta;
  double m_camPhi;

  //// Rendering ////
  
  double m_lightDir[3];

  int m_uiTextTopLabel2DId;
  int m_uiTextBottomLabel2DId;

  //// Physics ////
  
  double m_timeScale;
  PhysicsSystem::IntegrationMethod m_integrationMethod;

  //// Input ////

  enum InputMode {
    InputMode_Default = 0,
    InputMode_RotateCamera = 1
  };
  InputMode m_inputMode;
   
  //// Rendering (Global) ////

  // Lazy
  enum {PALETTE_SIZE = 5};
  sf::Vector3f m_colG[PALETTE_SIZE];
  sf::Vector3f m_colR[PALETTE_SIZE];
  sf::Vector3f m_colB[PALETTE_SIZE];

  uint32_t m_frameBufferId;
  uint32_t m_renderedTextureId;
  uint32_t m_depthRenderBufferId;

  sf::RenderWindow* m_window;
  // sf::Vectors are just plain old data. Eigen::Vectors are SSE magic.
  sf::Vector2i m_savedMousePos;

  bool m_hasFocus;

  //// Music ////

  sf::Music* m_music;
};

#endif	/* ORBITALSPACEAPP_H */

