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
#include "orPhysics.h"
#include "orCamera.h"
#include "orEntity.h"

#include <vector>

#include <SFML/Audio/Music.hpp>

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
  Vector3d CalcPlayerThrust(PhysicsSystem::ParticleBody const& playerBody);

private:
  Rnd64 m_rnd;

  double m_simTime;

  struct Config {
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

  int m_debugTextLabelId;

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

