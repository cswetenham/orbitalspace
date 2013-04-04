#include "orStd.h"
#include "orPlatform/window.h"

#include "orbitalSpaceApp.h"

#include "orProfile/perftimer.h"
#include "task.h"
#include "taskScheduler.h"
#include "taskSchedulerWorkStealing.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

#include "boost_begin.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "boost_end.h"

#include "constants.h"

OrbitalSpaceApp::OrbitalSpaceApp():
  App(),
  m_rnd(1123LL),
  m_simTime(0.0),
  m_paused(false),
  m_singleStep(false),
  m_wireframe(false),
  m_camOrig(true),
  m_camDist(-3.1855e7),
  m_camTheta(0.0),
  m_camPhi(0.0),
  m_camTarget(NULL),
  m_camTargetId(0),
  m_camMode(CameraMode_ThirdPerson),
  m_inputMode(InputMode_Default),
  m_playerShipId(0),
  m_integrationMethod(PhysicsSystem::IntegrationMethod_RK4),
  m_light(1, 1, 0),
  m_thrusters(0),
  m_hasFocus(false),
  m_music(),
  m_timeScale(1.0)
{
  // Create palette
  m_colG[0] = Vector3f(41,42,34)/255;
  m_colG[1] = Vector3f(77,82,50)/255;
  m_colG[2] = Vector3f(99,115,76)/255;
  m_colG[3] = Vector3f(151,168,136)/255;
  m_colG[4] = Vector3f(198,222,172)/255;

  for (int i = 0; i < PALETTE_SIZE; ++i)
  {
    m_colR[i] = Vector3f(m_colG[i].y(), m_colG[i].x(), m_colG[i].z());
    m_colB[i] = Vector3f(m_colG[i].x(), m_colG[i].z(), m_colG[i].y());
  }
  
  m_light /= m_light.norm();

  // TODO real-time date/time + time scale factor display

  // For now, give the moon a circular orbit

  double const muEarthMoon = (EARTH_MASS + MOON_MASS) * GRAV_CONSTANT;
  double const angularSpeed = MOON_PERIOD / M_TAU;

  double const earthMoonOrbitRadius = pow(muEarthMoon * angularSpeed * angularSpeed, 1.0/3.0); // meters

  // Distances from COM of Earth-Moon system
  double const earthOrbitRadius = earthMoonOrbitRadius * MOON_MASS / (EARTH_MASS + MOON_MASS);
  double const moonOrbitRadius = earthMoonOrbitRadius - earthOrbitRadius;
  
  Vector3d const earthPos = Vector3d(0.0, 0.0, -earthOrbitRadius);
  Vector3d const moonPos = Vector3d(0.0, 0.0, moonOrbitRadius);
    
  double const earthSpeed = earthOrbitRadius / angularSpeed;
  double const moonSpeed = moonOrbitRadius / angularSpeed;
    
  Vector3d const earthVel = Vector3d(-earthSpeed, 0.0, 0.0);
  Vector3d const moonVel = Vector3d(moonSpeed, 0.0, 0.0);

  // Create Earth
  
  m_camTargetNames.push_back("Earth"); // TODO how to implement camera now - point to all planets, then moons, then ships?

  PlanetEntity& earthPlanet = getPlanet(m_earthPlanetId = makePlanet());
  
  PhysicsSystem::GravBody& earthGravBody = m_physicsSystem.getGravBody(earthPlanet.m_gravBodyId = m_physicsSystem.makeGravBody());

  earthGravBody.m_mass = EARTH_MASS;
  earthGravBody.m_radius = EARTH_RADIUS;

  earthGravBody.m_pos = earthPos;
  earthGravBody.m_vel = earthVel;

  RenderSystem::Sphere& earthSphere = m_renderSystem.getSphere(earthPlanet.m_sphereId = m_renderSystem.makeSphere());
  earthSphere.m_radius = earthGravBody.m_radius;
  earthSphere.m_pos = earthGravBody.m_pos;
  earthSphere.m_col = m_colG[1];
  
  m_camTarget = &earthGravBody;
  
  // Create Moon

  m_camTargetNames.push_back("Moon");

  MoonEntity& moonMoon = getMoon(m_moonMoonId = makeMoon());
  
  PhysicsSystem::GravBody& moonGravBody = m_physicsSystem.getGravBody(moonMoon.m_gravBodyId = m_physicsSystem.makeGravBody());

  moonGravBody.m_mass = MOON_MASS;
  moonGravBody.m_radius = MOON_RADIUS;

  moonGravBody.m_pos = moonPos;
  moonGravBody.m_vel = moonVel;

  RenderSystem::Sphere& moonSphere = m_renderSystem.getSphere(moonMoon.m_sphereId = m_renderSystem.makeSphere());
  moonSphere.m_radius = moonGravBody.m_radius;
  moonSphere.m_pos = moonGravBody.m_pos;
  moonSphere.m_col = m_colG[1];

  RenderSystem::Orbit& moonOrbit = m_renderSystem.getOrbit(moonMoon.m_orbitId = m_renderSystem.makeOrbit());
  moonOrbit.m_col = m_colG[1];
  moonOrbit.m_pos = earthGravBody.m_pos;
  
  RenderSystem::Trail& moonTrail = m_renderSystem.getTrail(moonMoon.m_trailId = m_renderSystem.makeTrail(5000.0, moonGravBody.m_pos));
  moonTrail.m_colOld = m_colG[0];
  moonTrail.m_colNew = m_colG[4];
  
  // Create Earth-Moon COM

  RenderSystem::Point& comPoint = m_renderSystem.getPoint(m_comPointId = m_renderSystem.makePoint());
  comPoint.m_pos = Vector3d(0.0, 0.0, 0.0);
  comPoint.m_col = Vector3f(1.0, 0.0, 0.0);

  for (int i = 0; i < 5; ++i) {
    RenderSystem::Point& comPoint = m_renderSystem.getPoint(m_lagrangePointIds[i] = m_renderSystem.makePoint());
    comPoint.m_col = Vector3f(1.0, 0.0, 0.0);
  }
  
  // Create ships

  m_camTargetNames.push_back("Player");
  
  ShipEntity& playerShip = getShip(m_playerShipId = makeShip());

  PhysicsSystem::ParticleBody& playerBody = m_physicsSystem.getParticleBody(playerShip.m_particleBodyId = m_physicsSystem.makeParticleBody());
  
  playerBody.m_pos = Vector3d(0.0, 0.0, 1.3e7);
  playerBody.m_vel = Vector3d(5e3, 0.0, 0.0);
  
  RenderSystem::Orbit& playerOrbit = m_renderSystem.getOrbit(playerShip.m_orbitId = m_renderSystem.makeOrbit());
  playerOrbit.m_col = m_colB[2];
  playerOrbit.m_pos = earthGravBody.m_pos;
    
  RenderSystem::Trail& playerTrail = m_renderSystem.getTrail(playerShip.m_trailId = m_renderSystem.makeTrail(5000.0, playerBody.m_pos));
  playerTrail.m_colOld = m_colB[0];
  playerTrail.m_colNew = m_colB[4];

  RenderSystem::Point& playerPoint = m_renderSystem.getPoint(playerShip.m_pointId = m_renderSystem.makePoint());
  playerPoint.m_pos = playerBody.m_pos;
  playerPoint.m_col = m_colB[4];
  
  m_camTargetNames.push_back("Suspect");

  ShipEntity& suspectShip = getShip(m_suspectShipId = makeShip());

  PhysicsSystem::ParticleBody& suspectBody = m_physicsSystem.getParticleBody(suspectShip.m_particleBodyId = m_physicsSystem.makeParticleBody());
  
  suspectBody.m_pos = Vector3d(0.0, 0.0, 1.3e7);
  suspectBody.m_vel = Vector3d(5e3, 0.0, 0.0);
  
  RenderSystem::Orbit& suspectOrbit = m_renderSystem.getOrbit(suspectShip.m_orbitId = m_renderSystem.makeOrbit());
  suspectOrbit.m_col = m_colR[2];
  suspectOrbit.m_pos = earthGravBody.m_pos;
  
  RenderSystem::Trail& suspectTrail = m_renderSystem.getTrail(suspectShip.m_trailId = m_renderSystem.makeTrail(5000.0, suspectBody.m_pos));
  suspectTrail.m_colOld = m_colR[0];
  suspectTrail.m_colNew = m_colR[4];
  
  RenderSystem::Point& suspectPoint = m_renderSystem.getPoint(suspectShip.m_pointId = m_renderSystem.makePoint());
  suspectPoint.m_pos = suspectBody.m_pos;
  suspectPoint.m_col = m_colR[4];

  // Perturb all the ship orbits
  float* rnds = new float[6 * m_shipEntities.size()];
  UniformDistribution<float> dist(-1, +1);
  dist.Generate(&m_rnd, 6 * m_shipEntities.size(), &rnds[0]);
  for (int i = 0; i < (int)m_shipEntities.size(); ++i)
  {
    PhysicsSystem::ParticleBody& shipBody = m_physicsSystem.getParticleBody(m_shipEntities[i].m_particleBodyId);
    shipBody.m_pos += Vector3d(rnds[6*i  ], rnds[6*i+1], rnds[6*i+2]) * 6e4;
    shipBody.m_vel += Vector3d(rnds[6*i+3], rnds[6*i+4], rnds[6*i+5]) * 1e2;
  }
  delete[] rnds;
  

  m_music.openFromFile("music/spacething3_mastered_fullq.ogg");
  m_music.setLoop(true);
  // m_music.play();
}

OrbitalSpaceApp::~OrbitalSpaceApp()
{
}

// TODO put elsewhere
void runTests() {
  // Test FMod
  double const a = Util::FMod(3.0, 2.0);
  if ( a != 1.0 ) {
    DEBUGBREAK;
  }

  // Test Wrap
  double const b = Util::Wrap(3.5, 1.0, 2.0);
  if (b != 1.5) {
    DEBUGBREAK;
  }
}

void OrbitalSpaceApp::Run()
{
  runTests();

  while (m_running)
  {
    Timer::PerfTime const frameStart = Timer::GetPerfTime();
    
    {
      PERFTIMER("PollEvents");
      PollEvents();
    }

    if (m_hasFocus) {
      // Input handling
      if (m_inputMode == InputMode_RotateCamera) {
        sf::Vector2i const centerPos = sf::Vector2i(m_config.width/2, m_config.height/2);
        sf::Vector2i const mouseDelta = sf::Mouse::getPosition(*m_window) - centerPos;
        sf::Mouse::setPosition(centerPos, *m_window);
    
        m_camTheta += mouseDelta.x * M_TAU / 300.0;
        m_camTheta = Util::Wrap(m_camTheta, 0.0, M_TAU);
        m_camPhi += mouseDelta.y * M_TAU / 300.0;
        m_camPhi = Util::Clamp(m_camPhi, -.249 * M_TAU, .249 * M_TAU);
      }
    }

    {
      PERFTIMER("UpdateState");
      enum { UPDATES_PER_FRAME_HACK = 1 };
      for (int i = 0; i < UPDATES_PER_FRAME_HACK; ++i) {
        UpdateState(Timer::PerfTimeToMillis(m_lastFrameDuration));
      }
    }
    
    {
      PERFTIMER("BeginRender");
      BeginRender();
    }

    {
      PERFTIMER("RenderState");
      RenderState();
    }

    {
      PERFTIMER("EndRender");
      EndRender();
    }
    
    sf::sleep(sf::milliseconds(1)); // TODO sleep according to frame duration

    m_lastFrameDuration = Timer::GetPerfTime() - frameStart;
  }
}

void OrbitalSpaceApp::InitRender()
{
  App::InitRender();

  // TODO
  m_config.width = 1280;
  m_config.height = 768;
  
  sf::ContextSettings settings;
  settings.depthBits         = 24; // Request a 24 bits depth buffer
  settings.stencilBits       = 8;  // Request a 8 bits stencil buffer
  settings.antialiasingLevel = 2;  // Request 2 levels of antialiasing
  m_window = new sf::RenderWindow(sf::VideoMode(m_config.width, m_config.height, 32), "SFML OpenGL", sf::Style::Close, settings);
  
  sf::WindowHandle winHandle = m_window->getSystemHandle();
  orPlatform::FocusWindow(winHandle);
  m_hasFocus = true;

}

void OrbitalSpaceApp::ShutdownRender()
{
  App::ShutdownRender();
}

void OrbitalSpaceApp::InitState()
{
  App::InitState();
}

void OrbitalSpaceApp::ShutdownState()
{
  App::ShutdownState();
}

void OrbitalSpaceApp::HandleEvent(sf::Event const& _event)
{
  /* TODO allow?
  if (_event.type == sf::Event::Resized)
  {
    glViewport(0, 0, _event.size.width, _event.size.height);
  }
  */

  if (_event.type == sf::Event::MouseButtonPressed) {
    if (_event.mouseButton.button == sf::Mouse::Right) {
      m_inputMode = InputMode_RotateCamera;
      
      // When rotating the camera, hide the mouse cursor and center it. We'll then track how far it's moved off center each frame.
      sf::Vector2i const centerPos = sf::Vector2i(m_config.width/2, m_config.height/2);
      m_savedMousePos = sf::Mouse::getPosition(*m_window);
      m_window->setMouseCursorVisible(false);
      sf::Mouse::setPosition(centerPos, *m_window);
      
    }
  }
  
  if (_event.type == sf::Event::MouseButtonReleased) {
    if (_event.mouseButton.button == sf::Mouse::Right) {
      // Restore the old position of the cursor.
      m_inputMode = InputMode_Default;
      sf::Mouse::setPosition(m_savedMousePos, *m_window);
      m_window->setMouseCursorVisible(true);
    }
  }

  if (_event.type == sf::Event::MouseWheelMoved)
  {
    m_camDist *= pow(0.9, _event.mouseWheel.delta);
  }

  if (_event.type == sf::Event::KeyPressed)
  {
    if (_event.key.code == sf::Keyboard::Escape)
    {
      m_running = false;
    }

    if (_event.key.code == sf::Keyboard::Tab)
    {
      m_camTargetId++;
      // TODO maybe should point at renderables instead, but eh...
      if (m_camTargetId < m_physicsSystem.m_gravBodies.size()) {
        m_camTarget = &m_physicsSystem.getGravBody(m_camTargetId);
      } else if (m_camTargetId < m_physicsSystem.m_gravBodies.size() + m_physicsSystem.m_particleBodies.size()) {
        m_camTarget = &m_physicsSystem.getParticleBody(m_camTargetId - m_physicsSystem.m_gravBodies.size());
      } else {
        m_camTargetId = 0;
        m_camTarget = &m_physicsSystem.getGravBody(0);
      }
    }

    if (_event.key.code == sf::Keyboard::F1) {
      m_camMode = CameraMode_FirstPerson;
    }

    if (_event.key.code == sf::Keyboard::F2) {
      m_camMode = CameraMode_ThirdPerson;
    }

    if (_event.key.code == sf::Keyboard::PageDown) {
      m_integrationMethod = PhysicsSystem::IntegrationMethod((m_integrationMethod + 1) % PhysicsSystem::IntegrationMethod_Count);
    }
    
    if (_event.key.code == sf::Keyboard::R)
    {
      m_camOrig = !m_camOrig;
    }

    if (_event.key.code == sf::Keyboard::Add)
    {
      m_timeScale *= 2;
    }

    if (_event.key.code == sf::Keyboard::Subtract)
    {
      m_timeScale /= 2;
    }

    if (_event.key.code == sf::Keyboard::A)
    {
      m_thrusters |= ThrustLeft;
    }
    if (_event.key.code == sf::Keyboard::D)
    {
      m_thrusters |= ThrustRight;
    }
    if (_event.key.code == sf::Keyboard::W)
    {
      m_thrusters |= ThrustFwd;
    }
    if (_event.key.code == sf::Keyboard::S)
    {
      m_thrusters |= ThrustBack;
    }
    if (_event.key.code == sf::Keyboard::Up)
    {
      m_thrusters |= ThrustUp;
    }
    if (_event.key.code == sf::Keyboard::Down)
    {
      m_thrusters |= ThrustDown;
    }
  }

  if (_event.type == sf::Event::KeyReleased)
  {
    if (_event.key.code == sf::Keyboard::A)
    {
      m_thrusters &= ~ThrustLeft;
    }
    if (_event.key.code == sf::Keyboard::D)
    {
      m_thrusters &= ~ThrustRight;
    }
    if (_event.key.code == sf::Keyboard::W)
    {
      m_thrusters &= ~ThrustFwd;
    }
    if (_event.key.code == sf::Keyboard::S)
    {
      m_thrusters &= ~ThrustBack;
    }
    if (_event.key.code == sf::Keyboard::Up)
    {
      m_thrusters &= ~ThrustUp;
    }
    if (_event.key.code == sf::Keyboard::Down)
    {
      m_thrusters &= ~ThrustDown;
    }
  }

  if (_event.type == sf::Event::Closed)
  {
    m_running = false;
  }

  if (_event.type == sf::Event::LostFocus)
  {
    m_hasFocus = false;
  }

  if (_event.type == sf::Event::GainedFocus)
  {
    m_hasFocus = true;
  }
}

Vector3d OrbitalSpaceApp::CalcPlayerThrust(Vector3d p, Vector3d v)
{
  Vector3d origin = FindSOIGravBody(p).m_pos;
   
  // Calc acceleration due to gravity
  Vector3d const r = (origin - p);
  double const r_mag = r.norm();

  Vector3d const r_dir = r / r_mag;

  // Calc acceleration due to thrust
  double const thrustAccel = 10.0; // meters per second squared - TODO what is a realistic value?
      
  Vector3d thrustVec(0.0,0.0,0.0);
    
  Vector3d const fwd = v / v.norm(); // Prograde
  Vector3d const left = fwd.cross(r_dir); // name? (and is the order right?)
  Vector3d const dwn = left.cross(fwd); // name? (and is the order right?)

  if (m_thrusters & ThrustFwd)  { thrustVec += fwd; }
  if (m_thrusters & ThrustBack) { thrustVec -= fwd; }
  if (m_thrusters & ThrustDown)  { thrustVec += dwn; }
  if (m_thrusters & ThrustUp) { thrustVec -= dwn; }
  if (m_thrusters & ThrustLeft)  { thrustVec += left; }
  if (m_thrusters & ThrustRight) { thrustVec -= left; }

  Vector3d a_thrust = thrustAccel * thrustVec;

  return a_thrust;
}

void OrbitalSpaceApp::UpdateState(double const _dt)
{
  if (!m_paused) {
    double dt = m_timeScale * Util::Min(_dt, 100.0) / 1000.0; // seconds
        
    // Update player thrust
    PhysicsSystem::ParticleBody& playerShipBody = m_physicsSystem.getParticleBody(getShip(m_playerShipId).m_particleBodyId);
    playerShipBody.m_userAcc = CalcPlayerThrust(playerShipBody.m_pos, playerShipBody.m_vel);
    
    m_physicsSystem.update(m_integrationMethod, dt);

    // Update Planets
    for (int i = 0; i < (int)m_planetEntities.size(); ++i) {
      PlanetEntity& planet = getPlanet(i);

      PhysicsSystem::Body& body = m_physicsSystem.getGravBody(planet.m_gravBodyId);
      
      RenderSystem::Sphere& sphere = m_renderSystem.getSphere(planet.m_sphereId);
      sphere.m_pos = body.m_pos;
    }

    // Update Moons
    for (int i = 0; i < (int)m_moonEntities.size(); ++i) {
      MoonEntity& moon = getMoon(i);

      PhysicsSystem::Body& body = m_physicsSystem.getGravBody(moon.m_gravBodyId);

      RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(moon.m_orbitId);
      UpdateOrbit(body, orbit);

      RenderSystem::Trail& trail = m_renderSystem.getTrail(moon.m_trailId);
      trail.Update(_dt, body.m_pos);

      RenderSystem::Sphere& sphere = m_renderSystem.getSphere(moon.m_sphereId);
      sphere.m_pos = body.m_pos;
    }

    // Update the earth-moon COM
    RenderSystem::Point& com = m_renderSystem.getPoint(m_comPointId);
    PhysicsSystem::GravBody& earthBody = m_physicsSystem.getGravBody(getPlanet(m_earthPlanetId).m_gravBodyId);
    PhysicsSystem::GravBody& moonBody = m_physicsSystem.getGravBody(getMoon(m_moonMoonId).m_gravBodyId);
    double const totalMass = earthBody.m_mass + moonBody.m_mass;
    com.m_pos = (earthBody.m_pos * earthBody.m_mass / totalMass) + (moonBody.m_pos * moonBody.m_mass / totalMass);

    // Update the earth-moon Lagrange points
    // TODO some / all of these should be based on the COM and not earth position
    Vector3d const earthMoonVector = moonBody.m_pos - earthBody.m_pos;
    double const earthMoonOrbitRadius = earthMoonVector.norm();
    Vector3d const earthMoonDir = earthMoonVector / earthMoonOrbitRadius;
    double const massRatio = MOON_MASS / EARTH_MASS;
    double const r1 = earthMoonOrbitRadius * pow(massRatio / 3.0, 1.0/3.0);
    double const r3 = earthMoonOrbitRadius * (1.0 + (7.0/12.0) * massRatio); // extra 1.0 to make r3 a distand from Earth position rather than an offset from earthMoonOrbitRadius
    // Lagrange point 1
    m_renderSystem.getPoint(m_lagrangePointIds[0]).m_pos = moonBody.m_pos - earthMoonDir * r1;
    // Lagrange point 2
    m_renderSystem.getPoint(m_lagrangePointIds[1]).m_pos = moonBody.m_pos + earthMoonDir * r1;
    // Lagrange point 3
    m_renderSystem.getPoint(m_lagrangePointIds[2]).m_pos = earthBody.m_pos - earthMoonDir * r3;
    
    // L4 and L5 are on the Moon's orbit, 60 degrees ahead and 60 degrees behind.
    Vector3d orbitAxis = moonBody.m_vel.normalized().cross(earthMoonVector.normalized());
    Eigen::AngleAxisd rotation(M_TAU / 6.0, orbitAxis);
    // Lagrange point 4
    m_renderSystem.getPoint(m_lagrangePointIds[3]).m_pos = rotation           * earthMoonVector;
    // Lagrange point 5
    m_renderSystem.getPoint(m_lagrangePointIds[4]).m_pos = rotation.inverse() * earthMoonVector;

    // Update ships
    for (int i = 0; i < (int)m_shipEntities.size(); ++i) {
      ShipEntity& ship = getShip(i);

      PhysicsSystem::Body& body = m_physicsSystem.getParticleBody(ship.m_particleBodyId);
     
      RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(ship.m_orbitId);
      UpdateOrbit(body, orbit);

      RenderSystem::Trail& trail = m_renderSystem.getTrail(ship.m_trailId);
      trail.Update(_dt, body.m_pos);

      RenderSystem::Point& point = m_renderSystem.getPoint(ship.m_pointId);
      point.m_pos = body.m_pos;
    }

    m_simTime += dt;
  }

  if (m_singleStep)
  {
    m_singleStep = false;
    m_paused = true;
  }
}

PhysicsSystem::GravBody const& OrbitalSpaceApp::FindSOIGravBody(Vector3d const& p) {
  // TODO HACK
  // SOI really requires each body to have a "parent body" for the SOI computation.
  // For now will hack the earth-moon one.
  
  // TODO Id -> Id

  PhysicsSystem::GravBody const& earthBody = m_physicsSystem.getGravBody(getPlanet(m_earthPlanetId).m_gravBodyId);
  PhysicsSystem::GravBody const& moonBody = m_physicsSystem.getGravBody(getMoon(m_moonMoonId).m_gravBodyId);

  double const earthMoonOrbitRadius = (earthBody.m_pos - moonBody.m_pos).norm();

  // Distances from COM of Earth-Moon system
  double const earthOrbitRadius = earthMoonOrbitRadius * MOON_MASS / (EARTH_MASS + MOON_MASS);
  double const moonOrbitRadius = earthMoonOrbitRadius - earthOrbitRadius;

  double const moonSOI = moonOrbitRadius * pow(MOON_MASS / EARTH_MASS, 2.0/5.0);

  double const moonDistance = (p - moonBody.m_pos).norm();

  if (moonDistance < moonSOI) {
    return moonBody;
  } else {
    return earthBody;
  }
}

void OrbitalSpaceApp::UpdateOrbit(PhysicsSystem::Body const& body, RenderSystem::Orbit& o_params) {
  // TODO will want to just forward-project instead, this is broken with >1 body

  // Find body whose sphere of influence we are in
  // This is the one with the smallest sphere of influence

  // Compute Kepler orbit

  // HACK
  PhysicsSystem::GravBody const& parentBody =
    (&body == (PhysicsSystem::Body const*)&m_physicsSystem.getGravBody(getMoon(m_moonMoonId).m_gravBodyId))
      ? m_physicsSystem.getGravBody(getPlanet(m_earthPlanetId).m_gravBodyId)
      : FindSOIGravBody(body.m_pos);
  
  double const G = GRAV_CONSTANT;
  double const M = parentBody.m_mass;
    
  double const mu = M * G;

  Vector3d v = body.m_vel - parentBody.m_vel;

  Vector3d r = parentBody.m_pos - body.m_pos;
  double const r_mag = r.norm();

  Vector3d r_dir = r/r_mag;

  double const vr_mag = r_dir.dot(v);
  Vector3d vr = r_dir * vr_mag; // radial velocity
  Vector3d vt = v - vr; // tangent velocity
  double const vt_mag = vt.norm();
  Vector3d t_dir = vt/vt_mag;

  double const p = pow(r_mag * vt_mag, 2) / mu;
  double const v0 = sqrt(mu/p); // todo compute more accurately/efficiently?

  Vector3d ex = ((vt_mag - v0) * r_dir - vr_mag * t_dir) / v0;
  double const e = ex.norm();

  double const ec = (vt_mag / v0) - 1;
  double const es = (vr_mag / v0);
  double const theta = atan2(es, ec);

  Vector3d x_dir = cos(theta) * r_dir - sin(theta) * t_dir;
  Vector3d y_dir = sin(theta) * r_dir + cos(theta) * t_dir;

  o_params.e = e;
  o_params.p = p;
  o_params.theta = theta;
  o_params.x_dir = x_dir;
  o_params.y_dir = y_dir;
  o_params.m_pos = parentBody.m_pos;
}

void OrbitalSpaceApp::LookAt(Vector3d pos, Vector3d target, Vector3d up) {
  Vector3d camF = (target - pos).normalized();
  Vector3d camR = camF.cross(up).normalized();
  Vector3d camU = camF.cross(camR).normalized();

  Matrix3d camMat;
  camMat.col(0) = camR;
  camMat.col(1) = -camU;
  camMat.col(2) = -camF;

  Eigen::Affine3d camT;
  camT.linear() = camMat;
  camT.translation() = pos;

  glMultMatrix(camT.inverse());
}

Vector3d lerp(Vector3d const& _x0, Vector3d const& _x1, double const _a) {
    return _x0 * (1 - _a) + _x1 * _a;
}

void OrbitalSpaceApp::RenderState()
{
  m_window->resetGLStates();
  
  // Render debug text
  {
    sf::Font font(sf::Font::getDefaultFont());
    Eigen::Matrix<sf::Uint8, 3, 1> ct = (m_colG[4] * 255).cast<sf::Uint8>();
    
    uint32_t const fontSize = 14;
    sf::Text text(sf::String("Hello, World!"), font, fontSize);
    text.setColor(sf::Color(ct.x(), ct.y(), ct.z(), 255));
    text.setPosition(8, 8);

    std::ostringstream str;
    str.precision(3);
    str.width(7);
    str.flags(std::ios::right | std::ios::fixed);
    
    str << "Time Scale: " << m_timeScale << "\n";

    {
      using namespace boost::posix_time;
      using namespace boost::gregorian;

      // Astronomical Epoch: 1200 hours, 1 January 2000
      // Game start date:
      ptime epoch(date(2000, Jan, 1), hours(12));
      ptime gameStart(date(2025, Mar, 15), hours(1753));
      // Note: the (long) here limits us to ~68 years game time. Should be enough, otherwise just need to keep adding seconds to the dateTime to match the simTime
      ptime curDateTime = gameStart + seconds((long)m_simTime);
      str << "UTC DateTime: " << to_simple_string(curDateTime) << "\n";
    }
    
    str << "Cam Target: " << m_camTargetNames[m_camTargetId] << "\n";
    str << "Cam Dist: " << m_camDist << "\n";
    str << "Cam Theta:" << m_camTheta << "\n";
    str << "Cam Phi:" << m_camPhi << "\n";
    // double const shipDist = (m_ships[0].m_physics.m_pos - m_ships[1].m_physics.m_pos).norm();
    // str << "Intership Distance:" << shipDist << "\n";
    str << "Intership Distance: TODO\n";
    str << "Integration Method: " << m_integrationMethod << "\n";

    // TODO: better double value text formatting
    // TODO: small visualisations for the angle etc values
   
    text.setString(str.str());
    m_window->draw(text);
  }

  m_window->resetGLStates();

  glViewport(0, 0, m_config.width, m_config.height);

  Vector3f c = m_colG[0];
  glClearColor(c.x(), c.y(), c.z(), 0);
  glClearDepth(1.0);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  double const aspect = m_config.width / (double)m_config.height;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double const fov = 35.0; // degrees?
  gluPerspective(fov, aspect, 1.0, 1e11); // meters

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  Vector3d up(0.0, 1.0, 0.0);

  // Set camera
  
  assert(m_camTarget);
  Vector3d const camTarget = m_camTarget->m_pos;

  Vector3d camPos;
  
  if (m_camMode == CameraMode_FirstPerson) {
    camPos = m_physicsSystem.getParticleBody(getShip(m_playerShipId).m_particleBodyId).m_pos;
  } else if (m_camMode == CameraMode_ThirdPerson) {
    camPos = Vector3d(0.0, 0.0, m_camDist);

    Eigen::AngleAxisd thetaRot(m_camTheta, Vector3d(0.0, 1.0, 0.0));
    Eigen::AngleAxisd phiRot(m_camPhi, Vector3d(1.0, 0.0, 0.0));

    Eigen::Affine3d camMat1;
    camMat1.setIdentity();
    camMat1.rotate(thetaRot).rotate(phiRot);

    camPos = camMat1 * camPos;
    camPos += camTarget;
  } else {
    assert(false);
  }

  // TODO remove the gluLookAt? Or just have it off by default for now?
  if (m_camOrig) {
    gluLookAt(camPos.x(), camPos.y(), camPos.z(),
              camTarget.x(), camTarget.y(), camTarget.z(),
              up.x(), up.y(), up.z());
  } else {
    // Finally does the same as gluLookAt!
   LookAt(camPos, camTarget, up);
  }

  glEnable(GL_TEXTURE_2D);
  
  glLineWidth(1);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_LINE_SMOOTH);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // TODO clean up
  if (m_wireframe) {
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  } else {
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
  }
  
  m_renderSystem.render();
  
  printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime / 1000);
}
