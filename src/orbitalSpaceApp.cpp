#include "orStd.h"

#include "orbitalSpaceApp.h"

#include "perftimer.h"
#include "task.h"
#include "taskScheduler.h"
#include "taskSchedulerWorkStealing.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

// TODO #include <boost/posix_time.hpp>

// m^3 kg^-1 s^-2
#define GRAV_CONSTANT 6.6738480e-11

// meters
#define EARTH_RADIUS 6.371e6
#define MOON_RADIUS  1.737e6

// kg
#define EARTH_MASS 5.9742e24
#define MOON_MASS  7.3477e22

// meters
// #define MOON_APOGEE   3.6257e8
// #define MOON_PERIGEE  4.0541e8
// degrees
// #define MOON_INCLINATION 5.145

// seconds
#define MOON_PERIOD 2.3606e6

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
  m_camTargetIdx(0),
  m_camMode(CameraMode_ThirdPerson),
  m_inputMode(InputMode_Default),
  m_playerShipId(0),
  // m_integrationMethod(IntegrationMethod_ImplicitEuler),
  // TODO TEMP
  m_integrationMethod(IntegrationMethod_ExplicitEuler),
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

  // TODO
  // m_camTargetName[0] = "Earth";
  // m_camTargetName[1] = "Orbital Enforcer";
  // m_camTargetName[2] = "Suspect";

  // TODO real-time date/time + time scale factor display

  // Create Earth
  
  m_camTargetNames.push_back("Earth"); // TODO how to implement camera now - point to all planets, then moons, then ships?

  PlanetEntity& earthPlanet = getPlanet(m_earthPlanetId = makePlanet());
  
  GravBody& earthGravBody = getGravBody(earthPlanet.m_gravBodyIdx = makeGravBody());

  earthGravBody.m_mass = EARTH_MASS;
  earthGravBody.m_radius = EARTH_RADIUS;

  earthGravBody.m_pos = Vector3d(0.0, 0.0, 0.0);
  earthGravBody.m_vel = Vector3d(0.0, 0.0, 0.0);

  RenderableSphere& earthSphere = getSphere(earthPlanet.m_sphereIdx = makeSphere());
  earthSphere.m_radius = earthGravBody.m_radius;
  earthSphere.m_pos = earthGravBody.m_pos;
  earthSphere.m_col = m_colG[1];
  
  m_camTarget = &earthGravBody;
  
  // Create Moon

  m_camTargetNames.push_back("Moon");

  MoonEntity& moonMoon = getMoon(m_moonMoonId = makeMoon());
  
  GravBody& moonGravBody = getGravBody(moonMoon.m_gravBodyIdx = makeGravBody());

  // For now, give the moon a circular orbit

  double const muEarthMoon = (EARTH_MASS + MOON_MASS) * GRAV_CONSTANT;

  double const moonOrbitRadius = pow(muEarthMoon * (MOON_PERIOD / M_TAU) * (MOON_PERIOD / M_TAU), 1.0/3.0); // meters

  Vector3d const moonPos = Vector3d(0.0, 0.0, moonOrbitRadius);
  
  // NOTE computing from constants rather than using the radius - is it better or worse this way?
  double const moonSpeed = pow(muEarthMoon * (M_TAU / MOON_PERIOD), 1.0/3.0); // meters per second
  
  Vector3d const moonVel = Vector3d(moonSpeed, 0.0, 0.0);

  moonGravBody.m_mass = MOON_MASS;
  moonGravBody.m_radius = MOON_RADIUS;

  moonGravBody.m_pos = moonPos;
  moonGravBody.m_vel = moonVel;

  RenderableSphere& moonSphere = getSphere(moonMoon.m_sphereIdx = makeSphere());
  moonSphere.m_radius = moonGravBody.m_radius;
  moonSphere.m_pos = moonGravBody.m_pos;
  moonSphere.m_col = m_colG[1];

  RenderableOrbit& moonOrbit = getOrbit(moonMoon.m_orbitIdx = makeOrbit());
  moonOrbit.m_col = m_colG[1];
  moonOrbit.m_pos = earthGravBody.m_pos;
  
  RenderableTrail& moonTrail = getTrail(moonMoon.m_trailIdx = makeTrail());
  moonTrail.m_colOld = m_colG[0];
  moonTrail.m_colNew = m_colG[4];
  
  // Create ships

  m_camTargetNames.push_back("Player");
  
  ShipEntity& playerShip = getShip(m_playerShipId = makeShip());

  ParticleBody& playerBody = getParticleBody(playerShip.m_particleBodyIdx = makeParticleBody());
  
  playerBody.m_pos = Vector3d(0.0, 0.0, 1.3e7);
  playerBody.m_vel = Vector3d(5e3, 0.0, 0.0);
  
  RenderableOrbit& playerOrbit = getOrbit(playerShip.m_orbitIdx = makeOrbit());
  playerOrbit.m_col = m_colB[2];
  playerOrbit.m_pos = earthGravBody.m_pos;
    
  RenderableTrail& playerTrail = getTrail(playerShip.m_trailIdx = makeTrail());
  playerTrail.m_colOld = m_colB[0];
  playerTrail.m_colNew = m_colB[4];

  RenderablePoint& playerPoint = getPoint(playerShip.m_pointIdx = makePoint());
  playerPoint.m_pos = playerBody.m_pos;
  playerPoint.m_col = m_colB[4];
  
  m_camTargetNames.push_back("Suspect");

  ShipEntity& suspectShip = getShip(m_suspectShipId = makeShip());

  ParticleBody& suspectBody = getParticleBody(suspectShip.m_particleBodyIdx = makeParticleBody());
  
  suspectBody.m_pos = Vector3d(0.0, 0.0, 1.3e7);
  suspectBody.m_vel = Vector3d(5e3, 0.0, 0.0);
  
  RenderableOrbit& suspectOrbit = getOrbit(suspectShip.m_orbitIdx = makeOrbit());
  suspectOrbit.m_col = m_colR[2];
  suspectOrbit.m_pos = earthGravBody.m_pos;
  
  RenderableTrail& suspectTrail = getTrail(suspectShip.m_trailIdx = makeTrail());
  suspectTrail.m_colOld = m_colR[0];
  suspectTrail.m_colNew = m_colR[4];
  
  RenderablePoint& suspectPoint = getPoint(suspectShip.m_pointIdx = makePoint());
  suspectPoint.m_pos = suspectBody.m_pos;
  suspectPoint.m_col = m_colR[4];

  // Perturb all the ship orbits
  float* rnds = new float[6 * m_shipEntities.size()];
  UniformDistribution<float> dist(-1, +1);
  dist.Generate(&m_rnd, 6 * m_shipEntities.size(), &rnds[0]);
  for (int i = 0; i < (int)m_shipEntities.size(); ++i)
  {
    ParticleBody& shipBody = getParticleBody(m_shipEntities[i].m_particleBodyIdx);
    shipBody.m_pos += Vector3d(rnds[6*i  ], rnds[6*i+1], rnds[6*i+2]) * 6e4;
    shipBody.m_vel += Vector3d(rnds[6*i+3], rnds[6*i+4], rnds[6*i+5]) * 1e2;
  }
  delete[] rnds;
  

  m_music.openFromFile("spacething3_mastered_fullq.ogg");
  m_music.setLoop(true);
  m_music.play();
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
      UpdateState(Timer::PerfTimeToMillis(m_lastFrameDuration));
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
  
#ifdef _WIN32
  sf::WindowHandle hwnd = m_window->getSystemHandle();
  ::SetForegroundWindow(hwnd);
  ::SetActiveWindow(hwnd);
  ::SetFocus(hwnd);
  m_hasFocus = true;
#endif
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
      m_camTargetIdx++;
      // TODO maybe should point at renderables instead, but eh...
      if (m_camTargetIdx < m_gravBodies.size()) {
        m_camTarget = &getGravBody(m_camTargetIdx);
      } else if (m_camTargetIdx < m_gravBodies.size() + m_particleBodies.size()) {
        m_camTarget = &getParticleBody(m_camTargetIdx - m_gravBodies.size());
      } else {
        m_camTargetIdx = 0;
        m_camTarget = &getGravBody(0);
      }
    }

    if (_event.key.code == sf::Keyboard::F1) {
      m_camMode = CameraMode_FirstPerson;
    }

    if (_event.key.code == sf::Keyboard::F2) {
      m_camMode = CameraMode_ThirdPerson;
    }

    if (_event.key.code == sf::Keyboard::PageDown) {
      m_integrationMethod = IntegrationMethod((m_integrationMethod + 1) % IntegrationMethod_Count);
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

void OrbitalSpaceApp::CalcParticleAccel(int numParticles, Eigen::Array3Xd const& pp, Eigen::Array3Xd const& vp, int numGravBodies, Eigen::Array3Xd const& pg, Eigen::VectorXd const& mg, Eigen::Array3Xd& o_a)
{
  CalcParticleGrav(numParticles, pp, vp, numGravBodies, pg, mg, o_a);
  o_a.col(m_playerShipId) += CalcThrust(pp.col(m_playerShipId), vp.col(m_playerShipId)).array();
}

// TODO not data-oriented etc
// Calculates acceleration on first body by second body
void OrbitalSpaceApp::CalcParticleGrav(int numParticles, Eigen::Array3Xd const& pp, Eigen::Array3Xd const& vp, int numGravBodies, Eigen::Array3Xd const& pg, Eigen::VectorXd const& mg, Eigen::Array3Xd& o_a)
{
  double const G = GRAV_CONSTANT;

  for (int pi = 0; pi < numParticles; ++pi) {
    Vector3d a(0.0, 0.0, 0.0);
    for (int gi = 0; gi < numGravBodies; ++gi) {
      
      double const M = mg[gi];
   
      double const mu = M * G;

      // Calc acceleration due to gravity
      Vector3d const r = (pg.col(gi) - pp.col(pi));
      double const r_mag = r.norm();

      Vector3d const r_dir = r / r_mag;
      
      Vector3d const a_grav = r_dir * mu / (r_mag * r_mag);
      a += a_grav;
    }
    if (pi == m_playerShipId) {
      a += CalcThrust(pp.col(pi), vp.col(pi));
    }
    o_a.col(pi) = a;
  }
}

void OrbitalSpaceApp::CalcGravAccel(int numGravBodies, Eigen::Array3Xd const& pg, Eigen::Array3Xd const& vg, Eigen::VectorXd const& mg, Eigen::Array3Xd& o_a)
{
  double const G = GRAV_CONSTANT;
  
  for (int g1i = 0; g1i < numGravBodies; ++g1i) {
    for (int g2i = 0; g2i < numGravBodies; ++g2i) {
      if (g1i == g2i) { continue; }

      double const M = mg[g1i] + mg[g2i];
  
      double const mu = M * G;

      // Calc acceleration due to gravity
      Vector3d const r = (pg.col(g1i) - pg.col(g2i));
      double const r_mag = r.norm();

      Vector3d const r_dir = r / r_mag;
      
      Vector3d const a_grav = r_dir * mu / (r_mag * r_mag);

      o_a.col(g1i) = a_grav;
    }
  }
}

Vector3d OrbitalSpaceApp::CalcThrust(Vector3d p, Vector3d v)
{
  Vector3d origin = getGravBody(getPlanet(m_earthPlanetId).m_gravBodyIdx).m_pos;
   
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
    m_simTime += _dt;

    double dt = m_timeScale * Util::Min(_dt, 100.0) / 1000.0; // seconds
    
    // Explicit Euler:
    // p[t + 1] = p[t] + v[t] * dt
    // v[t + 1] = v[t] + a[t] * dt
    // a[t + 1] = calc_accel(p[t + 1], v[t + 1])
    
    // Implicit Euler (what we're currently using) (rearranged):
    // After rearranging the .5 factor is obviously missing in the position update...
    // "Time Adjusted" Verlet, rearranged (messing with position without touching velocity not guaranteed to work after rearranging) (From Physics for Flash Games)
    // Looks equivalent to Implicit Euler! And .5 factor is obviously missing
    // v[t + 1] = v[t] + a[t] * dt
    // p[t + 1] = p[t] + v[t] * dt + a[t] * dt * dt
    // a[t + 1] = calc_accel(p[t + 1], v[t + 1])

    // "Improved" (Midpoint?) Euler (From Physics for Flash Games)
    // v_temp = v[t] + a[t] * dt
    // p_temp = p[t] + v[t] * dt
    // a_temp = calc_accel(p_temp, v_temp)
    // p[t + 1] = p[t] + .5 * (v[t] + v_temp) * dt
    // v[t + 1] = v[t] + .5 * (a[t] + a_temp) * dt

    // lol.zoy.org - don't know what this is called, claims to be velocity verlet; assuming a[t] rather than a[t + 1] because we can't calc a[t + 1] early enough; article used constant a.
    // Rearranged, looks more sensible now...and obviously gives exactly correct results (numerical accuracy issues aside) regardless of timestep size in cases of constant acceleration.
    // v[t + 1] = v[t] + a[t] * dt
    // p[t + 1] = p[t] + v[t] * dt + .5 * a[t] * dt * dt
    // a[t + 1] = calc_accel(p[t + 1], v[t + 1])
    
    // Wikipedia - Velocity Verlet:
    // Assumes that a[t + 1] depends only on position p[t + 1], and not on velocity v[t + 1]. I can use this but I'll have to base thrust on outdated velocity info...
    // p[t + 1] = p[t] + v[t] * dt + .5 * a[t] * dt * dt
    // a[t + 1] = calc_accel(p[t + 1])
    // v[t + 1] = v[t] + .5 * (a[t] + a[t + 1]) * dt
    
    // TODO try these with floats instead of double

    // TODO have the physics update work with multiple buffers?
    // Needs to calculate all accelerations from an arbitrary world state

    switch (m_integrationMethod) {
      case IntegrationMethod_ExplicitEuler: { // Comically bad
        // Previous code, for reference:
        
        // Vector3d const a0 = CalcAccel(i, p0, v0);
        // Vector3d const p1 = p0 + v0 * dt;
        // Vector3d const v1 = v0 + a0 * dt;
          
        // pb.m_pos = p1;
        // pb.m_vel = v1;

        // Load Particle body data

        int numParticles = (int)m_particleBodies.size();
                
        Eigen::Array3Xd p0particles(3, numParticles);
        Eigen::Array3Xd v0particles(3, numParticles);

        for (int i = 0; i < numParticles; ++i) {
          Body& body = m_particleBodies[i];
          p0particles.col(i) = body.m_pos;
          v0particles.col(i) = body.m_vel;
        }

        // Load Grav body data

        int numGravs = (int)m_gravBodies.size();

        Eigen::Array3Xd p0gravs(3, numGravs);
        Eigen::Array3Xd v0gravs(3, numGravs);
        Eigen::VectorXd mgravs(numGravs);

        for (int i = 0; i < numGravs; ++i) {
          GravBody& body = m_gravBodies[i];
          p0gravs.col(i) = body.m_pos;
          v0gravs.col(i) = body.m_vel;
          mgravs[i] = body.m_mass;
        }


        Eigen::Array3Xd a0particles(3, numParticles);
        CalcParticleAccel(numParticles, p0particles, v0particles, numGravs, p0gravs, mgravs, a0particles);
        
        Eigen::Array3Xd a0gravs(3, numGravs);
        CalcGravAccel(numGravs, p0gravs, v0gravs, mgravs, a0gravs);

        Eigen::Array3Xd p1particles = p0particles + v0particles * dt;
        Eigen::Array3Xd v1particles = v0particles + a0particles * dt;
        
        Eigen::Array3Xd p1gravs = p0gravs + v0gravs * dt;
        Eigen::Array3Xd v1gravs = v0gravs + a0gravs * dt;

        // Store Particle body data

        for (int i = 0; i < numParticles; ++i) {
          Body& body = m_particleBodies[i];
          body.m_pos = p1particles.col(i);
          body.m_vel = v1particles.col(i);
        }

        // Store Grav body data

        for (int i = 0; i < numGravs; ++i) {
          Body& body = m_gravBodies[i];
          body.m_pos = p1gravs.col(i);
          body.m_vel = v1gravs.col(i);
        }
        

        break;
      }
#if 0
      case IntegrationMethod_ImplicitEuler: { // Visible creep
        Vector3d const a0 = CalcAccel(i, p0, v0);
        Vector3d const v1 = v0 + a0 * dt;
        Vector3d const p1 = p0 + v1 * dt;

        pb.m_pos = p1;
        pb.m_vel = v1;
        break;
      }
      case IntegrationMethod_ImprovedEuler: { // Looks perfect at low speeds. Really breaks down at 16k x speed... is there drift at slightly lower speeds than that?
        Vector3d const a0 = CalcAccel(i, p0, v0); // TODO this is wrong, needs to store the acceleration/thrust last frame
        Vector3d const vt = v0 + a0 * dt;
        Vector3d const pt = p0 + v0 * dt;
        Vector3d const at = CalcAccel(i, pt, vt);
        Vector3d const p1 = p0 + .5f * (v0 + vt) * dt;
        Vector3d const v1 = v0 + .5f * (a0 + at) * dt;

        pb.m_pos = p1;
        pb.m_vel = v1;
        break;
      }
      case IntegrationMethod_WeirdVerlet: { // Pretty bad - surprised that this is worse than implicit euler rather than better!
        Vector3d const a0 = CalcAccel(i, p0, v0);
        Vector3d const v1 = v0 + a0 * dt;
        Vector3d const p1 = p0 + v0 * dt + .5f * a0 * dt * dt;

        pb.m_pos = p1;
        pb.m_vel = v1;
        break;
      }
      case IntegrationMethod_VelocityVerlet: { // Looks perfect at low speeds. Breaks down at 32k x speed... is there drift at slightly lower speeds? Was less obvious than with IntegrationMethod_ImprovedEuler.
        Vector3d const a0 = CalcAccel(i, p0, v0); // TODO this is wrong, needs to store the acceleration/thrust last frame
        Vector3d const p1 = p0 + v0 * dt + .5f * a0 * dt * dt;
        Vector3d const a1 = CalcAccel(i, p1, v0); // TODO this is wrong, using thrust dir from old frame...
        Vector3d const v1 = v0 + .5f * (a0 + a1) * dt;

        pb.m_pos = p1;
        pb.m_vel = v1;
        break;
      }
#endif
      default: {
        orErr("Unknown Integration Method!");
        break;
      }
    }

    // Update Planets
    for (int i = 0; i < (int)m_planetEntities.size(); ++i) {
      PlanetEntity& planet = getPlanet(i);

      Body& body = getGravBody(planet.m_gravBodyIdx);
      
      RenderableSphere& sphere = getSphere(planet.m_sphereIdx);
      sphere.m_pos = body.m_pos;
    }

    // Update Moon
    for (int i = 0; i < (int)m_moonEntities.size(); ++i) {
      MoonEntity& moon = getMoon(i);

      Body& body = getGravBody(moon.m_gravBodyIdx);

      RenderableOrbit& orbit = getOrbit(moon.m_orbitIdx);
      UpdateOrbit(body, orbit);

      RenderableTrail& trail = getTrail(moon.m_trailIdx);
      trail.Update(_dt, body.m_pos);

      RenderableSphere& sphere = getSphere(moon.m_sphereIdx);
      sphere.m_pos = body.m_pos;
    }

    // Update ships
    for (int i = 0; i < (int)m_shipEntities.size(); ++i) {
      ShipEntity& ship = getShip(i);

      Body& body = getParticleBody(ship.m_particleBodyIdx);
     
      RenderableOrbit& orbit = getOrbit(ship.m_orbitIdx);
      UpdateOrbit(body, orbit);

      RenderableTrail& trail = getTrail(ship.m_trailIdx);
      trail.Update(_dt, body.m_pos);

      RenderablePoint& point = getPoint(ship.m_pointIdx);
      point.m_pos = body.m_pos;
    }
  }

  if (m_singleStep)
  {
    m_singleStep = false;
    m_paused = true;
  }
}

void OrbitalSpaceApp::UpdateOrbit(Body const& body, RenderableOrbit& o_params) {
  // TODO will want to just forward-project instead, this is broken with >1 body

  // Compute Kepler orbit

  GravBody const& earthBody = getGravBody(getPlanet(m_earthPlanetId).m_gravBodyIdx);

  double const G = GRAV_CONSTANT;
  double const M = earthBody.m_mass;
  Vector3d origin = earthBody.m_pos;
    
  double const mu = M * G;

  Vector3d v = body.m_vel;

  Vector3d r = (origin - body.m_pos);
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
  o_params.m_pos = earthBody.m_pos;
}

void OrbitalSpaceApp::DrawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks)
{
    int curStack, curSlice;

    /* Adjust z and radius as stacks and slices are drawn. */

    double r;
    double x,y,z;

    double const sliceInc = M_TAU / (-slices);
    double const stackInc = M_TAU / (2*stacks);
    
    /* Draw a line loop for each stack */
    for (curStack = 1; curStack < stacks; curStack++) {
        y = cos( curStack * stackInc );
        r = sin( curStack * stackInc );

        glBegin(GL_LINE_LOOP);

            for(curSlice = 0; curSlice <= slices; curSlice++) {
                x = cos( curSlice * sliceInc );
                z = sin( curSlice * sliceInc );

                glNormal3d(x,y,z);
                glVertex3d(x*r*radius + pos.x(), y*radius + pos.y(), z*r*radius + pos.z());
            }

        glEnd();
    }

    /* Draw a line loop for each slice */
    for (curSlice = 0; curSlice < slices; curSlice++) {
        glBegin(GL_LINE_STRIP);

            for (curStack = 1; curStack < stacks; curStack++) {
                x = cos( curSlice * sliceInc ) * sin( curStack * stackInc );
                z = sin( curSlice * sliceInc ) * sin( curStack * stackInc );
                y = cos( curStack * stackInc );

                glNormal3d(x,y,z);
                glVertex3d(x*radius + pos.x(), y*radius + pos.y(), z*radius + pos.z());
            }

        glEnd();
    }
}

void OrbitalSpaceApp::DrawCircle(double const radius, int const steps)
{
    /* Adjust z and radius as stacks and slices are drawn. */

    double x,y;

    double const stepInc = M_TAU / steps;
    
    /* Draw a line loop for each stack */
    glBegin(GL_LINE_LOOP);
    for (int curStep = 0; curStep < steps; curStep++) {
        x = cos( curStep * stepInc );
        y = sin( curStep * stepInc );

        glNormal3d(x,y,0.0);
        glVertex3d(x*radius, y*radius, 0.0);
    }
    glEnd();
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
    str << "Cam Target: " << m_camTargetNames[m_camTargetIdx] << "\n";
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
    camPos = getParticleBody(getShip(m_playerShipId).m_particleBodyIdx).m_pos;
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
  
  for (int si = 0; si < (int)m_renderableSpheres.size(); ++si) {
    RenderableSphere const& sphere = getSphere(si);
    Util::SetDrawColour(sphere.m_col);

    DrawWireSphere(sphere.m_pos, sphere.m_radius, 32, 32);
  }
  
  for (int pi = 0; pi < (int)m_renderablePoints.size(); ++pi) {
    RenderablePoint const& point = getPoint(pi);
    Util::SetDrawColour(point.m_col);
    
    glPointSize(10.0);
    glBegin(GL_POINTS);
    Vector3d p = point.m_pos;
    glVertex3d(p.x(), p.y(), p.z());
    glEnd();
    glPointSize(1.0);
  }

  for (int oi = 0; oi < (int)m_renderableOrbits.size(); ++oi) {
    RenderableOrbit const& orbit = getOrbit(oi);
    Util::SetDrawColour(orbit.m_col);

    int const steps = 10000;
    // e = 2.0; // TODO 1.0 sometimes works, > 1 doesn't - do we need to just
    // restrict the range of theta?
    double const delta = .0001;
    double const HAX_RANGE = .9; // limit range to stay out of very large values
    // TODO want to instead limit the range based on... some viewing area?
    // might be two visible segments, one from +ve and one from -ve theta, with
    // different visible ranges. Could determine 
    // TODO and want to take steps of fixed length/distance
    double range;
    if (orbit.e < 1 - delta) { // ellipse
        range = .5 * M_TAU;
    } else if (orbit.e < 1 + delta) { // parabola
        range = .5 * M_TAU * HAX_RANGE;
    } else { // hyperbola
        range = acos(-1/orbit.e) * HAX_RANGE;
    }
    double const mint = -range;
    double const maxt = range;
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= steps; ++i) {
        double const ct = Util::Lerp(mint, maxt, (double)i / steps);
        double const cr = orbit.p / (1 + orbit.e * cos(ct));

        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);
        Vector3d pos = (orbit.x_dir * x_len) + (orbit.y_dir * y_len) + orbit.m_pos;
        glVertex3d(pos.x(), pos.y(), pos.z());
    }
    glEnd();
  }

  for (int ti = 0; ti < (int)m_renderableTrails.size(); ++ti) {
    RenderableTrail const& trail = getTrail(ti);
    trail.Render();
  }
  
  printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime / 1000);
}

OrbitalSpaceApp::RenderableTrail::RenderableTrail(double const _duration) :
  m_duration(_duration), // TODO make sure this works as a value! // TODO what does this mean
  m_timeSinceUpdate(0.f),    
  m_headIdx(0),
  m_tailIdx(0)
{
  for (int i = 0; i < NUM_TRAIL_PTS; ++i)
  {
    m_trailPts[i] = Vector3d::Zero();
  }
}

void OrbitalSpaceApp::RenderableTrail::Update(double const _dt, Vector3d _pos)
{
  // TODO can have a list of past points and their durations, and cut up trail linearly

  // A -- 50ms -- B -- 10 ms -- C

  // So if we get several 10ms updates we would interpolate A towards B a proportional amount, then finally remove it.

  m_timeSinceUpdate += _dt;
      
  if (false) { // m_timeSinceUpdate < TODO) { // duration / NUM_TRAIL_PTS? Idea is to ensure queue always has space. This means we are ensuring a minimum time duration for each segment.
    // Not enough time elapsed. To avoid filling up trail, update the head point instead of adding a new one
    // m_trailPts[m_headIdx] = _pos;
    // m_trailDuration[m_headIdx] = 0.f;
  }
      
  m_headIdx++;
  if (m_headIdx >= NUM_TRAIL_PTS) { m_headIdx = 0; }
  m_trailPts[m_headIdx] = _pos;
  if (m_tailIdx == m_headIdx) { m_tailIdx++; }
      
  // assert(m_headIdx != m_tailIdx);
}

void OrbitalSpaceApp::RenderableTrail::Render() const
{
  glBegin(GL_LINE_STRIP);
  // TODO render only to m_tailIdx // TODO what does this mean
  for (int i = 0; i < RenderableTrail::NUM_TRAIL_PTS; ++i)
  {
    int idx = m_headIdx + i - RenderableTrail::NUM_TRAIL_PTS + 1;
    if (idx < 0) { idx += RenderableTrail::NUM_TRAIL_PTS; }
    Vector3d v = m_trailPts[idx];

    float const l = (float)i / RenderableTrail::NUM_TRAIL_PTS;
    Vector3f c = Util::Lerp(m_colOld, m_colNew, l);
    Util::SetDrawColour(c);

    glVertex3d(v.x(),v.y(),v.z());
  }
  glEnd();
}