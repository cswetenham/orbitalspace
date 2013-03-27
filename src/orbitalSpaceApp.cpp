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

// meters
#define EARTH_RADIUS 6.371e6
// kg
#define EARTH_MASS 5.9742e24
// ??
#define GRAV_CONSTANT 6.6738480e-11

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
  m_camTarget(&m_earthBody),
  m_camTargetIdx(0),
  m_camMode(CameraMode_ThirdPerson),
  m_inputMode(InputMode_Default),
  m_earthBody(),
  m_integrationMethod(IntegrationMethod_ImplicitEuler),
  m_light(1, 1, 0),
  m_thrusters(0),
  m_hasFocus(false),
  m_music(),
  m_timeScale(1.0)
{
  m_colG[0] = Vector3d(41,42,34)/255;
  m_colG[1] = Vector3d(77,82,50)/255;
  m_colG[2] = Vector3d(99,115,76)/255;
  m_colG[3] = Vector3d(151,168,136)/255;
  m_colG[4] = Vector3d(198,222,172)/255;

  m_camTargetName[0] = "Earth";
  m_camTargetName[1] = "Orbital Enforcer";
  m_camTargetName[2] = "Suspect";

  for (int i = 0; i < NUM_COLS; ++i)
  {
    m_colR[i] = Vector3d(m_colG[i].y(), m_colG[i].x(), m_colG[i].z());
    m_colB[i] = Vector3d(m_colG[i].x(), m_colG[i].z(), m_colG[i].y());
  }
  
  m_light /= m_light.norm();

  // TODO real-time date/time + time scale factor display

  m_earthBody.m_pos = Vector3d(0.0, 0.0, 0.0);
  m_earthBody.m_mass = EARTH_MASS; // kg
  m_earthBody.m_radius = EARTH_RADIUS; // m
  
  float rnds[6 * NUM_SHIPS];
  UniformDistribution<float> dist(-1, +1);
  dist.Generate(&m_rnd, 6 * NUM_SHIPS, &rnds[0]);
  for (int i = 0; i < NUM_SHIPS; ++i)
  {
    m_ships[i].m_physics.m_pos = Vector3d(0.0, 0.0, 1.3e7);
    m_ships[i].m_physics.m_vel = Vector3d(5e3, 0.0, 0.0);
    m_ships[i].m_physics.m_pos += Vector3d(rnds[6*i  ], rnds[6*i+1], rnds[6*i+2]) * 6e4;
    m_ships[i].m_physics.m_vel += Vector3d(rnds[6*i+3], rnds[6*i+4], rnds[6*i+5]) * 1e2;
  }

  m_music.openFromFile("spacething3_mastered_fullq.ogg");
  m_music.setLoop(true);
  m_music.play();
}

OrbitalSpaceApp::Ship::Ship() :
  m_physics(),
  m_trail(3.0)
{
}

OrbitalSpaceApp::~OrbitalSpaceApp()
{
}

void OrbitalSpaceApp::Run()
{
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
      // TODO add more bodies. Moon? That would be interesting, could then do sphere of influence code.
      // Could render sphere of influence for each body...
      if (m_camTargetIdx >= NUM_BODIES + NUM_SHIPS) {
        m_camTargetIdx = 0;
        m_camTarget = &m_earthBody;
      } else {
        m_camTarget = &m_ships[m_camTargetIdx - 1].m_physics;
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

// TODO not data-oriented etc
Vector3d OrbitalSpaceApp::CalcAccel(int i, Vector3d p, Vector3d v)
{
  double const G = GRAV_CONSTANT;
  double const M = m_earthBody.m_mass;
  Vector3d origin = m_earthBody.m_pos;
    
  double const mu = M * G;

  // Calc acceleration due to gravity
  Vector3d const r = (origin - p);
  double const r_mag = r.norm();

  Vector3d const r_dir = r / r_mag;
      
  Vector3d const a_grav = r_dir * mu / (r_mag * r_mag);

  // Calc acceleration due to thrust
  double const thrustAccel = 10.0; // meters per second squared - TODO what is a realistic value?
      
  Vector3d thrustVec(0.0,0.0,0.0);

  if (i == 0) { // TODO HAX, selecting first ship as controllable one
    Vector3d const fwd = v / v.norm(); // Prograde
    Vector3d const left = fwd.cross(r_dir); // name? (and is the order right?)
    Vector3d const dwn = left.cross(fwd); // name? (and is the order right?)

    if (m_thrusters & ThrustFwd)  { thrustVec += fwd; }
    if (m_thrusters & ThrustBack) { thrustVec -= fwd; }
    if (m_thrusters & ThrustDown)  { thrustVec += dwn; }
    if (m_thrusters & ThrustUp) { thrustVec -= dwn; }
    if (m_thrusters & ThrustLeft)  { thrustVec += left; }
    if (m_thrusters & ThrustRight) { thrustVec -= left; }
  }

  Vector3d a_thrust = thrustAccel * thrustVec;

  Vector3d a = a_grav + a_thrust;
  return a;
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

    for (int i = 0; i < NUM_SHIPS; ++i) {
      // Define directions
      PhysicsBody& pb = m_ships[i].m_physics;

      Vector3d const v0 = pb.m_vel;
      Vector3d const p0 = pb.m_pos;

      switch (m_integrationMethod) {
        case IntegrationMethod_ExplicitEuler: { // Comically bad
          Vector3d const a0 = CalcAccel(i, p0, v0);
          Vector3d const p1 = p0 + v0 * dt;
          Vector3d const v1 = v0 + a0 * dt;
          
          pb.m_pos = p1;
          pb.m_vel = v1;
          break;
        }
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
        default: {
          orErr("Unknown Integration Method!");
          break;
        }
      }

      // Update orbit
      ComputeKeplerParams(m_ships[i].m_physics, m_ships[i].m_orbit);
      
      // Update trail
      m_ships[i].m_trail.Update(_dt, pb.m_pos);
    }
  }

  if (m_singleStep)
  {
    m_singleStep = false;
    m_paused = true;
  }
}

void OrbitalSpaceApp::ComputeKeplerParams(PhysicsBody const& body, OrbitParams& o_params) {
  // Compute Kepler orbit
  double const G = GRAV_CONSTANT;
  double const M = m_earthBody.m_mass;
  Vector3d origin = m_earthBody.m_pos;
    
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
}

void OrbitalSpaceApp::DrawWireSphere(double const radius, int const slices, int const stacks)
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
                glVertex3d(x*r*radius,y*radius,z*r*radius);
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
                glVertex3d(x*radius,y*radius,z*radius);
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
    str << "Cam Target: " << m_camTargetName[m_camTargetIdx] << "\n";
    str << "Cam Dist: " << m_camDist << "\n";
    str << "Cam Theta:" << m_camTheta << "\n";
    str << "Cam Phi:" << m_camPhi << "\n";
    double const shipDist = (m_ships[0].m_physics.m_pos - m_ships[1].m_physics.m_pos).norm();
    str << "Intership Distance:" << shipDist << "\n";
    str << "Integration Method: " << m_integrationMethod << "\n";

    // TODO: better double value text formatting
    // TODO: small visualisations for the angle etc values
   
    text.setString(str.str());
    m_window->draw(text);
  }

  m_window->resetGLStates();

  glViewport(0, 0, m_config.width, m_config.height);

  Vector3d c = m_colG[0];
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

  assert(m_camTarget);
  Vector3d const camTarget = m_camTarget->m_pos;

  Vector3d camPos;
  
  if (m_camMode == CameraMode_FirstPerson) {
    camPos = m_ships[0].m_physics.m_pos;
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
  
  {
    Vector3f c = m_colG[1].cast<float>();
    Util::SetDrawColour(c);
  }
  DrawWireSphere(m_earthBody.m_radius, 32, 32);

  // TODO collision detection

  for (int s = 0; s < NUM_SHIPS; ++s) {
    // Draw ship
    glPointSize(10.0);
    glBegin(GL_POINTS);
    Vector3d p = m_ships[s].m_physics.m_pos;
    if (s == 0) {
      Util::SetDrawColour(m_colB[4]);
    } else {
      Util::SetDrawColour(m_colR[4]);
    }
    glVertex3d(p.x(), p.y(), p.z());
    glEnd();
    glPointSize(1.0);

    // Draw orbit
    {
      OrbitParams const& orbit = m_ships[s].m_orbit;
      
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
      if (s == 0) {
        Util::SetDrawColour(m_colB[2]);
      } else {
        Util::SetDrawColour(m_colR[2]);
      }
      for (int i = 0; i <= steps; ++i) {
          double const ct = Util::Lerp(mint, maxt, (double)i / steps);
          double const cr = orbit.p / (1 + orbit.e * cos(ct));

          double const x_len = cr * -cos(ct);
          double const y_len = cr * -sin(ct);
          Vector3d pos = (orbit.x_dir * x_len) + (orbit.y_dir * y_len);
          glVertex3d(pos.x(), pos.y(), pos.z());
      }
      glEnd();
    }

    // Draw trail
#if 0
    if (s == 0) {
      m_ships[s].m_trail.Render(m_colB[0], m_colB[4]);
    } else {
      m_ships[s].m_trail.Render(m_colR[0], m_colR[4]);
    }
#endif
  }
  
  printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime / 1000);
}

