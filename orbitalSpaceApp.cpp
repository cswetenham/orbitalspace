#include "orbitalSpaceApp.h"

#include "perftimer.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <Eigen/Geometry>

// TODO #include <boost/posix_time.hpp>

// meters
#define EARTH_RADIUS 6.371e9
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
  m_camDist(-3.1855e10),
  m_camTheta(0.0),
  m_camPhi(0.0),
  m_camTarget(&m_earthBody),
  m_camTargetIdx(0),
  m_earthBody(),
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
  UniformDistribution dist(-1, +1);
  dist.Generate(&m_rnd, 6 * NUM_SHIPS, &rnds[0]);
  for (int i = 0; i < NUM_SHIPS; ++i)
  {
    m_ships[i].m_physics.m_pos = Vector3d(0.0, 0.0, 1.3e10);
    m_ships[i].m_physics.m_vel = Vector3d(1.7e2, 0.0, 0.0);
    m_ships[i].m_physics.m_pos += Vector3d(rnds[6*i  ], rnds[6*i+1], rnds[6*i+2]) * 6e7;
    m_ships[i].m_physics.m_vel += Vector3d(rnds[6*i+3], rnds[6*i+4], rnds[6*i+5]) * 1e1;
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
  double const a = Util::FMod(3.0, 2.0);
  if ( a != 1.0 ) {
    __debugbreak();
  }

  double const b = Util::Wrap(3.5, 1.0, 2.0);
  if (b != 1.5) {
    __debugbreak();
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
      sf::Vector2i const centerPos = sf::Vector2i(m_config.width/2, m_config.height/2);
      sf::Vector2i const mouseDelta = sf::Mouse::getPosition(*m_window) - centerPos;
      sf::Mouse::setPosition(centerPos, *m_window);
    
      m_camTheta += mouseDelta.x * M_TAU / 300.0;
      m_camTheta = Util::Wrap(m_camTheta, 0.0, M_TAU);
      m_camPhi += mouseDelta.y * M_TAU / 300.0;
      m_camPhi = Util::Clamp(m_camPhi, -.249 * M_TAU, .249 * M_TAU);
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
  m_config.width = 800;
  m_config.height = 600;
  
  sf::ContextSettings settings;
  settings.depthBits         = 24; // Request a 24 bits depth buffer
  settings.stencilBits       = 8;  // Request a 8 bits stencil buffer
  settings.antialiasingLevel = 2;  // Request 2 levels of antialiasing
  m_window = new sf::RenderWindow(sf::VideoMode(m_config.width, m_config.height, 32), "SFML OpenGL", sf::Style::Close, settings);
  
  m_window->setMouseCursorVisible(false);
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
      if (m_camTargetIdx > NUM_SHIPS) {
        m_camTargetIdx = 0;
        m_camTarget = &m_earthBody;
      } else {
        m_camTarget = &m_ships[m_camTargetIdx - 1].m_physics;
      }
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

void OrbitalSpaceApp::UpdateState(double const _dt)
{
  if (!m_paused)
  {
    m_simTime += _dt;

    double dt = m_timeScale * Util::Min(_dt, 100.0) / 1000.0; // seconds
    
    double const G = GRAV_CONSTANT;
    double const M = m_earthBody.m_mass;
    Vector3d origin = m_earthBody.m_pos;
    
    double const mu = M * G;
    
    for (int i = 0; i < NUM_SHIPS; ++i)
    {
      // Define directions
      PhysicsBody& pb = m_ships[i].m_physics;

      Vector3d v = pb.m_vel;

      Vector3d r = (origin - pb.m_pos);
      double const r_mag = r.norm();

      Vector3d r_dir = r/r_mag;

      double const vr_mag = r_dir.dot(v);
      Vector3d vr = r_dir * vr_mag; // radial velocity
      Vector3d vt = v - vr; // tangent velocity
      double const vt_mag = vt.norm();
      Vector3d t_dir = vt/vt_mag;

      // Apply gravity
      Vector3d dv = dt * r_dir * mu / (r_mag * r_mag);

      // Vector3d dv = dt * HAX_SCALE_FACTOR * d * (G * M) / r;
      v += dv;

      // Apply thrust
    
      double const thrustAccel = 100.0;
      double const thrustDV = thrustAccel * dt;
      
      Vector3d thrustVec(0.0,0.0,0.0);

      Vector3d fwd = pb.m_vel / pb.m_vel.norm(); // Prograde
      Vector3d left = fwd.cross(r_dir); // name? (and is the order right?)
      Vector3d dwn = left.cross(fwd); // name? (and is the order right?)

      if (i == 0) // TODO HAX, selecting first ship as controllable one
      {
        if (m_thrusters & ThrustFwd)  { thrustVec += fwd; }
        if (m_thrusters & ThrustBack) { thrustVec -= fwd; }
        if (m_thrusters & ThrustDown)  { thrustVec += dwn; }
        if (m_thrusters & ThrustUp) { thrustVec -= dwn; }
        if (m_thrusters & ThrustLeft)  { thrustVec += left; }
        if (m_thrusters & ThrustRight) { thrustVec -= left; }
      }

      v += thrustDV * thrustVec;

      pb.m_vel = v;

      // Update position
      pb.m_pos += pb.m_vel * dt;

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
    for (curStack = 1; curStack < stacks; curStack++)
    {
        y = cos( curStack * stackInc );
        r = sin( curStack * stackInc );

        glBegin(GL_LINE_LOOP);

            for(curSlice = 0; curSlice <= slices; curSlice++)
            {
                x = cos( curSlice * sliceInc );
                z = sin( curSlice * sliceInc );

                glNormal3d(x,y,z);
                glVertex3d(x*r*radius,y*radius,z*r*radius);
            }

        glEnd();
    }

    /* Draw a line loop for each slice */
    for (curSlice = 0; curSlice < slices; curSlice++)
    {
        glBegin(GL_LINE_STRIP);

            for (curStack = 1; curStack < stacks; curStack++)
            {
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
    for (int curStep = 0; curStep < steps; curStep++)
    {
        x = cos( curStep * stepInc );
        y = sin( curStep * stepInc );

        glNormal3d(x,y,0.0);
        glVertex3d(x*radius, y*radius, 0.0);
    }
    glEnd();
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
    
    std::ostringstream str;

    uint32_t const fontSize = 14;
    sf::Text text(sf::String("Hello, World!"), font, fontSize);
    text.setColor(sf::Color(ct.x(), ct.y(), ct.z(), 255));
    text.setPosition(8, 8);

    str.precision(3);
    str.width(7);
    str.flags(std::ios::right + std::ios::fixed);
        
    str << "Time Scale: " << m_timeScale << "\n";
    str << "Cam Dist: " << m_camDist << "\n";
    str << "Cam Theta:" << m_camTheta << "\n";
    str << "Cam Phi:" << m_camPhi << "\n";

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
  gluPerspective(fov, aspect, 1.0, 1e14); // meters

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  Vector3d up(0.0, 1.0, 0.0);

  assert(m_camTarget);
  Vector3d const camTarget = m_camTarget->m_pos;

  Vector3d camPos = Vector3d(0.0, 0.0, m_camDist);

  Eigen::AngleAxisd thetaRot(m_camTheta, Vector3d(0.0, 1.0, 0.0));
  Eigen::AngleAxisd phiRot(m_camPhi, Vector3d(1.0, 0.0, 0.0));

  Eigen::Affine3d camMat1;
  camMat1.setIdentity();
  camMat1.rotate(thetaRot).rotate(phiRot);

  camPos = camMat1 * camPos;
  camPos += camTarget;

  // TODO remove the gluLookAt? Or just have it off by default for now?
  if (m_camOrig) {
    gluLookAt(camPos.x(), camPos.y(), camPos.z(),
              camTarget.x(), camTarget.y(), camTarget.z(),
              up.x(), up.y(), up.z());
  } else {
    // Finally does the same as gluLookAt!
    Vector3d camF = (camTarget - camPos).normalized();
    Vector3d camR = camF.cross(up).normalized();
    Vector3d camU = camF.cross(camR).normalized();

    Matrix3d camMat;
    camMat.col(0) = camR;
    camMat.col(1) = -camU;
    camMat.col(2) = -camF;

    Eigen::Affine3d camT;
    camT.linear() = camMat;
    camT.translation() = camPos;

    glMultMatrix(camT.inverse());
  }

  glEnable(GL_TEXTURE_2D);
  
  glLineWidth(1);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_LINE_SMOOTH);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // TODO clean up
  if (m_wireframe)
  {
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  }
  else
  {
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
  }
  
  {
    Vector3f c = m_colG[1].cast<float>();
    Util::SetDrawColour(c);
  }
  DrawWireSphere(m_earthBody.m_radius, 32, 32);

  // TODO collision detection

  for (int s = 0; s < NUM_SHIPS; ++s)
  {
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
      for (int i = 0; i <= steps; ++i)
      {
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
    if (s == 0) {
      m_ships[s].m_trail.Render(m_colB[0], m_colB[4]);
    } else {
      m_ships[s].m_trail.Render(m_colR[0], m_colR[4]);
    }
  }
  
  printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime / 1000);
}

