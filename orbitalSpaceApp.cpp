#include "orbitalSpaceApp.h"

#include "perftimer.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

OrbitalSpaceApp::OrbitalSpaceApp():
  App(),
  m_rnd(1123LL),
  m_simTime(0.f),
  m_paused(false),
  m_singleStep(false),
  m_wireframe(false),
  m_camZ(-1000),
  m_camTheta(0.f),
  m_camPhi(0.f),
  m_col1(41,42,34),
  m_col2(77,82,50),
  m_col3(99,115,76),
  m_col4(151,168,136),
  m_col5(198,222,172),
  m_shipPos(0.f, 0.f, 200.f),
  m_shipVel(100.f, 0.f, 0.f),
  m_trailIdx(0)
{
  m_col1/=255;
  m_col2/=255;
  m_col3/=255;
  m_col4/=255;
  m_col5/=255;

  for (int i = 0; i < NUM_TRAIL_PTS; ++i)
  {
    m_trailPts[i] = m_shipPos;
  }
}

OrbitalSpaceApp::~OrbitalSpaceApp()
{
}

void OrbitalSpaceApp::Run()
{
  while (m_running)
  {
    Timer::PerfTime const frameStart = Timer::GetPerfTime();
    
    {
      PERFTIMER("PollEvents");
      PollEvents();
    }

    // Input handling
    sf::Vector2i const centerPos = sf::Vector2i(m_config.width/2, m_config.height/2);
    sf::Vector2i const mouseDelta = sf::Mouse::getPosition(*m_window) - centerPos;
    sf::Mouse::setPosition(centerPos, *m_window);
    
    m_camTheta += mouseDelta.x * M_TAU / 100.f;
    Util::Wrap(m_camTheta, 0.f, M_TAU);
    m_camPhi += mouseDelta.y * M_TAU / 100.f;
    Util::Wrap(m_camPhi, 0.f, M_TAU);

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
  m_window = new sf::Window(sf::VideoMode(m_config.width, m_config.height, 32), "SFML OpenGL", sf::Style::Close, settings);
  
  m_window->setMouseCursorVisible(false);

  glViewport(0, 0, m_config.width, m_config.height);

  Vector3f c = m_col1;
  glClearColor(c.x(), c.y(), c.z(), 0);
  glClearDepth(1.0f);
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

  // TODO mouse capture isn't ideal?

  if (_event.type == sf::Event::MouseWheelMoved)
  {
    m_camZ += 10.f * _event.mouseWheel.delta;
  }

  if (_event.type == sf::Event::KeyPressed)
  {
    if (_event.key.code == sf::Keyboard::Escape)
    {
      m_running = false;
    }
  }

  if (_event.type == sf::Event::Closed)
  {
    m_running = false;
  }
}

void OrbitalSpaceApp::UpdateState(float const _dt)
{
  if (!m_paused)
  {
    m_simTime += _dt;

    float const dt = _dt / 1000.f;
    
    m_shipPos += m_shipVel * dt;
    
    Vector3f origin(0,0,0);
    float const G = 6.6738480e-11f;
    float const M = 5.9742e24f;
    Vector3f d = (origin - m_shipPos);
    float const r = d.norm();
    // TODO there are better ways of doing this I'm sure
    // TODO get scales, distances right. Maybe need scaling matrix for rendering?
    float const HAX_SCALE_FACTOR = 0.00000001f;
    Vector3f n = d/r;
    Vector3f dv = dt * HAX_SCALE_FACTOR * n * (G * M) / (r * r);
    m_shipVel += dv;

    m_trailIdx++;
    if (m_trailIdx >= NUM_TRAIL_PTS) { m_trailIdx = 0; }
    m_trailPts[m_trailIdx] = m_shipPos;
  }

  if (m_singleStep)
  {
    m_singleStep = false;
    m_paused = true;
  }
}

void OrbitalSpaceApp::DrawWireSphere(float const radius, int const slices, int const stacks)
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

void OrbitalSpaceApp::DrawCircle(float const radius, int const steps)
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

        glNormal3d(x,y,0.f);
        glVertex3d(x*radius, y*radius, 0.f);
    }
    glEnd();
}

void OrbitalSpaceApp::RenderState()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Units: cm
  float const aspect = m_config.width / (float)m_config.height;
  float const height = 500.f; // cm
  float const width = height * aspect;
  Vector2f size(width, height);
  Vector2f tl = -.5f * size;
  Vector2f br = .5f * size;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  float const fov = 35.f;
  gluPerspective(fov, aspect, 1.0f, 10000.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  Vector3f eye(0.0, 0.0, m_camZ);
  Vector3f focus(0.0, 0.0, 0.0);
  Vector3f up(0.0, 1.0, 0.0);

  glTranslatef(0.f, 0.f, m_camZ);
  glRotatef(m_camTheta, 0.0f, 1.0f, 0.0f);
  glRotatef(m_camPhi, 1.0f, 0.0f, 0.0f);
  glTranslatef(0.f, 0.f, -m_camZ);

  gluLookAt(eye.x(), eye.y(), eye.z(),
            focus.x(), focus.y(), focus.z(),
            up.x(), up.y(), up.z());

  glEnable(GL_TEXTURE_2D);
  
  glLineWidth(1);

  if (m_wireframe)
  {
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
  }
  else
  {
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
  }
  
  SetDrawColour(m_col3);
  DrawWireSphere(100.0f, 32, 32);

  SetDrawColour(m_col5);
  // DrawCircle(200.0f, 32);
  glBegin(GL_LINE_STRIP);
    for (int i = 0; i < NUM_TRAIL_PTS; ++i)
    {
      int idx = m_trailIdx + i - NUM_TRAIL_PTS + 1;
      if (idx < 0) { idx += NUM_TRAIL_PTS; }
      Vector3f v = m_trailPts[idx];
      glVertex3f(v.x(),v.y(),v.z());
    }
  glEnd();

  printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime);
}

void OrbitalSpaceApp::SetDrawColour(Vector3f const _c)
{
  glColor3f(_c.x(), _c.y(), _c.z());
}
