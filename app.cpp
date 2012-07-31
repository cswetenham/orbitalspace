#include "app.h"

#include "perftimer.h"
#include "util.h"

# include <GL/gl.h>
# include <GL/glu.h>

App::App():
  m_window(NULL),
  m_lastFrameDuration(0),
  m_running(true)
{
  PerfTimer::StaticInit(); // TODO terrible code
}

App::~App()
{
  delete m_window; m_window = NULL;
  PerfTimer::StaticShutdown(); // TODO terrible code
}

void App::Init()
{
  InitState();
  InitRender();
}

void App::Shutdown()
{
  PerfTimer::Print();

  ShutdownRender();
  ShutdownState();
}

void App::InitRender()
{
}

void App::ShutdownRender()
{
  m_window->close();
  delete m_window; m_window = NULL;
}

void App::PollEvents()
{
    sf::Event event;
    while (m_window->pollEvent(event))
    {
      HandleEvent(event);
    }
}

void App::InitState()
{
}

void App::ShutdownState()
{
}

void App::Run()
{
  // TODO need to run event loop while (window.isOpen())?
  // TODO this doesn't get called :|
    
  while (m_running)
  {
    Timer::PerfTime const frameStart = Timer::GetPerfTime();
    double dt = Timer::PerfTimeToMillis(m_lastFrameDuration)/1000;
    dt = Util::Min(dt, .1); // Clamp simulation stepsize
    
    {
      PERFTIMER("PollEvents");
      PollEvents();
    }

    {
      PERFTIMER("UpdateState");
      UpdateState(dt);
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

void App::BeginRender()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
}

void App::EndRender()
{
  m_window->display();
}