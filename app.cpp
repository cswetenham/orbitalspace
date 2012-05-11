#include "app.h"

#include "perftimer.h"
#include "util.h"

#ifdef _WIN32
# include <SDL.h>
# include <SDL_opengl.h>
#else
# include <SDL/SDL.h>
# include <SDL/SDL_opengl.h>
# include <GL/gl.h>
# include <GL/glu.h>
#endif

App::App():
  m_display(NULL),
  m_lastFrameDuration(0),
  m_running(true)
{
  PerfTimer::StaticInit(); // TODO terrible code
}

App::~App()
{
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
  if(SDL_Init(SDL_INIT_EVERYTHING | SDL_INIT_NOPARACHUTE) < 0) {
    exit(1);
  }

  SDL_GL_SetAttribute( SDL_GL_DOUBLEBUFFER, 1 );
  if((m_display = SDL_SetVideoMode(640, 480, 32, SDL_HWSURFACE | SDL_OPENGL)) == NULL) {
    exit(3);
  }
  
  glClearColor(0, 0, 0, 0);
  glClearDepth(1.0f);
}

void App::ShutdownRender()
{
  SDL_FreeSurface(m_display);
  m_display = NULL;
  SDL_Quit();
}

void App::InitState()
{
}

void App::ShutdownState()
{
}

void App::Run()
{
  while (m_running)
  {
    Timer::PerfTime const frameStart = Timer::GetPerfTime();
    float dt = Timer::PerfTimeToMillis(m_lastFrameDuration)/1000.f;
    dt = Util::Min(dt, .1f); // Clamp simulation stepsize
    
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

void App::PollEvents()
{
  SDL_Event event;
  while(SDL_PollEvent(&event)) {
    HandleEvent(event);
  }
}

void App::BeginRender()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
}

void App::EndRender()
{
  SDL_GL_SwapBuffers();
}