#include "orStd.h"

#include "orApp.h"

#include "orProfile/perftimer.h"

#if 0
#include "task.h"
#include "taskScheduler.h"
#include "taskSchedulerWorkStealing.h"
#endif

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

// SDL
#include <SDL.h>
#include <SDL_log.h>

#include <Eigen/Geometry>

#include "boost_begin.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include "boost_end.h"

#include "constants.h"

#include "util.h"

#include "orGfx.h"

#include "orPlatform/window.h"

orApp::orApp(Config const& config):
  m_appScreen(Screen_Title),
  m_lastFrameDuration(0),
  m_running(true),
  m_rnd(1123LL),
  m_simTime(0.0),
  m_config(config),
  m_paused(false),
  m_singleStep(false),

  m_cameraSystem(),
  m_camMode(CameraMode_ThirdPerson),
  m_cameraId(0),
  m_cameraTargetId(0),
  m_camParams(-3.1855e7),

  m_renderSystem(),
  m_frameBuffer(),
  m_wireframe(false),

  m_physicsSystem(),
  m_timeScale(1.0),
  m_integrationMethod(PhysicsSystem::IntegrationMethod_RK4),

  m_entitySystem(m_cameraSystem, m_renderSystem, m_physicsSystem),
  m_playerShipId(0),

  m_inputMode(InputMode_Default),
  m_thrusters(0),
  m_hasFocus(false),
  m_window(NULL),
  m_music(NULL)
{
  orLog("Starting init\n");

  PerfTimer::StaticInit();

  Init();

#if 0
  m_music = new Music();
  m_music->openFromFile("music/spacething3_mastered_fullq.ogg");
  m_music->setLoop(true);
  m_music->play();
#endif
  SDL_LogWarn(SDL_LOG_CATEGORY_ERROR, "TODO NYI Music");
  orLog("Init complete\n");
}

orApp::~orApp()
{
  orLog("Starting shutdown\n");
  Shutdown();

  delete m_music; m_music = NULL;

  PerfTimer::StaticShutdown(); // TODO terrible code

  orLog("Shutdown complete\n");
}

void orApp::Init()
{
  // TODO init audio
  // TODO init...joystick? gamecontroller? need to look up what each of those do
  if (SDL_Init(SDL_INIT_EVENTS|SDL_INIT_TIMER|SDL_INIT_VIDEO) != 0) {
    fprintf(stderr,
      "\nUnable to initialize SDL:  %s\n",
      SDL_GetError()
    );
  }

  // Show anything with priority of Info and above
  SDL_LogSetAllPriority(SDL_LOG_PRIORITY_INFO);

  InitState();
  InitRender();
}

void orApp::Shutdown()
{
  ShutdownRender();
  ShutdownState();
  SDL_Quit();
}

// TODO put elsewhere
void runTests() {
  // Test FMod
  double const a = orFMod(3.0, 2.0);
  if ( a != 1.0 ) {
    DEBUGBREAK;
  }

  // Test Wrap
  double const b = orWrap(3.5, 1.0, 2.0);
  if (b != 1.5) {
    DEBUGBREAK;
  }
}

void orApp::PollEvents()
{
  SDL_Event event;
  while (SDL_PollEvent(&event))
  {
    HandleEvent(event);
  }
}

void orApp::Run()
{
  runTests();

  while (m_running)
  {
    Timer::PerfTime const frameStart = Timer::GetPerfTime();

    RunOneStep();

    SDL_Delay(1); // milliseconds

    m_lastFrameDuration = Timer::GetPerfTime() - frameStart;
  }
}

void orApp::RunOneStep()
{
  {
    PERFTIMER("PollEvents");
    PollEvents();
  }

  {
    PERFTIMER("HandleInput");
    HandleInput();
  }

  {
    PERFTIMER("UpdateState");
    UpdateState();
  }

  {
    PERFTIMER("RenderState");
    RenderState();
  }
}

void orApp::HandleInput()
{
  if (!m_hasFocus) {
    return;
  }

  // Input handling

  // TODO Relative mouse mode doesn't work right on linux;
  // implement manually with Grab and GetState
  if (m_hasFocus && m_inputMode == InputMode_RotateCamera) {
    int x;
    int y;
    SDL_GetMouseState(&x, &y);
    SDL_WarpMouseInWindow(m_window, m_config.windowWidth / 2, m_config.windowHeight / 2);
    double const dx = (x - m_config.windowWidth / 2) * M_TAU / 300.0;
    m_camParams.theta = orWrap(m_camParams.theta + dx, 0.0, M_TAU);
    double const dy = (y - m_config.windowHeight / 2) * M_TAU / 300.0;
    m_camParams.phi = Util::Clamp(m_camParams.phi + dy, -.249 * M_TAU, .249 * M_TAU);
  }
}

// TODO cleanup
char* file_contents(const char* filename, GLint* length)
{
    FILE* f = fopen(filename, "r");
    char* buffer;

    if (!f) {
        fprintf(stderr, "Unable to open %s for reading\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = (char*)malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    buffer[*length] = '\0';

    return buffer;
}

// TODO cleanup
static void show_info_log(
    GLuint object,
    PFNGLGETSHADERIVPROC glGet__iv,
    PFNGLGETSHADERINFOLOGPROC glGet__InfoLog
)
{
    GLint log_length;
    char* log;

    glGet__iv(object, GL_INFO_LOG_LENGTH, &log_length);
    log = (char*)malloc(log_length);
    glGet__InfoLog(object, log_length, NULL, log);
    fprintf(stderr, "%s", log);
    free(log);
}

// TODO cleanup
static GLuint make_shader(GLenum type, const char* filename)
{
    GLint length;
    GLchar* source = file_contents(filename, &length);
    GLuint shader;
    GLint shader_ok;

    if (!source)
        return 0;

    shader = glCreateShader(type);
    glShaderSource(shader, 1, (const GLchar**)&source, &length);
    free(source);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);
    if (!shader_ok) {
        fprintf(stderr, "Failed to compile %s:\n", filename);
        show_info_log(shader, glGetShaderiv, glGetShaderInfoLog);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// TODO cleanup
static GLuint make_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLint program_ok;

    GLuint program = glCreateProgram();

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &program_ok);
    if (!program_ok) {
        fprintf(stderr, "Failed to link shader program:\n");
        show_info_log(program, glGetProgramiv, glGetProgramInfoLog);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void orApp::InitRender()
{
  // TODO move to render system

  // TODO z-ordering seems bad on text labels (Body label appears in front of everything else - not sure if this is what I want, would be good to have the option)
  // TODO 3d labels show when behind the camera as well as when they are in front.
  // TODO trails are bugged when switching camera targets.

  // TODO SDL logging not flushing?
#if 0
  sf::ContextSettings settings;
  settings.depthBits         = 24; // Request a 24 bits depth buffer
  settings.stencilBits       = 8;  // Request a 8 bits stencil buffer
  settings.antialiasingLevel = 2;  // Request 2 levels of antialiasing
#endif
  // TODO backbuffer settings as above?

  // Window mode MUST include SDL_WINDOW_OPENGL for use with OpenGL.
  m_window = SDL_CreateWindow(
    "Orbital Space", 0, 0, m_config.windowWidth, m_config.windowHeight,
    SDL_WINDOW_OPENGL);

  // Create an OpenGL context associated with the window.
  m_gl_context = SDL_GL_CreateContext(m_window);

  // TODO some part of the SDL init causes a spurious error.
  // Find it, and add a loop calling glGetError() until it returns 0...

  GLenum err = glewInit();
  if (GLEW_OK != err) {
    /* Problem: glewInit failed, something is seriously wrong. */
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }

  m_frameBuffer = m_renderSystem.makeFrameBuffer(m_config.renderWidth, m_config.renderHeight);

  m_renderSystem.initRender();

#ifdef WIN32 // TODO linux
  sf::WindowHandle winHandle = m_window->getSystemHandle();
  orPlatform::FocusWindow(winHandle);
#endif
  m_hasFocus = true;
}

void orApp::ShutdownRender()
{
  m_renderSystem.shutdownRender();

  // TODO free opengl resources

  // Once finished with OpenGL functions, the SDL_GLContext can be deleted.
  SDL_GL_DeleteContext(m_gl_context);

  SDL_DestroyWindow(m_window);
  m_window = NULL;
}

int orApp::spawnBody(
  std::string const& name,
  double const radius,
  double const mass,
  orEphemerisJPL const& ephemeris_jpl,
  int const parent_grav_body_id
)
{
  orEphemerisCartesian ephemeris_cart;

  Eigen::Vector3d parent_pos(0, 0, 0);
  Eigen::Vector3d parent_vel(0, 0, 0);
  if (parent_grav_body_id) {
    PhysicsSystem::GravBody const& parentBody = m_physicsSystem.getGravBody(parent_grav_body_id);
    parent_pos = parentBody.m_pos;
    parent_vel = parentBody.m_vel;
  }

  ephemerisCartesianFromJPL(
    ephemeris_jpl,
    m_simTime,
    ephemeris_cart
  );

  orVec3 pos(ephemeris_cart.pos + parent_pos);
  orVec3 vel(ephemeris_cart.vel + parent_vel);

  int body_id;
  EntitySystem::Body& body = m_entitySystem.getBody(body_id = m_entitySystem.makeBody());

  {
    PhysicsSystem::GravBody& gravBody = m_physicsSystem.getGravBody(body.m_gravBodyId = m_physicsSystem.makeGravBody());
    gravBody.m_ephemeris = ephemeris_jpl;
    gravBody.m_mass = mass;
    gravBody.m_radius = radius;
    gravBody.m_pos = pos;
    gravBody.m_vel = vel;
    gravBody.m_soiParentBody = parent_grav_body_id;
  }

  {
    RenderSystem::Sphere& sphere = m_renderSystem.getSphere(body.m_sphereId = m_renderSystem.makeSphere());
    sphere.m_radius = radius;
    sphere.m_pos = pos;
    sphere.m_col = RenderSystem::Colour(1.0, 1.0, 0); // TODO use this type throughout
  }

  // Best to init parents before children for this to work!
  if (parent_grav_body_id)
  {
    RenderSystem::Orbit& orbit = m_renderSystem.getOrbit(body.m_orbitId = m_renderSystem.makeOrbit());
    // Orbit pos is pos of parent body
    orbit.m_pos = m_physicsSystem.getGravBody(parent_grav_body_id).m_pos;
    orbit.m_col = m_colG[1];
  }

  {
    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(body.m_cameraTargetId = m_cameraSystem.makeTarget());
    camTarget.m_pos = pos;
    camTarget.m_name = name;
  }

  {
    RenderSystem::Label3D& label = m_renderSystem.getLabel3D(body.m_label3DId = m_renderSystem.makeLabel3D());
    label.m_pos = pos;
    label.m_col = orVec3(1.0, 1.0, 0.0);
    label.m_text = name;
  }

  return body_id;
}

void orApp::InitState()
{
  typedef RenderSystem::Colour Colour;
  // Create NES-ish palette (selected colour sets from the NES palettes, taken from Wikipedia)
  m_colR[0] = Colour(0,0,0)/255.f;
  m_colR[1] = Colour(136,20,0)/255.f;
  m_colR[2] = Colour(228,92,16)/255.f;
  m_colR[3] = Colour(252,160,68)/255.f;
  m_colR[4] = Colour(252,224,168)/255.f;

  m_colG[0] = Colour(0,0,0)/255.f;
  m_colG[1] = Colour(0,120,0)/255.f;
  m_colG[2] = Colour(0,184,0)/255.f;
  m_colG[3] = Colour(184,248,24)/255.f;
  m_colG[4] = Colour(216,248,120)/255.f;

  m_colB[0] = Colour(0,0,0)/255.f;
  m_colB[1] = Colour(0,0,252)/255.f;
  m_colB[2] = Colour(0,120,248)/255.f;
  m_colB[3] = Colour(60,188,252)/255.f;
  m_colB[4] = Colour(164,228,252)/255.f;

  // Make camera

  CameraSystem::Camera& camera = m_cameraSystem.getCamera(m_cameraId = m_cameraSystem.makeCamera());
  camera.m_fov = 35.0; // degrees? Seems low...this is the vertical fov though...

  // Make debug text label3D

  // RenderSystem::Label2D& debugTextLabel2D = m_renderSystem.getLabel2D(m_debugTextLabel2DId = m_renderSystem.makeLabel2D());
  m_uiTextTopLabel2DId = m_renderSystem.makeLabel2D();
  RenderSystem::Label2D& m_uiTextTopLabel2D = m_renderSystem.getLabel2D(m_uiTextTopLabel2DId);

  m_uiTextTopLabel2D.m_pos = orVec2(0, 0);

  m_uiTextTopLabel2D.m_col = m_colG[4];

  m_uiTextBottomLabel2DId = m_renderSystem.makeLabel2D();
  RenderSystem::Label2D& m_uiTextBottomLabel2D = m_renderSystem.getLabel2D(m_uiTextBottomLabel2DId);

  // TODO HAX - 8 is the bitmap font height, shouldn't be hardcoded
  m_uiTextBottomLabel2D.m_pos = orVec2(0, m_config.renderHeight - 8);

  m_uiTextBottomLabel2D.m_col = m_colG[4];

  // TODO s_jpl_elements_t0[ephemeris_id]

  // Create Sun
  {
    orEphemerisJPL sun_ephemeris = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    m_sunBodyId = spawnBody("Sun", SUN_RADIUS, SUN_MASS, sun_ephemeris, 0);
  }

  // Set initial camera target to be the Sun
  m_cameraTargetId = m_entitySystem.getBody(m_sunBodyId).m_cameraTargetId;

  // Create Mercury
  {
    int const ephemeris_idx = 0;
    m_mercuryBodyId = spawnBody("Mercury", MERCURY_RADIUS, MERCURY_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Venus
  {
    int const ephemeris_idx = 1;
    m_venusBodyId = spawnBody("Venus", VENUS_RADIUS, VENUS_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Earth
  // TODO earth and moon should orbit about common barycenter, but I don't have good data on that.
  {
    int const ephemeris_idx = 2;
    m_earthBodyId = spawnBody("Earth", EARTH_RADIUS, EARTH_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Moon
  {
    int const ephemeris_idx = 9;
    m_moonBodyId = spawnBody("Moon", MOON_RADIUS, MOON_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_earthBodyId).m_gravBodyId);
  }

  // Create Mars
  {
    int const ephemeris_idx = 3;
    m_marsBodyId = spawnBody("Mars", MARS_RADIUS, MARS_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Jupiter
  {
    int const ephemeris_idx = 4;
    m_jupiterBodyId = spawnBody("Jupiter", JUPITER_RADIUS, JUPITER_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Saturn
  {
    int const ephemeris_idx = 5;
    m_saturnBodyId = spawnBody("Saturn", SATURN_RADIUS, SATURN_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Neptune
  {
    int const ephemeris_idx = 6;
    m_neptuneBodyId = spawnBody("Neptune", NEPTUNE_RADIUS, NEPTUNE_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Uranus
  {
    int const ephemeris_idx = 7;
    m_uranusBodyId = spawnBody("Uranus", URANUS_RADIUS, URANUS_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Uranus
  {
    int const ephemeris_idx = 8;
    m_plutoBodyId = spawnBody("Pluto", PLUTO_RADIUS, PLUTO_MASS, s_jpl_elements_t0[ephemeris_idx], m_entitySystem.getBody(m_sunBodyId).m_gravBodyId);
  }

  // Create Earth-Body COM and Lagrange points
#if 0
  EntitySystem::Poi& comPoi = m_entitySystem.getPoi(m_comPoiId = m_entitySystem.makePoi());

  orVec3 comPos = ephemeris[2].pos;

  {
    RenderSystem::Point& comPoint = m_renderSystem.getPoint(comPoi.m_pointId = m_renderSystem.makePoint());
    comPoint.m_pos = comPos;
    comPoint.m_col = RenderSystem::Colour(1.0, 0.0, 0.0);
  }

  {
    CameraSystem::Target& comCamTarget = m_cameraSystem.getTarget(comPoi.m_cameraTargetId = m_cameraSystem.makeTarget());
    comCamTarget.m_pos = comPos;
    comCamTarget.m_name = std::string("Earth-Body COM");
  }

  for (int i = 0; i < 5; ++i) {
    EntitySystem::Poi& lagrangePoi = m_entitySystem.getPoi(m_lagrangePoiIds[i] = m_entitySystem.makePoi());

    // Correct positions will get computed in update step
    {
      RenderSystem::Point& lagrangePoint = m_renderSystem.getPoint(lagrangePoi.m_pointId = m_renderSystem.makePoint());
      lagrangePoint.m_pos = orVec3(0, 0, 0);
      lagrangePoint.m_col = RenderSystem::Colour(1.0, 0.0, 0.0);
    }

    {
      CameraSystem::Target& lagrangeCamTarget = m_cameraSystem.getTarget(lagrangePoi.m_cameraTargetId = m_cameraSystem.makeTarget());
      lagrangeCamTarget.m_pos = orVec3(0, 0, 0);
      std::stringstream builder;
      builder << "Earth-Body L" << (i + 1);
      lagrangeCamTarget.m_name = builder.str();
    }
  }
#endif

  // Create ships
  PhysicsSystem::GravBody const& earthBody = m_physicsSystem.getGravBody(m_entitySystem.getBody(m_earthBodyId).m_gravBodyId);
  Vector3d earthPos = earthBody.m_pos;
  Vector3d earthVel = earthBody.m_vel;
  {
    EntitySystem::Ship& playerShip = m_entitySystem.getShip(m_playerShipId = m_entitySystem.makeShip());

    orVec3 playerPos(earthPos + Vector3d(0.0, 0.0, 1.3e7));
    orVec3 playerVel(earthVel + Vector3d(5e3, 0.0, 0.0));
    {
      PhysicsSystem::ParticleBody& playerBody = m_physicsSystem.getParticleBody(playerShip.m_particleBodyId = m_physicsSystem.makeParticleBody());
      playerBody.m_pos = playerPos;
      playerBody.m_vel = playerVel;
      playerBody.m_userAcc = orVec3(0, 0, 0);
    }

    {
      RenderSystem::Orbit& playerOrbit = m_renderSystem.getOrbit(playerShip.m_orbitId = m_renderSystem.makeOrbit());
      playerOrbit.m_pos = earthPos;
      playerOrbit.m_col = m_colB[2];
    }

#if 0
    {
      RenderSystem::Trail& playerTrail = m_renderSystem.getTrail(playerShip.m_trailId = m_renderSystem.makeTrail());
      playerTrail.Init(5000.0, playerPos, m_cameraSystem.getTarget(m_cameraTargetId).m_pos);
      playerTrail.m_colOld = m_colB[0];
      playerTrail.m_colNew = m_colB[4];
    }
#endif

    {
      RenderSystem::Point& playerPoint = m_renderSystem.getPoint(playerShip.m_pointId = m_renderSystem.makePoint());
      playerPoint.m_pos = playerPos;
      playerPoint.m_col = m_colB[4];
    }

    {
      CameraSystem::Target& playerCamTarget = m_cameraSystem.getTarget(playerShip.m_cameraTargetId = m_cameraSystem.makeTarget());
      playerCamTarget.m_pos = playerPos;
      playerCamTarget.m_name = std::string("Player");
    }
  }

  {
    EntitySystem::Ship& suspectShip = m_entitySystem.getShip(m_suspectShipId = m_entitySystem.makeShip());

    orVec3 suspectPos(earthPos + Vector3d(0.0, 0.0, 1.3e7));
    orVec3 suspectVel(earthVel + Vector3d(5e3, 0.0, 0.0));
    {
      PhysicsSystem::ParticleBody& suspectBody = m_physicsSystem.getParticleBody(suspectShip.m_particleBodyId = m_physicsSystem.makeParticleBody());
      suspectBody.m_pos = suspectPos;
      suspectBody.m_vel = suspectVel;
      suspectBody.m_userAcc = orVec3(0.0, 0.0, 0.0);
    }

    {
      RenderSystem::Orbit& suspectOrbit = m_renderSystem.getOrbit(suspectShip.m_orbitId = m_renderSystem.makeOrbit());
      suspectOrbit.m_pos = earthPos;
      suspectOrbit.m_col = m_colR[2];
    }

#if 0
    {
      RenderSystem::Trail& suspectTrail = m_renderSystem.getTrail(suspectShip.m_trailId = m_renderSystem.makeTrail());
      suspectTrail.Init(5000.0, suspectPos, m_cameraSystem.getTarget(m_cameraTargetId).m_pos);
      suspectTrail.m_colOld = m_colR[0];
      suspectTrail.m_colNew = m_colR[4];
    }
#endif

    {
      RenderSystem::Point& suspectPoint = m_renderSystem.getPoint(suspectShip.m_pointId = m_renderSystem.makePoint());
      suspectPoint.m_pos = suspectPos;
      suspectPoint.m_col = m_colR[4];
    }

    {
      CameraSystem::Target& suspectCamTarget = m_cameraSystem.getTarget(suspectShip.m_cameraTargetId = m_cameraSystem.makeTarget());
      suspectCamTarget.m_pos = suspectPos;
      suspectCamTarget.m_name = std::string("Suspect");
    }
  }
  // Perturb all the ship orbits
  // TODO this should update all the other positions too, or happen earlier!
  // TODO shouldn't iterate through IDs outside a system.
#if 0
  float* rnds = new float[6 * m_entitySystem.numShips()];
  UniformDistribution<float> dist(-1, +1);
  dist.Generate(&m_rnd, 6 * m_entitySystem.numShips(), &rnds[0]);
  for (int i = 0; i < m_entitySystem.numShips(); ++i)
  {
    PhysicsSystem::ParticleBody& shipBody = m_physicsSystem.getParticleBody(m_entitySystem.getShip(i).m_particleBodyId);
    // could round-trip to vector, or do it in components...
    shipBody.m_pos[0] += 6e4 * rnds[6*i  ];
    shipBody.m_pos[1] += 6e4 * rnds[6*i+1];
    shipBody.m_pos[2] += 6e4 * rnds[6*i+2];

    shipBody.m_vel[0] += 1e2 * rnds[6*i+3];
    shipBody.m_vel[1] += 1e2 * rnds[6*i+4];
    shipBody.m_vel[2] += 1e2 * rnds[6*i+5];
  }
  delete[] rnds;
#endif
}


void orApp::ShutdownState()
{
}

void orApp::HandleEvent(SDL_Event const& _event)
{
  // TODO allow resize?

  switch(_event.type) {
    case SDL_MOUSEBUTTONDOWN: {
      if (_event.button.button == SDL_BUTTON_RIGHT) {
        m_inputMode = InputMode_RotateCamera;
        SDL_ShowCursor(SDL_DISABLE);
        SDL_GetMouseState(&m_savedMouseX, &m_savedMouseY);
        SDL_WarpMouseInWindow(m_window, m_config.windowWidth / 2, m_config.windowHeight / 2);
      }
      break;
    }

    case SDL_MOUSEBUTTONUP: {
      if (_event.button.button == SDL_BUTTON_RIGHT) {
        m_inputMode = InputMode_Default;
        SDL_WarpMouseInWindow(m_window, m_savedMouseX, m_savedMouseY);
        SDL_ShowCursor(SDL_ENABLE);
      }
      break;
    }

    case SDL_MOUSEWHEEL: {
      double const WHEEL_SPEED = 1.0;
      m_camParams.dist *= pow(0.9, WHEEL_SPEED * _event.wheel.y);
      break;
    }

    case SDL_KEYDOWN: {
      if (_event.key.keysym.sym == SDLK_ESCAPE)
      {
        m_running = false;
      }

      if (_event.key.keysym.sym == SDLK_TAB)
      {
        m_cameraTargetId = m_cameraSystem.nextTarget(m_cameraTargetId);
        // TODO clear all trails
        // TODO make all trails relative to camera target
        // Maybe make a physics frame object, with position + velocity?
        // Make Body the frame?
        // Make frame its own system, have render, physics etc share the frame?
      }

      if (_event.key.keysym.sym == SDLK_F1) {
        m_camMode = CameraMode_FirstPerson;
      }

      if (_event.key.keysym.sym == SDLK_F2) {
        m_camMode = CameraMode_ThirdPerson;
      }

      if (_event.key.keysym.sym == SDLK_PAGEDOWN) {
        m_integrationMethod = PhysicsSystem::IntegrationMethod((m_integrationMethod + 1) % PhysicsSystem::IntegrationMethod_Count);
      }

      if (_event.key.keysym.sym == SDLK_PLUS || _event.key.keysym.sym == SDLK_EQUALS)
      {
        m_timeScale *= 2;
        if (m_timeScale > 1<<16) {
          m_timeScale = 1<<16;
        }
      }

      if (_event.key.keysym.sym == SDLK_MINUS || _event.key.keysym.sym == SDLK_UNDERSCORE)
      {
        m_timeScale /= 2;
        if (m_timeScale < 1) {
          m_timeScale = 1;
        }
      }

      if (_event.key.keysym.sym == SDLK_a)
      {
        m_thrusters |= ThrustLeft;
      }
      if (_event.key.keysym.sym == SDLK_d)
      {
        m_thrusters |= ThrustRight;
      }
      if (_event.key.keysym.sym == SDLK_w)
      {
        m_thrusters |= ThrustFwd;
      }
      if (_event.key.keysym.sym == SDLK_s)
      {
        m_thrusters |= ThrustBack;
      }
      if (_event.key.keysym.sym == SDLK_UP)
      {
        m_thrusters |= ThrustUp;
      }
      if (_event.key.keysym.sym == SDLK_DOWN)
      {
        m_thrusters |= ThrustDown;
      }
      break;
    }

    case SDL_KEYUP: {
      if (_event.key.keysym.sym == SDLK_a)
      {
        m_thrusters &= ~ThrustLeft;
      }
      if (_event.key.keysym.sym == SDLK_d)
      {
        m_thrusters &= ~ThrustRight;
      }
      if (_event.key.keysym.sym == SDLK_w)
      {
        m_thrusters &= ~ThrustFwd;
      }
      if (_event.key.keysym.sym == SDLK_s)
      {
        m_thrusters &= ~ThrustBack;
      }
      if (_event.key.keysym.sym == SDLK_UP)
      {
        m_thrusters &= ~ThrustUp;
      }
      if (_event.key.keysym.sym == SDLK_DOWN)
      {
        m_thrusters &= ~ThrustDown;
      }
      break;
    }

    case SDL_QUIT: {
      m_running = false;
      break;
    }

    case SDL_WINDOWEVENT: {
      if (_event.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
      {
        m_hasFocus = false;
      }

      if (_event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
      {
        m_hasFocus = true;
      }
      break;
    }

    default:
      break;
  } // switch (_event.type)
}

Vector3d orApp::CalcPlayerThrust(PhysicsSystem::ParticleBody const& playerBody)
{
  Vector3d const origin(m_physicsSystem.findSOIGravBody(playerBody).m_pos);
  Vector3d const playerPos(playerBody.m_pos);
  Vector3d const playerVel(playerBody.m_vel);

  // Calc acceleration due to gravity
  Vector3d const r = origin - playerPos;
  double const r_mag = r.norm();

  Vector3d const r_dir = r / r_mag;

  // Calc acceleration due to thrust
  double const thrustAccel = 10.0; // meters per second squared - TODO what is a realistic value?

  Vector3d thrustVec(0.0,0.0,0.0);

  Vector3d const fwd = playerVel.normalized(); // Prograde
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

void orApp::UpdateState_Bodies(double const dt)
{
  // Update player thrust
  PhysicsSystem::ParticleBody& playerShipBody = m_physicsSystem.getParticleBody(m_entitySystem.getShip(m_playerShipId).m_particleBodyId);
  Vector3d const userAcc = CalcPlayerThrust(playerShipBody);

  playerShipBody.m_userAcc = userAcc;

  m_physicsSystem.update(m_integrationMethod, m_simTime, dt);

  // TODO eaghghgh not clear where these should live
#if 0
  PhysicsSystem::GravBody& earthBody = m_physicsSystem.getGravBody(m_entitySystem.getBody(m_earthBodyId).m_gravBodyId);
  EntitySystem::Body& moon = m_entitySystem.getBody(m_moonBodyId);
  PhysicsSystem::GravBody& moonBody = m_physicsSystem.getGravBody(moon.m_gravBodyId);

  Vector3d const earthPos(earthBody.m_pos);
  Vector3d const earthVel(earthBody.m_vel);
  Vector3d const moonPos(moonBody.m_pos);
  Vector3d const moonVel(moonBody.m_vel);

  // Update the moon's label
  RenderSystem::Label3D& moonLabel3D = m_renderSystem.getLabel3D(moon.m_label3DId);
  moonLabel3D.m_pos = moonPos;

  // Update the earth-moon COM
  double const totalMass = earthBody.m_mass + moonBody.m_mass;

  Vector3d const comPos = (earthPos * earthBody.m_mass / totalMass) + (moonPos * moonBody.m_mass / totalMass);

  EntitySystem::Poi& comPoi = m_entitySystem.getPoi(m_comPoiId);

  RenderSystem::Point& comPoint = m_renderSystem.getPoint(comPoi.m_pointId);
  {
    comPoint.m_pos = comPos;
  }

  CameraSystem::Target& comTarget = m_cameraSystem.getTarget(comPoi.m_cameraTargetId);
  {
    comTarget.m_pos = comPos;
  }

  // Update the earth-moon Lagrange points
  Vector3d const earthMoonVector = moonPos - earthPos;
  double const earthMoonOrbitRadius = earthMoonVector.norm();
  Vector3d const earthMoonDir = earthMoonVector / earthMoonOrbitRadius;
  double const massRatio = MOON_MASS / EARTH_MASS;
  double const r1 = earthMoonOrbitRadius * pow(massRatio / 3.0, 1.0/3.0);
  double const r3 = earthMoonOrbitRadius * (1.0 + (7.0/12.0) * massRatio); // extra 1.0 to make r3 a distand from Earth position rather than an offset from earthMoonOrbitRadius

  Vector3d lagrangePos[5];
  // Lagrange point 1
  lagrangePos[0] = moonPos - earthMoonDir * r1;
  // Lagrange point 2
  lagrangePos[1] = moonPos + earthMoonDir * r1;
  // Lagrange point 3
  lagrangePos[2] = earthPos - earthMoonDir * r3;

  // L4 and L5 are on the Body's orbit, 60 degrees ahead and 60 degrees behind.
  Vector3d orbitAxis = moonVel.normalized().cross(earthMoonVector.normalized());
  Eigen::AngleAxisd rotation(M_TAU / 6.0, orbitAxis);

  // Lagrange point 4
  lagrangePos[3] = rotation           * earthMoonVector;
  // Lagrange point 5
  lagrangePos[4] = rotation.inverse() * earthMoonVector;

  for (int i = 0; i < 5; ++i) {
    EntitySystem::Poi& lagrangePoi = m_entitySystem.getPoi(m_lagrangePoiIds[i]);
    RenderSystem::Point& lagrangePoint = m_renderSystem.getPoint(lagrangePoi.m_pointId);
    CameraSystem::Target& lagrangeTarget = m_cameraSystem.getTarget(lagrangePoi.m_cameraTargetId);

    lagrangePoint.m_pos = lagrangePos[i];
    lagrangeTarget.m_pos = lagrangePos[i];
  }
#endif
}

void orApp::UpdateState_CamTargets(double const dt) {
  m_entitySystem.updateCamTargets(dt, m_cameraSystem.getCamera(m_cameraId).m_pos);
}
void orApp::UpdateState_RenderObjects(double const dt) {
  m_entitySystem.updateRenderObjects(dt, m_cameraSystem.getCamera(m_cameraId).m_pos);
}

Vector3d orApp::CamPosFromCamParams(OrbitalCamParams const& params)
{
  Eigen::AngleAxisd thetaRot(params.theta, Vector3d(0.0, 0.0, 1.0));
  Eigen::AngleAxisd phiRot(params.phi, Vector3d(1.0, 0.0, 0.0));

  Eigen::Affine3d mat;
  mat.setIdentity();
  mat.rotate(thetaRot).rotate(phiRot);

  return mat * Vector3d(0.0, params.dist, 0.0);
}

void orApp::UpdateState()
{
  double const dt = m_timeScale * Util::Min((double)Timer::PerfTimeToMillis(m_lastFrameDuration), 100.0) / 1000.0; // seconds

  if (!m_paused) {
    UpdateState_Bodies(dt);

    m_simTime += dt;
  }

  if (m_singleStep) {
    // Pause simulation after this step
    m_singleStep = false;
    m_paused = true;
  }

  UpdateState_CamTargets(dt);

  {
    // Update camera

    CameraSystem::Target& camTarget = m_cameraSystem.getTarget(m_cameraTargetId);
    Vector3d const camTargetPos(camTarget.m_pos);

    Vector3d camPos;

    if (m_camMode == CameraMode_FirstPerson) {
      camPos = Vector3d(m_physicsSystem.getParticleBody(m_entitySystem.getShip(m_playerShipId).m_particleBodyId).m_pos);
    } else if (m_camMode == CameraMode_ThirdPerson) {
      // Camera position is based on its target's position
      // TODO so now we need the physics update to happen before the camera
      // but the render update to happen afterwards. This would also make sense
      // since it would potentially move the camera while paused.
      camPos = CamPosFromCamParams(m_camParams) + camTargetPos;
    } else {
      assert(false);
    }

    CameraSystem::Camera& camera = m_cameraSystem.getCamera(m_cameraId);

    const double* const camPosData = camPos.data();

    camera.m_pos[0] = camPosData[0];
    camera.m_pos[1] = camPosData[1];
    camera.m_pos[2] = camPosData[2];
  }

  UpdateState_RenderObjects(dt);

  // Update debug text
  {
    PERFTIMER("DebugText");

    // Top of screen
    {
      std::ostringstream str;
      str.precision(3);
      str.flags(std::ios::right | std::ios::fixed);

      str << calendarDateFromSimTime(m_simTime) << "\n";

      str << "Time Scale: " << (int)m_timeScale << "\n";

      // str << "FPS: " << (int)(1000.0 / Timer::PerfTimeToMillis(m_lastFrameDuration)) << "\n";
      // str << "Cam Dist: " << m_camDist << "\n";
      // str << "Cam Theta:" << m_camTheta << "\n";
      // str << "Cam Phi:" << m_camPhi << "\n";
      // double const shipDist = (m_ships[0].m_physics.m_pos - m_ships[1].m_physics.m_pos).norm();
      // str << "Intership Distance:" << shipDist << "\n";
      // str << "Intership Distance: TODO\n";
      // str << "Integration Method: " << m_integrationMethod << "\n";

      // TODO: better double value text formatting
      // TODO: small visualisations for the angle etc values

      RenderSystem::Label2D& m_uiTextTopLabel2D = m_renderSystem.getLabel2D(m_uiTextTopLabel2DId);
      m_uiTextTopLabel2D.m_text = str.str();
    }

    // Bottom of screen
    {
      std::ostringstream str;
      str.precision(3);
      str.flags(std::ios::right | std::ios::fixed);

      CameraSystem::Target& camTarget = m_cameraSystem.getTarget(m_cameraTargetId);
      str << "Cam Target: " << camTarget.m_name << "\n";

      RenderSystem::Label2D& m_uiTextBottomLabel2D = m_renderSystem.getLabel2D(m_uiTextBottomLabel2DId);
      m_uiTextBottomLabel2D.m_text = str.str();
    }
  }
}

Vector3d lerp(Vector3d const& _x0, Vector3d const& _x1, double const _a) {
    return _x0 * (1 - _a) + _x1 * _a;
}

char const* orApp::s_jpl_names[] = {
  "Mercury",
  "Venus",
  "EarthBody",
  "Mars",
  "Jupiter",
  "Saturn",
  "Uranus",
  "Neptune",
  "Pluto"
};

orEphemerisJPL orApp::s_jpl_elements_t0[] = {
  {  0.38709843, 0.20563661,  7.00559432, 252.25166724,  77.45771895,  48.33961819,  0.00000000,  0.00002123, -0.00590158, 149472.67486623,  0.15940013, -0.12214182,           0,           0,           0,           0 },
  {  0.72332102, 0.00676399,  3.39777545, 181.97970850, 131.76755713,  76.67261496, -0.00000026, -0.00005107,  0.00043494,  58517.81560260,  0.05679648, -0.27274174,           0,           0,           0,           0 },
  {  1.00000018, 0.01673163, -0.00054346, 100.46691572, 102.93005885,  -5.11260389, -0.00000003, -0.00003661, -0.01337178,  35999.37306329,  0.31795260, -0.24123856,           0,           0,           0,           0 },
  {  1.52371243, 0.09336511,  1.85181869,  -4.56813164, -23.91744784,  49.71320984,  0.00000097,  0.00009149, -0.00724757,  19140.29934243,  0.45223625, -0.26852431,           0,           0,           0,           0 },
  {  5.20248019, 0.04853590,  1.29861416,  34.33479152,  14.27495244, 100.29282654, -0.00002864,  0.00018026, -0.00322699,   3034.90371757,  0.18199196,  0.13024619, -0.00012452,  0.06064060, -0.35635438, 38.35125000 },
  {  9.54149883, 0.05550825,  2.49424102,  50.07571329,  92.86136063, 113.63998702, -0.00003065, -0.00032044,  0.00451969,   1222.11494724,  0.54179478, -0.25015002,  0.00025899, -0.13434469,  0.87320147, 38.35125000 },
  { 19.18797948, 0.04685740,  0.77298127, 314.20276625, 172.43404441,  73.96250215, -0.00020455, -0.00001550, -0.00180155,    428.49512595,  0.09266985,  0.05739699,  0.00058331, -0.97731848,  0.17689245,  7.67025000 },
  { 30.06952752, 0.00895439,  1.77005520, 304.22289287,  46.68158724, 131.78635853,  0.00006447,  0.00000818,  0.00022400,    218.46515314,  0.01009938, -0.00606302, -0.00041348,  0.68346318, -0.10162547,  7.67025000 },
  { 39.48686035, 0.24885238, 17.14104260, 238.96535011, 224.09702598, 110.30167986,  0.00449751,  0.00006016,  0.00000501,    145.18042903, -0.00968827, -0.00809981, -0.01262724,           0,           0,           0 },
  { // Moon TODO this is wrt earth ecliptic
    384400.0 * 1000.0 / METERS_PER_AU, // semi-major axis, AU
    0.0554, // eccentricity
    5.16, // inclination, deg
    135.27 + 125.08 + 318.15, // mean longitude, deg = mean anomaly + longitude of periapsis = Mean anomaly + longitude of ascending node + argument of periapsis
    125.08 + 318.15, // longitude of periapsis, deg
    125.08, // longitude of ascending node, deg
    0, // semi major axis, AU per C
    0, // eccentricity, per C
    0, // inclination, deg per C
    13.176358 * DAYS_PER_CENTURY, // mean longitude, deg per C
    0, // longitude of perihelion, deg per C
    0, // longitude of ascending node, deg per C
    0, // error b deg
    0, // error c deg
    0, // error s deg
    0  // error f deg
  }
};

void orApp::RenderState()
{
  PERFTIMER("RenderState");

  // Projection matrix (GL_PROJECTION)
  double const minZ = 1e6; // meters
  double const maxZ = 1e13; // meters

  double const aspect = m_config.windowWidth / (double)m_config.windowHeight;
  Eigen::Matrix4d const projMatrix = m_cameraSystem.calcProjMatrix(m_cameraId, m_config.renderWidth, m_config.renderHeight, minZ, maxZ, aspect);

  // TODO for better accuracy, want to avoid using a camera matrix for the translation
  // Instead, subtract the camera position from everything before passing it to the render system
  // TODO check that the camera position is updated at the correct time in the frame
  // before we use it to update the render positions!

  // Camera matrix (GL_MODELVIEW)
  Vector3d up = Vector3d::UnitZ();
  Eigen::Matrix4d const camMatrix = m_cameraSystem.calcCameraMatrix(m_cameraId, m_cameraTargetId, up);

  // Used to translate a 3d position into a 2d framebuffer position
  Eigen::Matrix4d const screenMatrix = m_cameraSystem.calcScreenMatrix(m_config.renderWidth, m_config.renderHeight);

  RenderSystem::Colour clearCol = m_colG[0];

  m_renderSystem.render(m_frameBuffer, clearCol, minZ, screenMatrix, projMatrix, camMatrix);

  {
    PERFTIMER("PostEffect");

    // Render from 2D framebuffer to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_config.windowWidth, m_config.windowHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);

    float const scale = 1.0;
    float const uv_scale = 1.0;

    glBindTexture(GL_TEXTURE_2D, m_frameBuffer.colorTextureId);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-scale, -scale, 0.0);
    glTexCoord2f(uv_scale, 0.0);
    glVertex3f(+scale, -scale, 0.0);
    glTexCoord2f(uv_scale, uv_scale);
    glVertex3f(+scale, +scale, 0.0);
    glTexCoord2f(0.0, uv_scale);
    glVertex3f(-scale, +scale, 0.0);
    glEnd();

    glDisable(GL_TEXTURE_2D);
  }

  SDL_GL_SwapWindow(m_window);

  // printf("Frame Time: %04.1f ms Total Sim Time: %04.1f s \n", Timer::PerfTimeToMillis(m_lastFrameDuration), m_simTime / 1000);
}
