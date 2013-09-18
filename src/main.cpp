#include "orStd.h"

#include "orApp.h"

#include <SFML/Window.hpp>

// TODO Entity and other systems also need to support destroying objects.
// Need a policy for reuse of Ids/Handles
// - Valid until end of frame?
// - Have Ids keep incrementing so time till reuse is maximised?

// Strategy:
// - First make IDs into struct System::Id { uint32 value; }; or struct System::Type::Id { uint32 value; };
// Conversion to a uint32 is easy, but won't be incrementing them by accident.
// Fix any resulting errors. Remove numFoo(). Maybe an isValid() type method for debugging at least?
// - Then add a new function for destroy
// - construct/destruct
// - allocate/free
// - create/destroy
// - claim/release
// - add/remove
// - insert/erase
// - new/delete
// - do/undo
// - resume/yield
// make/unmake? raze? lose? end? finish? desist? give up? yield? demolish? dismantle? dispose? disavow? disown?
// release kind of works semantically but it suggests reference counting

// TODO Need a way to tie gameplay objects in save files to system objects; GUID?

// TODO Perftimer shouldn't use StaticInit/StaticShutdown

// TODO we start a seemingly huge number of threads on startup, and many of them die after a certain amount
// of time. The report that they have died comes when we call m_window->pollEvent(event).
// Would like to figure out where these come from.
// Idea: set ourselves up to poll events before every system has been brought up, bring them up one by one
// and see what happens with threads starting and ending.

// TODO finish/use orTask

// TODO Load solar system, ships, etc from a config files
// TODO give moon, earth correct orbit, inclination, rotation period, axial tilt.
// TODO give earth, moon rotation and orbit correct phase for J2000.
// TODO add Sun, other planets and moons

// TODO make the opaque IDs into strong typedefs for safety? (enum - not perfect. struct containing just an int?)

// TODO improve palettes

// TODO don't like the fact that EntitySystem depends on PhysicsSystem and RenderSystem

// TODO not happy "UpdateOrbit" lives in EntitySystem

// TODO move orGfx to orPlatform

// TODO update both Linux and Windows versions to SFML 2.0 final

// TODO figure out better packaging solution for Linux; just require devs to install SFML 2.0 to system location?

// TODO GravBody required to be told what its parent body is; want to remove that

// TODO Want to figure out a better way of storing and simulating large-scale stuff.
// For serveral purposed (camera, visualisation, etc) want to have notion of frames (inertial and rotating)
// Bodies could still have a "parent" body or system, deduced from masses and positions...
// Pos and vel could be stored relative to parent body.

// TODO Fix trail:
// Make methods into methods on RenderSystem instead.
// Fix case of switching camera: easiest solution is to just clear the trail.
// Switching it to a new frame of reference would actually mean storing the whole history of the system for a duration
// If trail is for a fixed length of wall-time, would want to clear the trail on time accel change too.

// TODO want to achieve orbits around L-points, maybe also Weak Stability Boundary low-energy transfers.
// For that need tools to allow achieving that.

// TODO far too easy to accelerate too much at the moment; ideally, would scale acceleration with time scale so you can
// always have the same level of responsiveness.

// TODO RenderSystem::render2D, RenderSystem::renderLabels have terrible params.

// TODO Should we have a renderer which just knows about the offscreen texture,
// and then a screen effect which renders it to screen with scaling, scanlines etc?

// TODO set up CONFIG_DEBUG, CONFIG_PROFILE

// TODO clean up rnd.h, put in namespace, template better
// TODO clean up util.h, put in namespace

// TODO want ability to record past simulation states and extrapolate future ones.
// Will need to support entities being created or destroyed; that means our history isn't just a big fixed-sized array for each frame.
// Will want to support AI; we'll want to be able to run the AI on predicted future states too.

// TODO UpdateOrbit needs to be broken up into several things.
// A simple keplerian projector
// Something that detects SOI switches
// Something that detects SOI switches and finds the actual switchover point to sub-frame accuracy
// An n-body forward projector

int CDECL main()
{
  orApp::Config appConfig;

#if 0
  appConfig.windowWidth = 1280;
  appConfig.windowHeight = 768;
  appConfig.renderWidth = 320;
  appConfig.renderHeight = 200;
#elif ENABLE_FRAMEBUFFER
  appConfig.windowWidth = 4 * 320;
  appConfig.windowHeight = 3 * 320;
  appConfig.renderWidth = 256;
  appConfig.renderHeight = 240;
#else
  appConfig.windowWidth = 4 * 240;
  appConfig.windowHeight = 3 * 240;
  appConfig.renderWidth = appConfig.windowWidth;
  appConfig.renderHeight = appConfig.windowHeight;
#endif

  orApp app(appConfig);
  app.Run();

  return 0;
}
