#include "orStd.h"

#include "orApp.h"

#include <SFML/Window.hpp>

int CDECL main()
{
  // TODO we start a seemingly huge number of threads on startup, and many of them die after a certain amount
  // of time. The report that they have died comes when we call m_window->pollEvent(event).

  // Would like to figure out where these come from.

  // Idea: set ourselves up to poll events before every system has been brought up, bring them up one by one
  // and see what happens with threads starting and ending.
  orApp app;
  app.Run();

  return 0;
}
