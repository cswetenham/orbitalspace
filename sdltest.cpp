
#include "orbitalSpaceApp.h"

#ifdef _WIN32
# include <Windows.h>
#endif

// wtf sdl
#undef main
int ENTRY_FN main(int, char const**)
{
#ifdef _WIN32
  ::SetThreadAffinityMask(GetCurrentThread(), 1);
#endif

  OrbitalSpaceApp app;

  app.Init();
  app.Run();
  app.Shutdown();

  return 0;
}
