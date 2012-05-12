
#include "orbitalSpaceApp.h"

#include <SFML/Window.hpp>

int __cdecl main()
{
    OrbitalSpaceApp app;
    app.Init();
    app.Run();
    app.Shutdown();

    return 0;
}
