#include "orStd.h"

#include "orbitalSpaceApp.h"

#include <SFML/Window.hpp>

int CDECL main()
{
    OrbitalSpaceApp app;
    app.Init();
    app.Run();
    app.Shutdown();

    return 0;
}
