#include "orStd.h"

#include "orApp.h"

int CDECL main()
{
  Timer::StaticInit();

  orApp::Config appConfig;

  // 2:1 pixel aspect ratio

  appConfig.windowWidth = 2*640;
  appConfig.windowHeight = 2*400;

  appConfig.renderWidth = 640;
  appConfig.renderHeight = 200;

  orApp app(appConfig);
  app.Run();

  return 0;
}
