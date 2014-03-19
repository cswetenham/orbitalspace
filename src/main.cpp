#include "orStd.h"

#include "orApp.h"

int CDECL main()
{
  Timer::StaticInit();

  orApp::Config appConfig;

  // 2:1 pixel aspect ratio

  appConfig.windowWidth = 2*640;
  appConfig.windowHeight = 2*400;

  appConfig.renderWidth = 2*640;
  appConfig.renderHeight = 2*200;

  orApp app(appConfig);
  app.Run();

  return 0;
}
