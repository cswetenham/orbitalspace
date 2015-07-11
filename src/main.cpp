#include "orStd.h"

#include "orApp.h"

#include "Fixed64.h"

#include <memory>

int main()
{
  Timer::StaticInit();

#if 0
  run_tests();
#endif

  orApp::Config appConfig;

  // 2:1 pixel aspect ratio

  appConfig.windowWidth = 2*640;
  appConfig.windowHeight = 2*400;

  appConfig.renderWidth = 2*640;
  appConfig.renderHeight = 2*400;
  // appConfig.renderWidth = 160;
  // appConfig.renderHeight = 144;

  auto app = std::make_unique<orApp>(appConfig);
  app->Run();

  return 0;
}
