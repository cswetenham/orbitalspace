#include <stdint.h>

#ifdef _WIN32
// for QueryPerformanceCounter()
# include <windows.h>
#else
// for clock_gettime()
# include <time.h>
#endif

#include "util/timer.h"

namespace orbital {
namespace timer {

namespace {
  perftime_t x_initTime = 0;
  perftime_t x_perfFreq = 0; // Bogus value in case we don't call Init();
}  // namespace

// store time at start of program run
void Init() {
  x_initTime = Now();
#ifdef _WIN32
  LARGE_INTEGER wf;
  ::QueryPerformanceFrequency(&wf);
  x_perfFreq = (perftime_t)wf.QuadPart;
#else  // !def _WIN32
  x_perfFreq = 1000000000;
#endif // !def _WIN32
}

perftime_t Uptime() {
  return Now() - x_initTime;
}

perftime_t Now()
{
#ifdef _WIN32
  LARGE_INTEGER wt;
  ::QueryPerformanceCounter(&wt);
  return (PerfTime)wt.QuadPart;
#else  // !def _WIN32
  timespec ts_current;
  int const ret = clock_gettime(CLOCK_MONOTONIC, &ts_current);
  (void)ret;
  uint64_t const ds = ts_current.tv_sec;
  uint64_t const dns = ts_current.tv_nsec;

  return dns + (ds * 1000000000);
#endif  // !def _WIN32
}

float PerfTimeToSeconds( perftime_t t )
{
  return (float)t / (float)x_perfFreq;
}

}  // namespace timer
}  // namespace orbital