#include "orStd.h"

#include "timer.h"

#include <assert.h>
#ifdef _WIN32
// for QueryPerformanceCounter()
# include <windows.h>
#else
// for gettimeofday()
# include <sys/types.h>
# include <sys/time.h>
#endif

// store time at start of program run
static Timer::PerfTime x_baseTime;

static struct MagicInit
{
  MagicInit() { x_baseTime = Timer::GetPerfTime(); }
} x_magicInit;

#ifdef _WIN32
Timer::PerfTime Timer::GetPerfTime()
{
  LARGE_INTEGER wt;
  ::QueryPerformanceCounter(&wt);
  return (PerfTime)wt.QuadPart;
}

float Timer::PerfTimeToMillis( PerfTime const _t )
{
  LARGE_INTEGER wf;
  ::QueryPerformanceFrequency(&wf);
  return 1000.f * ((float)_t / wf.QuadPart);
}

#else // !def _WIN32

Timer::PerfTime Timer::GetPerfTime()
{
  timeval curTime;
  int const ret = gettimeofday(&curTime, 0);
  assert(ret == 0);

  uint64_t const ds = curTime.tv_sec;
  uint64_t const dus = curTime.tv_usec;
  
  return dus + (ds * 1000000);
}

float Timer::PerfTimeToMillis( PerfTime const _t )
{
  return _t / 1000.0f;
}

Timer::PerfTime Timer::UptimePerfTime()
{
  return GetPerfTime() - x_baseTime;
}
#endif // !def _WIN32