#ifndef _TIMER_H
#define _TIMER_H

#include <stdint.h>

class Timer
{
public:
  typedef uint64_t PerfTime; // Platform-native time unit. Can be added and subtracted.

  static void StaticInit();

  static PerfTime GetPerfTime();
  static float PerfTimeToMillis( PerfTime const _t );

  static uint64_t UptimePerfTime(); // Uptime in native units
  // UptimeMillis() gives the time in milliseconds since the start of the program run.
  static float UptimeMillis() { return PerfTimeToMillis(UptimePerfTime()); }

private:
  static Timer::PerfTime x_baseTime;
};

#endif // _TIMER_H
