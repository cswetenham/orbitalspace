#ifndef _TIMER_H
#define _TIMER_H

# include <stdint.h>
# include <math.h>

class Timer
{
public:
  typedef uint64_t PerfTime; // Platform-native time unit. Can be added and subtracted.
  
  static PerfTime GetPerfTime();
  static float PerfTimeToMillis( PerfTime const _t );
    
  static uint64_t UptimePerfTime(); // Uptime in native units
  // UptimeMillis() gives the time in milliseconds since the start of the program run.
  static float UptimeMillis() { return PerfTimeToMillis(UptimePerfTime()); }
};

#endif // _TIMER_H
