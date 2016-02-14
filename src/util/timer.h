/*
 * File:   timer.h
 * Author: cswetenham
 *
 * Created on 13 February 2016, 17:57
 */

#ifndef ORBITALSPACE_TIME_H
#define	ORBITALSPACE_TIME_H

namespace orbital {
namespace timer {

typedef uint64_t perftime_t;

void Init();

perftime_t Now();    // Current time in native units
perftime_t Uptime(); // Uptime (current time - start time) in native units

float PerfTimeToSeconds( perftime_t );

inline float UptimeSeconds() { return PerfTimeToSeconds(Uptime()); }

}  // namespace timer
}  // namespace orbital

#endif	/* ORBITALSPACE_TIME_H */

