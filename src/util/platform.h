/*
 * File:   platform.h
 * Author: cswetenham
 *
 * Created on 13 February 2016, 17:24
 */

#ifndef ORBITALSPACE_PLATFORM_H
#define	ORBITALSPACE_PLATFORM_H

#include <stdint.h>

#pragma once

namespace orbital {

#ifdef _MSC_VER
# define ENTRY_FN __cdecl
#else
# define ENTRY_FN
#endif

#ifdef _MSC_VER
# define NORETURN __declspec(noreturn)
#else
# define NORETURN /* TODO: GCC, etc */
#endif

#ifdef CDECL
#  undef CDECL
#endif

#ifdef _MSC_VER
  #define CDECL __cdecl
#else
  #define CDECL __attribute__((__cdecl__))
#endif

#ifdef _MSC_VER
  #define snprintf _snprintf
#endif

#ifdef _MSC_VER
  extern "C" __declspec(dllimport) void __stdcall ::DebugBreak(void);
  extern "C" __declspec(dllimport) void __stdcall ::FatalExit(int);
#else  // !defined(_MSC_VER)
  extern "C" int raise(int sig) throw ();
#endif

inline void DebugBreak() {
#ifdef _MSC_VER
  ::DebugBreak();
#else
  // #define	SIGTRAP		5	/* Trace trap (POSIX). */
  raise(5);
#endif
}

inline void FatalExit() {
#ifdef _MSC_VER
  ::FatalExit(3);
#else
  __builtin_trap();
#endif
}

}  // namespace orbital

#endif	/* ORBITALSPACE_PLATFORM_H */

