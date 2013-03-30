/* 
 * File:   orStd.h
 * Author: fib
 *
 * Created on 26 December 2012, 12:22
 */

#ifndef ORSTD_H
#define	ORSTD_H

// TODO argh
#ifdef _WIN32
# include <Windows.h>
#endif

# include <stdint.h>
# include <limits.h>

# include <assert.h>
# include <malloc.h>

# include <cstdio>

# include <stdlib.h>
# include <stdarg.h>

#ifdef _WIN32
# define ENTRY_FN __cdecl
#else
# define ENTRY_FN
#endif

#ifdef _WIN32
# define NORETURN __declspec(noreturn)
#else
# define NORETURN /* TODO: GCC, etc */
#endif

#ifdef _WIN32
  extern "C" __declspec(dllimport) void __stdcall DebugBreak(void);
# define DEBUGBREAK do { DebugBreak(); } while (0)
#else
# define DEBUGBREAK /* TODO: GCC, etc */
#endif

#ifdef _WIN32
  extern "C" __declspec(dllimport) void __stdcall FatalExit(int);
# define FATAL do { FatalExit(3); } while (0)
#else
# define FATAL /* TODO: GCC, etc */
#endif

#ifdef CDECL
#  undef CDECL
#endif

#ifdef _WIN32
  #define CDECL __cdecl
#else
  #define CDECL __attribute__((__cdecl__))
#endif

#endif	/* ORSTD_H */

