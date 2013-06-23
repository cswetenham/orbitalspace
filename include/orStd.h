/*
 * File:   orStd.h
 * Author: fib
 *
 * Created on 26 December 2012, 12:22
 */

#ifndef ORSTD_H
#define	ORSTD_H

# include <stdint.h>
# include <limits.h>
# include <float.h>

# include <assert.h>
# include <malloc.h>

# include <cstdio>

# include <stdlib.h>
# include <stdarg.h>

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

#ifdef _MSC_VER
  extern "C" __declspec(dllimport) void __stdcall DebugBreak(void);
# define DEBUGBREAK do { DebugBreak(); } while (0)
#else
  extern "C" int raise (int __sig) __THROW;
  // #define	SIGTRAP		5	/* Trace trap (POSIX). */
# define DEBUGBREAK do { raise(5); } while (0)
#endif

#ifdef _MSC_VER
  extern "C" __declspec(dllimport) void __stdcall FatalExit(int);
# define FATAL do { FatalExit(3); } while (0)
#else
# define FATAL do { __builtin_trap(); } while (0)
#endif

#ifdef CDECL
#  undef CDECL
#endif

#ifdef _MSC_VER
  #define CDECL __cdecl
#else
  #define CDECL __attribute__((__cdecl__))
#endif

// orLog()
#define orLog( _FMT, ... ) do { fprintf(stdout, "[%08d] " _FMT, (int)Timer::UptimeMillis(), ## __VA_ARGS__); } while (0)

// orErr()
#define orErr( _FMT, ... ) do { fprintf(stderr, "[%08d] " _FMT, (int)Timer::UptimeMillis(), ## __VA_ARGS__); } while (0)

#define allocat( _T, _S ) (_T*)alloca(_S * sizeof(_T))

template <typename T> inline void ignore(T const&) {}; // To explicitly ignore return values without warning

// TODO portable implementations

inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line) {
  if (!_cond) {
    printf("%s(%d): Assertion failed: %s\n", _file, _line, _condStr);
    DEBUGBREAK;
    FATAL;
  }
}

inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line, char const* _msg, ...) {
  if (!_cond) {
    printf("%s(%d): Assertion failed: %s\n", _file, _line, _condStr);
    printf("%s(%d): ", _file, _line);
    va_list vargs;
    va_start(vargs, _msg);
    vprintf(_msg, vargs);
    va_end(vargs);
    DEBUGBREAK;
    FATAL;
  }
}

// TODO set up CONFIG_DEBUG, CONFIG_PROFILE
#ifdef _DEBUG
# define ensure(_cond, ...) ensure_impl(_cond, #_cond, __FILE__, __LINE__, ##__VA_ARGS__)
#else
# define ensure(...)
#endif

#endif	/* ORSTD_H */

