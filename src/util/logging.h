/*
 * File:   logging.h
 * Author: cswetenham
 *
 * Created on 13 February 2016, 17:10
 */

#pragma once

#ifndef ORBITALSPACE_LOGGING_H
#define	ORBITALSPACE_LOGGING_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util/platform.h"
#include "util/timer.h"

namespace orbital {

namespace logging {

enum Level {
  INFO = 1,
  WARN,
  ERROR,
  FATAL
};

#ifdef _MSC_VER
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA( char const* lpOutputString );
extern "C" __declspec(dllimport) int _vsnprintf_s( char *buffer, size_t count, const char *format, va_list argptr );
#endif

inline void log(Level level, char const* fmt, ...) {
  va_list args;
  va_start(args, fmt);

#ifdef _MSC_VER
  enum { BufSize = 1024 };
  char buffer[BufSize];
  (void)_vsnprintf_s(buffer, BufSize-1, fmt, args);
  ::OutputDebugStringA(buffer);
#else  // !_MSC_VER
  struct _IO_FILE* out = stdout;
  if (level >= ERROR) {
    out = stderr;
  }
  (void)vfprintf(out, fmt, args);
  (void)fprintf(out, "\n");
  (void)fflush(out);
#endif  // !_MSC_VER

  va_end(args);

  if (level == FATAL) {
    ::orbital::FatalExit();
  }
}

#define LOGINFO( _FMT, ... ) do { ::orbital::logging::log(::orbital::logging::INFO, "[I %05.3f] " _FMT, ::orbital::timer::UptimeSeconds(), ## __VA_ARGS__); } while (0)
#define LOGWARN( _FMT, ... ) do { ::orbital::logging::log(::orbital::logging::WARN, "[W %05.3f] " _FMT, ::orbital::timer::UptimeSeconds(), ## __VA_ARGS__); } while (0)
#define LOGERR( _FMT, ... ) do { ::orbital::logging::log(::orbital::logging::ERROR, "[E %05.3f] " _FMT, ::orbital::timer::UptimeSeconds(), ## __VA_ARGS__); } while (0)
#define LOGFATAL( _FMT, ... ) do { ::orbital::logging::log(::orbital::logging::FATAL, "[F %05.3f] " _FMT, ::orbital::timer::UptimeSeconds(), ## __VA_ARGS__); FatalExit(); } while (0)

#define LOGINFO_IF( _COND, _FMT, ... ) if ( _COND ) { LOGINFO( _FMT, ## __VA_ARGS__); }
#define LOGWARN_IF( _COND, _FMT, ... ) if ( _COND ) { LOGWARN( _FMT, ## __VA_ARGS__); }
#define LOGERR_IF( _COND, _FMT, ... ) if ( _COND ) { LOGERR( _FMT, ## __VA_ARGS__); }
#define LOGFATAL_IF( _COND, _FMT, ... ) if ( _COND ) { LOGFATAL( _FMT, ## __VA_ARGS__); }

// TODO LOGINFO_IF, LOGWARN_IF, LOGERR_IF, LOGFATAL_IF

}  // namespace logging

}  // namespace orbital

#endif	/* ORBITALSPACE_LOGGING_H */

