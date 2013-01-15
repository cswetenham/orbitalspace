/*
Logging and verification utility macros and functions
*/

#ifndef UTIL_H
#define UTIL_H

#include "orStd.h"
#include "orMath.h"

/******************************** Defines *********************************/

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

#ifdef _WIN32
  #define CDECL __cdecl
#else
  #define CDECL __attribute__((__cdecl__))
#endif
  
/******************************** Structures *********************************/

#define M_TAU      6.28318530717958647693
#define M_TAU_F    6.28318530717958647693f

// TODO namespace
class Util
{
public:
  // ShouldExit() returns true if a SIGTERM signal has been received.
  static bool ShouldExit();

  // Installs some signal handlers
  static void InstallHandlers();

  // Dumps the stack to standard error
  static void DumpStack();

  // Floating-point modulus - handles negative values correctly
  // From http://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
  template <typename T>
  static T FMod(T const _a, T const _b)
  {
    if (_b == 0) { return _a; } // Welp

    return _a - _b * floor(_a / _b);
  }
  
  template <typename T>
  static T Clamp(T const _v, T const _min, T const _max)
  {
    return (_v < _min) ? _min :
           (_max < _v) ? _max :
                         _v;
  }

  /* Wrap a value into the given range, for example -.5f*M_TAU to +.5f*M_TAU */
  template <typename T>
  static T Wrap(T const _x, T const _min, T const _max) {
      return FMod(_x - _min, _max - _min) + _min;
  }

  template <typename T>
  static T Max(T const _a, T const _b) {
    return (_a > _b) ? _a : _b;
  }
  
  template <typename T>
  static T Min(T const _a, T const _b) {
      return (_a < _b) ? _a : _b;
  }

  template <typename T, typename A>
  static T Lerp(T const _x0, T const _x1, A const _a) {
      return _x0 * (1 - _a) + _x1 * _a;
  }

  // SmootherStep interpolation function from http://en.wikipedia.org/wiki/Smoothstep
  template <typename T>
  static T SmootherStep(T const _x, T const _edge0, T const _edge1) {
      // Scale, and clamp x to 0..1 range
      T const x = Util::Clamp((_x - _edge0) / (_edge1 - _edge0), 0, 1);
      // Evaluate polynomial
      return x * x * x * (x * (x * 6 - 15) + 10);
  }

  // Pad a size to the nearest multiple of some value
  static size_t PadSize(size_t const _n, size_t const _pad)
  {
    return _pad * ((_n + _pad - 1) / _pad);
  }

  static void SleepMicros(uint32_t const _usecs);

  static void SetDrawColour(Vector3f const& _c);
  static void SetDrawColour(Vector3d const& _c);

private:
  static void SigAbrt(int);
  static void SigQuit(int);
  static void SigTerm(int);
  static void SigSegv(int);
  static void SigStop(int);
  static void SigInt(int);
};


/******************************** Macros *********************************/
 
// kLog()
#define kLog( _FMT, ... ) do { fprintf(stdout, "[%08d] " _FMT, (int)Timer::UptimeMillis(), ## __VA_ARGS__); } while (0)

// kErr()
#define kErr( _FMT, ... ) do { fprintf(stderr, "[%08d] " _FMT, (int)Timer::UptimeMillis(), ## __VA_ARGS__); } while (0)

#define allocat( _T, _S ) (_T*)alloca(_S * sizeof(_T))

// TODO portable implementations

NORETURN inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line) {
  if (!_cond) {
    printf("%s(%d): Assertion failed: %s\n", _file, _line, _condStr);
    DEBUGBREAK;
    FATAL;
  }
}

NORETURN inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line, char const* _msg, ...) {
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
# define ensure(_cond, ...) ensure_impl(_cond, #_cond, __FILE__, __LINE__, __VA_ARGS__)
#else
# define ensure(...)
#endif

#endif // UTIL_H

