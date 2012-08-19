/*
Logging and verification utility macros and functions
*/

#include "util.h"

// for assert()
#include <assert.h>

// for signal()
#include <signal.h>

#ifdef _WIN32
// for Sleep()
# include <Windows.h>
#else
// for backtrace()
# include <execinfo.h>
// for usleep()
# include <unistd.h>
#endif

// For glColor3f()
#include <Eigen/OpenGLSupport>

static bool x_shouldExit;

static struct MagicInit
{
  MagicInit() { x_shouldExit = false; }
} x_magicInit;

void Util::DumpStack()
{
#ifndef _WIN32
  static void *backbuf[ 50 ];
  int levels;

  levels = backtrace( backbuf, 50 );
  backtrace_symbols_fd( backbuf, levels, 2 );
#endif
}

void Util::InstallHandlers()
{
#ifndef _WIN32
  // signal(SIGABRT, SigAbrt);
  // signal(SIGQUIT, SigQuit);
  // signal(SIGTERM, SigTerm);
  signal(SIGSEGV, SigSegv);
  // signal(SIGSTOP, SigStop);
  // signal(SIGINT,  SigInt);
  printf("Signal handlers installed.\n");
#endif
}

bool Util::ShouldExit() { return x_shouldExit; }

void Util::SleepMicros(uint32_t const _usecs)
{
#ifdef _WIN32
  Sleep(_usecs / 1000);
#else
  usleep(_usecs);
#endif
}

void Util::SetDrawColour(Vector3f const& _c)
{
  glColor3f(_c.x(), _c.y(), _c.z());
}

void Util::SetDrawColour(Vector3d const& _c)
{
  glColor3d(_c.x(), _c.y(), _c.z());
}

void Util::SigAbrt(int)
{
  printf("Received SIGABRT!\n");
  fflush(stdout);
  fflush(stderr);
  x_shouldExit = true;
}

void Util::SigQuit(int)
{
  printf("Received SIGQUIT!\n");
  fflush(stdout);
  fflush(stdin);
  x_shouldExit = true;
}

void Util::SigTerm(int)
{
  printf("Received SIGTERM\n");
  fflush(stdout);
  fflush(stdin);
  x_shouldExit = true;
}

void Util::SigSegv(int)
{
  printf("Received SIGSEGV\n");
  DumpStack();
  fflush(stdout);
  fflush(stdin);
  x_shouldExit = true;
  signal(SIGSEGV, NULL);
}

void Util::SigStop(int)
{
  printf("Received SIGSTOP\n");
  fflush(stdout);
  fflush(stdin);
  x_shouldExit = true;
}

void Util::SigInt(int)
{
  printf("Received SIGINT\n");
  fflush(stdout);
  fflush(stdin);
  x_shouldExit = true;
}
