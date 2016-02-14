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

# include "timer.h"

inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line) {
  if (!_cond) {
    orErr("%s(%d): Assertion failed: %s\n", _file, _line, _condStr);
    DEBUGBREAK;
    FATAL;
  }
}

inline void ensure_impl(bool _cond, char const* _condStr, char const* _file, int _line, char const* _msg, ...) {
  if (!_cond) {
    orErr("%s(%d): Assertion failed: %s\n", _file, _line, _condStr);
    orErr("%s(%d): ", _file, _line);
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

#define allocat( _T, _S ) (_T*)alloca(_S * sizeof(_T))

template <typename T> inline void ignore(T const&) {}; // To explicitly ignore return values without warning


#endif	/* ORSTD_H */

