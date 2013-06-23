/*
 * File:   orGfx.h
 * Author: fib
 *
 * Created on 26 December 2012, 12:22
 */

#ifndef ORGFX_H
#define	ORGFX_H

// TODO move to orPlatform

# include "orStd.h"
# include "orMath.h"

#ifdef WINGDIAPI
#error
#endif

# ifdef _MSC_VER
#   define WINGDIAPI __declspec(dllimport)
#   define APIENTRY __stdcall
#   define CALLBACK __stdcall
# endif

# define GLEW_STATIC
# include <GL/glew.h>
# undef GLEW_STATIC
# include <GL/gl.h>
# include <GL/glu.h>

# ifdef _MSC_VER
#   undef WINGDIAPI
#   undef APIENTRY
#   undef CALLBACK
# endif


# include <Eigen/OpenGLSupport>

#endif	/* ORGFX_H */

