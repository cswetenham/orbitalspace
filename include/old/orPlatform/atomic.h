#pragma once

#include "orStd.h"

#ifdef _MSC_VER
# include "win32/atomic_win32.h"
# else
# include "linux/atomic_linux.h"
#endif

