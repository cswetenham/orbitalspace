#include "orStd.h"

#include "orProfile/perftimer.h"

std::stack<PerfTimer::Entry*> PerfTimer::s_stack;
PerfTimer::Entry PerfTimer::s_parent;

