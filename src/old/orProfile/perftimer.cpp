#include "orStd.h"

#include "orProfile/perftimer.h"

std::stack<PerfTimer::Entry*> PerfTimer::s_stack;
PerfTimer::Entry PerfTimer::s_parent;
Timer::PerfTime PerfTimer::s_startTime;

void PerfTimer::Print(Entry const* const _entry, std::string const& _indent)
{
  float const parentTime = Timer::PerfTimeToMillis(_entry->time);
  std::string const childIndent = _indent + std::string("  ");
  for (Iter i = _entry->children.begin(); i != _entry->children.end(); ++i)
  {
    std::string const& childName = i->first;
    Entry const* const childEntry = &i->second;
    float const childTime = Timer::PerfTimeToMillis(childEntry->time);
    float const childPc = 100.f * childTime / parentTime;
    orLog("%s%s: %3.3f ms (%f%%)\n", _indent.c_str(), childName.c_str(), childTime, childPc);
    Print(childEntry, childIndent);
  }
}

