#include "orStd.h"

#include "orProfile/perftimer.h"

std::stack<PerfTimer::Entry*> PerfTimer::s_stack;
PerfTimer::Entry PerfTimer::s_parent;

void PerfTimer::Print(Entry const* const _entry, std::string const& _indent)
{
  std::string const childIndent = _indent + std::string("  ");
  for (Iter i = _entry->children.begin(); i != _entry->children.end(); ++i)
  {
    std::string const& childName = i->first;
    Entry const* const childEntry = &i->second;
      
    orLog("%s%s: %3.3f ms\n", _indent.c_str(), childName.c_str(), Timer::PerfTimeToMillis(childEntry->time));
    Print(childEntry, childIndent);
  }
}

