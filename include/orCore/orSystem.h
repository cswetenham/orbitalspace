#ifndef ORSYSTEM_H
#define	ORSYSTEM_H

#include "SDL_assert.h"

// TODO make() invalidates existing references but I'm holding on to them!!
#define DECLARE_SYSTEM_TYPE(T_SINGULAR, T_PLURAL)\
public:\
  int num ## T_PLURAL() const { return (int)m_instanced ## T_PLURAL.size(); }\
  int make ## T_SINGULAR() { m_instanced ## T_PLURAL.push_back(T_SINGULAR()); return num ## T_PLURAL(); }\
  T_SINGULAR&       get ## T_SINGULAR(int i)       { SDL_assert(0 < i && i <= num ## T_PLURAL()); return m_instanced ## T_PLURAL[i - 1]; }\
  T_SINGULAR const& get ## T_SINGULAR(int i) const { SDL_assert(0 < i && i <= num ## T_PLURAL()); return m_instanced ## T_PLURAL[i - 1]; }\
private:\
  std::vector<T_SINGULAR> m_instanced ## T_PLURAL;\
public:

#endif	/* ORSYSTEM_H */

