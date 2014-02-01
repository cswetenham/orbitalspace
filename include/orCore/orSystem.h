#ifndef ORSYSTEM_H
#define	ORSYSTEM_H

#define DECLARE_SYSTEM_TYPE(T_SINGULAR, T_PLURAL)\
public:\
  int num ## T_PLURAL() const { return (int)m_instanced ## T_PLURAL.size(); }\
  int make ## T_SINGULAR() { m_instanced ## T_PLURAL.push_back(T_SINGULAR()); return num ## T_PLURAL() - 1; }\
  T_SINGULAR&       get ## T_SINGULAR(int i)       { return m_instanced ## T_PLURAL[i]; }\
  T_SINGULAR const& get ## T_SINGULAR(int i) const { return m_instanced ## T_PLURAL[i]; }\
private:\
  std::vector<T_SINGULAR> m_instanced ## T_PLURAL;\
public:

#endif	/* ORSYSTEM_H */

