#ifndef _RND_H
#define _RND_H

#include "util.h"

#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <malloc.h>

/* Based on code from Numerical Recipes 3rd Edition */
class Rnd64
{
public:
  struct State
  {
    uint64_t u;
    uint64_t v;
    uint64_t w;
  };
    
public:
  Rnd64
  (
    uint64_t const _seed
  ) :
    m_state(StateFromSeed(_seed))
  { }
    
  static State StateFromSeed(uint64_t const _seed)
  {
    State state;
    state.u = 0;
    state.v = 4101842887655102017LL;
    state.w = 1;
    
    state.u = _seed ^ state.v;
    state = Next(state);
    state.v = state.u;
    state = Next(state);
    state.w = state.v;
    state = Next(state);
    
    return state;
  }

  inline static State Next(State const& _state)
  {
    uint64_t const u0 = _state.u;
    uint64_t const v0 = _state.v;
    uint64_t const w0 = _state.w;

    // Mixing states
    uint64_t const u1 = u0 * 2862933555777941757LL + 7046029254386353087LL;
    
    uint64_t const v1 = v0 ^ (v0 >> 17);
    uint64_t const v2 = v1 ^ (v1 << 31);
    uint64_t const v3 = v2 ^ (v2 >> 8);
    
    uint64_t const w1 = 4294957665U * (w0 & 0xffffffff) + (w0 >> 32);
    
    State res;
    res.u = u1;
    res.v = v3;
    res.w = w1;
    
    return res;
  }
  
  inline static uint64_t UInt64FromState(State const& _state)
  {
    // Generate sample based on state
    uint64_t const u = _state.u;
    uint64_t const v = _state.v;
    uint64_t const w = _state.w;

    uint64_t const x0 = u ^ (u << 21);
    uint64_t const x1 = x0 ^ (x0 >> 35);
    uint64_t const x2 = x1 ^ (x1 << 4);
    
    uint64_t const r = (x2 + v) ^ w;
    
    return r;
  }
  
  inline static double DoubleFromUInt64(uint64_t const _r) // 0.0 to 1.0
  {
    return 5.42101086242752217E-20 * _r;
  }

  inline static float FloatFromUInt32(uint32_t const _r)
  {
    return 2.32830643653869629E-10f * _r;
  }
 
  void gen_doubles( int const _n, double* const __restrict _out )
  {
    for (int i = 0; i < _n; ++i)
    {
      m_state = Next(m_state);

      // Generate sample based on state
      uint64_t const r = UInt64FromState(m_state);

      _out[i] = DoubleFromUInt64(r);
    }
  }
  
  void gen_floats( int const _n, float* const __restrict _out )
  {
    assert((_n & 1) == 0);
    for (int i = 0; i < _n; i += 2)
    {
      m_state = Next(m_state);

      // Generate sample based on state
      uint64_t const r = UInt64FromState(m_state);

      _out[i] = FloatFromUInt32((uint32_t)(r));
      _out[i+1] = FloatFromUInt32((uint32_t)(r >> 32));
    }
  }
  
private:
  State m_state;
};

class BernoulliDistribution
{
public:
  struct Config
  {
    Config(float const _p):p(_p) {}
    float p;
  };
  
  BernoulliDistribution() : m_config(.5f) {}
  BernoulliDistribution(float const _p) : m_config(_p) {}
  
  void Generate(Rnd64* _r, int const _n, bool* const _out) const
  {
    // Make sure corner cases do the right thing
    if (m_config.p == 0.f)
    {
      for (int i = 0; i < _n; ++i)
      {
        _out[i] = false;
      }
    }
    else if (m_config.p == 1.f)
    {
      for (int i = 0; i < _n; ++i)
      {
        _out[i] = true;
      }
    }
    else
    {
      int const paddedCount = Util::PadSize(_n, 2);
      float* const temp = (float*)alloca(paddedCount * sizeof(float));
      
      _r->gen_floats(paddedCount, temp);

      for (int i = 0; i < _n; ++i)
      {
        _out[i] = temp[i] < m_config.p;
      }
    }
  }
  
  // TODO Evaluate many?
  float Evaluate(bool const _in) const
  {
    return _in ? m_config.p : 1.f - m_config.p;
  }

private:
  Config m_config;
};

class UniformDistribution
{
public:
  struct Config
  {
    Config(float const _min, float const _max) : min(_min), max(_max) {}
    float min;
    float max;
  };
  
  UniformDistribution() : m_config(0.f, 1.f) {}
  UniformDistribution(float const _min, float const _max) : m_config(_min, _max) {}
  
  void Generate(Rnd64* _r, int const _n, float* const _out) const
  {
    int const paddedCount = Util::PadSize(_n, 2);
    float* const temp = (float*)alloca(paddedCount * sizeof(float));
    
    _r->gen_floats(paddedCount, temp);
    
    for (int i = 0; i < _n; ++i)
    {
      _out[i] = temp[i] * (m_config.max - m_config.min) + m_config.min;
    }
  }
  
  // TODO Evaluate many?
  float Evaluate(float const _in) const
  {
    if (m_config.min <= _in && _in <= m_config.max)
    {
      return 1 / (m_config.max - m_config.min);
    }
    else
    {
      return 0;
    }
  }

private:
  Config m_config;
};

class NormalDistribution
{
public:
  struct Config
  {
    Config(float const _mean, float const _std):mean(_mean), std(_std) {}
    float mean;
    float std;
  };
  
  NormalDistribution() : m_config(0.f, 1.f) {}
  NormalDistribution(float const _mean, float const _std) : m_config(_mean, _std) {}
  
  void Generate(Rnd64* _r, int const _n, float* const __restrict _out) const
  {
    for (int i = 0; i < _n; ++i)
    {
      float u;
      float v;
      bool reject;
      // TODO - generate larger number of deviates, reject afterward?
      do
      {
        float data[2];
        _r->gen_floats(2, &data[0]);
        u = data[0];
        float const v0 = data[1];
        v = 1.7156f * (v0 - 0.5f);
        float const x = u - 0.449871f;
        float const y = fabsf(v) + 0.386595f;
        float const q = (x * x) + y * (0.19600f * y - 0.25472f * x);
        reject = q > 0.27597f && (q > 0.27846f || (v * v) > -4.f * logf(u) * u * u);
      } while (reject);

      _out[i] = m_config.mean + m_config.std * (v/u);
    }
  }
  
  // TODO Evaluate many?
  float Evaluate(float const _in) const
  {
    if (m_config.std == 0.f)
    {
      return _in == m_config.mean ? 1.f : 0.f;
    }

    float const u = (_in - m_config.mean) / m_config.std;
    float const k = 1.f / (sqrtf(M_TAU_F) * m_config.std);
    return k * expf(-.5f * u * u);
  }
  
private:
  Config m_config;
};

#endif // _RND_H