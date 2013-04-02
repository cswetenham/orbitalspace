#pragma once

#include <stdint.h>

// Forward refs instead of including Windows.h
#if 1
  long __cdecl _InterlockedIncrement( long volatile* lpAddend );
  long __cdecl _InterlockedDecrement( long volatile* lpAddend );
  long __cdecl _InterlockedCompareExchange( long volatile* Destination, long Exchange, long Comperand );
  __int64 __cdecl _InterlockedCompareExchange64( __int64 volatile* Destination, __int64 Exchange, __int64 Comperand );
#pragma warning( push )
#pragma warning( disable : 4793 )
  __forceinline void MemoryBarrier ( void ) { long Barrier; __asm { xchg Barrier, eax } };
#pragma warning( pop )
#endif

namespace orPlatform {

// Atomically increments the value at p, and returns the *previous* value.
inline int atomicInc( int32_t* const p ) {
  return ::_InterlockedIncrement((long volatile*)p) - 1;
}

// Atomically decrements the value at p, and returns the *previous* value.
inline int atomicDec( int32_t* const p ) {
  return ::_InterlockedDecrement((long volatile*)p) + 1;
}

// Attempts to atomically swap the value at p from oldVal to newVal.
// Returns the value seen at p before the swap. If this is oldVal, the swap succeeded, otherwise it failed.
inline uint32_t atomicCompareAndSwap( uint32_t* const p, uint32_t const oldVal, uint32_t const newVal ) {
  return (uint32_t)::_InterlockedCompareExchange((long volatile*)p, (int32_t)oldVal, (int32_t)newVal);
}
inline int32_t atomicCompareAndSwap( int32_t* const p, int32_t const oldVal, int32_t const newVal ) {
  return (int32_t)::_InterlockedCompareExchange((long volatile*)p, (int32_t)oldVal, (int32_t)newVal);
}
inline uint64_t atomicCompareAndSwap( uint64_t* const p, uint64_t const oldVal, uint64_t const newVal ) {
  return (uint64_t)::_InterlockedCompareExchange64((__int64 volatile*)p, (int64_t)oldVal, (int64_t)newVal);
}
inline int64_t atomicCompareAndSwap( int64_t* const p, int64_t const oldVal, int64_t const newVal ) {
  return (int64_t)::_InterlockedCompareExchange64((__int64 volatile*)p, (int64_t)oldVal, (int64_t)newVal);
}
template <typename T>
inline T* atomicCompareAndSwap( T** const p, T* const oldVal, T* const newVal ) {
  return (T*)atomicCompareAndSwap((uintptr_t*)p, (uintptr_t)oldVal, (uintptr_t)newVal);
}

inline void readBarrier() {
  ::MemoryBarrier();
}

inline void writeBarrier() {
  ::MemoryBarrier();
}

inline void fullBarrier() {
  ::MemoryBarrier();
}

} // namespace orPlatform