#ifndef ATOMIC_H
#define ATOMIC_H

// TODO have platform-specific stuff in a platform/ folder?
// TODO include forward refs instead of including Windows.h?
#ifdef _MSC_VER
# include <Windows.h>
#endif

namespace orCore {

// Atomically increments the value at p, and returns the *previous* value.
inline int atomicInc( int* const p ) {
# ifdef _MSC_VER
    return InterlockedIncrement((unsigned*)p) - 1;
# else
    return __sync_fetch_and_add(p, 1);
# endif
}

// Atomically decrements the value at p, and returns the *previous* value.
inline int atomicDec( int* const p ) {
# ifdef _MSC_VER
    return InterlockedDecrement((unsigned*)p) + 1;
# else
    return __sync_fetch_and_sub(p, 1);
# endif
}

// Attempts to atomically swap the value at p from oldVal to newVal.
// Returns the value seen at p before the swap. If this is oldVal, the swap succeeded, otherwise it failed.
# ifdef _MSC_VER
inline uint32_t atomicCompareAndSwap( uint32_t* const p, uint32_t const oldVal, uint32_t const newVal ) {
  return InterlockedCompareExchange(p, oldVal, newVal);
}
inline int32_t atomicCompareAndSwap( int32_t* const p, int32_t const oldVal, int32_t const newVal ) {
  return (int32_t)InterlockedCompareExchange((uint32_t*)p, (uint32_t)oldVal, (uint32_t)newVal);
}
inline uint64_t atomicCompareAndSwap( uint64_t* const p, uint64_t const oldVal, uint64_t const newVal ) {
  return InterlockedCompareExchange(p, oldVal, newVal);
}
inline int64_t atomicCompareAndSwap( int64_t* const p, int64_t const oldVal, int64_t const newVal ) {
  return (int64_t)InterlockedCompareExchange((uint64_t*)p, (uint64_t)oldVal, (uint64_t)newVal);
}
template <typename T>
inline T* atomicCompareAndSwap( T** const p, T* const oldVal, T* const newVal ) {
  return (T*)atomicCompareAndSwap((uintptr_t*)p, (uintptr_t)oldVal, (uintptr_t)newVal);
}
# else // !_MSC_VER
template <typename T>
inline T atomicCompareAndSwap( T* const p, T const oldVal, T const newVal ) {
  return __sync_val_compare_and_swap(p, oldVal, newVal);
}
# endif // !_MSC_VER

inline void readBarrier() {
# ifdef _MSC_VER
    MemoryBarrier();
# else
    __sync_synchronize();
# endif
}

inline void writeBarrier() {
# ifdef _MSC_VER
    MemoryBarrier();
# else
    __sync_synchronize();
# endif
}

inline void fullBarrier() {
# ifdef _MSC_VER
    MemoryBarrier();
# else
    __sync_synchronize();
# endif
}
} // namespace orCore



#endif /* ATOMIC_H */

