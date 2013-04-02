#pragma once

namespace orPlatform {

// Atomically increments the value at p, and returns the *previous* value.
inline int atomicInc( int* const p ) {
  return __sync_fetch_and_add(p, 1);
}

// Atomically decrements the value at p, and returns the *previous* value.
inline int atomicDec( int* const p ) {
  return __sync_fetch_and_sub(p, 1);
}


// Attempts to atomically swap the value at p from oldVal to newVal.
// Returns the value seen at p before the swap. If this is oldVal, the swap succeeded, otherwise it failed.
template <typename T>
inline T atomicCompareAndSwap( T* const p, T const oldVal, T const newVal ) {
  return __sync_val_compare_and_swap(p, oldVal, newVal);
}

inline void readBarrier() {
  __sync_synchronize();
}

inline void writeBarrier() {
  __sync_synchronize();
}

inline void fullBarrier() {
  __sync_synchronize();
}

} // namespace orPlatform