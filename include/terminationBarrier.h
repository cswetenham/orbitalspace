#ifndef TERMINATIONBARRIER_H
#define	TERMINATIONBARRIER_H

#include "atomic.h"

namespace orTask {

/* Termination barrier to detect when a group of threads have all run out of work.
 * Based on `The Art of Multiprocessor Programming'.
 */
class TerminationBarrier {
public:
  // Call with 0 if threads start inactive, or number of threads if they start active
  TerminationBarrier(int startCount) : activeCount_(startCount) {}

  // Warning: not idempotent! Call only once per inactive->active transition!
  void incActive() {
    orCore::atomicInc(&activeCount_);
    orCore::fullBarrier();
  }

  // Warning: not idempotent! Call only once per active->inactive transition!
  void decActive() {
    orCore::fullBarrier();
    orCore::atomicDec(&activeCount_);
  }

  bool allInactive() const {
    return activeCount_ == 0;
  }

private:
  int activeCount_;
};

} // namespace orTask

#endif	/* TERMINATIONBARRIER_H */

