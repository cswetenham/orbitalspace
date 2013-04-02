/*
Copyright (c) 2012, Christopher William Geib All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef VICTIMPICKER_H
#define	VICTIMPICKER_H

#include "orStd.h"
#include "orMath.h"
#include "rnd.h"

#include "boost_begin.h"
#include <boost/thread.hpp>
#include "boost_end.h"

#include "orPlatform/atomic.h"


namespace orTask {
  class VictimPicker {
  private:
    Rnd64 rng_;
    UniformDistribution<uint32_t> gen_;

    boost::mutex mutex_;
    size_t size_;
    int* buffer_;
    size_t readPos_;

  public:
    VictimPicker(int min, int max, size_t size) :
      rng_(0U),
      gen_(min, max),
      mutex_(),
      size_(size),
      buffer_(new int[size_]),
      readPos_(size_)
    {
      fill();
    }

    ~VictimPicker() { delete[] buffer_; }

    int pick() {
      while (true) {
        size_t const oldPos = readPos_;

        if (oldPos == size_) {
          fill();
          continue;
        }

        size_t const newPos = oldPos + 1;

        orPlatform::readBarrier();
        int const seenVal = buffer_[oldPos];

        size_t const seenPos = orPlatform::atomicCompareAndSwap(&readPos_, oldPos, newPos);

        if (seenPos == oldPos) {
          return seenVal;
        }
      }
    }
  private:
    void fill() {
      boost::mutex::scoped_lock lock(mutex_);
      // Another thread can have filled the buffer while we acquired the
      // lock. If the buffer is still empty, this thread is responsible for
      // filling it.
      if (readPos_ < size_) { return; }

      gen_.Generate(&rng_, size_, (uint32_t*)&buffer_[0]);

      orPlatform::writeBarrier();

      readPos_ = 0;
    }
  };
} // namespace orTask

#endif	/* VICTIMPICKER_H */

