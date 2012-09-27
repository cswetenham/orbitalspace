/*
Copyright (c) 2012, Christopher William Geib and Chris Swetenham All rights reserved.

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

#ifndef PROBLEMPROCESSOR_H
#define PROBLEMPROCESSOR_H

#include <list>
using std::list;

#include "boost_begin.h"
#include <boost/thread.hpp>
#include "boost_end.h"

#include "util.h"
#include "util2.h"
#include "atomic.h"

namespace orTask {
  // Receives the userdata specified for the task.
  typedef void (TaskFn) (uint32_t threadIdx, void* userData);
  struct TaskGroup;
  struct Task;
  
  // Abstract base class ensuring virtual destructor
  class TaskScheduler {
  public:
    TaskScheduler(int numThreads, int maxBatch) :
      numThreads_(numThreads), maxBatch_(maxBatch)
    {}
    virtual ~TaskScheduler() {}

    void submitTask(int threadIdx, TaskFn* fn, void* ud) { submitTaskForGroup(threadIdx, NULL, fn, ud); }
    virtual void submitTaskForGroup(int threadIdx, TaskGroup* group, TaskFn* fn, void* ud) = 0;
    virtual void waitForTaskGroup(int threadIdx, TaskGroup* group) = 0;

    int getNumThreads() const { return numThreads_; }
    int getMaxBatch() const { return maxBatch_; }

  protected:  
    int numThreads_;
    int maxBatch_;
  };
  
  struct Task
  {
    Task() : group(NULL), exit(false) {}
    TaskGroup* group;
    bool exit;
    TaskFn* taskUserFn;
    void* taskUserData;
  };
  
  struct TaskGroup
  {
    TaskGroup() : tasks(0) {}
    int tasks;
    
    void addTask() { orCore::atomicInc(&tasks); }
    void remTask() { orCore::atomicDec(&tasks); }
  };
  
  struct ThreadDataBase
  {
    // Shared between all schedulers
    // Must be filled in
    int threadIdx;
    int numThreads;

    // These are default-initialised to valid values
    boost::posix_time::time_duration activeTime;
    boost::posix_time::time_duration inactiveTime;
  };
  
} // namespace orTask

#endif /* PROBLEMPROCESSOR_H */

