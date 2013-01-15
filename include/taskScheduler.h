#ifndef TASKSCHEDULER_H
#define TASKSCHEDULER_H

#include <list>
using std::list;

#include "boost_begin.h"
#include <boost/thread.hpp>
#include "boost_end.h"

#include "util.h"
#include "util2.h"
#include "atomic.h"
#include "task.h"

namespace orTask {
  // Receives the userdata specified for the task.
  struct TaskGroup;
  struct Task;
  
  // Abstract base class ensuring virtual destructor
  class TaskScheduler {
  public:
    TaskScheduler(int numThreads) :
      numThreads_(numThreads)
    {}
    virtual ~TaskScheduler() {}

    void submitTask(int threadIdx, TaskFn* fn, void* ud) { submitTaskForGroup(threadIdx, NULL, fn, ud); }
    virtual void submitTaskForGroup(int threadIdx, TaskGroup* group, TaskFn* fn, void* ud) = 0;
    virtual void waitForTaskGroup(int threadIdx, TaskGroup* group) = 0;

    int getNumThreads() const { return numThreads_; }

  protected:  
    int numThreads_; // TODO move to implementation
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

#endif /* TASKSCHEDULER_H */

