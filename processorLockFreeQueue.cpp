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

#include "processorLockFreeQueue.h"

#include "boost_begin.h"
#include "boost/date_time/posix_time/ptime.hpp"
#include "boost/date_time/posix_time/time_formatters.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost_end.h"

void orTask::TaskSchedulerWorkStealing::thread_fn(ThreadData* threadData) {
  // TODO refactor out common code with waitForTaskGroup
  while (threadData->curState != ThreadData::STATE_EXIT) {
    ThreadData::State startState = threadData->curState;
    
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::universal_time();
  
    threadData->curState = thread_step(threadData);
    
    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::universal_time();
    
    if (startState == ThreadData::STATE_WORKING) {
      threadData->activeTime += (endTime - startTime);
    } else {
      threadData->inactiveTime += (endTime - startTime);
    }
  }
}

void orTask::TaskSchedulerWorkStealing::waitForTaskGroup(int threadIdx, TaskGroup* group) {
  ThreadData* curThreadData = &threadData[threadIdx];
  while (group->tasks != 0 && curThreadData->curState != ThreadData::STATE_EXIT) {
    orCore::readBarrier(); // ensures read to group->tasks isn't optimised out
    
    ThreadData::State startState = curThreadData->curState;
    
    boost::posix_time::ptime startTime = boost::posix_time::microsec_clock::universal_time();
  
    curThreadData->curState = thread_step(curThreadData);
    
    boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::universal_time();
    
    if (startState == ThreadData::STATE_WORKING) {
      curThreadData->activeTime += (endTime - startTime);
    } else {
      curThreadData->inactiveTime += (endTime - startTime);
    }
  }
  // We got an exit task on the waiting thread (Why?)
  // So just sleep until the group is completed
  boost::posix_time::millisec const sleepTime = boost::posix_time::millisec(1);
  while (group->tasks != 0) {
    orCore::readBarrier();
    boost::this_thread::sleep(sleepTime);
  }
}

orTask::TaskSchedulerWorkStealing::ThreadData::State orTask::TaskSchedulerWorkStealing::thread_exectask(Task const& task, ThreadData* threadData)
{
  TaskGroup* group = task.group;
  bool exit = task.exit;
  TaskFn* taskFn = task.taskUserFn;
  void* taskData = task.taskUserData;
  
  // TODO can forbid null taskFn in submit, use null to indicate exit instead of storing bool.
  // TODO assert group null?
  if (exit) {
    threadData->barrier->decActive();
    return ThreadData::STATE_EXIT;
  }
    
  (*taskFn)(threadData->threadIdx, taskData);
    
  if (group) {
    group->remTask();
  }
  
  return ThreadData::STATE_WORKING;
}

orTask::TaskSchedulerWorkStealing::ThreadData::State orTask::TaskSchedulerWorkStealing::thread_step(ThreadData* threadData)
{
  VictimPicker* victimPicker = threadData->victimPicker;

  std::vector< Queue* >& queues = *threadData->queues;
  int id = threadData->threadIdx;
  TerminationBarrier* barrier = threadData->barrier;

  boost::posix_time::millisec const sleepTime = boost::posix_time::millisec(1);
  
  switch (threadData->curState) {
    case ThreadData::STATE_WORKING: {
      // Take work from own queue
      
      Task task;
      if (!queues[id]->popBottom(&task)) {
        barrier->decActive();
        return ThreadData::STATE_STEALING;
      }

      return thread_exectask(task, threadData);
    }
    case ThreadData::STATE_STEALING: {
      // Try stealing work
      
      int victim = victimPicker->pick();

      if (queues[victim]->isEmpty()) {
        // Queue was empty
        // TODO Could probably have idle threads sleep:
        // just sleep longer and check barrier->allInactive()...
        // if (barrier->allInactive()) {
        //   return WorkerData::STATE_IDLE;
        // }
        // Other threads are still active; we've remained in the stealing state, but didn't manage to steal anything this time. Sleep before retrying.
        boost::this_thread::sleep(sleepTime);
        return ThreadData::STATE_STEALING;
      }
      
      // Mark ourselves active before actual steal attempt - required for correct termination
      barrier->incActive();

      Task task;
      if (!queues[victim]->popTop(&task)) {
        // Queue was non-empty but we didn't manage to steal - 
        barrier->decActive();
        return ThreadData::STATE_STEALING;
      }

      // TODO should this execution count towards active time?
      // What does active/inactive time mean across different schedulers?
      // It's certainly not the same as time spent in job vs time spent in scheduler
      return thread_exectask(task, threadData);
    }
    case ThreadData::STATE_IDLE: {
      // TODO need to do something useful instead. Sleep, woken on event? Then need to think about race conditions.
      // ensure(false);
      // return;
      boost::this_thread::sleep(sleepTime);
      return ThreadData::STATE_IDLE;
    }
    case ThreadData::STATE_EXIT:
    default:
    {
      ensure(false);
      return ThreadData::STATE_EXIT;
    }
  }
}

void orTask::TaskSchedulerWorkStealing::submitExit(int threadIdx) {
  Task task;
  task.exit = true;
  task.group = NULL;
  task.taskUserFn = NULL;
  task.taskUserData = NULL;
  
  queues[threadIdx]->pushBottom( task );
}

void orTask::TaskSchedulerWorkStealing::submitTaskForGroup(int threadIdx, TaskGroup* group, TaskFn* fn, void* userData) {
  Task task;
  task.exit = false;
  task.group = group;
  task.taskUserFn = fn;
  task.taskUserData = userData;
  
  if (group) {
    group->addTask();
  }
  
  queues[threadIdx]->pushBottom( task );
}

// TODO inline & remove
void orTask::TaskSchedulerWorkStealing::initThreads()
{
  // Start the worker threads, index [1 -- numThreads]
  for (int threadIdx = 1; threadIdx < numThreads_; ++threadIdx) {
    threads.add_thread( new boost::thread(&thread_fn, &threadData[threadIdx]) );
  }
}

// TODO inline & remove
void orTask::TaskSchedulerWorkStealing::exitThreads()
{
  for (int threadIdx = 0; threadIdx < numThreads_; ++threadIdx) {
    submitExit(threadIdx);
  }

  threads.join_all();
}
