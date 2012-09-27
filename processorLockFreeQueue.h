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

#ifndef PROCESSORLOCKFREEQUEUE_H
#define PROCESSORLOCKFREEQUEUE_H

#include "problemProcessor.h"
#include "victimPicker.h"
#include "workStealingQueue.h"
#include "terminationBarrier.h"


namespace orTask {

  class TaskSchedulerWorkStealing :
    public TaskScheduler
  {
  private:

    typedef WorkStealingQueue<Task> Queue;

    struct ThreadData : public ThreadDataBase
    {
      // Specific to this scheduler
      std::vector< Queue* >* queues;
      TerminationBarrier* barrier;
      
      VictimPicker* victimPicker;
      
      enum State { STATE_WORKING, STATE_STEALING, STATE_IDLE, STATE_EXIT };
      State curState;
    };
    
    int numWorkers() const { return numThreads_ - 1; }
    
    static void thread_fn(ThreadData* threadData);
    static ThreadData::State thread_step(ThreadData* threadData);
    static ThreadData::State thread_exectask(Task const& task, ThreadData* threadData);

    std::vector< ThreadData > threadData;
    TerminationBarrier barrier;

    enum { QUEUE_LOGCAPACITY_INITIAL = 2 };
    boost::thread_group threads;

    std::vector< Queue* > queues;
    VictimPicker victimPicker;

    Queue::HazardList hazardList;

  public:
    TaskSchedulerWorkStealing(int numThreads, int maxBatch) :
      TaskScheduler(numThreads, maxBatch),
      threadData(numThreads),
      barrier(numThreads),
      victimPicker(0, numThreads-1, 32)
    {
      for (int i = 0; i < numThreads; ++i) {
        queues.push_back(new Queue(QUEUE_LOGCAPACITY_INITIAL, &hazardList));
        threadData[i].threadIdx = i;
        threadData[i].numThreads = numThreads_;
        
        threadData[i].queues = &queues;
        threadData[i].barrier = &barrier;
        
        threadData[i].victimPicker = &victimPicker;
        threadData[i].curState = ThreadData::STATE_WORKING;
      }
      
      initThreads();
    }

    ~TaskSchedulerWorkStealing() {
      exitThreads();
      
      for (int threadIdx = 0; threadIdx < numThreads_; ++threadIdx) {
        delete queues.back();
        queues.pop_back();
      }
    }
    
  public:
    // TODO needs to work on main thread too! so main thread needs to fill in as a worker when we're waiting
    // TODO passing in own threadIdx is not safe - hard to detect if we get it wrong. But how to hide it?
    // I don't really like using thread-local storage if I can avoid it.
    // From TaskScheduler
    void submitTaskForGroup(int threadIdx, TaskGroup* group, TaskFn* fn, void* ud);
    void waitForTaskGroup(int threadIdx, TaskGroup* group);
  
  private:
    void initThreads();
    void exitThreads();
    void submitExit(int threadIdx);
  };

} // namespace orTask


#endif /* PROCESSORLOCKFREEQUEUE_H */

