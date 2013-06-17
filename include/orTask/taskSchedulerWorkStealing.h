#ifndef TASKSCHEDULERWORKSTEALING_H
#define TASKSCHEDULERWORKSTEALING_H

#include "taskScheduler.h"
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
    TaskSchedulerWorkStealing(int numThreads) :
      TaskScheduler(numThreads),
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


#endif /* TASKSCHEDULERWORKSTEALING_H */

