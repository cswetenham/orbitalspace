/* 
 * File:   task.h
 * Author: s1149322
 *
 * Created on 08 December 2011, 13:56
 */

#ifndef TASK_H
#define	TASK_H

#include <stdint.h>
#include "workStealingQueue.h"
#include "terminationBarrier.h"

namespace orTask {

typedef uint64_t TaskId;

TaskId const TaskId_None = 0;

typedef uint32_t ThreadAversion; // Thread mask; a logically negated affinity. Value of all 0s means the task can run on all threads.

typedef uint32_t ThreadId;

typedef void TaskFn(void*);

struct Task
{
  TaskFn* func;
  void*   data;
};

struct TaskMeta
{
  Task task;

  TaskId id;
  TaskId parent;
  TaskId startDep; // Each task has a single dependency which must be complete before this task can start. Can be TaskId_None (0). To have multiple dependencies, add them all as children of an empty task, and use that as dep.
  uint32_t startDepCount; // dependency count before starting. Can start executing task once this reaches 0.
  uint32_t endDepCount; // dendency count before ending. Can mark task as completed once this reaches 0.

  ThreadAversion aversion; // Bitmask. Currently must be either all 0s or a single 0 cleared.
};

class TaskQueue; // Two queues for each thread: one for its own affinity, where all can push but only it can pop; and one work-stealing queue, where it pushes and pops and others steal.

class TaskScheduler {
  TaskId m_lastId;
public:
  TaskScheduler() : m_lastId(TaskId_None) {}

  TaskId next_id() {
    m_lastId++;
    return m_lastId;
  }
  
  void wait(TaskId _id) { /* TODO */ ensure(false, "Not implemented!"); }
  void add_tasks(size_t _n, TaskMeta* _tasks) { /* TODO */ ensure(false, "Not implemented!"); }
};

class TaskSubmitter;

class TaskBuilder { // implicit conversion to TaskId; methods return self.
  TaskSubmitter& m_submitter;
  TaskMeta& m_task;
public:
  TaskBuilder(TaskSubmitter& _submitter, TaskMeta& _task) : m_submitter(_submitter), m_task(_task) {}
  operator TaskId() { return m_task.id; }

  TaskBuilder& affinity(ThreadId threadId) {
    m_task.aversion = ~(1 << threadId);
    return *this;
  }
  
  inline TaskBuilder& depends_on(TaskId dep);

  inline TaskBuilder& add_child(TaskId child);
};

class TaskSubmitter {
  TaskScheduler& m_scheduler;
  std::vector<TaskMeta> m_pending;
public:
  explicit TaskSubmitter(TaskScheduler& _scheduler) : m_scheduler(_scheduler), m_pending(0) {}
  ~TaskSubmitter() { m_scheduler.add_tasks(m_pending.size(), m_pending.data()); }

  TaskBuilder add(Task const& task) {
    TaskMeta& taskData = add_task();
    taskData.task = task;
    return TaskBuilder(*this, taskData);
  }

  TaskBuilder add_empty() {
    TaskMeta& taskData = add_task();
    return TaskBuilder(*this, taskData);
  }

  // TODO only to be used by TaskBuilder
  TaskMeta& get_pending(TaskId id) {
    for (uint32_t i = 0; i < m_pending.size(); ++i) {
      if (m_pending[i].id == id) {
        return m_pending[i];
      }
    }
    ensure(false, "Task not found!");
  }

private:
  TaskMeta& add_task() {
    m_pending.push_back(TaskMeta());
    // I think c++ standard should ensure this is 0-initialised, but putting in asserts for now and TODO check if it's true
    TaskMeta& taskData = m_pending.back();
    assert(taskData.id == 0);
    assert(taskData.parent == 0);
    assert(taskData.startDep == 0);
    assert(taskData.startDepCount == 0);
    assert(taskData.endDepCount == 0);
    assert(taskData.task.func == 0);
    assert(taskData.task.data == 0);
    assert(taskData.aversion == 0);

    taskData.id = m_scheduler.next_id();

    return taskData;
  }
};

TaskBuilder& TaskBuilder::depends_on(TaskId dep) {
  assert(m_task.startDep == TaskId_None);
  m_task.startDepCount++;
  m_task.startDep = dep;
  return *this;
}

TaskBuilder& TaskBuilder::add_child(TaskId child) {
  m_task.endDepCount++;
  TaskMeta& childTask = m_submitter.get_pending(child);
  childTask.parent = m_task.id;
  return *this;
}

// TODO find better naming scheme for work items vs tasks...
// Compile test func
// TODO remove, or move to tests
void World_update(TaskScheduler& scheduler, ThreadId render_thread, Task const& animWork, Task const& sceneWork, Task const& guiWork, Task const& renderWork, Task const& soundWork)
{
  // Child tasks are tasks which must complete before their parent is allowed to complete.
  // Child tasks are implemented by giving tasks an "completionDepCount" fiend and "parentTask" field.
  // Tasks cannot complete until completionDepCount reaches 0.
  // When a tasks' work item has been run, it decrements completionDepCount.
  // When a task completes, it decrements the completionDepCount of its parent.
  // Whichever thread decrements comletionDepCount to 0 is resonbible for marking the task as complete, and decrementing the count of its parent, etc.

  // Tasks cannot be allowed to complete until their children have been added.
  // Children cannot be allowed to complete until their parents have been added.
  // We can prevent completion by incrementing the completionDepCount until the task has been fully configured.
  // Tasks cannot be allowed to start until their dependencies have been added.
  // Tasks cannot be allowed to start until their work item (if any) has been added.
  // We can prevent starting by not adding the task until the method chain ends, and only allowing the above operations while a method chain is in progress.

  // TODO We could allow task workitems to add new child tasks to themselves; that way, e.g the animation task can spawn a subtree of tasks, then deschedule itself rather than having to wait on them.

  // TaskIds could be used to wait for a task, poll for task completion (do we ever want to allow this though?), add a task as a child, add a task as a dependency...
  
  TaskId done;
  {
    // TODO instead:
    // TaskParams guiTask;
    // guiTask.fn = ...
    // guiTask.data = ...
    // guiTask.affinity = render_thread;
    // TaskId gui = submitter.add( guiTask );

    // TaskParams gui_scene; // implicitly empty? or make explicit? TaskParams::Empty(), TaskParams::Task(fn, data)?
    // gui_scene.add_child( scene_graph );

    // etc

    TaskSubmitter submitter(scheduler); // RAII class; added tasks are submitted on destruction

    TaskId animation = submitter.add( animWork );

    TaskId gui = submitter.add( guiWork )
                          .affinity( render_thread );
  
    TaskId scene_graph = submitter.add( sceneWork )
                                  .depends_on(animation);

    TaskId gui_scene = submitter.add_empty()
                                .add_child( scene_graph )
                                .add_child( gui );
   
    TaskId render = submitter.add( renderWork )
                             .affinity( render_thread )
                             .depends_on( gui_scene );

    TaskId sound = submitter.add( soundWork );
    
    done = submitter.add_empty()
                    .add_child(render)
                    .add_child(sound);
  }
  scheduler.wait(done); // Should give an error if tasks not yet submitted
}

} // namespace orTask

#endif // TASK_H