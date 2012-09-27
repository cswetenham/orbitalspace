#ifndef WORKSTEALINGQUEUE_H
#define WORKSTEALINGQUEUE_H

#include "atomic.h"
#include "util.h"
#include "util2.h"
#include <list>

namespace orTask {

// Hazard pointers: each thread has a list of hazard pointers that are
// references which prevent collection, and a list of pointers to be collected.
// The thread which retires a pointer is the one which is responsible for
// collecting it; if the thread ends before collecting, it should hand off its
// collection list to another thread.

// Since each thread will have one queue and at most one hazard pointer, we can
// store the hazard pointer records in the queues.

// Since the main thread creates and destroys the queues themselves, and only
// destroys them once all other threads have ended, the queue destructor could
// collect any remaining items in the collection list.

template <typename T> struct HazardPointerRecord;

template <typename T>
struct HazardPointerList 
{
  HazardPointerList() : head(NULL) {};
  ~HazardPointerList() // only destroy when every user of the records is finished and all hazards cleared!
  {
    HazardPointerRecord<T>* hprec = head;
    while (hprec != NULL) {
      ensure(hprec->pointer == NULL);
      HazardPointerRecord<T>* nextRec = hprec->next;
      delete hprec;
      hprec = nextRec;
    }
  }
  HazardPointerRecord<T>* head;
};

template <typename T>
struct RetiredList
{
  HazardPointerList<T>* hazardList;
  std::list<T*> retiredList;
  
  
  RetiredList(HazardPointerList<T>* list) : hazardList(list) {}
  
  ~RetiredList() { Scan(); ensure(retiredList.empty()); } // Assuming the lists are destroyed after all threads have finished
  
  void RetireNode(T* node) {
    retiredList.push_back(node);
    Scan();
  };
  
  void Scan() {
    // Stage 1: Scan each thread's hazard pointer record and insert non-null values in hazardList
    std::list<T*> localHazardList;
    HazardPointerRecord<T>* hprec = hazardList->head;
    while (hprec != NULL) {
      T* hptr = hprec->pointer;
      if (hptr != NULL) {
        localHazardList.push_back(hptr);
      }
      hprec = hprec->next;
    }

    // Stage 2: Search hazardList
    std::list<T*> tmpList;
    tmpList.splice(tmpList.end(), retiredList);
    
    while (!tmpList.empty()) {
      T* node = tmpList.front();
      tmpList.pop_front();
      
      if(std::find(localHazardList.begin(), localHazardList.end(), node) != localHazardList.end()) {
        retiredList.push_back(node);
      } else {
        delete node;
      }
    }
  }
};

// One record per thread
template <typename T>
struct HazardPointerRecord
{
  HazardPointerRecord(HazardPointerList<T>* list) :
    pointer(NULL),
    next(NULL)
  {
    HazardPointerRecord* oldHead;
    do {
      oldHead = list->head;
      next = oldHead;
      orCore::writeBarrier();
    } while (orCore::atomicCompareAndSwap(&list->head, oldHead, this) != oldHead);
  }
          
  T* pointer;
  HazardPointerRecord* next;
};

template <typename T>
class LockFreeCircularArray
{
  // T must be default-constructible
private:
  size_t const logCapacity_;
  std::vector<T> currentTasks_;

public:
  explicit LockFreeCircularArray(size_t logCapacity) :
    logCapacity_(logCapacity),
    currentTasks_(1 << logCapacity_)
  {}

  // resizing copy ctor
  LockFreeCircularArray(LockFreeCircularArray const& oldArray, size_t bottom, size_t top) :
    logCapacity_(oldArray.logCapacity_ + 1),
    currentTasks_(1 << logCapacity_)
  {
    for (size_t i = top; i < bottom; ++i) {
      put(i, oldArray.get(i));
    }
  }

  ~LockFreeCircularArray() {}

  size_t capacity() const { return 1 << logCapacity_; }

  T& get(size_t idx) {
    return currentTasks_[idx % capacity()];
  }

  T const& get(size_t idx) const {
    return currentTasks_[idx % capacity()];
  }

  void put(size_t idx, T const& v) {
    get(idx) = v;
  }
};

template <typename T>
class WorkStealingQueue
{
  // T must be Default-Constructible, for the inner circular queue.
  // T must be Assignable, for when we resize the inner circular queue.
private:
  typedef LockFreeCircularArray<T> Array;

public:
  typedef HazardPointerList< Array > HazardList;

private:
  RetiredList< Array > retired; // Retired list
  HazardPointerRecord< Array >* hazard; // Hazard pointer
  
  LockFreeCircularArray<T>* tasks;
  size_t bottom;
  size_t top;
public:
  WorkStealingQueue(size_t logCapacity, HazardList* hazardList) :
    retired(hazardList),
    hazard(new HazardPointerRecord<Array>(hazardList)), // freed by hazardList's dtor
    tasks(new LockFreeCircularArray<T>(logCapacity)),
    bottom(0),
    top(0)
  { }
    
  ~WorkStealingQueue() {
    orCore::fullBarrier();
    retired.RetireNode(tasks);
  }

  bool isEmpty() const {
    size_t const localTop = top;
    size_t const localBottom = bottom;
    orCore::readBarrier(); // not strictly necessary...
    return localBottom <= localTop;
  }

  // One thread, the owner of the queue, can call pushBottom and popBottom().
  // Concurrently with this, other threads can call popTop().
  void pushBottom(T const& v) {
    size_t const oldBottom = bottom;
    size_t const oldTop = top;
    LockFreeCircularArray<T>* oldTasks = tasks;
    
    size_t const newBottom = oldBottom + 1;
    size_t const size = oldBottom - oldTop;
    LockFreeCircularArray<T>* newTasks = oldTasks;
    
    if (size + 1 >= oldTasks->capacity()) {
      newTasks = new LockFreeCircularArray<T>(*oldTasks, oldBottom, oldTop);
      tasks = newTasks;
      orCore::fullBarrier(); // Ensure the write replacing tasks is visible before we check any other thread's hazard pointers
      retired.RetireNode(oldTasks);
    }
   
    newTasks->put(oldBottom, v);
    orCore::writeBarrier(); // Write buffer contents before writing bottom (allowing reading of the contents)
    
    bottom = newBottom;
  }

  bool popTop(T* v) {
    size_t const oldTop = top;
    orCore::readBarrier(); // Read top before reading bottom TODO why this particular order?
    // Guarantees that if the CAS below succeeds, bottom had a value at least as up to date as top.
    size_t const oldBottom = bottom;
    orCore::readBarrier(); // Read bottom before reading tasks, ensures that we don't get a value of bottom for a task that was pushed into a new array, but the old value of the array.
    LockFreeCircularArray<T>* oldTasks;
    
    // Store tasks in hazard pointer
    do {
      oldTasks = tasks;
      hazard->pointer = oldTasks;
      orCore::fullBarrier(); // ensure we write the hazard pointer before we read tasks again
    } while (oldTasks != tasks);
    
    int const size = oldBottom - oldTop; // int because we test for < 0. We allow bottom and top to wrap, so the best we could do is test (int64)(bottom - oldTop) < 0.

    if (size <= 0) {
      hazard->pointer = NULL;
      return false;
    }
    
    orCore::readBarrier(); // Read bottom before reading buffer contents
    
    *v = oldTasks->get(oldTop);
    
    orCore::fullBarrier(); // Read buffer contents before writing top (if we succeed, we allow overwriting of the contents) or releasing the hazard pointer (allowing replacing the array)
    
    hazard->pointer = NULL;
    
    size_t const newTop = oldTop + 1;
    if (orCore::atomicCompareAndSwap(&top, oldTop, newTop) == oldTop) {
      return true;
    }

    return false;
  }

  bool popBottom(T* v) {
    size_t const oldBottom = bottom;
    size_t const newBottom = oldBottom - 1;
    bottom = newBottom;
    
    orCore::fullBarrier(); // Write bottom before reading from top. Ensures we've claimed an item in the queue.
    
    size_t const oldTop = top;
    
    LockFreeCircularArray<T>* const oldTasks = tasks;
    
    int const newSize = newBottom - oldTop; // int because we test for < 0. We allow bottom and top to wrap, so the best we could do is test (int64)(bottom - oldTop) < 0.

    if (newSize < 0) {
      // We tried to claim an item in the queue, but there wasn't one
      bottom = oldTop; // Reset bottom to give an empty queue
      return false;
    }
    
    // No read barrier here, since only this thread will have written bottom and buffer contents
    
    *v = oldTasks->get(newBottom);
    
    // No barrier here, since only this thread would modify the buffer contents after this point
    
    if (newSize > 0) {
      return true;
    }
    
    // newSize == 0, so only one item in the queue. Out of a set of concurrent
    // popTop() calls, only one can have an outdated value of bottom (before we
    // claimed the item with bottom--) AND succeed in its CAS, because bottom is
    // read after top in popTop(). So we only need to worry about this one
    // concurrent popTop() conflicting with this call. We attempt to CAS the
    // value of top; this prevents a concurrent popTop call from seeing
    // successfully stealing the same item. If we succeed, we got the item,
    // otherwise a concurrent stealing thread got it.
    size_t const newTop = oldTop + 1;
        
    if (orCore::atomicCompareAndSwap(&top, oldTop, newTop) != oldTop) {
      bottom = newTop;
      return false;
    }

    bottom = newTop; // either way, the result is an empty queue
    return true;
  }
};

} // namespace lexRec

#endif /* WORKSTEALINGQUEUE_H */

