#ifndef ORCORE_REFCOUNT_H
#define ORCORE_REFCOUNT_H

#include "orPlatform/atomic.h"

namespace orCore {

// Reference counting

template <class T> class RefCountPtr;

class RefCounted {
protected:
  // NOTE: starts with 0 references.
  // Number of claim() calls should match number of release() calls.
  RefCounted() : refCount_(0) {}
  RefCounted(RefCounted const&) : refCount_(0) {}
  virtual ~RefCounted() { ensure(refCount_ == 0); }

  template <class U> friend class RefCountPtr;
  
private:
  void claim() {
    orPlatform::atomicInc(&refCount_);
  }
  
  void release() {
    ensure(refCount_ > 0);
    if (orPlatform::atomicDec(&refCount_) == 1) { delete this; }
  }
  
private:
  int refCount_;
};

// Non-member functions which accept NULL
template <class T> RefCountPtr<T> wrapWithClaim(T* ptr);
template <class T> RefCountPtr<T> wrapNoClaim(T* ptr);

template <class T>
class RefCountPtr
{
  friend RefCountPtr<T> wrapWithClaim<>(T*);
  friend RefCountPtr<T> wrapNoClaim<>(T*);
  template <class U> friend class RefCountPtr;

private:
  // Doesn't claim. Called by wrapWithClaim and wrapNoClaim.
  template <class U>
  explicit RefCountPtr(U* u) : ptr_(u) {}
  
public:
  RefCountPtr() : ptr_(NULL) {}
  RefCountPtr(RefCountPtr const& r) : ptr_(r.ptr_) { claim(r.ptr_); }
  ~RefCountPtr() { release(ptr_); }
  
  RefCountPtr& operator=(RefCountPtr const& r) {
    claim(r.ptr_);
    release(ptr_);
    ptr_ = r.ptr_;
    return *this;
  }
  
  template <class U>
  bool operator==( RefCountPtr<U> const& r ) const { return ptr_ == r.ptr_; }
  template <class U>
  bool operator!=( RefCountPtr<U> const& r ) const { return !operator==(r); }
  
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  
  // Intentionally omitting operator bool() so that RefCountPtr<A> == RefCountPtr<B> doesn't compile.
  bool operator!() const { return ptr_ == NULL; }
  
  // Needed for std::map
  bool operator<(RefCountPtr const& r) const { return ptr_ < r.ptr_; }
  
  template <class U>
  operator RefCountPtr<U>() const {
    U* const u = ptr_;
    return wrapWithClaim(u);
  }

  template <class U>
  RefCountPtr<U> staticCast() const {
    U* const u = static_cast<U*>(ptr_);
    return wrapWithClaim(u);
  }
  
private:
  inline static void claim(RefCounted* const r) { if (r) { r->claim(); } }
  inline static void release(RefCounted* const r) { if (r) { r->release(); } }
  
private:
  T* ptr_;
};

template <class T> RefCountPtr<T> wrapWithClaim(T* ptr) { RefCountPtr<T>::claim(ptr); return RefCountPtr<T>(ptr); }
template <class T> RefCountPtr<T> wrapNoClaim(T* ptr) { return RefCountPtr<T>(ptr); }

} // namespace orCore

#endif /* ORCORE_REFCOUNT_H */

