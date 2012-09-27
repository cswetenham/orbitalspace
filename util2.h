#ifndef UTIL_H
#define UTIL_H

#include "atomic.h"

namespace orCore {

// Base class to prevent operator== being generated
class NoEq
{
private:
  bool operator==(NoEq const&);
  bool operator!=(NoEq const&);
};

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
    atomicInc(&refCount_);
  }
  
  void release() {
    ensure(refCount_ > 0);
    if (atomicDec(&refCount_) == 1) { delete this; }
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

// Non-member function which accepts NULL
template <typename T> inline T* clone(T* t) { if (t) { return t->clone(); } return NULL; }

template <typename T> inline void ignore(T const&) {}; // To explicitly ignore return values without warning

} // namespace lexCore

#endif /* UTIL_H */

