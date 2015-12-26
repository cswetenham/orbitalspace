#ifndef ORSYSTEM_H
#define	ORSYSTEM_H

#include "SDL_assert.h"
#include "orStd.h"
#include <stdint.h>

// TODO make() invalidates existing references but I'm holding on to them!! (while constructing objects)
// go through code and make sure I don't hold onto references unsafely

// TODO make it possible to free objects without invalidating all handles
// TODO make it possible to detect stale handles - if we free an object and then
// allocate a new one, old handles pointing to the old object should be detected
// as invalid if they are ever used.
// means objects will need a 'living'/'allocated' flag that is checked every time
// we loop over the objects?
// I remember seeing this done in bitsquid's impl with the following differences:
// - a system has one id type for all of its subtypes (e.g. Render would have one handle type for Orbit, Sphere, etc.)
// - a layer of indirection was used between the handle and the index of the actual allocation - the mapping was being
// maintained in an array in the system.
// - might have involved the trick of 'freeing' from an arbitrary position in an
// array by moving the last element into the position of the freed element, and
// therefore keeping 'living' element contiguous? (Means allocation is O(1)
// instead of O(N), and we don't have to maintain and check a 'living' flag when
// looping over objects)
// So freeing would mean moving the last object into the freed position, and
// updating the indirection table for the handle for the last object to point
// to the new position. We can maintain the index of the handle for each object
// index in a separate array that also gets moved on delete, and use it to update
// the indirection table entry for the object...
// Handle invalidation? Either we would keep the invalid handle around as long
// as possible, or we would use a larger space to allocate handles from (sequentially or randomly)
// and have some way of knowing which are still valid ('generation' number in handle, wraps, incremented on allocation? map of valid handles?)
// Handle: { uint16_t gen; uint16_t type; uint32_t idx; }
// was the intermediate an array of handles by idx, or of idx by handle?
// [ 1, 0, 2, 5 ]
// [ {}, {}, 0, {}, 0, {} ]
// operations: lookup (handle to T&), create, delete, loop over Ts
// handle: type, uid OR uid only, have type in inner array...
// let's use the move-last-item trick, keep the type out of the handle, have the
// handle be an idx, and have the indirection array contain a 'real' idx and a
// type id (which selects the concrete array)
// that all works except how do we handle handle invalidation?

namespace orbital {

// TODO make zero-initializable
template <typename T>
struct Id {
  inline constexpr Id() noexcept :
    generation { 0 },
    sparse_idx { UINT16_MAX }
  {}

  inline constexpr Id(uint16_t generation_, uint16_t sparse_idx_) noexcept :
    generation { generation_ },
    sparse_idx { sparse_idx_ }
  {}

  inline constexpr explicit operator bool() const noexcept { return sparse_idx != UINT16_MAX; }

  uint16_t generation;
  uint16_t sparse_idx;
};
  
template <typename T, uint32_t MAX_OBJECTS = 32*1024>
struct IdArray {
  // TODO: select index type from MAX_OBJECTS size at compile time
  // TODO: think carefully about edge case where we have exactly the max number of objects; we use UINT16_MAX as an invalid value
  static_assert(MAX_OBJECTS <= UINT16_MAX, "Object count too large for IdArray!");
  
  
  struct Index {
    uint16_t generation;
    uint16_t dense_idx;
    uint16_t next_free;
  };
  
  uint16_t _num_objects;
  Index _indices[MAX_OBJECTS];
  
  T _objects[MAX_OBJECTS];
  uint16_t _sparse_from_dense[MAX_OBJECTS];
  
  uint16_t _freelist_enqueue;
  uint16_t _freelist_dequeue;
  
  IdArray() :
    _num_objects {0},
    _freelist_enqueue { MAX_OBJECTS-1 },
    _freelist_dequeue { 0 }
  {
    for (uint16_t i = 0; i < MAX_OBJECTS; ++i) {
      _indices[i].generation = 0;
      _indices[i].dense_idx = 0;
      _indices[i].next_free = i+1;
      _sparse_from_dense[i] = UINT16_MAX;
    }
  }

  // TODO TEMP because I can't get the free function versions in id_array to be found
    // TODO try putting them in orbital:: instead of orbital::id_array::
  inline T* begin() {
    return &_objects[0];
  }
  inline T const* begin() const {
    return &_objects[0];
  }
  inline T* end() {
    return &_objects[_num_objects];
  }
  inline T const* end() const {
    return &_objects[_num_objects];
  }
  
};

namespace id_array {
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto num_objects(IdArray<T, MAX_OBJECTS> const& a) noexcept {
    return a._num_objects;
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto objects(IdArray<T, MAX_OBJECTS>& a) noexcept {
    return &a._objects[0];
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto objects(IdArray<T, MAX_OBJECTS> const& a) noexcept {
    return &a._objects[0];
  }
 
  template<typename T, uint32_t MAX_OBJECTS>
  inline T* begin(IdArray<T, MAX_OBJECTS>& a) {
    return &a._objects[0];
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline T const* begin(IdArray<T, MAX_OBJECTS> const& a) {
    return &a._objects[0];
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline T* end(IdArray<T, MAX_OBJECTS>& a) {
    return &a._objects[a._num_objects];
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline T const* end(IdArray<T, MAX_OBJECTS> const& a) {
    return &a._objects[a._num_objects];
  }
  
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline bool has(IdArray<T, MAX_OBJECTS> const& a, Id<T> const id) {
    ensure(id.sparse_idx < MAX_OBJECTS);
    auto& in = a._indices[id.sparse_idx];
		return in.generation == id.generation
        && in.dense_idx != UINT16_MAX;
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline bool has(IdArray<T, MAX_OBJECTS> const& a, T const* const p) { // TODO think carefully about type of p - could be a const& for instance, but don't want to allow a temporary? Or does it not matter?
    return &a._objects[0] < p && p <= &a._objects[MAX_OBJECTS-1] // contained
        && (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(&a._objects[0])) % sizeof(T) == 0; // aligned
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline T& get_object(IdArray<T, MAX_OBJECTS>& a, Id<T> const id) {
    ensure(id.sparse_idx < MAX_OBJECTS);
    ensure(has(a, id));
    auto& in = a._indices[id.sparse_idx];
		return a._objects[in.dense_idx];
	}
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline T const& get_object(IdArray<T, MAX_OBJECTS> const& a, Id<T> const id) {
    ensure(id.sparse_idx < MAX_OBJECTS);
    ensure(has(a, id));
    auto& in = a._indices[id.sparse_idx];
		return a._objects[in.dense_idx];
	}
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto get_idx(IdArray<T, MAX_OBJECTS> const& a, Id<T> const id) {
    ensure(id.sparse_idx < MAX_OBJECTS);
    ensure(has(a, id));
    auto& in = a._indices[id.sparse_idx];
		return in.dense_idx;
	}
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto get_id(IdArray<T, MAX_OBJECTS> const& a, T const* const p) { // TODO think carefully about type of p - could be a const& for instance, but don't want to allow a temporary? Or does it not matter?
    ensure(has(a, p));
    ptrdiff_t dense_idx = p - &a._objects[0];
    ensure(dense_idx < MAX_OBJECTS);
    Id<T> id;
    id.sparse_idx = a._sparse_from_dense[dense_idx];
    id.generation = a._indices[id.sparse_idx].generation;
    return id;
  }
  
  template<typename T, uint32_t MAX_OBJECTS>
  inline auto add(IdArray<T, MAX_OBJECTS>& a) {
		ensure(a._freelist_dequeue < MAX_OBJECTS); // out of space!
    
    auto const sparse_idx = a._freelist_dequeue;
    
    auto& in = a._indices[sparse_idx];
    
    auto const dense_idx = a._num_objects;
    
    a._num_objects += 1;
    a._freelist_dequeue = in.next_free;
    		    
    in.generation += 1;
    in.dense_idx = dense_idx;
    
    Id<T> id;
    id.generation = in.generation;
    id.sparse_idx = sparse_idx;
    
    // T& o = a._objects[dense_idx];
    a._sparse_from_dense[dense_idx] = sparse_idx;

    return id;
	}
	
  // TODO write tests
  
  // id must have been tested with has()
  template<typename T, uint32_t MAX_OBJECTS>
	inline void remove(IdArray<T, MAX_OBJECTS>& a, Id<T> const id) {
    ensure(has(a, id));
    
    auto const sparse_idx = id.sparse_idx;
    auto const dense_idx = a._indices[sparse_idx].dense_idx;
		
    // Compact dense array by copying last object into free slot
    // TODO: based only on the assumptions we make in this class, can we do
    // better for a wider range of types if we use std::move? Would need to think
    // in more detail about construction + destruction and making this a real allocator.
    auto const moved_dense_idx = a._num_objects-1;
    auto const moved_sparse_idx = a._sparse_from_dense[moved_dense_idx];
    
    a._num_objects -= 1;
    
    // If we are removing the last dense idx, we will copy it in place and then
    // invalidate it, which is fine
    
    a._objects[dense_idx] = a._objects[moved_dense_idx];
    a._sparse_from_dense[dense_idx] = a._sparse_from_dense[moved_dense_idx];
    
    a._sparse_from_dense[moved_dense_idx] = UINT16_MAX;
        
    // Update the sparse entry for the moved object (only the dense_idx has changed)
    a._indices[moved_sparse_idx].dense_idx = dense_idx;
		
    // Clear the sparse entry for the removed object
		a._indices[sparse_idx].dense_idx = UINT16_MAX;
		a._indices[a._freelist_enqueue].next_free = sparse_idx;
		a._freelist_enqueue = sparse_idx;
	}
  
  // This is a simple way of iterating through all ids by dense_idx, not super efficient for a loop but handy for camera stuff for now.
  // In future could store a list of camera ids / target ids in the cycle order we want.
  template<typename T, uint32_t MAX_OBJECTS>
  auto next_id(IdArray<T, MAX_OBJECTS> const& a, Id<T> id) {
    ensure(has(a, id));
    uint32_t dense_idx = get_idx(a, id);
    uint32_t next_dense_idx = (dense_idx + 1) % num_objects(a);
    return get_id(a, &objects(a)[next_dense_idx]);
  }
}

} // namespace orbital

// TODO get rid of the macro
// TODO can I express both const overloads in one in C++14?
#define DECLARE_SYSTEM_TYPE(T_SINGULAR, T_PLURAL)\
public:\
  uint32_t num ## T_PLURAL () const { return ::orbital::id_array::num_objects( m_instanced ## T_PLURAL ); }\
  auto make ## T_SINGULAR () { return ::orbital::id_array::add( m_instanced ## T_PLURAL ); }\
  T_SINGULAR&       get ## T_SINGULAR ( ::orbital::Id<T_SINGULAR> id)       { return ::orbital::id_array::get_object( m_instanced ## T_PLURAL, id ); }\
  T_SINGULAR const& get ## T_SINGULAR ( ::orbital::Id<T_SINGULAR> id) const { return ::orbital::id_array::get_object( m_instanced ## T_PLURAL, id ); }\
  auto next ## T_SINGULAR ( ::orbital::Id< T_SINGULAR > id ) const { return ::orbital::id_array::next_id( m_instanced ## T_PLURAL, id ); }\
private:\
  ::orbital::IdArray<T_SINGULAR> m_instanced ## T_PLURAL;\
public:


#endif	/* ORSYSTEM_H */

