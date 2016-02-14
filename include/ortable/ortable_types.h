/*
 * File:   ortable_types.h
 * Author: cswetenham
 *
 * Created on 07 February 2016, 18:30
 */

#ifndef ORTABLE_TYPES_H
#define	ORTABLE_TYPES_H

namespace ortable {

struct Table {
  void* buffer;
  size_t size;
  size_t capacity;

};

struct Attribute {
  size_t size;
  size_t count;
  size_t offset;
  size_t stride;
};

}  // namespace ortable

#endif	/* ORTABLE_TYPES_H */

