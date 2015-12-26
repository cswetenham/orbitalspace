/* 
 * File:   geom.h
 * Author: s1149322
 *
 * Created on 08 December 2011, 13:56
 */

#ifndef GEOM_H
#define	GEOM_H

#include "util.h"
#include "vector.h"

struct Ray2D
{
  Vector2f pos;
  Vector2f dir; // Unit
};

struct Segment2D
{
  Vector2f start;
  Vector2f end;
};

struct Circle2D
{
  Vector2f center;
  float radius;
};

struct RayCastResult
{
  bool col;
  float dist;
};

// Algorithm from Computational Geometry in C (Second Edition)
inline void TestRaySegment(Ray2D const& _ray, Segment2D const& _segment, RayCastResult* o_result)
{
  Vector2f const dr = _ray.dir;
  Vector2f const ds = _segment.end - _segment.start;

  Vector2f const u = _ray.pos - _segment.start;

  // Might be negative!
  float const scale = (dr.y() * ds.x() - dr.x() * ds.y());

  float const r = (u.x() * ds.y() - u.y() * ds.x()) / scale;
  float const s = (u.x() * dr.y() - u.y() * dr.x()) / scale;

  o_result->col = (0.f < r) & (0.f < s) & (s < 1.f);
  o_result->dist = r;
}

inline void TestRaysSegment(int const _n, Ray2D const* const _ray, Segment2D const& _segment, RayCastResult* const o_results)
{
  Vector2f const ds = _segment.end - _segment.start;
  
  for (int i = 0; i < _n; ++i)
  {
    Vector2f const dr = _ray[i].dir;

    Vector2f const u = _ray[i].pos - _segment.start;

    // Might be negative!
    float const scale = (dr.y() * ds.x() - dr.x() * ds.y());

    float const r = (u.x() * ds.y() - u.y() * ds.x()) / scale;
    float const s = (u.x() * dr.y() - u.y() * dr.x()) / scale;

    o_results[i].col = (0.f < r) & (0.f < s) & (s < 1.f);
    o_results[i].dist = r;
  }
}

inline void TestRayCircle(Ray2D const& _ray, Circle2D const& _circle, RayCastResult* o_result)
{
  // Algo taken from http://stackoverflow.com/questions/1073336/circle-line-collision-detection
  Vector2f d = _ray.dir;
  Vector2f f = _ray.pos - _circle.center;
  float r = _circle.radius;
    
  float a = d.Dot( d ) ;
  float b = 2*f.Dot( d ) ;
  float c = f.Dot( f ) - r*r ;

  float discriminant = b*b-4*a*c;
  
  o_result->col = false;
  o_result->dist = 999999999.f;

  if( discriminant < 0 )
  {
    return;
  }

  // ray didn't totally miss sphere,
  // so there is a solution to
  // the equation.
  
  discriminant = sqrtf( discriminant );
  float t1 = (-b + discriminant)/(2*a);
  float t2 = (-b - discriminant)/(2*a);
    
  if ( t1 >= 0 )
  {
    o_result->col = true;
    o_result->dist = Util::Min(t1, o_result->dist);
  }
  
  if ( t2 >= 0 )
  {
    o_result->col = true;
    o_result->dist = Util::Min(t2, o_result->dist);
  }
}

struct CircleTestResult
{
  bool col;
  float dist;
  Vector2f colNormal;
};

inline void TestCircleSegment(Circle2D const& _circle, Segment2D const& _segment, CircleTestResult* const o_result)
{
  // TODO HACK: assuming infinite walls (lines) instead of segments
  // TODO: check whether closest point falls within segment; if not,
  // check distance to each end of the segment (or just the end the closest point is on the side of)

  Vector2f const segVec = _segment.end - _segment.start;
  Vector2f const segDir = segVec / segVec.GetLength();
  Vector2f p = _circle.center - _segment.start;
  float r = fabsf(segDir.x() * p.y() - p.x() * segDir.y());

  o_result->dist = r;
  o_result->col = (r < _circle.radius);
  o_result->colNormal = Vector2f(-segDir.y(), segDir.x());
}

inline void TestCircleCircle(Circle2D const& _c1, Circle2D const& _c2, CircleTestResult* const o_result)
{
  Vector2f colVec = _c2.center - _c1.center;
  float dist = colVec.GetLength();

  
  o_result->dist = dist - _c2.radius;
  o_result->col = (dist < (_c1.radius + _c2.radius));
  o_result->colNormal = colVec / colVec.GetLength();
}

#endif	/* GEOM_H */

