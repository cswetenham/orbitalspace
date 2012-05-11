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
  Vec2 pos;
  Vec2 dir; // Unit
};

struct Segment2D
{
  Vec2 start;
  Vec2 end;
};

struct Circle2D
{
  Vec2 center;
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
  Vec2 const dr = _ray.dir;
  Vec2 const ds = _segment.end - _segment.start;

  Vec2 const u = _ray.pos - _segment.start;

  // Might be negative!
  float const scale = (dr.m_y * ds.m_x - dr.m_x * ds.m_y);

  float const r = (u.m_x * ds.m_y - u.m_y * ds.m_x) / scale;
  float const s = (u.m_x * dr.m_y - u.m_y * dr.m_x) / scale;

  o_result->col = (0.f < r) & (0.f < s) & (s < 1.f);
  o_result->dist = r;
}

inline void TestRaysSegment(int const _n, Ray2D const* const _ray, Segment2D const& _segment, RayCastResult* const o_results)
{
  Vec2 const ds = _segment.end - _segment.start;
  
  for (int i = 0; i < _n; ++i)
  {
    Vec2 const dr = _ray[i].dir;

    Vec2 const u = _ray[i].pos - _segment.start;

    // Might be negative!
    float const scale = (dr.m_y * ds.m_x - dr.m_x * ds.m_y);

    float const r = (u.m_x * ds.m_y - u.m_y * ds.m_x) / scale;
    float const s = (u.m_x * dr.m_y - u.m_y * dr.m_x) / scale;

    o_results[i].col = (0.f < r) & (0.f < s) & (s < 1.f);
    o_results[i].dist = r;
  }
}

inline void TestRayCircle(Ray2D const& _ray, Circle2D const& _circle, RayCastResult* o_result)
{
  // Algo taken from http://stackoverflow.com/questions/1073336/circle-line-collision-detection
  Vec2 d = _ray.dir;
  Vec2 f = _ray.pos - _circle.center;
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
  Vec2 colNormal;
};

inline void TestCircleSegment(Circle2D const& _circle, Segment2D const& _segment, CircleTestResult* const o_result)
{
  // HACK: assuming infinite walls (lines) instead of segments
  // TODO: check whether closest point falls within segment; if not,
  // check distance to each end of the segment (or just the end the closest point is on the side of)

  Vec2 const segVec = _segment.end - _segment.start;
  Vec2 const segDir = segVec / segVec.GetLength();
  Vec2 p = _circle.center - _segment.start;
  float r = fabsf(segDir.m_x * p.m_y - p.m_x * segDir.m_y);

  o_result->dist = r;
  o_result->col = (r < _circle.radius);
  o_result->colNormal = Vec2(-segDir.m_y, segDir.m_x);
}

inline void TestCircleCircle(Circle2D const& _c1, Circle2D const& _c2, CircleTestResult* const o_result)
{
  Vec2 colVec = _c2.center - _c1.center;
  float dist = colVec.GetLength();

  
  o_result->dist = dist - _c2.radius;
  o_result->col = (dist < (_c1.radius + _c2.radius));
  o_result->colNormal = colVec / colVec.GetLength();
}

#endif	/* GEOM_H */

