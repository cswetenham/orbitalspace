#ifndef _VEC2_H
#define _VEC2_H

#include <math.h>

/*
class Vec1; class Vec1H;
class Vec2; class Vec2H;
class Vec3; class Vec3H;
class Vec4;
*/
// template <typename T> class Vec1T<T>;
// template <int N> class VecN<N>; template <int N> class VecNH<N>;
// VecNf is specialisation of VecN<T>
// Can specialise the template and then typedef it. Only disadvantage is that
// you can't forward-declare the typedef, but inheriting from the specialisation
// would be worse.
// Vec3f is specialisation of VecNf<3>
// Vectors are specialisations of matrices...
// Matrices can have all the above options...
// Could have column and row vectors
// TODO - Vec2, Pos2, Dir2(Unit2)? Mat22, Rot2, Orientation2...Normal2?
// Learning Modern 3D Graphics Programming calls delta vectors direction vectors...
// But to me direction would be a unit vector
// "Displacement Vector" seems to be more common
// Displacement2? Or Translation2.
// + Homogeneous versions: Vec3H -> xyzw
// In general:
// structure types: T array, float array, T fields, float fields, SSE field.
// base math/algebraic type (angle, vector, matrix)
// position/object type (point, heading/orientation, pose)
// difference/transformation type (rotation, translation/vector, transformation matrix)
// special types: normal vector
// constrained types: unit vector, diagonal matrix
// compound geometric objects: line, plane, etc.
// Can't add or scale positions per se but can take a weighted sum as long as the weights sum to 1...


/* Simple 2D vector type */
class Vec2
{
public:
  float m_x;
  float m_y;

  Vec2() : m_x(0.f), m_y(0.f) {}
  Vec2(float const _x, float const _y) : m_x(_x), m_y(_y) {}
  Vec2(Vec2 const& _v) : m_x(_v.m_x), m_y(_v.m_y) {}

  Vec2 const& operator=(Vec2 const& _v) { m_x = _v.m_x; m_y = _v.m_y; return *this; }

  static Vec2 XAxis() { return Vec2(1.f, 0.f); }
  static Vec2 YAxis() { return Vec2(0.f, 1.f); }

  float GetLengthSquared() const
  {
    return m_x * m_x + m_y * m_y;
  }
  
  float GetLength() const
  {
    return sqrt(GetLengthSquared());
  }
  
  float GetDir() const 
  { 
      return atan2(m_y, m_x); 
  }
  
  Vec2& operator+=(Vec2 const& _r);
  Vec2& operator-=(Vec2 const& _r);
  Vec2& operator*=(float const _s);
  Vec2& operator/=(float const _s);
  
  inline static Vec2 UnitFromDir(float const _dir);
  inline static Vec2 FromDirLen(float const _dir, float const _len );
  
  Vec2 RotatedBy(float const _theta) const
  {
    return Vec2(
        m_x * cos(_theta) - m_y * sin(_theta),
        m_x * sin(_theta) + m_y * cos(_theta)
    );
  }

  float Dot(Vec2 const& _r) const
  {
    return m_x * _r.m_x + m_y * _r.m_y;
  }
};

inline Vec2 operator-(Vec2 const& _v)
{
  return Vec2(-_v.m_x, -_v.m_y);
}

inline Vec2 operator+(Vec2 const& _v)
{
  return _v;
}

inline Vec2 operator+(Vec2 const& _l, Vec2 const& _r)
{
  return Vec2(_l.m_x + _r.m_x, _l.m_y + _r.m_y);
}

inline Vec2 operator-(Vec2 const& _l, Vec2 const& _r)
{
  return Vec2(_l.m_x - _r.m_x, _l.m_y - _r.m_y);
}

inline Vec2 operator*(float const _s, Vec2 const& _v)
{
  return Vec2(_v.m_x * _s, _v.m_y * _s);
}

inline Vec2 operator*(Vec2 const& _v, float const _s)
{
  return _s * _v;
}

inline Vec2 operator/(Vec2 const& _v, float const _s)
{
  return (1.f / _s) * _v;
}

inline Vec2& Vec2::operator+=(Vec2 const& _r)
{
  *this = *this + _r;
  return *this;
}

inline Vec2& Vec2::operator-=(Vec2 const& _r)
{
  *this = *this - _r;
  return *this;
}

inline Vec2& Vec2::operator*=(float const _s)
{
  *this = *this * _s;
  return *this;
}

inline Vec2& Vec2::operator/=(float const _s)
{
  *this = *this / _s;
  return *this;
}

Vec2 Vec2::UnitFromDir(float const _dir) 
{ 
    return Vec2(cos(_dir), sin(_dir)); 
}

Vec2 Vec2::FromDirLen(float const _dir, float const _len ) 
{ 
    return _len * UnitFromDir(_dir); 
}

/* Simple 3D vector type */
class Vec3
{
public:
  float m_x;
  float m_y;
  float m_z;

  Vec3() : m_x(0.f), m_y(0.f), m_z(0.f) {}
  Vec3(float const _x, float const _y, float const _z) : m_x(_x), m_y(_y), m_z(_z) {}
  Vec3(Vec3 const& _v) : m_x(_v.m_x), m_y(_v.m_y), m_z(_v.m_z) {}

  Vec3 const& operator=(Vec3 const& _v) { m_x = _v.m_x; m_y = _v.m_y; m_z = _v.m_z; return *this; }

  static Vec3 XAxis() { return Vec3(1.f, 0.f, 0.f); }
  static Vec3 YAxis() { return Vec3(0.f, 1.f, 0.f); }
  static Vec3 ZAxis() { return Vec3(0.f, 0.f, 1.f); }

  float GetLengthSquared() const
  {
    return m_x * m_x + m_y * m_y + m_z * m_z;
  }
  
  float GetLength() const
  {
    return sqrt(GetLengthSquared());
  }
  
  Vec3& operator+=(Vec3 const& _r);
  Vec3& operator-=(Vec3 const& _r);
  Vec3& operator*=(float const _s);
  Vec3& operator/=(float const _s);
};

inline Vec3 operator-(Vec3 const& _v)
{
  return Vec3(-_v.m_x, -_v.m_y, -_v.m_z);
}

inline Vec3 operator+(Vec3 const& _v)
{
  return _v;
}

inline Vec3 operator+(Vec3 const& _l, Vec3 const& _r)
{
  return Vec3(_l.m_x + _r.m_x, _l.m_y + _r.m_y, _l.m_z + _r.m_z);
}

inline Vec3 operator-(Vec3 const& _l, Vec3 const& _r)
{
  return Vec3(_l.m_x - _r.m_x, _l.m_y - _r.m_y, _l.m_z - _r.m_z);
}

inline Vec3 operator*(float const _s, Vec3 const& _v)
{
  return Vec3(_v.m_x * _s, _v.m_y * _s, _v.m_z * _s);
}

inline Vec3 operator*(Vec3 const& _v, float const _s)
{
  return _s * _v;
}

inline Vec3 operator/(Vec3 const& _v, float const _s)
{
  return (1.f / _s) * _v;
}

inline Vec3& Vec3::operator+=(Vec3 const& _r)
{
  *this = *this + _r;
  return *this;
}

inline Vec3& Vec3::operator-=(Vec3 const& _r)
{
  *this = *this - _r;
  return *this;
}

inline Vec3& Vec3::operator*=(float const _s)
{
  *this = *this * _s;
  return *this;
}

inline Vec3& Vec3::operator/=(float const _s)
{
  *this = *this / _s;
  return *this;
}

#endif // _VEC2_H
