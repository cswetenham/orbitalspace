/* 
 * File:   pose.h
 * Author: s1149322
 *
 * Created on 07 December 2011, 19:30
 */

#ifndef POSE_H
#define	POSE_H

#include "vector.h"

struct Pose
{
  Vec2 pos;
  float dir;
  
  Vec2 WorldFromLocal(Vec2 const _p) const
  {
    return _p.RotatedBy(dir) + pos;
  }
  
  Pose WorldFromLocal(Pose const _p) const
  {
    Pose res;
    res.pos = pos + _p.pos.RotatedBy(dir);
    res.dir = dir + _p.dir;
    return res;
  }
  
  void Update(float const _turn, float const _distance)
  {
    dir = Util::Wrap(dir + _turn, -.5f * M_TAU, +.5f * M_TAU);
    pos += Vec2::FromDirLen(dir, _distance);
  }
  
  static void Update(int const _n, float const* const _turns, float const* const _distances, Pose const* const i_prevPoses, Pose* const o_currPoses)
  {
    for (int i = 0; i < _n; ++i)
    {
      float const newDir = Util::Wrap(i_prevPoses[i].dir + _turns[i], 0.f, M_TAU);
      o_currPoses[i].dir = newDir;
      o_currPoses[i].pos = i_prevPoses[i].pos + Vec2::FromDirLen(newDir, _distances[i]);
    }
  }
};

#endif	/* POSE_H */

