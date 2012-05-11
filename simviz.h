/* 
 * File:   simviz.h
 * Author: s1149322
 *
 * Created on 08 December 2011, 13:57
 */

#ifndef SIMVIZ_H
#define	SIMVIZ_H

#include "vector.h"
#include "geom.h"

#include "rnd.h"
#include "pose.h"

#include <algorithm>

// HACK
#define SSRM_SECTION 4

namespace SimViz {
  
typedef Segment2D Wall;

struct Motion
{
  Motion() : fwdSpeed(0.f), angSpeed(0.f) {}
  float fwdSpeed;
  float angSpeed;
};

// State shared between all particles, and simulated robots if there are any.
class World
{
public:
  World();
  ~World();
  void Update(Rnd64* const _rnd, float const _dt);
  
  // Sensor poses - the same between all particles
  enum { NUM_SENSORS = 25 };
  Pose m_sensorPose[NUM_SENSORS];

  NormalDistribution m_IRSensorNoiseDistrib;

  // Map
  float m_width;
  float m_height;
  
  enum { NUM_WALLS = 4 };

  Wall m_wall[NUM_WALLS];
};

struct RobotStates
{
  Pose* prevPoses;
  Pose* currPoses;

  void Swap() { std::swap(prevPoses, currPoses); }
};

struct SensorValues
{
  bool bumpSensor;
  float* sensorDist;
};

class RobotModel
{
public:
  RobotStates states;
  SensorValues readings;
  
  void Update(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, int const _n, SimViz::Motion const& _motion, float const _dt);

  void UpdatePoses(Rnd64* const _rnd, World const& _world, int const _n, Pose const* const i_prevPoses, Pose* const o_currPoses, bool* const o_bump, SimViz::Motion const& _motion, float const _dt);
  void UpdateIRSensors(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, int const _n, Pose const& _sensorPose, Pose const* const _poses, float* const o_dists);
};

class RobotAgent
{
public:
  RobotAgent();
  ~RobotAgent();
    
  void InitState(Rnd64* const _rnd, World const& _world);
  void UpdateMotion(Rnd64* const _rnd, SimViz::Motion* const o_motion);
  
  void Update(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, SimViz::Motion const& _motion, float const _dt);
  
  void Kidnap(Rnd64* const _rnd, World const& _world);
  
  enum Behaviour { Behaviour_Fwd, Behaviour_Turn };
  Behaviour m_behaviour;

  RobotModel m_model;
};

} // namespace SimViz

#endif	/* SIMVIZ_H */

