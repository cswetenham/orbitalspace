#include "simviz.h"

#include "util.h"
#include "timer.h"
#include "perftimer.h"

#include <math.h>

SimViz::World::World() :
#if SSRM_SECTION == 2
  m_width(600.f),
  m_height(125.f),
#else
  m_width(600.f),
  m_height(400.f),
#endif
#if SSRM_SECTION == 3
  m_IRSensorNoiseDistrib(NormalDistribution(0, sqrtf(3.f)))
#else
  m_IRSensorNoiseDistrib(NormalDistribution(0, 0))
#endif
{
  // TODO this is a bad place for all this initialisation...
  
  for (int i = 0; i < NUM_SENSORS; ++i)
  {
    float theta = i * 2.f * (float)M_PI / NUM_SENSORS;
    m_sensorPose[i].dir = theta;
    m_sensorPose[i].pos = Vec2(10.f, 0.f).RotatedBy(theta);
  }
   
  m_wall[0].start = Vec2(-.5f*m_width, -.5f*m_height);
  m_wall[0].end = Vec2(-.5f*m_width, .5f*m_height);
 
  m_wall[1].start = Vec2(-.5f*m_width, .5f*m_height);
  m_wall[1].end = Vec2(.5f*m_width, .5f*m_height);

  m_wall[2].start = Vec2(.5f*m_width, .5f*m_height);
  m_wall[2].end = Vec2(.5f*m_width, -.5f*m_height);
  
  m_wall[3].start = Vec2(.5f*m_width, -.5f*m_height);
  m_wall[3].end = Vec2(-.5f*m_width, -.5f*m_height);
}

SimViz::World::~World()
{
}

void SimViz::World::Update(Rnd64* const _rnd, float const _dt)
{
}

SimViz::RobotAgent::RobotAgent() : m_behaviour(Behaviour_Fwd)
{
  m_model.states.prevPoses = new Pose[1];
  m_model.states.currPoses = new Pose[1];
  
  m_model.readings.sensorDist = new float[World::NUM_SENSORS];
}

SimViz::RobotAgent::~RobotAgent()
{
  delete[] m_model.readings.sensorDist;
  
  delete[] m_model.states.prevPoses;
  delete[] m_model.states.currPoses;
}

void SimViz::RobotAgent::InitState(Rnd64* const _rnd, World const& _world)
{
  Kidnap(_rnd, _world);

  for (int i = 0; i < World::NUM_SENSORS; ++i)
  {
    m_model.readings.sensorDist[i] = 9999.f;
  }
}

void SimViz::RobotAgent::UpdateMotion(Rnd64* const _rnd, SimViz::Motion* const o_motion)
{
  BernoulliDistribution behaviourDist(.05f);
  bool change;
  behaviourDist.Generate(_rnd, 1, &change);
  if (change)
  {
    // TODO ugh
    float r[2];
    _rnd->gen_floats(2, &r[0]);
    if (m_behaviour == Behaviour_Fwd)
    {
      if (r[0] < 0.7f)
      {
        m_behaviour = Behaviour_Fwd;
      }
      else
      {
        m_behaviour = Behaviour_Turn;
      }
    }
    else
    {
      if (r[0] < 0.9f)
      {
        m_behaviour = Behaviour_Fwd;
      }
      else
      {
        m_behaviour = Behaviour_Turn;
      }
    }
  }

  SimViz::Motion newMotion;
  
  switch(m_behaviour)
  {
    case Behaviour_Fwd:
    {
      newMotion.fwdSpeed = 25.f; // cm/s
      break;
    }
    case Behaviour_Turn:
    {
      newMotion.angSpeed = 2.f * (float)M_PI * 100.f / 360.f; // rad/s
      break;
    }
  }
  
  *o_motion = newMotion;
}

void SimViz::RobotAgent::Update(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, SimViz::Motion const& _motion, float const _dt)
{
  m_model.Update(_rnd, _world, _otherBot, 1, _motion, _dt);
  
  if (m_model.readings.bumpSensor)
  {
    m_behaviour = Behaviour_Turn;
  }
}

void SimViz::RobotModel::Update(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, int const _n, SimViz::Motion const& _motion, float const _dt)
{
  states.Swap();
  
  {
    PERFTIMER("UpdatePoses");

    UpdatePoses(
            _rnd,
            _world,
            _n,
            &states.prevPoses[0],
            &states.currPoses[0],
            &readings.bumpSensor,
            _motion,
            _dt
    );
  }
  
  {
    PERFTIMER("UpdateSensors");  
  
    for (int i = 0; i < World::NUM_SENSORS; ++i)
    {
      UpdateIRSensors(_rnd, _world, _otherBot, _n, _world.m_sensorPose[i], &states.currPoses[0], &readings.sensorDist[i]);
    }
  }
}

void SimViz::RobotModel::UpdatePoses(Rnd64* const _rnd, World const& _world, int const _n, Pose const* const i_prevPoses, Pose* const o_currPoses, bool* const o_bump, SimViz::Motion const& _motion, float const _dt)
{
  int const paddedCount = Util::PadSize(_n, 2);
  float* const fwd = (float*)alloca(paddedCount * sizeof(float));
  float* const ang = (float*)alloca(paddedCount * sizeof(float));
  
  // Speeds
  float const fwdError = _motion.fwdSpeed * 0.2f;
  float const angError = _motion.angSpeed * 0.4f;
  NormalDistribution fwdMotionDistrib(_motion.fwdSpeed, fwdError);
  NormalDistribution angMotionDistrib(_motion.angSpeed, angError);
  fwdMotionDistrib.Generate(_rnd, _n, fwd);
  angMotionDistrib.Generate(_rnd, _n, ang);
  
  // Distances
  for (int i = 0; i < _n; ++i)
  {
    fwd[i] *= _dt;
    ang[i] *= _dt;
  }
  
  Pose::Update(_n, ang, fwd, i_prevPoses, o_currPoses);

  // Collision detection and resolution
  // HACK - this isn't the best place to be testing collisions
  // TODO - on collision, change behaviour to turning

  for (int i = 0; i < _n; ++i)
  {
    for (int j = 0; j < World::NUM_WALLS; ++j)
    {
      Wall w = _world.m_wall[j];
      // HACK: assuming infinite walls (lines) instead of segments
      Circle2D bot;
      bot.center = o_currPoses[i].pos;
      bot.radius = 25.f;

      CircleTestResult result;
      TestCircleSegment(bot, w, &result);

      *o_bump = result.col;

      if (result.col)
      {
        // Unit, perpendicular to wall
        // Move away from wall, with a bit of histeresis...might be bad though
        Vec2 new_pos = o_currPoses[i].pos + (result.dist - 26.f) * result.colNormal;
        o_currPoses[i].pos = new_pos;
      }
    }
  }
}

void SimViz::RobotModel::UpdateIRSensors(Rnd64* const _rnd, World const& _world, Vec2 _otherBot, int const _n, Pose const& _sensorPose, Pose const* const _poses, float* const o_dists)
{
  Ray2D* const ray = (Ray2D*)alloca(_n * sizeof(Ray2D));
  
  for (int i = 0; i < _n; ++i)
  {
    Pose sensorPose = _poses[i].WorldFromLocal(_sensorPose);
    ray[i].pos = sensorPose.pos;
    ray[i].dir = Vec2::UnitFromDir(sensorPose.dir);
  }
  
  for (int i = 0; i < _n; ++i)
  {
    o_dists[i] = 9999999999.f;
  }
  
  RayCastResult* const results = (RayCastResult*)alloca(_n * sizeof(RayCastResult));
  
  for (int j = 0; j < World::NUM_WALLS; ++j)
  {
    TestRaysSegment(_n, ray, _world.m_wall[j], results);
    
    for (int i = 0; i < _n; ++i)
    {
      if (results[i].col && results[i].dist < o_dists[i])
      {
        o_dists[i] = results[i].dist;
      }
    }
  }
  
  
  for (int i = 0; i < _n; ++i)
  {
    Circle2D otherBot;
    otherBot.center = _otherBot;
    otherBot.radius = 25.f;
    TestRayCircle(ray[0], otherBot, &results[i]);
    if (results[i].col && results[i].dist < o_dists[i])
    {
      o_dists[i] = results[i].dist;
    }
  }
  
  for (int i = 0; i < _n; ++i)
  {
    float error;
    _world.m_IRSensorNoiseDistrib.Generate(_rnd, 1, &error);
    // TODO no error or clamping for now
    o_dists[i] = Util::Clamp(o_dists[i] + error, 0.f, 999999.f);
  }
}

void SimViz::RobotAgent::Kidnap(Rnd64* const _rnd, World const& _world)
{
  float const w2 = .5f * _world.m_width;
  float const h2 = .5f * _world.m_height;
  
  UniformDistribution xDist(-w2, +w2);
  UniformDistribution yDist(-h2, +h2);
  UniformDistribution angDist(-(float)M_PI, +(float)M_PI);
  float xs;
  float ys;
  float ts;
  xDist.Generate(_rnd, 1, &xs);
  yDist.Generate(_rnd, 1, &ys);
  angDist.Generate(_rnd, 1, &ts);

  m_model.states.currPoses[0].pos = Vec2(xs, ys);
  m_model.states.currPoses[0].dir = ts;
  m_model.states.prevPoses[0] = m_model.states.currPoses[0];
}


