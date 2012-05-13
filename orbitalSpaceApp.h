/* 
 * File:   ssrm1app.h
 * Author: s1149322
 *
 * Created on 12 December 2011, 17:11
 */

#ifndef SSRM1APP_H
#define	SSRM1APP_H

#include "simviz.h"
#include "sv_particlefilter.h"
#include "app.h"

class OrbitalSpaceApp :
  public App
{
public:
  OrbitalSpaceApp();
  virtual ~OrbitalSpaceApp();
  
  // From App
public:
  virtual void Run();

protected:
  virtual void InitRender();
  virtual void ShutdownRender();

  virtual void InitState();
  virtual void ShutdownState();

  virtual void HandleEvent(sf::Event const& _event);
  virtual void UpdateState(float const _dt);
  
  virtual void RenderState();

private:
  void DrawWireSphere(float const radius, int const slices, int const stacks);

  void DrawRobots(int const _n, Pose const* const _poses, float const* const _weights, Vec3 const& _frontCol, Vec3 const& backCol);
  
  void DrawSensors(int const _n, Pose const* const _poses, float const* const _weights, Pose const* const _sensorPose, Vec3 const& _lineCol);

  void DrawWalls(int const _n, SimViz::Wall const* const _walls);

  void UpdateContacts(int const _n, Pose const& _sensorPose, Pose const* const _poses, float const* const _dists, Vec2* const o_points);

  void DrawPoints(int const _n, Vec2 const* const _points, float const* const _weights, Vec3 const _col);
  void DrawTrail(int const _n, Pose const* const _poses, Vec3 const _col);

private:
  inline float AlphaFromProb(float const _p);
  inline static void SetDrawColour(Vec3 const _c);

private:
  Rnd64 m_rnd;
  
  float m_simTime;

  struct Config
  {
    int width;
    int height;
  };
  
  Config m_config;

  // Simulation options
  bool m_paused;
  bool m_singleStep;
  // Rendering options
  bool m_useWeights;
  bool m_wireframe;
     
  SimViz::RobotAgent m_alice;
#if SSRM_SECTION == 4
  SimViz::RobotAgent m_bob;
#endif
  SimViz::World m_world;

  float m_camZ;
  float m_camTheta;

  enum { NUM_STEPS = 30000 };
  enum { STEP_SIZE_MS = 32 }; // Duration of each step in simulated time

  int m_curStep;
  
  Pose m_poseHist[NUM_STEPS];
  float m_sensorHist[NUM_STEPS][SimViz::World::NUM_SENSORS];

  Vec2 m_sensorContact[SimViz::World::NUM_SENSORS];
};

#endif	/* SSRM1APP_H */

