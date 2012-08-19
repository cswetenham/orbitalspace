/* 
 * File:   ssrm1app.h
 * Author: s1149322
 *
 * Created on 12 December 2011, 17:11
 */

#ifndef SSRM1APP_H
#define	SSRM1APP_H

#include "app.h"
#include "rnd.h"
#include "util.h"
#include <Eigen/Eigen>
#include <Eigen/OpenGLSupport>

EIGEN_USING_MATRIX_TYPEDEFS;

#include <SFML/Audio/Music.hpp>

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
  virtual void UpdateState(double const _dt);
  
  virtual void RenderState();

private:
  struct PhysicsBody;
  struct OrbitParams;
  void ComputeKeplerParams(PhysicsBody const& body, OrbitParams& o_params);
  void LookAt(Vector3d pos, Vector3d target, Vector3d up);

  void DrawCircle(double const radius, int const steps);
  void DrawWireSphere(double const radius, int const slices, int const stacks);

private:
  enum { NUM_BODIES = 1 };
  enum { NUM_SHIPS = 2 };

  Rnd64 m_rnd;
  
  double m_simTime;

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
  bool m_wireframe;
  bool m_camOrig;

  double m_camDist;
  double m_camTheta;
  double m_camPhi;
   
  struct Body
  {
    Vector3d m_pos;
  };

  struct MassiveBody : public Body
  {
    double m_radius;
    double m_mass;
  };

  struct PhysicsBody : public Body
  {
    Vector3d m_vel;
  };

  Body* m_camTarget;
  size_t m_camTargetIdx;
  std::string m_camTargetName[NUM_SHIPS+1];

  MassiveBody m_earthBody;

  enum Thrusters
  {
    ThrustFwd = 1 << 0,
    ThrustBack = 1 << 1,
    ThrustLeft = 1 << 2,
    ThrustRight = 1 << 3,
    ThrustUp = 1 << 4,
    ThrustDown = 1 << 5
  };

  uint32_t m_thrusters;

  struct OrbitParams
  {
    double p;
    double e;
    double theta;
    Vector3d x_dir;
    Vector3d y_dir;
  };

  struct Trail
  {
    Trail(double const _duration) :
      m_duration(_duration), // TODO make sure this works as a value!
      m_timeSinceUpdate(0.f),    
      m_headIdx(0),
      m_tailIdx(0)
    {
      for (int i = 0; i < NUM_TRAIL_PTS; ++i)
      {
        m_trailPts[i] = Vector3d::Zero();
      }
    }

    void Update(double const _dt, Vector3d _pos)
    {
      // TODO can have a list of past points and their durations, and cut up trail linearly

      // A -- 50ms -- B -- 10 ms -- C

      // So if we get several 10ms updates we would interpolate A towards B a proportional amount, then finally remove it.

      m_timeSinceUpdate += _dt;
      
      if (false) { // m_timeSinceUpdate < TODO) { // duration / NUM_TRAIL_PTS? Idea is to ensure queue always has space. This means we are ensuring a minimum time duration for each segment.
        // Not enough time elapsed. To avoid filling up trail, update the head point instead of adding a new one
        // m_trailPts[m_headIdx] = _pos;
        // m_trailDuration[m_headIdx] = 0.f;
      }
      
      m_headIdx++;
      if (m_headIdx >= NUM_TRAIL_PTS) { m_headIdx = 0; }
      m_trailPts[m_headIdx] = _pos;
      if (m_tailIdx == m_headIdx) { m_tailIdx++; }
      
      // assert(m_headIdx != m_tailIdx);
    }

    void Render(Vector3d const& _col0, Vector3d const& _col1)
    {
      glBegin(GL_LINE_STRIP);
      int prevIdx = 0;
      // TODO render only to m_tailIdx
      for (int i = 0; i < Trail::NUM_TRAIL_PTS; ++i)
      {
        int idx = m_headIdx + i - Trail::NUM_TRAIL_PTS + 1;
        if (idx < 0) { idx += Trail::NUM_TRAIL_PTS; }
        Vector3d v = m_trailPts[idx];

        double const l = (double)i / Trail::NUM_TRAIL_PTS;
        Vector3d cd = Util::Lerp(_col0, _col1, l);
        Vector3f c = cd.cast<float>();
        Util::SetDrawColour(c);

        glVertex3d(v.x(),v.y(),v.z());

        prevIdx = idx;
      }
      glEnd();
    }

    // TODO this stores a fixed number of frames, not the best approach
    // On the other hand the break in the ellipse is a good way of seeing ship location for now
    enum { NUM_TRAIL_PTS = 1000 };
    double m_duration;
    double m_timeSinceUpdate;

    int m_headIdx;
    int m_tailIdx;
    Vector3d m_trailPts[NUM_TRAIL_PTS];
    double m_trailDuration[NUM_TRAIL_PTS];
  };

  struct Ship
  {
    Ship();

    PhysicsBody m_physics;
    OrbitParams m_orbit;
    Trail m_trail;
  };
  
  Ship m_ships[NUM_SHIPS];
  
  // TODO make into a palette array.
  // TODO Convert to HSV so can modify the hue to make new palettes.
  enum {NUM_COLS = 5};
  Vector3d m_colG[NUM_COLS];
  Vector3d m_colR[NUM_COLS];
  Vector3d m_colB[NUM_COLS];
  
  Vector3d m_light;

  bool m_hasFocus;

  sf::Music m_music;

  double m_timeScale;
};

#endif	/* SSRM1APP_H */

