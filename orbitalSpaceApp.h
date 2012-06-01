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
  void DrawCircle(float const radius, int const steps);
  void DrawWireSphere(float const radius, int const slices, int const stacks);

private:
  inline static void SetDrawColour(Vector3f const& _c);

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
  bool m_wireframe;
     
  float m_camZ;
  float m_camTheta;
  float m_camPhi;

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

  struct Ship
  {
    Ship();

    Vector3f m_pos;
    Vector3f m_vel;

    // TODO this stores a fixed number of frames, not the best approach
    // On the other hand the break in the ellipse is a good way of seeing ship location for now
    enum { NUM_TRAIL_PTS = 1000 };
    Vector3f m_trailPts[NUM_TRAIL_PTS];
    int m_trailIdx;

    struct OrbitParams
    {
      float p;
      float e;
      float theta;
      Vector3f x_dir;
      Vector3f y_dir;
    };

    OrbitParams m_orbit;
  };
  
  enum { NUM_SHIPS = 1 };
  Ship m_ships[NUM_SHIPS];

  Vector3f m_col1;
  Vector3f m_col2;
  Vector3f m_col3;
  Vector3f m_col4;
  Vector3f m_col5;

  Vector3f m_light;
};

#endif	/* SSRM1APP_H */

