/* 
 * File:   ssrm1app.h
 * Author: s1149322
 *
 * Created on 12 December 2011, 17:11
 */

#ifndef SSRM1APP_H
#define	SSRM1APP_H

#include "app.h"
#include "vector.h"
#include "rnd.h"
#include "util.h"

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
  bool m_wireframe;
     
  float m_camZ;
  float m_camTheta;
  float m_camPhi;

  Vec3 m_shipPos;
  Vec3 m_shipVel;

  enum { NUM_TRAIL_PTS = 1000 };
  Vec3 m_trailPts[NUM_TRAIL_PTS];
  int m_trailIdx;

  Vec3 m_col1;
  Vec3 m_col2;
  Vec3 m_col3;
  Vec3 m_col4;
  Vec3 m_col5;
};

#endif	/* SSRM1APP_H */

