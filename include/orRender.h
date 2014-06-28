#pragma once

#include "orStd.h"
#include "orMath.h"
#include "orGfx.h"
#include "orCore/orSystem.h"

#include <vector>

struct SDL_Surface;

class RenderSystem {
public:
  RenderSystem();
  ~RenderSystem();

  void initRender();
  void shutdownRender();

  typedef Eigen::Vector3d Colour; // TODO maybe use different type?

  struct FrameBuffer {
    FrameBuffer() : width(0), height(0), frameBufferId(0), colorTextureId(0), depthBufferId(0) {}
    int width;
    int height;
    uint32_t frameBufferId;
    uint32_t colorTextureId;
    uint32_t depthBufferId;
  };

  FrameBuffer makeFrameBuffer(int width, int height);
  void freeFrameBuffer(FrameBuffer&) { /* TODO */ }

  struct Point {
    Point(): m_pos(), m_col() {}
    orVec3 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Point, Points);

  struct Label2D {
    Label2D() : m_text(), m_pos(), m_col() {}

    std::string m_text;

    orVec2 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Label2D, Label2Ds);

  struct Label3D {
    Label3D() : m_text(), m_pos(), m_col() {}

    std::string m_text;

    orVec3 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Label3D, Label3Ds);

  struct Sphere {
    Sphere() : m_radius(0), m_pos(), m_col() {}

    double m_radius;
    orVec3 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Sphere, Spheres);

  struct Orbit {
    Orbit() : m_params(), m_pos(), m_col() {}

    orOrbitParams m_params;

    orVec3 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Orbit, Orbits);

#if 0
  struct Trail
  {
    Trail() : m_duration(0), m_headIdx(0), m_HACKorigin(), m_colOld(), m_colNew() {}

    void Init(double const _duration, const orVec3 _initPos, const orVec3 _initOrigin);
    void Update(double const _dt, Vector3d const _pos);

    // TODO this stores a fixed number of frames, not the best approach
    enum { NUM_TRAIL_PTS = 32 };
    double m_duration; // TODO not obeyed at the moment; effective duration is NUM_TRAIL_PTS * minAge

    int m_headIdx;
    double m_trailPts[NUM_TRAIL_PTS*3];
    double m_trailPointAge[NUM_TRAIL_PTS];

    orVec3 m_HACKorigin;

    orVec3 m_colOld;
    orVec3 m_colNew;
  };

  DECLARE_SYSTEM_TYPE(Trail, Trails);
#endif

  void render2D(int w_px, int h_px, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx); // TODO not the best params...
  void render3D();

  void render(FrameBuffer const& frameBuffer, Colour clearCol, float clearDepth, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx); // TODO not the best params...

  void checkGLErrors();

private:
  void drawCircle(double const radius, int const steps) const;
  void drawSolidSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;
  void drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;
  void drawAxes(Vector3d const pos, double const size) const;

  void drawString(std::string const& str, int pos_x, int pos_y);

  void projectLabel3Ds(Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx);

  void renderPoints() const;
  void renderLabels( int w_px, int h_px );
  void renderSpheres() const;
  void renderOrbits() const;
#if 0
  void renderTrails() const;
#endif

private:
  // 2D labels for this frame
  std::vector<Label2D> m_label2DBuffer;

  uint32_t m_fontTextureId;

  SDL_Surface* m_fontImage;
}; // class RenderSystem
