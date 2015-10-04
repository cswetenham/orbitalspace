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

    orEphemerisHybrid m_params;

    orVec3 m_pos;

    orVec3 m_col;
  };

  DECLARE_SYSTEM_TYPE(Orbit, Orbits);
  
  // TODO move all code to old_style/ and new_style/ folders; new style is
  // structs in headers, then headers/cpps for different operations on data,
  // with everything namespaced, one folder per system.
  
  // TODO as an exercise, is there a nice way of doing this *without* callbacks?
  // NOTE yes:
  
  // Client code would allocate event ids from GUI system (we could instead allow an
  // arbitrary uint32_t, but that would couple different systems using GUI since
  // they would have to disambiguate IDs. Can always associate arbitrary data to
  // an event on the user side.)
  
  // Client code would request association of event ids to certain events for
  // certain UI elements:
  // - selectable item selected
  // - toggleable item toggled
  // - slider item changed
  // - text item edited
  
  // GUI system would produce an event stream each frame which could just be
  // a stream of opaque IDs, perhaps with some associated value for toggles and
  // sliders. Client code would parse stream for event ids it is interested in.
  
  // Something like hiding/showing nested menus could in theory be handled
  // entirely by client code but it feels more like a concern of GUI code;
  // client code just cares about actual user toggles etc.
  
   
  /*
  struct MenuEntry {
    std::string m_text;
    boost::function<void()> m_cb;
  };
  
  struct Menu {
    Menu() {}
    
    std::vector<MenuEntry> entries;
    int m_focused_entry_idx;
  };
  
  DECLARE_SYSTEM_TYPE(Menu, Menus);
  */
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

  void drawLine(Vector3d const start, Vector3d const end, Vector3d const col) const;
  void render2D(int w_px, int h_px, Eigen::Matrix4d const& screenFromWorld); // TODO not the best params...
  void render3D();

  void render(FrameBuffer const& frameBuffer, Colour clearCol, float clearDepth, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx); // TODO not the best params...

  void checkGLErrors();

private:
  void drawCircle(double const radius, int const steps) const;
  void drawSolidSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;
  void drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;
  void drawAxes(Vector3d const pos, double const size) const;

  void drawString(std::string const& str, int pos_x, int pos_y);

  void projectLabel3Ds(Eigen::Matrix4d const& screenFromWorld);

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
