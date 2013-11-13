#pragma once

#include "orStd.h"
#include "orMath.h"
#include "orGfx.h"

#include <vector>

namespace sf { class RenderWindow; class Font; class Image; }

class RenderSystem {
public:
  RenderSystem();
  ~RenderSystem();

  void initRender();
  void shutdownRender();

  struct Point {
    double m_pos[3];

    float m_col[3];
  };

  int numPoints() const { return m_points.size(); }
  int makePoint() { m_points.push_back(Point()); return numPoints() - 1; }
  Point&       getPoint(int i)       { return m_points[i]; }
  Point const& getPoint(int i) const { return m_points[i]; }


  struct Label2D {
    std::string m_text;

    int m_pos[2];

    float m_col[3];
  };

  int numLabel2Ds() const { return m_label2Ds.size(); }
  int makeLabel2D() { m_label2Ds.push_back(Label2D()); return numLabel2Ds() - 1; }
  Label2D&       getLabel2D(int i)       { ensure(0 <= i && i < numLabel2Ds()); return m_label2Ds[i]; }
  Label2D const& getLabel2D(int i) const { ensure(0 <= i && i < numLabel2Ds()); return m_label2Ds[i]; }

  struct Label3D {
    std::string m_text;

    double m_pos[3];

    float m_col[3];
  };

  int numLabel3Ds() const { return m_label3Ds.size(); }
  int makeLabel3D() { m_label3Ds.push_back(Label3D()); return numLabel3Ds() - 1; }
  Label3D&       getLabel3D(int i)       { ensure(0 <= i && i < numLabel3Ds()); return m_label3Ds[i]; }
  Label3D const& getLabel3D(int i) const { ensure(0 <= i && i < numLabel3Ds()); return m_label3Ds[i]; }

  struct Sphere {
    double m_radius;
    double m_pos[3];

    float m_col[3];
  };

  int numSpheres() const { return (int)m_spheres.size(); }
  int makeSphere() { m_spheres.push_back(Sphere()); return numSpheres() - 1; }
  Sphere&       getSphere(int i)       { return m_spheres[i]; }
  Sphere const& getSphere(int i) const { return m_spheres[i]; }

  struct Orbit {
    double p;
    double e;
    double theta;
    double x_dir[3];
    double y_dir[3];

    double m_pos[3];

    float m_col[3];
  };

  int numOrbits() const { return (int)m_orbits.size(); }
  int makeOrbit() { m_orbits.push_back(Orbit()); return numOrbits() - 1; }
  Orbit&       getOrbit(int i)       { return m_orbits[i]; }
  Orbit const& getOrbit(int i) const { return m_orbits[i]; }

  struct Trail
  {
    Trail(double const _duration, const double _initPos[3], const double _initOrigin[3]);

    void Update(double const _dt, Vector3d const _pos);

    // TODO this stores a fixed number of frames, not the best approach
    enum { NUM_TRAIL_PTS = 32 };
    double m_duration; // TODO not obeyed at the moment; effective duration is NUM_TRAIL_PTS * minAge

    int m_headIdx;
    double m_trailPts[NUM_TRAIL_PTS*3];
    double m_trailPointAge[NUM_TRAIL_PTS];

    double m_HACKorigin[3];

    float m_colOld[3];
    float m_colNew[3];
  };

  int numTrails() const { return (int)m_trails.size(); }
  int makeTrail( double const _duration, const double _initPos[3], const double _origin[3] ) { m_trails.push_back(Trail(_duration, _initPos, _origin)); return numTrails() - 1; }
  Trail&       getTrail(int i)       { return m_trails[i]; }
  Trail const& getTrail(int i) const { return m_trails[i]; }

  void beginRender() { m_label2DBuffer.clear(); }
  void endRender() {}

  void render2D(int w_px, int h_px, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx); // TODO not the best params...
  void render3D();

private:
  void drawCircle(double const radius, int const steps) const;
  void drawSolidSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;
  void drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;

  void drawString(std::string const& str, int pos_x, int pos_y);

  void projectLabel3Ds(Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx);

  void renderPoints() const;
  void renderLabels( int w_px, int h_px );
  void renderSpheres() const;
  void renderOrbits() const;
  void renderTrails() const;

private:
  std::vector<Point> m_points;
  std::vector<Label2D> m_label2Ds;
  std::vector<Label3D> m_label3Ds;

  // 2D labels for this frame
  std::vector<Label2D> m_label2DBuffer;

  std::vector<Sphere> m_spheres;
  std::vector<Orbit> m_orbits;
  std::vector<Trail> m_trails;

  uint32_t m_fontTextureId;
  sf::Image* m_fontImage;
}; // class RenderSystem
