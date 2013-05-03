#pragma once

#include "orStd.h"
#include "orMath.h"
#include "orGfx.h"

#include <vector>

namespace sf { class RenderWindow; }

class RenderSystem {
public:
  struct Point {
    Vector3d m_pos;

    Vector3f m_col;
  };

  int numPoints() const { return m_points.size(); }
  int makePoint() { m_points.push_back(Point()); return numPoints() - 1; }
  Point&       getPoint(int i)       { return m_points[i]; }
  Point const& getPoint(int i) const { return m_points[i]; }

  struct Label {
    Vector3d m_pos;
    std::string m_text;

    Vector3f m_col;
  };

  // TODO this is currently implemented as 2D text, want 3D labels too
  int numLabels() const { return m_labels.size(); }
  int makeLabel() { m_labels.push_back(Label()); return numLabels() - 1; }
  Label&       getLabel(int i)       { ensure(0 < i && i < numLabels()); return m_labels[i]; }
  Label const& getLabel(int i) const { ensure(0 < i && i < numLabels()); return m_labels[i]; }

  struct Sphere {
    double m_radius;
    Vector3d m_pos;

    Vector3f m_col;
  };

  int numSpheres() const { return (int)m_spheres.size(); }
  int makeSphere() { m_spheres.push_back(Sphere()); return numSpheres() - 1; }
  Sphere&       getSphere(int i)       { return m_spheres[i]; }
  Sphere const& getSphere(int i) const { return m_spheres[i]; }

  struct Orbit {
    double p;
    double e;
    double theta;
    Vector3d x_dir;
    Vector3d y_dir;

    Vector3d m_pos;

    Vector3f m_col;
  };

  int numOrbits() const { return (int)m_orbits.size(); }
  int makeOrbit() { m_orbits.push_back(Orbit()); return numOrbits() - 1; }
  Orbit&       getOrbit(int i)       { return m_orbits[i]; }
  Orbit const& getOrbit(int i) const { return m_orbits[i]; }

  struct Trail
  {
    // TODO make into methods on RenderSystem instead
    Trail(double const _duration, Vector3d const _initPos, Vector3d const _initOrigin);
    void Update(double const _dt, Vector3d const _pos);

    // TODO this stores a fixed number of frames, not the best approach
    enum { NUM_TRAIL_PTS = 32 };
    double m_duration; // TODO not obeyed at the moment; effective duration is NUM_TRAIL_PTS * minAge

    int m_headIdx;
    Vector3d m_trailPts[NUM_TRAIL_PTS];
    double m_trailPointAge[NUM_TRAIL_PTS];

    Vector3d m_HACKorigin;

    Vector3f m_colOld;
    Vector3f m_colNew;
  };

  int numTrails() const { return (int)m_trails.size(); }
  int makeTrail( double const _duration, Vector3d const _initPos, Vector3d const _origin ) { m_trails.push_back(Trail(_duration, _initPos, _origin)); return numTrails() - 1; }
  Trail&       getTrail(int i)       { return m_trails[i]; }
  Trail const& getTrail(int i) const { return m_trails[i]; }

  void render2D(sf::RenderWindow* window);
  void render3D(sf::RenderWindow* window);

private:
  void setDrawColour(Vector3f const& _c) const;
  void setDrawColour(Vector3d const& _c) const;

  void drawCircle(double const radius, int const steps) const;
  void drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;

  void renderLabels(sf::RenderWindow* window) const;

  void renderPoints() const;
  void renderSpheres() const;
  void renderOrbits() const;
  void renderTrails() const;

private:
  std::vector<Point> m_points;
  std::vector<Label> m_labels;
  std::vector<Sphere> m_spheres;
  std::vector<Orbit> m_orbits;
  std::vector<Trail> m_trails;
}; // class RenderSystem
