#ifndef ORRENDER_H
#define	ORRENDER_H

# include "orStd.h"
# include "orMath.h"
# include "orGfx.h"

# include <vector>

class RenderSystem {
public:
  struct Point {
    Vector3d m_pos;

    Vector3f m_col;
  };
    
  int makePoint() { m_renderablePoints.push_back(Point()); return m_renderablePoints.size() - 1; }
  Point& getPoint(int i) { return m_renderablePoints[i]; }
  Point const& getPoint(int i) const { return m_renderablePoints[i]; }

  struct Sphere {
    double m_radius;
    Vector3d m_pos;

    Vector3f m_col;
  };
    
  int makeSphere() { m_renderableSpheres.push_back(Sphere()); return m_renderableSpheres.size() - 1; }
  Sphere& getSphere(int i) { return m_renderableSpheres[i]; }
  Sphere const& getSphere(int i) const { return m_renderableSpheres[i]; }

  struct Orbit
  {
    double p;
    double e;
    double theta;
    Vector3d x_dir;
    Vector3d y_dir;
    Vector3d m_pos;

    Vector3f m_col;
  };
    
  int makeOrbit() { m_renderableOrbits.push_back(Orbit()); return m_renderableOrbits.size() - 1; }
  Orbit& getOrbit(int i) { return m_renderableOrbits[i]; }
  Orbit const& getOrbit(int i) const { return m_renderableOrbits[i]; }

  struct Trail
  {
    // TODO make into method on app instead
    Trail(double const _duration);
    void Update(double const _dt, Vector3d _pos);
    void Render() const;

    // TODO this stores a fixed number of frames, not the best approach
    enum { NUM_TRAIL_PTS = 1000 };
    double m_duration;
    double m_timeSinceUpdate;

    int m_headId;
    int m_tailId;
    Vector3d m_trailPts[NUM_TRAIL_PTS];
    double m_trailDuration[NUM_TRAIL_PTS];

    Vector3f m_colOld;
    Vector3f m_colNew;
  };
    
  int makeTrail() { m_renderableTrails.push_back(Trail(3.0)); return m_renderableTrails.size() - 1; }
  Trail& getTrail(int i) { return m_renderableTrails[i]; }
  Trail const& getTrail(int i) const { return m_renderableTrails[i]; }

  void render();

private:
  void setDrawColour(Vector3f const& _c) const;
  void setDrawColour(Vector3d const& _c) const;

  void drawCircle(double const radius, int const steps) const;
  void drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const;

  void renderPoints() const;
  void renderSpheres() const;
  void renderOrbits() const;
  void renderTrails() const;

private:
  std::vector<Point> m_renderablePoints;
  std::vector<Sphere> m_renderableSpheres;
  std::vector<Orbit> m_renderableOrbits;
  std::vector<Trail> m_renderableTrails;
};

#endif // ORRENDER_H