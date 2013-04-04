#include "orGfx.h"
#include "util.h"

#include "orRender.h"


void RenderSystem::setDrawColour(Vector3f const& _c) const
{
  glColor3f(_c.x(), _c.y(), _c.z());
}

void RenderSystem::setDrawColour(Vector3d const& _c) const
{
  glColor3d(_c.x(), _c.y(), _c.z());
}

void RenderSystem::drawCircle(double const radius, int const steps) const
{
    /* Adjust z and radius as stacks and slices are drawn. */

    double x,y;

    double const stepInc = M_TAU / steps;
    
    /* Draw a line loop for each stack */
    glBegin(GL_LINE_LOOP);
    for (int curStep = 0; curStep < steps; curStep++) {
        x = cos( curStep * stepInc );
        y = sin( curStep * stepInc );

        glNormal3d(x,y,0.0);
        glVertex3d(x*radius, y*radius, 0.0);
    }
    glEnd();
}

void RenderSystem::drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const
{
    int curStack, curSlice;

    /* Adjust z and radius as stacks and slices are drawn. */

    double r;
    double x,y,z;

    double const sliceInc = M_TAU / (-slices);
    double const stackInc = M_TAU / (2*stacks);
    
    /* Draw a line loop for each stack */
    for (curStack = 1; curStack < stacks; curStack++) {
        y = cos( curStack * stackInc );
        r = sin( curStack * stackInc );

        glBegin(GL_LINE_LOOP);

            for(curSlice = 0; curSlice <= slices; curSlice++) {
                x = cos( curSlice * sliceInc );
                z = sin( curSlice * sliceInc );

                glNormal3d(x,y,z);
                glVertex3d(x*r*radius + pos.x(), y*radius + pos.y(), z*r*radius + pos.z());
            }

        glEnd();
    }

    /* Draw a line loop for each slice */
    for (curSlice = 0; curSlice < slices; curSlice++) {
        glBegin(GL_LINE_STRIP);

            for (curStack = 1; curStack < stacks; curStack++) {
                x = cos( curSlice * sliceInc ) * sin( curStack * stackInc );
                z = sin( curSlice * sliceInc ) * sin( curStack * stackInc );
                y = cos( curStack * stackInc );

                glNormal3d(x,y,z);
                glVertex3d(x*radius + pos.x(), y*radius + pos.y(), z*radius + pos.z());
            }

        glEnd();
    }
}

void RenderSystem::renderPoints() const
{
  for (int pi = 0; pi < (int)m_renderablePoints.size(); ++pi) {
    RenderSystem::Point const& point = getPoint(pi);
    setDrawColour(point.m_col);
    
    glPointSize(10.0);
    glBegin(GL_POINTS);
    Vector3d p = point.m_pos;
    glVertex3d(p.x(), p.y(), p.z());
    glEnd();
    glPointSize(1.0);
  }
}

void RenderSystem::renderSpheres() const
{
  for (int si = 0; si < (int)m_renderableSpheres.size(); ++si) {
    RenderSystem::Sphere const& sphere = getSphere(si);
    setDrawColour(sphere.m_col);

    drawWireSphere(sphere.m_pos, sphere.m_radius, 32, 32);
  }
}

void RenderSystem::renderOrbits() const
{
  for (int oi = 0; oi < (int)m_renderableOrbits.size(); ++oi) {
    RenderSystem::Orbit const& orbit = getOrbit(oi);
    setDrawColour(orbit.m_col);

    int const steps = 10000;
    // e = 2.0; // TODO 1.0 sometimes works, > 1 doesn't - do we need to just
    // restrict the range of theta?
    double const delta = .0001;
    double const HAX_RANGE = .9; // limit range to stay out of very large values
    // TODO want to instead limit the range based on... some viewing area?
    // might be two visible segments, one from +ve and one from -ve theta, with
    // different visible ranges. Could determine 
    // TODO and want to take steps of fixed length/distance
    double range;
    if (orbit.e < 1 - delta) { // ellipse
        range = .5 * M_TAU;
    } else if (orbit.e < 1 + delta) { // parabola
        range = .5 * M_TAU * HAX_RANGE;
    } else { // hyperbola
        range = acos(-1/orbit.e) * HAX_RANGE;
    }
    double const mint = -range;
    double const maxt = range;
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= steps; ++i) {
        double const ct = Util::Lerp(mint, maxt, (double)i / steps);
        double const cr = orbit.p / (1 + orbit.e * cos(ct));

        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);
        Vector3d pos = (orbit.x_dir * x_len) + (orbit.y_dir * y_len) + orbit.m_pos;
        glVertex3d(pos.x(), pos.y(), pos.z());
    }
    glEnd();
  }
}

void RenderSystem::renderTrails() const
{
  for (int ti = 0; ti < (int)m_renderableTrails.size(); ++ti) {
    Trail const& trail = getTrail(ti);
    glBegin(GL_LINE_STRIP);
    
    for (int i = 0; i < Trail::NUM_TRAIL_PTS; ++i)
    {
      // Render from tail to head
      int tailIdx = (trail.m_headIdx + 1) % Trail::NUM_TRAIL_PTS;
      int idx = (tailIdx + i) % Trail::NUM_TRAIL_PTS;
      Vector3d v = trail.m_trailPts[idx];

      float const l = (float)(trail.m_trailPointAge[idx] / trail.m_duration);
      Vector3f c = Util::Lerp(trail.m_colNew, trail.m_colOld, l);
      setDrawColour(c);

      glVertex3d(v.x(),v.y(),v.z());
    }
    
    glEnd();


    // Debugging trail: show the trail points
#if 0
    setDrawColour(Vector3f(0.0, 0.0, 1.0));

    for (int i = 0; i < Trail::NUM_TRAIL_PTS; ++i)
    {
      // Render from tail to head
      int tailIdx = (trail.m_headIdx + 1) % Trail::NUM_TRAIL_PTS;
      int idx = (tailIdx + i) % Trail::NUM_TRAIL_PTS;
      Vector3d v = trail.m_trailPts[idx];

      glPointSize(10.0);
      glBegin(GL_POINTS);
      glVertex3d(v.x(), v.y(), v.z());
      glEnd();
      glPointSize(1.0);
    }
#endif
  }
}

void RenderSystem::render()
{
  renderPoints();
  renderSpheres();
  renderOrbits();
  renderTrails();
}

RenderSystem::Trail::Trail(double const _duration, Vector3d const _initPos) :
  m_duration(_duration), 
  m_headIdx(0)
{
  for (int i = 0; i < NUM_TRAIL_PTS; ++i) {
    m_trailPts[i] = _initPos;
    m_trailPointAge[i] = 0.0;
  }
}

// TODO grumble not sure this should really be here...
void RenderSystem::Trail::Update(double const _dt, Vector3d const _pos)
{
  // TODO can have a list of past points and their durations, and cut up trail linearly

  // A -- 50ms -- B -- 10 ms -- C

  // So if we get several 10ms updates we would interpolate A towards B a proportional amount, then finally remove it.

  for (int i = 0; i < NUM_TRAIL_PTS; ++i) {
    m_trailPointAge[i] += _dt;
  }
  
  double const minAge = 100.0; // 100 ms;
  double const maxAge = m_duration;

  int prevHeadIdx = (m_headIdx - 1 + NUM_TRAIL_PTS) % NUM_TRAIL_PTS;
  if (m_trailPointAge[prevHeadIdx] > minAge) {
    // Push a new point
    m_headIdx = (m_headIdx + 1) % NUM_TRAIL_PTS;
  }

  m_trailPts[m_headIdx] = _pos;
  m_trailPointAge[m_headIdx] = 0.f;
    
  // From the head, skip over points that are younger than max age;
#if 0
  int tailIdx = (m_headIdx + 1) % NUM_TRAIL_PTS;
  int prevTailIdx = (tailIdx + 1) % NUM_TRAIL_PTS;
  if (m_trailPointAge[tailIdx] > maxAge) {
    // Interpolate to target position
    double const t = m_trailPointAge[tailIdx] - m_trailPointAge[prevTailIdx];
    double const tt = maxAge - m_trailPointAge[prevTailIdx];
    Vector3d const d = m_trailPts[tailIdx] - m_trailPts[prevTailIdx];
    m_trailPts[tailIdx] = m_trailPts[prevTailIdx] + (d / t) * tt;
    m_trailPointAge[tailIdx] = maxAge;
  }
#endif
}