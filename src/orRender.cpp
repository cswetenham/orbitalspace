#include "orGfx.h"
#include "util.h"

#include "orRender.h"

// For Text rendering
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "orProfile/perftimer.h"

RenderSystem::RenderSystem() :
  m_font(new sf::Font)
{
  // *m_font = sf::Font::getDefaultFont();
  m_font->loadFromFile("Fonts/m01/m01.ttf");
}

RenderSystem::~RenderSystem()
{
  delete m_font; m_font = NULL;
}

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
  PERFTIMER("RenderPoints");
  for (int pi = 0; pi < (int)m_points.size(); ++pi) {
    RenderSystem::Point const& point = getPoint(pi);

    setDrawColour(Vector3f(point.m_col));

    glPointSize(3.0);
    glBegin(GL_POINTS);
    Vector3d p(point.m_pos);
    // TODO isn't there Eigen opengl support included?
    glVertex3d(p.x(), p.y(), p.z());
    glEnd();
    glPointSize(1.0);
  }
}


void RenderSystem::projectLabel3Ds(Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx)
{
  PERFTIMER("ProjectLabel3Ds");
  for (int li = 0; li < (int)m_label3Ds.size(); ++li) {
    RenderSystem::Label3D const& label3D = getLabel3D(li);
    
    Vector4d pos3d;
    pos3d.x() = label3D.m_pos[0]; // is this really the best way?
    pos3d.y() = label3D.m_pos[1];
    pos3d.z() = label3D.m_pos[2];
    pos3d.w() = 1.0;

    Vector4d pos2d = screenMtx * projMtx * camMtx * pos3d;

    pos2d /= pos2d.w();

    float x = (float)pos2d.x();
    float y = (float)pos2d.y();
    
    m_label2DBuffer.push_back(Label2D());
    Label2D& label2D = m_label2DBuffer.back();

    label2D.m_text = label3D.m_text;

    label2D.m_col[0] = label3D.m_col[0];
    label2D.m_col[1] = label3D.m_col[1];
    label2D.m_col[2] = label3D.m_col[2];

    label2D.m_pos[0] = x;
    label2D.m_pos[1] = y;
  }
}

void RenderSystem::renderLabels(sf::RenderWindow* window)
{
  PERFTIMER("RenderLabels");
  // uint32_t const fontSize = 14;
  uint32_t const fontSize = 8;

  for (int li = 0; li < (int)m_label2Ds.size(); ++li) {
    RenderSystem::Label2D const& label2D = getLabel2D(li);
    sf::Text text("", *m_font, fontSize);

    {
      PERFTIMER("ConfigText");
      {
        PERFTIMER("ConfigString");
        text.setString(label2D.m_text);
      }

      {
        PERFTIMER("ConfigColor");
        text.setColor(sf::Color(
          uint8_t(label2D.m_col[0] * 255),
          uint8_t(label2D.m_col[1] * 255),
          uint8_t(label2D.m_col[2] * 255),
          255
        ));
      }

      {
        PERFTIMER("ConfigPos");
        text.setPosition(label2D.m_pos[0], label2D.m_pos[1]);
      }
    }
    
    {
      PERFTIMER("DrawText");
      window->draw(text);
    }
  }

  for (int li = 0; li < (int)m_label2DBuffer.size(); ++li) {
    RenderSystem::Label2D const& label2D = m_label2DBuffer[li];
    sf::Text text("", *m_font, fontSize);

    {
      PERFTIMER("ConfigText");
      {
        PERFTIMER("ConfigString");
        text.setString(label2D.m_text);
      }

      {
        PERFTIMER("ConfigColor");
        text.setColor(sf::Color(
          uint8_t(label2D.m_col[0] * 255),
          uint8_t(label2D.m_col[1] * 255),
          uint8_t(label2D.m_col[2] * 255),
          255
        ));
      }

      {
        PERFTIMER("ConfigPos");
        text.setPosition(label2D.m_pos[0], label2D.m_pos[1]);
      }
    }

    {
      PERFTIMER("DrawText");
      window->draw(text);
    }
  }
}

void RenderSystem::renderSpheres() const
{
  PERFTIMER("RenderSpheres");
  for (int si = 0; si < (int)m_spheres.size(); ++si) {
    RenderSystem::Sphere const& sphere = getSphere(si);
    setDrawColour(Vector3f(sphere.m_col));

    drawWireSphere(Vector3d(sphere.m_pos), sphere.m_radius, 32, 32);
  }
}

void RenderSystem::renderOrbits() const
{
  PERFTIMER("RenderOrbits");
  for (int oi = 0; oi < (int)m_orbits.size(); ++oi) {
    RenderSystem::Orbit const& orbit = getOrbit(oi);
    setDrawColour(Vector3f(orbit.m_col));

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
    
    Vector3d const orbit_x(orbit.x_dir);
    Vector3d const orbit_y(orbit.y_dir);
    Vector3d const orbit_pos(orbit.m_pos);
    
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= steps; ++i) {
#if 0 // Original version (correct implementation)
        double const ct = Util::Lerp(mint, maxt, (double)i / steps);
        double const cr = orbit.p / (1 + orbit.e * cos(ct));

        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);
        Vector3d const pos = (orbit_x * x_len) + (orbit_y * y_len) + orbit_pos;
        const double* const posData = pos.data();
        glVertex3d(posData[0], posData[1], posData[2]);
#elif 0 // No trig version (broken, for testing)
        Vector3d const pos = (orbit_x * 1.0) + (orbit_y * 1.0) + orbit_pos;
        const double* const posData = pos.data();
        glVertex3d(posData[0], posData[1], posData[2]);
#elif 1 // No vector version (correct implementation)
        double const ct = Util::Lerp(mint, maxt, (double)i / steps);
        double const cr = orbit.p / (1 + orbit.e * cos(ct));

        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);
        double posData[3];
        posData[0] = (orbit.x_dir[0] * x_len) + (orbit.y_dir[0] * y_len) + orbit_pos[0];
        posData[1] = (orbit.x_dir[1] * x_len) + (orbit.y_dir[1] * y_len) + orbit_pos[1];
        posData[2] = (orbit.x_dir[2] * x_len) + (orbit.y_dir[2] * y_len) + orbit_pos[2];
        glVertex3d(posData[0], posData[1], posData[2]);
#elif 0 // No vector no trig version (broken, for testing)
        double posData[3];
        posData[0] = (orbit.x_dir[0] * 1.0) + (orbit.y_dir[0] * 1.0) + orbit_pos[0];
        posData[1] = (orbit.x_dir[1] * 1.0) + (orbit.y_dir[1] * 1.0) + orbit_pos[1];
        posData[2] = (orbit.x_dir[2] * 1.0) + (orbit.y_dir[2] * 1.0) + orbit_pos[2];
        glVertex3d(posData[0], posData[1], posData[2]);
#else // glVertex3d only version (broken, for testing)
        glVertex3d(orbit.m_pos[0], orbit.m_pos[2], orbit.m_pos[2]);
#endif
    }
    glEnd();
  }
}

void RenderSystem::renderTrails() const
{
  PERFTIMER("RenderTrails");
  for (int ti = 0; ti < (int)m_trails.size(); ++ti) {
    Trail const& trail = getTrail(ti);
    glBegin(GL_LINE_STRIP);

    for (int i = 0; i < Trail::NUM_TRAIL_PTS; ++i)
    {
      // Render from tail to head
      int tailIdx = (trail.m_headIdx + 1) % Trail::NUM_TRAIL_PTS;
      int idx = (tailIdx + i) % Trail::NUM_TRAIL_PTS;
      Vector3d v = Vector3d(&trail.m_trailPts[3*idx]) + Vector3d(trail.m_HACKorigin);

      float const l = (float)(trail.m_trailPointAge[idx] / trail.m_duration);
      Vector3f c = Util::Lerp(Vector3f(trail.m_colNew), Vector3f(trail.m_colOld), l);
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
      Vector3d v = trail.m_trailPts[idx] + trail.m_HACKorigin;

      glPointSize(10.0);
      glBegin(GL_POINTS);
      glVertex3d(v.x(), v.y(), v.z());
      glEnd();
      glPointSize(1.0);
    }
#endif
  }
}

void RenderSystem::render2D(sf::RenderWindow* window, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx)
{
  projectLabel3Ds(screenMtx, projMtx, camMtx);
  renderLabels(window);
}

void RenderSystem::render3D(sf::RenderWindow* window)
{
  renderPoints();
  renderSpheres();
  renderOrbits();
  renderTrails();
}

RenderSystem::Trail::Trail(double const _duration, const double _initPos[3], const double _initOrigin[3]) :
  m_duration(_duration),
  m_headIdx(0)
{
  // TODO need to set m_HACKorigin from _initOrigin? _initOrigin never read right now, m_HACKorigin never set...
  Vector3d initPos(_initPos);
  Vector3d initOrigin(_initOrigin);

  Vector3d initPt = initPos - Vector3d(m_HACKorigin);
  const double* const initPtData = initPt.data();

  for (int i = 0; i < NUM_TRAIL_PTS; ++i) {
    m_trailPts[3*i  ] = initPtData[0];
    m_trailPts[3*i+1] = initPtData[1];
    m_trailPts[3*i+2] = initPtData[2];
  }

  for (int i = 0; i < NUM_TRAIL_PTS; ++i) {
    m_trailPointAge[i] = 0.0;
  }
}

// TODO grumble not sure this should really be here...
void RenderSystem::Trail::Update(double const _dt, Vector3d const _pos)
{
  // So if we get several 10ms updates we would interpolate A towards B a proportional amount, then finally remove it.

  Vector3d const pos = _pos - Vector3d(m_HACKorigin);

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

  Eigen::Map<Vector3d> v(&m_trailPts[3*m_headIdx]);
  v = pos;
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