#include "orGfx.h"
#include "util.h"

#include "orRender.h"

#include "orProfile/perftimer.h"

#include "SDL_log.h"
#include "SDL_surface.h"

RenderSystem::RenderSystem() :
  m_fontImage(NULL)
{
}

RenderSystem::~RenderSystem()
{
  SDL_FreeSurface(m_fontImage); m_fontImage = NULL;
}

void RenderSystem::initRender()
{
  m_fontImage = SDL_LoadBMP("fonts/dos-ascii-8x8.bmp");

  if (!m_fontImage) {
    orErr("Could not load '%s': %s", "fonts/dos-ascii-8x8.bmp", SDL_GetError());
    SDL_ClearError();
    return;
  }

  SDL_LogInfo(SDL_LOG_CATEGORY_RENDER, "Image format: %s", SDL_GetPixelFormatName(m_fontImage->format->format));

  uint32_t width = m_fontImage->w;
  uint32_t height = m_fontImage->h;

  glGenTextures(1, &m_fontTextureId);
  glBindTexture(GL_TEXTURE_2D, m_fontTextureId);

  glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_INTENSITY,
    width,
    height,
    0,
    GL_BGR,
    GL_UNSIGNED_BYTE,
    m_fontImage->pixels
  );

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void RenderSystem::shutdownRender()
{
  // TODO free opengl resources
}


RenderSystem::FrameBuffer RenderSystem::makeFrameBuffer(int width, int height)
{
  FrameBuffer result;

  result.width = width;
  result.height = height;

  {
    // Based on OpenGL Tutorial 14 http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
    checkGLErrors();

    // Allocate a texture ID for the texture we're going to render to
    glGenTextures(1, &result.colorTextureId); checkGLErrors();
    // Bind the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, result.colorTextureId); checkGLErrors();
    // Create an empty texture (last '0' is a null image data pointer)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0); checkGLErrors();
    // Make sure to use nearest-neighbour filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); checkGLErrors();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); checkGLErrors();

    // The depth buffer
    glGenRenderbuffers(1, &result.depthBufferId); checkGLErrors();
    glBindRenderbuffer(GL_RENDERBUFFER, result.depthBufferId); checkGLErrors();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height); checkGLErrors();

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    glGenFramebuffers(1, &result.frameBufferId); checkGLErrors();
    glBindFramebuffer(GL_FRAMEBUFFER, result.frameBufferId); checkGLErrors();
    // Attach our texture and depth buffers to the frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, result.colorTextureId, 0); checkGLErrors();
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, result.depthBufferId); checkGLErrors();
    // Enable the buffers
    GLenum drawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT};
    glDrawBuffers(sizeof(drawBuffers) / sizeof(drawBuffers[0]), drawBuffers); checkGLErrors();

    // Check that our framebuffer is set up correctly
    ensure(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  }

  return result;
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

void RenderSystem::drawSolidSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const
{
  GLfloat mat_ones[]={ 1.0, 1.0, 1.0, 1.0 };
  glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, &mat_ones[0] );
  // glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, &mat_ones[0] );
#if 0
  glPushMatrix();
  GLUquadric* quad = gluNewQuadric();
  gluQuadricDrawStyle(quad, GLU_FILL);
  gluQuadricNormals(quad, GLU_SMOOTH);
  gluQuadricOrientation(quad, GLU_OUTSIDE);
  gluQuadricTexture(quad, GL_TRUE);
  glTranslated(pos.x(), pos.y(), pos.z());
  gluSphere(quad, radius, slices, stacks);
  gluDeleteQuadric(quad);
  glPopMatrix();
#else
  Vector3d const center(pos.x(), pos.y(), pos.z());
  float const off_H = ( M_PI ) / float(stacks);
  float const off_R = ( M_PI * 2.0 ) / float(slices);

  // draw the tips as tri_fans
  {
    glBegin( GL_TRIANGLE_FAN );
    Vector3d n(
      sin( 0.0 ) * sin( 0.0 ),
      cos( 0.0 ) * sin( 0.0 ),
      cos( 0.0 )
    );
    Vector3d p = center + n * radius;
    glNormal3dv( n.data() );
    glVertex3dv( p.data() );

    for ( int sl=0; sl<slices+1; sl++ )
    {
      float a = float(sl)*off_R;
      Vector3d n(
        sin( a ) * sin( off_H ),
        cos( a ) * sin( off_H ),
        cos( off_H )
      );
      Vector3d p = center + n * radius;
      glNormal3dv( n.data() );
      glVertex3dv( p.data() );
    }
    glEnd();
  }

  {
    glBegin( GL_TRIANGLE_FAN );
    Vector3d n(
      sin( 0.0 ) * sin( M_PI ),
      cos( 0.0 ) * sin( M_PI ),
      cos( M_PI )
    );
    Vector3d p = center + n * radius;
    glNormal3dv( n.data() );
    glVertex3dv( p.data() );

    for ( int sl=slices; sl>=0; sl-- )
    {
      float a = float(sl)*off_R;
      Vector3d n(
        sin( a ) * sin( M_PI-off_H ),
        cos( a ) * sin( M_PI-off_H ),
        cos( M_PI-off_H )
      );
      Vector3d p = center + n * radius;
      glNormal3dv( n.data() );
      glVertex3dv( p.data() );
    }
    glEnd();
  }

  for ( int st=1; st<stacks-1; st++ )
  {
    float b = float(st)*off_H;
    glBegin( GL_QUAD_STRIP );
    for ( int sl=0; sl<slices+1; sl++ )
    {
      float a = float(sl)*off_R;
      {
        Vector3d n(
          sin( a ) * sin( b ),
          cos( a ) * sin( b ),
          cos( b )
        );
        Vector3d p = center + n * radius;
        glNormal3dv( n.data() );
        glVertex3dv( p.data() );
      }

      {
        Vector3d n(
          sin( a ) * sin( b+off_H ),
          cos( a ) * sin( b+off_H ),
          cos( b+off_H )
        );
        Vector3d p = center + n * radius;
        glNormal3dv( n.data() );
        glVertex3dv( p.data() );
      }
    }
    glEnd();
  }
#endif
  glNormal3d( 0.0, 0.0, 1.0 );
}

void RenderSystem::drawWireSphere(Vector3d const pos, double const radius, int const slices, int const stacks) const
{
  int curStack, curSlice;

  glPushMatrix();
  glTranslated(pos.x(), pos.y(), pos.z());

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
      glVertex3d(x*r*radius, y*radius, z*r*radius);
    }

    glEnd();
  }

  /* Draw a line loop for each slice */
  for (curSlice = 0; curSlice < slices; curSlice++) {
    double cosSlice = cos( curSlice * sliceInc );
    double sinSlice = sin( curSlice * sliceInc );

    glBegin(GL_LINE_STRIP);

    for (curStack = 1; curStack < stacks; curStack++) {
      double sinStack = sin( curStack * stackInc );
      x = cosSlice * sinStack;
      z = sinSlice * sinStack;
      y = cos( curStack * stackInc );

      glNormal3d(x,y,z);
      glVertex3d(x*radius, y*radius, z*radius);
    }

    glEnd();
  }

  glPopMatrix();
}

void RenderSystem::renderPoints() const
{
  PERFTIMER("RenderPoints");
  glDisable(GL_LIGHTING);
  for (int pi = 0; pi < (int)m_instancedPoints.size(); ++pi) {
    RenderSystem::Point const& point = getPoint(pi);

    glColor3d(point.m_col[0], point.m_col[1], point.m_col[2]);

    glPointSize(3.0);
    glBegin(GL_POINTS);
    glVertex3d(point.m_pos[0], point.m_pos[1], point.m_pos[2]);
    glEnd();
    glPointSize(1.0);
  }
}


void RenderSystem::projectLabel3Ds(Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx)
{
  PERFTIMER("ProjectLabel3Ds");
  for (int li = 0; li < (int)m_instancedLabel3Ds.size(); ++li) {
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

    label2D.m_pos[0] = (int)x;
    label2D.m_pos[1] = (int)y;
  }
}

void RenderSystem::renderLabels( int w_px, int h_px )
{
  PERFTIMER("RenderLabels");

  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, w_px, h_px, 0, 0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glBindTexture(GL_TEXTURE_2D, m_fontTextureId);

  glBegin(GL_QUADS);

  // Font texture:
  // 8x8 pixels per char
  // 16x16 chars
  // 128x128 pixels total

  // Maps directly to ASCII

  // Need a better naming scheme!

  // thing_measure[_space]_unit?

  for (int li = 0; li < (int)m_instancedLabel2Ds.size(); ++li) {
    RenderSystem::Label2D const& label2D = getLabel2D(li);

    glColor3d(label2D.m_col[0], label2D.m_col[1], label2D.m_col[2]);

    drawString(label2D.m_text, label2D.m_pos[0], label2D.m_pos[1]);
  }

  for (int li = 0; li < (int)m_label2DBuffer.size(); ++li) {
    RenderSystem::Label2D const& label2D = m_label2DBuffer[li];

    glColor3d(label2D.m_col[0], label2D.m_col[1], label2D.m_col[2]);

    drawString(label2D.m_text, label2D.m_pos[0], label2D.m_pos[1]);
  }

  glEnd();
}

void RenderSystem::drawString(std::string const& str, int pos_x, int pos_y)
{
  int const char_w_px = 8;
  int const char_h_px = 8;
  int const font_w_chars = 16;
  int const font_h_chars = 16;

  float const font_l_px = 0;
  float const font_r_px = 128;

  float const font_t_px = 0;
  float const font_b_px = 128;

  float const font_l_tx = 0.0;
  float const font_r_tx = 1.0;

  float const font_t_tx = 0.0;
  float const font_b_tx = 1.0;

  float const u_scale = (font_r_tx - font_l_tx) / (font_r_px - font_l_px);
  float const v_scale = (font_b_tx - font_t_tx) / (font_b_px - font_t_px);

  float const char_w_tx = char_w_px * u_scale;
  float const char_h_tx = char_h_px * v_scale;

  int char_x_px = pos_x;
  int char_y_px = pos_y;

  for (int i = 0; i < (int)str.length(); ++i) {
    int const char_idx = str[i];

    if (char_idx == '\n') {
      char_x_px = pos_x;
      char_y_px += char_h_px;
      continue;
    }

    int const char_x_idx = char_idx % font_w_chars;
    int const char_y_idx = char_idx / font_w_chars;

    float const char_x_tex_px = char_x_idx * char_w_px - font_l_px;
    float const char_y_tex_px = char_y_idx * char_h_px - font_t_px;

    float const char_l_tx = char_x_tex_px * u_scale;
    float const char_t_tx = char_y_tex_px * v_scale;

    glTexCoord2f( char_l_tx,             char_t_tx               );
    glVertex3d(   char_x_px,             char_y_px,             0);

    glTexCoord2f( char_l_tx + char_w_tx, char_t_tx               );
    glVertex3d(   char_x_px + char_w_px, char_y_px,             0);

    glTexCoord2f( char_l_tx + char_w_tx, char_t_tx + char_h_tx   );
    glVertex3d(   char_x_px + char_w_px, char_y_px + char_h_px, 0);

    glTexCoord2f( char_l_tx,             char_t_tx + char_h_tx   );
    glVertex3d(   char_x_px,             char_y_px + char_h_px, 0);

    char_x_px += char_w_px;
  }
}

void RenderSystem::renderSpheres() const
{
  glEnable(GL_LIGHTING);

  PERFTIMER("RenderSpheres");
  for (int si = 0; si < (int)m_instancedSpheres.size(); ++si) {
    RenderSystem::Sphere const& sphere = getSphere(si);

    glColor3d(sphere.m_col[0], sphere.m_col[1], sphere.m_col[2]);

    drawSolidSphere(Vector3d(sphere.m_pos), sphere.m_radius, 16, 16);
  }
}

void RenderSystem::renderOrbits() const
{
  PERFTIMER("RenderOrbits");
  glDisable(GL_LIGHTING);
  for (int oi = 0; oi < (int)m_instancedOrbits.size(); ++oi) {
    RenderSystem::Orbit const& orbit = getOrbit(oi);

    glColor3d(orbit.m_col[0], orbit.m_col[1], orbit.m_col[2]);

    int const steps = 1000;
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
#else // No vector version (correct implementation, faster...)
        double const ct = Util::Lerp(mint, maxt, (double)i / steps);
        double const cr = orbit.p / (1 + orbit.e * cos(ct));

        double const x_len = cr * -cos(ct);
        double const y_len = cr * -sin(ct);
        double posData[3];
        posData[0] = (orbit.x_dir[0] * x_len) + (orbit.y_dir[0] * y_len) + orbit_pos[0];
        posData[1] = (orbit.x_dir[1] * x_len) + (orbit.y_dir[1] * y_len) + orbit_pos[1];
        posData[2] = (orbit.x_dir[2] * x_len) + (orbit.y_dir[2] * y_len) + orbit_pos[2];
        glVertex3d(posData[0], posData[1], posData[2]);
#endif
        // TODO normals? If we want lighting. Otherwise, disable lighting.
    }
    glEnd();
  }
}

void RenderSystem::renderTrails() const
{
  PERFTIMER("RenderTrails");
  for (int ti = 0; ti < (int)m_instancedTrails.size(); ++ti) {
    Trail const& trail = getTrail(ti);
    glBegin(GL_LINE_STRIP);

    for (int i = 0; i < Trail::NUM_TRAIL_PTS; ++i)
    {
      // Render from tail to head
      int tailIdx = (trail.m_headIdx + 1) % Trail::NUM_TRAIL_PTS;
      int idx = (tailIdx + i) % Trail::NUM_TRAIL_PTS;
      Vector3d v = Vector3d(&trail.m_trailPts[3*idx]) + Vector3d(trail.m_HACKorigin);

      float const l = (float)(trail.m_trailPointAge[idx] / trail.m_duration);
      Vector3d c = Util::Lerp(Vector3d(trail.m_colNew), Vector3d(trail.m_colOld), l);

      glColor3d(c.x(), c.y(), c.z());

      glVertex3d(v.x(),v.y(),v.z());
    }

    glEnd();


    // Debugging trail: show the trail points
#if 0
    glColor3d(0.0, 0.0, 1.0);

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

void RenderSystem::render2D(int w_px, int h_px, Eigen::Matrix4d const& screenMtx, Eigen::Matrix4d const& projMtx, Eigen::Matrix4d const& camMtx)
{
  projectLabel3Ds(screenMtx, projMtx, camMtx);
  renderLabels(w_px, h_px);
}

void RenderSystem::render3D()
{
  renderPoints();
  renderSpheres();
  renderOrbits();
  renderTrails();
}

void RenderSystem::checkGLErrors()
{
  int gl_err = glGetError();
  if(gl_err != GL_NO_ERROR) {
    fprintf(stderr, "Error: %d %s", gl_err, (char const*)gluErrorString(gl_err));
  }
}

#if 0
void RenderSystem::clearBuffer(
  FrameBuffer const& frameBuffer,
  sf::Vector3d clearCol,
  float clearDepth
) {
  // Render to our framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer.frameBufferId);
  glBindRenderbuffer(GL_RENDERBUFFER, frameBuffer.depthRenderBufferId);

   // Render on the whole framebuffer, complete from the lower left corner to the upper right
  glViewport(0, 0, frameBuffer.width, frameBuffer.height);

  // TODO what?
  // This is visibly not clearing the offscreen frame buffer, it's clearing the default one...
  glClearColor(clearCol.x, clearCol.y, clearCol.z, 0);
  glClearDepth(clearDepth);
}
#endif

void RenderSystem::render(
  FrameBuffer const& frameBuffer,
  Colour clearCol,
  float clearDepth,
  Eigen::Matrix4d const& screenFromProj,
  Eigen::Matrix4d const& projFromCam,
  Eigen::Matrix4d const& camFromWorld
)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();

  m_label2DBuffer.clear();

  {
    PERFTIMER("Prepare3D");

    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer.frameBufferId);
    glBindRenderbuffer(GL_RENDERBUFFER, frameBuffer.depthBufferId);

    // Render on the whole framebuffer, complete from the lower left corner to the upper right
    glViewport(0, 0, frameBuffer.width, frameBuffer.height);

    // This is visibly not clearing the offscreen frame buffer, it's clearing the default one...
    // glClearColor(clearCol.x, clearCol.y, clearCol.z, 0);
    // glClearDepth(clearDepth);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMultMatrix( projFromCam );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrix( camFromWorld );

    glEnable(GL_TEXTURE_2D);

    GLfloat ambient[] = { 0.0, 0.2, 0.0, 1.0 };
    GLfloat diffuse[] = { 1.0, 0.0, 0.0, 1.0 };
    // GLfloat specular[] = { 0.0, 0.0, 1.0, 1.0 };

    // TODO NOTE XXX HACK this lights the orbits fine when the w is 0.0,
    // lights the sphere when the w is 1.0, but not the other way around.
    // Even when the sphere is lit, the light position doesn't seem to matter
    // but the sphere is lit from behind from certain camera positions
    // and not lit at all otherwise.
    // Possibly something to do with normals? gluSphere code is setting normals
    // of some kind, and my orbit-drawing code too.
    // The polygon-based sphere I copy-pasted from the internet gives the same
    // results!

    // Seems to be related to normals and lighting, made some fixes...

    // TODO clean up mode changes, move more into the Render system

    GLfloat light_pos[] = { 0.0, 0.0, 25000000.0, 1.0 };

    glShadeModel( GL_SMOOTH );
    glLightfv( GL_LIGHT0, GL_AMBIENT, &ambient[0] );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, &diffuse[0] );
    glLightfv( GL_LIGHT0, GL_POSITION, &light_pos[0] );
    glEnable( GL_LIGHT0 );
    glEnable( GL_LIGHTING );

    glNormal3d(0,0,1);

    glLineWidth(1.0);
#if 0
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
#endif
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    glDisable(GL_TEXTURE_2D);
  }

  {
    PERFTIMER("Render3D");
    render3D();
  }

  {
    PERFTIMER("Render2D");
    render2D(frameBuffer.width, frameBuffer.height, screenFromProj, projFromCam, camFromWorld);
  }
}

void RenderSystem::Trail::Init(double const _duration, const orVec3 _initPos, const orVec3 _initOrigin)
{
  m_duration = _duration;
  m_headIdx = 0;
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