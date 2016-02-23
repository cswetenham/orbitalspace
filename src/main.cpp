// C headers

#include <stdio.h>

// C++ headers
#include <chrono>

// GL headers
// OpenGL loader - loads/defines everything
#include "glad/glad.h"

// Include GLU for gluErrorString - not sure if this will break things after GLAD include...
#include "GL/glu.h"

// GLM
#define GLM_FORCE_RADIANS 1
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// SDL headers
#include <SDL.h>

// Image loading
// Do this before you include this file in *one* C or C++ file to create the implementation.
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "util/platform.h"
#include "util/logging.h"
#include "util/timer.h"

#include "imgui.h"
#include "imgui_impl_sdl_gl3.h"

// TODO cleanup
#define ENABLE_GL_CHECK 1
#if ENABLE_GL_CHECK
#define GL_CHECK_R(expr) _gl_check(expr, __FILE__, __LINE__, #expr)
#define GL_CHECK(expr) expr; _gl_check(0, __FILE__, __LINE__, #expr);
#else
#define GL_CHECK_R(expr) expr
#define GL_CHECK(expr) expr
#endif

template <typename T>
inline T _gl_check(T t, char const* file, int line, char const* expr) {
  GLenum err = glGetError();
  while(err != GL_NO_ERROR) {
    LOGERR("OpenGL Error: %d %s (%s:%d)", err, (char const*)gluErrorString(err), file, line);
    err = glGetError();
  }
  return t;
}

// TODO include Lua, use Lua object format for config files + saving/loading

// TODO cleanup
// Allocates a new string buffer for the contents of filename, and returns the
// length in lenggh
char* file_contents(const char* filename, GLint* length)
{
  // TODO better logging + error handling
  assert(filename);
  assert(length);

  FILE* f = fopen(filename, "r");
  char* buffer;

  if (!f) {
    LOGERR("Unable to open %s for reading\n", filename);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  *length = ftell(f);
  fseek(f, 0, SEEK_SET);

  buffer = (char*)malloc(*length+1);
  *length = fread(buffer, 1, *length, f); // TODO why reassign length? what can go wrong here?
  fclose(f);
  buffer[*length] = '\0';

  return buffer;
}

// TODO cleanup
void show_shader_info_log(
  char const* shader_path,
  GLuint object
) {
  GLint log_length;
  GL_CHECK(glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length));
  char* log = (char*)malloc(log_length);
  GL_CHECK(glGetShaderInfoLog(object, log_length, NULL, log));
  LOGERR("Error compiling shader %s: %s", shader_path, log);
  free(log);
}

void show_program_info_log(
  char const* vertex_path,
  char const* fragment_path,
  GLuint object
) {
  GLint log_length;
  GL_CHECK(glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length));
  char* log = (char*)malloc(log_length);
  GL_CHECK(glGetProgramInfoLog(object, log_length, NULL, log));
  LOGERR("Error linking shaders %s, %s: %s", vertex_path, fragment_path, log);
  free(log);
}

static GLuint make_shader(GLenum type, const char* filename)
{
  GLint length;
  GLchar* source = file_contents(filename, &length);
  if (!source) {
    return 0;
  }

  GLuint shader = GL_CHECK(glCreateShader(type));
  GL_CHECK(glShaderSource(shader, 1, (const GLchar**)&source, &length));
  free(source);
  GL_CHECK(glCompileShader(shader));

  GLint shader_ok;
  GL_CHECK(glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok));
  if (!shader_ok) {
    LOGERR("Failed to compile %s:\n", filename);
    show_shader_info_log(filename, shader);
    GL_CHECK(glDeleteShader(shader));
    return 0;
  }
  return shader;
}

struct ShaderInfo
{
  char const* shader_path;
  GLuint shader_id;
};

// TODO cleanup
static GLuint make_program(int shader_count, ShaderInfo const* shader_infos)
{
    GLuint program = GL_CHECK(glCreateProgram());

    for (int i = 0; i < shader_count; ++i) {
      GL_CHECK(glAttachShader(program, shader_infos[i].shader_id));
    }

    // TODO well hrm we'll need to pass in an array or something
    // Needs to be called before glLinkProgram and after glAttachShader I guess
    glBindFragDataLocation(program, 0, "outColor"); // not strictly needed since only 1 output

    GL_CHECK(glLinkProgram(program));

    GLint program_ok;
    GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &program_ok));
    if (!program_ok) {
      LOGERR("Failed to link shader program:\n");
      for (int i = 0; i < shader_count; ++i) {
        LOGERR("- %s", shader_infos[i].shader_path);
      }
      show_program_info_log("", "", program);
      // TODO hmm
      // GL_CHECK(glDeleteProgram(program));
      // return 0;
    }

    for (int i = 0; i < shader_count; ++i) {
      // Can detach shaders after link
      GL_CHECK(glDetachShader(program, shader_infos[i].shader_id));
    }

    return program;
}

void barf_floats(size_t size, float const* data) {
  size_t count = size / sizeof(float);
  LOGINFO("size: %d", size);
  LOGINFO("count: %d", count);

  size_t idx = 0;
  while (idx < count) {
    LOGINFO("{%03.3f, %03.3f, %03.3f, %03.3f}", data[idx], data[idx+1], data[idx+2], data[idx+3]);
    idx += 4;
  }
}

// A note on scale
// Currently not planning anything outside the solar system
// Voyager is outside the heliopause at 134 AU

// A 64-bit int can represent distances up to +/-100 AU at a resolution of
// 0.0015 mm. Or to keep units scaled in powers of 2 of SI units, if 2^44 m =
// 2^63 units then 1 unit is about 0.0019 mm and we can represent over 100AU.

// Minimum time resolution I'd simulate at might be 120 fps / 8ms; if we say
// 1 unit = 1/2^10 seconds then 1 unit is about 1 ms and max is 292277266 years.

// What about speeds?
// Voyager's fastest speed was 17 km/s. Sun-grazing comets can reach 570 km/s.
// If we scale for speeds up to +/- 1000 km/s, we can represent speeds at a
// resolution of 1.08420217x10^-13 m/s. (not the same scale as distances -
// would lead to constants in various calculations I might have to derive...)
// Or if I keep the above resolutions, 1 unit distance per 1 unit time is
// about 0.0019 m/s, and maximum is 2^64 units per 1 unit = something truly
// absurd. What about scaling by 2^32? 2^31 units per 1 unit is still 8000 km/s
// and will give good resolution.

// Finally what about accelerations, and other quantities (orbital elements?)
// TODO for now, maybe just think about the position aspect, and how we'd go
// from fixed-point to rendering values in single-precision floats in shaders;
// whatever I decide will also apply to using doubles then rendering to floats.
// Want high-precision rendering of orbits - geometry shader to generate
// vertices based on the view frustum and target resolution.

// One option first of all, is to allow the camera matrices to use doubles by
// pre-computing them, and then submitting only screen-space vertices - but
// this does mean re-computing the vertices every time the camera changes.

// TODO debug camera vs game camera (model-view transform vs LOD/culling frustum)

extern "C"
int main(int argc, char *argv[])
{
  using namespace orbital;

  timer::Init();

  SDL_Init(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

  // No idea what effect these have on performance!
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 5);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  SDL_Window* window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1600, 900, SDL_WINDOW_OPENGL);

  SDL_GLContext context = SDL_GL_CreateContext(window);

  // Load OpenGL and OpenGL functions
  gladLoadGLLoader(SDL_GL_GetProcAddress);
  LOGINFO("OpenGL loaded");
  LOGINFO("Vendor:   %s", GL_CHECK_R(glGetString(GL_VENDOR)));
  LOGINFO("Renderer: %s", GL_CHECK_R(glGetString(GL_RENDERER)));
  LOGINFO("Version:  %s", GL_CHECK_R(glGetString(GL_VERSION)));

  // Setup ImGui binding
  ImGui_ImplSdlGL3_Init(window);

  // TODO enable sRGB framebuffer + textures, and use it

  // Explicitly set culling orientation and disable/enable culling
  GL_CHECK(glFrontFace(GL_CCW));
  // GL_CHECK(glFrontFace(GL_CW));
  // TODO temp
  // GL_CHECK(glDisable(GL_CULL_FACE));
  GL_CHECK(glEnable(GL_CULL_FACE));
  // Possibly needed for logarithmic depth buffer calculations.
  // "If enabled, the -wc <= zc <= wc plane equation is ignored by view volume
  // clipping (effectively, there is no near or far plane clipping).
  // See glDepthRange."
  // Note that with the logarithmic shaders, the near and far planes as set by
  // the perspective matrix still seem to be used for clipping...
  // TODO figure out what is going on
  GL_CHECK(glEnable(GL_DEPTH_CLAMP));

  GL_CHECK(glEnable(GL_DEPTH_TEST));

  // Vertex array object:
  GLuint vao;
  GL_CHECK(glGenVertexArrays(1, &vao));
  GL_CHECK(glBindVertexArray(vao));

  GLuint vbo;
  GL_CHECK(glGenBuffers(1, &vbo)); // Generate 1 buffer

  struct Vertex {
    glm::vec3 pos;
    glm::vec3 col;
  };

  Vertex vertices[] = {
    { { -1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f } }, // Black Back-right-bottom
    { { -1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f, 0.0f } }, // Red Back-right-top
    { { -1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f, 0.0f } }, // Green Back-left-bottom
    { { -1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f, 0.0f } }, // Yellow Back-left-top
    { {  1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f, 1.0f } }, // Blue Front-right-bottom
    { {  1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f, 1.0f } }, // Magenta Front-right-top
    { {  1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f, 1.0f } }, // Cyan Front-left-bottom
    { {  1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f, 1.0f } }  // White Front-left-top
  };

  barf_floats(sizeof(vertices), (float*)vertices);

  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vbo));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

  // Element buffer object
  GLuint ebo;
  GL_CHECK(glGenBuffers(1, &ebo));
  GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo));

  uint32_t elements[] = {
    2, 3, 6,
    6, 3, 7,
    1, 5, 3,
    3, 5, 7,
    6, 4, 2,
    2, 4, 0,
    0, 1, 2,
    2, 1, 3,
    4, 5, 0,
    0, 5, 1,
    6, 7, 4,
    4, 7, 5
  };

  GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW));

  // A basic clip-space shader

  // TODO better resource loading
  GLuint vertexShader = make_shader(GL_VERTEX_SHADER, "shaders/vertexcol.v.glsl");
  GLuint fragmentShader = make_shader(GL_FRAGMENT_SHADER, "shaders/vertexcol.f.glsl");
  ShaderInfo shader_infos[2];
  shader_infos[0].shader_path = "shaders/vertexcol.v.glsl";
  shader_infos[0].shader_id = vertexShader;
  shader_infos[1].shader_path = "shaders/vertexcol.f.glsl";
  shader_infos[1].shader_id = fragmentShader;
  GLuint shaderProgram = make_program(2, shader_infos);
  GL_CHECK(glUseProgram(shaderProgram));

  // Bind position attribute to vertex array
  GLint posAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "position"));
  LOGINFO("posAttrib=%d", posAttrib);
  GL_CHECK(glEnableVertexAttribArray(posAttrib));
  // (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
  GL_CHECK(glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos)));

  GLint colAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "color"));
  LOGINFO("colAttrib=%d", colAttrib);
  GL_CHECK(glEnableVertexAttribArray(colAttrib));
  GL_CHECK(glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, col)));

  // Set up matrices
  GLint uniModel = GL_CHECK_R(glGetUniformLocation(shaderProgram, "model"));
  LOGINFO("uniModel=%d", uniModel);
  GLint uniView = GL_CHECK_R(glGetUniformLocation(shaderProgram, "view"));
  LOGINFO("uniView=%d", uniView);
  GLint uniProj = GL_CHECK_R(glGetUniformLocation(shaderProgram, "proj"));
  LOGINFO("uniProj=%d", uniProj);

  GLint uniFcoef = GL_CHECK_R(glGetUniformLocation(shaderProgram, "Fcoef"));
  LOGINFO("uniFcoef=%d", uniFcoef);
  GLint uniFcoef_half = GL_CHECK_R(glGetUniformLocation(shaderProgram, "Fcoef_half"));
  LOGINFO("uniFcoef_half=%d", uniFcoef_half);


  // GUI state
  bool show_test_window = false;
  bool wireframe = false;
  ImVec4 clear_color = ImColor(10, 10, 10);
  ImVec4 eye_pos = ImVec4(3.0f, 3.0f, 3.0f, 1.0f);

  float fov_y = 90.0f;

  float nearplane = 1.0f;
  float farplane = 100.0f;

  ImVec2 size(1500.0f, 260.0f);
  ImVec2 size_inner(1000.0f, 620.0f);
  ImVec2 size_entities(100.0f, 620.0f);

  float external_scroll = 0.0f;
  float external_scroll_max = 1000.0f;

  LOGINFO("Starting main loop");

  while (true) {
    // Input handling
    SDL_Event windowEvent;
    if (SDL_PollEvent(&windowEvent)) {
      if (windowEvent.type == SDL_QUIT) {
        break;
      }
      if (
        windowEvent.type == SDL_KEYUP &&
        windowEvent.key.keysym.sym == SDLK_ESCAPE
      ) {
        break;
      }
    }

    ImGui_ImplSdlGL3_NewFrame();

    // 1. Show a simple window
    // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
    {
        ImGui::Text("Hello, world!");
        ImGui::SliderFloat("fov_y", &fov_y, 10.f, 140.0f);
        // TODO near plane
        // TOOD far plane
        ImGui::ColorEdit3("clear color", (float*)&clear_color);
        ImGui::DragFloat3("Eye Pos", (float*)&eye_pos, 0.1f, -10.0f, 10.0f);
        if (ImGui::Button("Test Window")) show_test_window ^= 1;
        ImGui::Checkbox("Wireframe", &wireframe);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    }

    // 2. Show timeline window
    {
      ImGui::Begin("Timeline", NULL, ImGuiWindowFlags_ShowBorders);
        ImGui::SliderFloat2("size", &size.x, 0.0f, ImGui::GetIO().DisplaySize.x);
        ImGui::SliderFloat2("size_inner", &size_inner.x, 0.0f, ImGui::GetIO().DisplaySize.x);
        ImGui::SliderFloat2("size_entities", &size_entities.x, 0.0f, ImGui::GetIO().DisplaySize.x);
        ImGui::BeginChild("inner_timeline", size, true, ImGuiWindowFlags_ShowBorders | ImGuiWindowFlags_ForceVerticalScrollbar);
          // ImGui::Columns(2, NULL, true);
          // TODO entity list
          // ImGui::PushItemWidth(60.0f);
          ImGui::BeginChild("entities_column", size_entities, true, ImGuiWindowFlags_ShowBorders | ImGuiWindowFlags_NoScrollbar);
          ImGui::Text("Entity 1");
          ImGui::Text("Entity 2");
          ImGui::Text("Entity 3");
          // ImGui::NextColumn();
          ImGui::EndChild();
          ImGui::SameLine();
          // ImGui::PopItemWidth();
          ImGui::BeginChild("timeline_timeline", size_inner, true, ImGuiWindowFlags_ShowBorders | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_HorizontalScrollbar ); // | ImGuiWindowFlags_ForceHorizontalScrollbar);
            ImGui::SetScrollX(external_scroll);
            external_scroll_max = ImGui::GetScrollMaxX();
            // TODO entity timeline
            ImGui::Text("Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline Image of a Timeline");
          ImGui::EndChild();
          // ImGui::Columns(1);
        ImGui::EndChild();
        ImGui::SliderFloat("Time", &external_scroll, 0.0f, glm::max(external_scroll_max, 0.0f));
      ImGui::End();
    }

    // 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
    if (show_test_window)
    {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
        ImGui::ShowTestWindow(&show_test_window);
    }

    // Rendering
    glViewport(0, 0, (int)ImGui::GetIO().DisplaySize.x, (int)ImGui::GetIO().DisplaySize.y);

    // Clear the screen to black
    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));


    // Render game

    glPolygonMode( GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL );

    // Bind textures to samplers
    GL_CHECK(glUseProgram(shaderProgram));

    GL_CHECK(glBindVertexArray(vao));

    glm::mat4 view = glm::lookAt(
        glm::vec3(eye_pos.x, eye_pos.y, eye_pos.z),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GL_CHECK(glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view)));

    glm::mat4 proj = glm::perspective(
        glm::radians(fov_y),
        ImGui::GetIO().DisplaySize.x / ImGui::GetIO().DisplaySize.y,
        nearplane,
        farplane
    );
    GL_CHECK(glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj)));

    float Fcoef = 2.0 / glm::log2(farplane + 1.0);
    GL_CHECK(glUniform1f(uniFcoef, Fcoef));
    GL_CHECK(glUniform1f(uniFcoef_half, Fcoef * 0.5));

    for (int x = -10; x < 10; ++x) {
      for (int y = -10; y < 10; ++y) {
        glm::mat4 model;
        model = glm::rotate(
            model,
            0.0f,
            glm::vec3(0.0f, 0.0f, 1.0f)
        );
        model = glm::translate(model, glm::vec3(3.0f * x, 3.0f * y, 0.0f));
        GL_CHECK(glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model)));

        GL_CHECK(glDrawElements(GL_TRIANGLES, sizeof(elements) / (sizeof(uint32_t)), GL_UNSIGNED_INT, 0));
      }
    }

    // Render GUI
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

    ImGui::Render();

    // Swap
    SDL_GL_SwapWindow(window);
  }

  LOGINFO("Exiting...");

  // Cleanup

  GL_CHECK(glDeleteProgram(shaderProgram));
  GL_CHECK(glDeleteShader(fragmentShader));
  GL_CHECK(glDeleteShader(vertexShader));

  GL_CHECK(glDeleteBuffers(1, &ebo));
  GL_CHECK(glDeleteBuffers(1, &vbo));

  GL_CHECK(glDeleteVertexArrays(1, &vao));

  ImGui_ImplSdlGL3_Shutdown();

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
