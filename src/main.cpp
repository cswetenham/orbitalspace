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

  int idx = 0;
  while (idx < count) {
    LOGINFO("{%03.3f, %03.3f, %03.3f, %03.3f}", data[idx], data[idx+1], data[idx+2], data[idx+3]);
    idx += 4;
  }

}

extern "C"
int main(int argc, char *argv[])
{
  using namespace orbital;

  timer::Init();

  SDL_Init(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

  SDL_Window* window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024, 768, SDL_WINDOW_OPENGL);

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
  // TODO temp
  GL_CHECK(glDisable(GL_CULL_FACE));
  //GL_CHECK(glEnable(GL_CULL_FACE));

  glEnable(GL_DEPTH_TEST);

  // Vertex array object:
  GLuint vao;
  GL_CHECK(glGenVertexArrays(1, &vao));
  GL_CHECK(glBindVertexArray(vao));

  GLuint vbo;
  GL_CHECK(glGenBuffers(1, &vbo)); // Generate 1 buffer

  struct Vertex {
    glm::vec4 pos;
    glm::vec4 col;
  };

#if 1
  Vertex vertices[] = {
    { { -1.0f, -1.0f, -1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f, 0.5f } }, // Back-right-bottom
    { { -1.0f, -1.0f,  1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f, 0.5f } }, // Back-right-top
    { { -1.0f,  1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f, 0.5f } }, // Back-left-bottom
    { { -1.0f,  1.0f,  1.0f, 1.0f }, { 1.0f, 1.0f, 0.0f, 0.5f } }, // Back-left-top
    { {  1.0f, -1.0f, -1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f, 0.5f } }, // Front-right-bottom
    { {  1.0f, -1.0f,  1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f, 0.5f } }, // Front-right-top
    { {  1.0f,  1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, 1.0f, 0.5f } }, // Front-left-bottom
    { {  1.0f,  1.0f,  1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 0.5f } }  // Front-left-top
  };
#else
  GLfloat vertices[] = {
        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f
    };
#endif

  barf_floats(sizeof(vertices), (float*)vertices);

  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vbo));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));

  // Element buffer object
  GLuint ebo;
  GL_CHECK(glGenBuffers(1, &ebo));
  GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo));

  GLuint elements[] = {
    0, 1, 2, 2, 1, 3,
    4, 5, 0, 0, 5, 1,
    6, 7, 4, 4, 7, 5,
    2, 3, 6, 6, 7, 3,
    1, 5, 3, 3, 5, 7,
    6, 2, 4, 4, 0, 2
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
  GL_CHECK(glVertexAttribPointer(posAttrib, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos)));

  GLint colAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "color"));
  LOGINFO("colAttrib=%d", colAttrib);
  GL_CHECK(glEnableVertexAttribArray(colAttrib));
  GL_CHECK(glVertexAttribPointer(colAttrib, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, col)));

  // Set up matrices
  GLint uniModel = GL_CHECK_R(glGetUniformLocation(shaderProgram, "model"));
  LOGINFO("uniModel=%d", uniModel);
  GLint uniView = GL_CHECK_R(glGetUniformLocation(shaderProgram, "view"));
  LOGINFO("uniView=%d", uniView);
  GLint uniProj = GL_CHECK_R(glGetUniformLocation(shaderProgram, "proj"));
  LOGINFO("uniProj=%d", uniProj);

  // Load textures

  GLuint texKitten;
  {
    GL_CHECK(glGenTextures(1, &texKitten));

    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texKitten));

    // Set a border color in case we want to use GL_CLAMP_TO_BORDER
    GLfloat border_color[] = { 1.0f, 0.0f, 1.0f, 1.0f };
    GL_CHECK(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color));
    // Set wrapping
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
    // Set filtering
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    // load texture
    int tex_width, tex_height, tex_channels;
    uint8_t* tex_data = stbi_load("images/sample.png", &tex_width, &tex_height, &tex_channels, 3);
    // TODO assert macro - or just use GLog?
    assert(tex_data);
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data));
    // Free the image data
    stbi_image_free(tex_data);
    // Generate mipmaps after loading
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
  }

  GLuint texPuppy;
  {
    GL_CHECK(glGenTextures(1, &texPuppy));

    GL_CHECK(glActiveTexture(GL_TEXTURE1));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, texPuppy));

    // Set a border color in case we want to use GL_CLAMP_TO_BORDER
    GLfloat border_color[] = { 1.0f, 0.0f, 1.0f, 1.0f };
    GL_CHECK(glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color));
    // Set wrapping
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));
    // Set filtering
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    // load texture
    int tex_width, tex_height, tex_channels;
    uint8_t* tex_data = stbi_load("images/sample2.png", &tex_width, &tex_height, &tex_channels, 3);
    // TODO assert macro - or just use GLog?
    assert(tex_data);
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_width, tex_height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data));
    // Free the image data
    stbi_image_free(tex_data);
    // Generate mipmaps after loading
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
  }

  // GUI state
  bool show_test_window = false;
  bool show_another_window = false;
  ImVec4 clear_color = ImColor(114, 144, 154);

  ImVec4 eye_pos = ImVec4(3.0f, 3.0f, 3.0f, 1.0f);

  float fov_y = 60.0f;

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
        ImGui::ColorEdit3("clear color", (float*)&clear_color);
        ImGui::DragFloat3("Eye Pos", (float*)&eye_pos, 1.0f, -10.0f, 10.0f);
        if (ImGui::Button("Test Window")) show_test_window ^= 1;
        if (ImGui::Button("Another Window")) show_another_window ^= 1;
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    }

    // 2. Show another simple window, this time using an explicit Begin/End pair
    if (show_another_window)
    {
        ImGui::SetNextWindowSize(ImVec2(200,100), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Another Window", &show_another_window);
        ImGui::Text("Hello");
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
    // Bind textures to samplers
    GL_CHECK(glUseProgram(shaderProgram));

    GL_CHECK(glBindVertexArray(vao));

    glm::mat4 model;
    model = glm::rotate(
        model,
        0.0f,
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GL_CHECK(glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model)));

    glm::mat4 view = glm::lookAt(
        glm::vec3(eye_pos.x, eye_pos.y, eye_pos.z),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GL_CHECK(glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view)));

    glm::mat4 proj = glm::perspective(
        glm::radians(fov_y),
        1024.0f / 768.0f,
        0.01f,
        100.0f
    );
    GL_CHECK(glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj)));

#if 1
    GL_CHECK(glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0));
#else
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 36));
#endif

    // Render GUI
    ImGui::Render();

    // Swap
    SDL_GL_SwapWindow(window);
  }

  LOGINFO("Exiting...");

  // Cleanup

  GL_CHECK(glDeleteProgram(shaderProgram));
  GL_CHECK(glDeleteShader(fragmentShader));
  GL_CHECK(glDeleteShader(vertexShader));

  GL_CHECK(glDeleteTextures(1, &texKitten));
  GL_CHECK(glDeleteTextures(1, &texPuppy));

  GL_CHECK(glDeleteBuffers(1, &ebo));
  GL_CHECK(glDeleteBuffers(1, &vbo));

  GL_CHECK(glDeleteVertexArrays(1, &vao));

  ImGui_ImplSdlGL3_Shutdown();

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
