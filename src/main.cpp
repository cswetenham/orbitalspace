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

// SDL headers
#include <SDL.h>

// Image loading
// Do this before you include this file in *one* C or C++ file to create the implementation.
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "glog/logging.h"

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
    LOG(ERROR) << "OpenGL Error: " << err << " " << (char const*)gluErrorString(err) << "(" << file << ":" << line << ")";
    err = glGetError();
  }
  return t;
}

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
    fprintf(stderr, "Unable to open %s for reading\n", filename);
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
)
{
  GLint log_length;
  GL_CHECK(glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length));
  char* log = (char*)malloc(log_length);
  GL_CHECK(glGetShaderInfoLog(object, log_length, NULL, log));
  fprintf(stderr, "Error compiling shader %s: %s", shader_path, log);
  free(log);
}

void show_program_info_log(
  char const* vertex_path,
  char const* fragment_path,
  GLuint object
)
{
  GLint log_length;
  GL_CHECK(glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length));
  char* log = (char*)malloc(log_length);
  GL_CHECK(glGetProgramInfoLog(object, log_length, NULL, log));
  fprintf(stderr, "Error linking shaders %s, %s: %s", vertex_path, fragment_path, log);
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
    fprintf(stderr, "Failed to compile %s:\n", filename);
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
      fprintf(stderr, "Failed to link shader program:\n");
      for (int i = 0; i < shader_count; ++i) {
        fprintf(stderr, shader_infos[i].shader_path);
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

extern "C"
int main(int argc, char *argv[])
{
  SDL_Init(SDL_INIT_VIDEO);
  
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

  SDL_Window* window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_OPENGL);
  
  SDL_GLContext context = SDL_GL_CreateContext(window);
  
  // Load OpenGL and OpenGL functions
  gladLoadGLLoader(SDL_GL_GetProcAddress);
  printf("OpenGL loaded\n");
  printf("Vendor:   %s\n", GL_CHECK_R(glGetString(GL_VENDOR)));
  printf("Renderer: %s\n", GL_CHECK_R(glGetString(GL_RENDERER)));
  printf("Version:  %s\n", GL_CHECK_R(glGetString(GL_VERSION)));
  
  // TODO enable sRGB framebuffer + textures, and use it
  
  // Explicitly set culling orientation and disable/enable culling
  GL_CHECK(glFrontFace(GL_CCW));
  //GL_CHECK(glDisable(GL_CULL_FACE));
  GL_CHECK(glEnable(GL_CULL_FACE));
  
  // Vertex array object: relates vbos to a shader program somehow??
  GLuint vao;
  GL_CHECK(glGenVertexArrays(1, &vao));
  GL_CHECK(glBindVertexArray(vao));

  GLuint vbo;
  GL_CHECK(glGenBuffers(1, &vbo)); // Generate 1 buffer
  
  GLfloat vertices[] = {
  //   X,     Y,    R,    G,    B,    S,    T
   -0.5f,  0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // Top-left 
    0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // Top-right
    0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // Bottom-right
   -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f  // Bottom-left
  };
  
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vbo));
  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW));
  
  GLuint ebo;
  GL_CHECK(glGenBuffers(1, &ebo));
  
  GLuint elements[] = {
    0, 2, 1,
    0, 3, 2
  };
  
  GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo));
  GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW));
  
  
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
  
  // A basic clip-space shader
  
  GLuint vertexShader = make_shader(GL_VERTEX_SHADER, "shaders/ch2.v.glsl");
  GLuint fragmentShader = make_shader(GL_FRAGMENT_SHADER, "shaders/ch2.f.glsl");
  ShaderInfo shader_infos[2];
  shader_infos[0].shader_path = "shaders/ch2.v.glsl";
  shader_infos[0].shader_id = vertexShader;
  shader_infos[1].shader_path = "shaders/ch2.f.glsl";
  shader_infos[1].shader_id = fragmentShader;
  GLuint shaderProgram = make_program(2, shader_infos);
  GL_CHECK(glUseProgram(shaderProgram));

  // Bind position attribute to vertex array
  GLint posAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "position"));
  GL_CHECK(glEnableVertexAttribArray(posAttrib));
  GL_CHECK(glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 7*sizeof(GLfloat), 0));
  
  GLint colAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "color"));
  GL_CHECK(glEnableVertexAttribArray(colAttrib));
  GL_CHECK(glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 7*sizeof(GLfloat), (void*)(2*sizeof(GLfloat))));
  
  GLint texAttrib = GL_CHECK_R(glGetAttribLocation(shaderProgram, "texcoord"));
  GL_CHECK(glEnableVertexAttribArray(texAttrib));
  GL_CHECK(glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 7*sizeof(GLfloat), (void*)(5*sizeof(GLfloat))));
  
  // Bind textures to samplers
  glUniform1i(glGetUniformLocation(shaderProgram, "texKitten"), 0);
  glUniform1i(glGetUniformLocation(shaderProgram, "texPuppy"), 1);
  
  SDL_Event windowEvent;
  while (true) {
    // Input handling
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
    
    // Clear the screen to black
    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
    
    // Rendering
    GL_CHECK(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0));
    // GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 3));

    // Swap
    SDL_GL_SwapWindow(window);
  }
  
  GL_CHECK(glDeleteProgram(shaderProgram));
  GL_CHECK(glDeleteShader(fragmentShader));
  GL_CHECK(glDeleteShader(vertexShader));
  
  GL_CHECK(glDeleteTextures(1, &texKitten));
  GL_CHECK(glDeleteTextures(1, &texPuppy));
  
  GL_CHECK(glDeleteBuffers(1, &ebo));
  GL_CHECK(glDeleteBuffers(1, &vbo));
  
  GL_CHECK(glDeleteVertexArrays(1, &vao));
  
  SDL_GL_DeleteContext(context);

  SDL_Quit();

  return 0;
}
