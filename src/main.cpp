// C headers

#include <stdio.h>

// C++ headers

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

// TODO cleanup
#define GL_CHECK(expr) expr; _gl_check(__FILE__, __LINE__, #expr);

inline void _gl_check(char const* file, int line, char const* expr) {
  GLenum err = glGetError();
  while(err!=GL_NO_ERROR) {
    fprintf(stderr, "OpenGL Error: %d %s (%s:%d)\n", err, (char const*)gluErrorString(err), file, line);
    err = glGetError();
  }
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

// TODO cleanup
static GLuint make_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint program = GL_CHECK(glCreateProgram());

    GL_CHECK(glAttachShader(program, vertex_shader));
    GL_CHECK(glAttachShader(program, fragment_shader));
    
    // TODO well hrm we'll need to pass in an array or something
    glBindFragDataLocation(program, 0, "outColor"); // not strictly needed since only 1 output
    
    GL_CHECK(glLinkProgram(program));

    GLint program_ok;
    GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &program_ok));
    if (!program_ok) {
        fprintf(stderr, "Failed to link shader program:\n"); // TODO report source filenames
        show_program_info_log("", "", program);
        GL_CHECK(glDeleteProgram(program));
        return 0;
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
  printf("Vendor:   %s\n", glGetString(GL_VENDOR));
  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("Version:  %s\n", glGetString(GL_VERSION));
  
  // A triangle
  float vertices[] = {
    0.0f,  0.5f, // Vertex 1 (X, Y)
    0.5f, -0.5f, // Vertex 2 (X, Y)
   -0.5f, -0.5f  // Vertex 3 (X, Y)
  };

  GLuint vbo;
  glGenBuffers(1, &vbo); // Generate 1 buffer
  
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  
  // A basic clip-space shader
  
  GLuint vertexShader = make_shader(GL_VERTEX_SHADER, "shaders/ch2.v.glsl");
  GLuint fragmentShader = make_shader(GL_FRAGMENT_SHADER, "shaders/ch2.f.glsl");
  GLuint shaderProgram = make_program(vertexShader, fragmentShader);
  glUseProgram(shaderProgram);
  
  // Vertex array object: binding of 1 or more vertex buffer objects to a shader program
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  
  GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
  glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(posAttrib);
  
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
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Rendering
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Swap
    SDL_GL_SwapWindow(window);
  }
  
  SDL_GL_DeleteContext(context);

  SDL_Quit();

  return 0;
}
