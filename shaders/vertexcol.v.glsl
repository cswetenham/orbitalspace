#version 150 core

in vec3 position;
in vec3 color;

out vec3 Color;
out float flogz;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

// See http://outerra.blogspot.co.uk/2013/07/logarithmic-depth-buffer-optimizations.html

uniform float Fcoef;

void main()
{
  Color = color;
  gl_Position = proj * view * model * vec4(position, 1.0);
  // assuming gl_Position was already computed
  // log-z buffer
  gl_Position.z = log2(max(1e-6, 1.0 + gl_Position.w)) * Fcoef - 1.0;
  // interpolated value for fragment shader
  flogz = 1.0 + gl_Position.w;
}