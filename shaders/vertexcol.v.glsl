#version 150 core

in vec4 position;
in vec4 color;

out vec4 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
  Color = color;
  gl_Position = proj * view * model * position;
}