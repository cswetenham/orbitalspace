#version 150

in vec3 Color;
in vec2 Texcoord;

out vec4 outColor;

uniform sampler2D tex; // I don't have to bind/set this externally??

void main()
{
  outColor = texture(tex, Texcoord) * vec4(Color, 1.0);
}