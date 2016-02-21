#version 150 core

in vec3 Color;
in float flogz;

out vec4 outColor;

uniform float Fcoef_half;

void main()
{
  outColor = vec4(Color, 1.0);
  gl_FragDepth = log2(flogz) * Fcoef_half;
}