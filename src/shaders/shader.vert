#version 450

layout (binding = 0) uniform Uniform {
  float looped;
} u;

layout (location = 0) in vec2 inPosition;

layout (location = 0) out vec3 fragColor;

void main() {
  gl_Position = vec4(inPosition + u.looped, 0.0, 1.0);
  fragColor = vec3(0.0, 1.0, 0.0);
}
