#version 450

struct SSBO {
  vec2 pos;
  vec2 vel;
};

layout (binding = 0) uniform Uniform {
  float delta_time;
} u;

layout (binding = 1) readonly buffer SSBOs {
  SSBO data [ ];
} ssbos;

layout (location = 0) out vec3 fragColor;

// Angle is of course in radians
// Rotates in an anti-clockwise direction
vec2 rotate_vec(vec2 v, float angle)
{
  float x = cos(angle)*v.x - sin(angle)*v.y;
  float y = sin(angle)*v.x - cos(angle)*v.y;

  return vec2(x,y);
}

void main() {
  int index = gl_VertexIndex / 3;
  vec2 position = ssbos.data[index].pos;
  vec2 vel = ssbos.data[index].vel;
  int triangle_vertex_idx = gl_VertexIndex % 3;

  vec2 direction = normalize(vel);
  float triangle_radius = 0.01;

  vec2 offset = vec2(0,0);

  if (triangle_vertex_idx == 0) {
    //offset = rotate_vec(direction, radians(240.));
    offset = vec2(0., -1.);
  } else if (triangle_vertex_idx == 1) {
    //offset = rotate_vec(direction, radians(120.));
    offset = normalize(vec2(1., 1.));
  } else {
    //offset = rotate_vec(direction, radians(0.));
    offset = normalize(vec2(-1., 1.));
  }

  position += offset*triangle_radius;

  gl_Position = vec4(position, 0.0, 1.0);
  fragColor = vec3(direction.y, direction.x, 1.0);
}
