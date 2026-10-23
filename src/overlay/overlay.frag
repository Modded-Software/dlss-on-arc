#version 450

layout(set = 0, binding = 0) uniform sampler2D atlas_tex;

layout(push_constant) uniform Push {
    vec2 size;
    float premult;
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

void main() {
    float cov = v_uv.x < 0.0 ? 1.0 : texture(atlas_tex, v_uv).r;
    float a = v_color.a * cov;
    vec3 rgb = v_color.rgb * mix(1.0, a, pc.premult);
    out_color = vec4(rgb, a);
}
