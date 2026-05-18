#[compute]
#version 450

// ─────────────────────────────────────────────────────────────────────────────
// Brush Paint Shader — writes directly into the current ping-pong field.
//
// This shader is unchanged from the 9×9 version; it has no dependency on
// the kernel system.  It runs before the simulation step each frame.
//
// Blend mode: smooth alpha falloff from centre to edge, controllable via
// the `hardness` push constant (1.0 = hard-edged, 0.0 = fully feathered).
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Current field — read-write (no ping-pong; brush paints in-place)
layout(set = 0, binding = 0, rgba16f) uniform image2D field_img;

// Push constants — 12 floats = 48 bytes
layout(push_constant) uniform PushConstants {
    float grid_w;
    float grid_h;
    float brush_x;       // centre x in grid pixels
    float brush_y;       // centre y in grid pixels
    float brush_radius;  // radius in grid pixels
    float color_r;
    float color_g;
    float color_b;
    float hardness;      // 1.0 = sharp edge, 0.0 = fully feathered
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = ivec2(int(pc.grid_w), int(pc.grid_h));

    if (coord.x >= size.x || coord.y >= size.y) return;

    // Distance from brush centre in grid-pixel space
    float dx   = float(coord.x) - pc.brush_x;
    float dy   = float(coord.y) - pc.brush_y;
    float dist = sqrt(dx * dx + dy * dy);

    // Early discard — outside brush circle entirely.
    // This branch is coherent across the workgroup, so divergence cost is
    // minimal; it also saves the imageLoad/Store bandwidth for distant threads.
    if (dist > pc.brush_radius) return;

    // Alpha: linear falloff in [0,1], then blended between feathered and hard
    // by the hardness parameter.
    float t     = 1.0 - clamp(dist / pc.brush_radius, 0.0, 1.0);
    float alpha = mix(t, step(0.0, t), pc.hardness);

    vec4 current = imageLoad(field_img, coord);
    vec4 painted  = vec4(pc.color_r, pc.color_g, pc.color_b, 1.0);

    imageStore(field_img, coord, mix(current, painted, alpha));
}
