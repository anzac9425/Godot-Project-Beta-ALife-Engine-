#[compute]
#version 450

// ─────────────────────────────────────────────────────────────────────────────
// Procedural Radial-Kernel Lenia Simulation — Vulkan Compute Shader ( Adreno 740 Targetted )
// Workgroup: 16×16 threads
//
// ARCHITECTURE:
//
//   CPU precomputes a per-channel radial kernel table and uploads it as a 2D
//   texture. The shader only fetches kernel weights — no per-tap sqrt/exp,
//   no per-pixel kernel normalization, and no branchy disk cutoff.
//
//   Each output channel R/G/B uses its own kernel weights to convolve the full
//   RGB field, then applies its own growth function (mu_r/g/b, sigma_r/g/b).
//
// PER-CHANNEL CONVOLUTION:
//
//   conv_kr = weighted_avg(field_rgb, kernel_r)  → feeds R output
//   conv_kg = weighted_avg(field_rgb, kernel_g)  → feeds G output
//   conv_kb = weighted_avg(field_rgb, kernel_b)  → feeds B output
//
//   The kernel texture stores normalized weights in RGB:
//     .r = kernel for R output
//     .g = kernel for G output
//     .b = kernel for B output
//
// CROSS-CHANNEL MIX:
//
//   Each output channel's neighbourhood signal is the dot of its own
//   convolved RGB vector with its row of the coefficient matrix:
//
//     u_r = dot(conv_kr, vec3(coeff_rr, coeff_rg, coeff_rb))
//     u_g = dot(conv_kg, vec3(coeff_gr, coeff_gg, coeff_gb))
//     u_b = dot(conv_kb, vec3(coeff_br, coeff_bg, coeff_bb))
//
// GROWTH FUNCTION (per-channel mu, sigma):
//
//   G_c(u) = 2·exp(−0.5·((u − mu_c) / sigma_c)²) − 1
//
// EULER STEP:
//
//   next_c = clamp(cur_c + dt · G_c(u_c), 0, 1)
//
// NOTES:
//
//   • Kernel radii are no longer used in the shader.
//   • The kernel texture dimension defines the convolution radius.
//   • The host must upload a kernel texture whose size is odd:
//       side = 2 * R_max + 1
//   • The kernel texture must be normalized on the CPU.
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// ── Bindings ──────────────────────────────────────────────────────────────────
// 0: source field texture, toroidal repeat, sampled with nearest filtering
// 1: destination storage image
// 2: precomputed kernel texture, sampled with nearest filtering
layout(set = 0, binding = 0) uniform sampler2D src_tex;
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D dst_img;
layout(set = 0, binding = 2) uniform sampler2D kernel_tex;

// ── Push Constants ────────────────────────────────────────────────────────────
// 18 floats = 72 bytes
//
// Layout:
//   [0] grid_w
//   [1] grid_h
//   [2] dt
//   [3] mu_r
//   [4] mu_g
//   [5] mu_b
//   [6] sigma_r
//   [7] sigma_g
//   [8] sigma_b
//   [9]  coeff_rr
//   [10] coeff_rg
//   [11] coeff_rb
//   [12] coeff_gr
//   [13] coeff_gg
//   [14] coeff_gb
//   [15] coeff_br
//   [16] coeff_bg
//   [17] coeff_bb
layout(push_constant) uniform PC {
    float grid_w;
    float grid_h;
    float dt;

    float mu_r;
    float mu_g;
    float mu_b;

    float sigma_r;
    float sigma_g;
    float sigma_b;

    float coeff_rr;
    float coeff_rg;
    float coeff_rb;

    float coeff_gr;
    float coeff_gg;
    float coeff_gb;

    float coeff_br;
    float coeff_bg;
    float coeff_bb;
} pc;

// ─────────────────────────────────────────────────────────────────────────────
// Growth function — Lenia Gaussian bell, range [−1, +1].
//   G(u) = 2·exp(−0.5·((u − mu) / sigma)²) − 1
// ─────────────────────────────────────────────────────────────────────────────
float growth(float u, float mu, float sigma) {
    float z = (u - mu) / sigma;
    return 2.0 * exp(-0.5 * z * z) - 1.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims  = ivec2(int(pc.grid_w), int(pc.grid_h));

    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }

    vec2 inv_dims  = 1.0 / vec2(dims);
    vec2 center_uv = (vec2(coord) + 0.5) * inv_dims;

    // Kernel table size defines the convolution radius.
    int R_max = (textureSize(kernel_tex, 0).x - 1) / 2;

    // Accumulate RGB field separately for each output kernel.
    vec3 conv_kr = vec3(0.0);
    vec3 conv_kg = vec3(0.0);
    vec3 conv_kb = vec3(0.0);

    for (int dy = -R_max; dy <= R_max; ++dy) {
        for (int dx = -R_max; dx <= R_max; ++dx) {
            vec3 w    = texelFetch(kernel_tex, ivec2(dx + R_max, dy + R_max), 0).rgb;
            vec3 smpl = texture(src_tex, center_uv + vec2(dx, dy) * inv_dims).rgb;

            conv_kr += smpl * w.r;
            conv_kg += smpl * w.g;
            conv_kb += smpl * w.b;
        }
    }

    // Cross-channel linear mix.
    float u_r = dot(conv_kr, vec3(pc.coeff_rr, pc.coeff_rg, pc.coeff_rb));
    float u_g = dot(conv_kg, vec3(pc.coeff_gr, pc.coeff_gg, pc.coeff_gb));
    float u_b = dot(conv_kb, vec3(pc.coeff_br, pc.coeff_bg, pc.coeff_bb));

    // Per-channel growth.
    vec3 g_vec = vec3(
        growth(u_r, pc.mu_r, pc.sigma_r),
        growth(u_g, pc.mu_g, pc.sigma_g),
        growth(u_b, pc.mu_b, pc.sigma_b)
    );

    // Euler integration.
    vec3 cur  = texture(src_tex, center_uv).rgb;
    vec3 next = clamp(cur + pc.dt * g_vec, 0.0, 1.0);

    imageStore(dst_img, coord, vec4(next, 1.0));
}