#[compute]
#version 450

// ─────────────────────────────────────────────────────────────────────────────
// Procedural Radial-Kernel Lenia Simulation — Vulkan Compute Shader
// Workgroup: 8×8 threads
//
// ARCHITECTURE:
//
//   Three independent per-channel radial kernels.  Each output channel R/G/B
//   uses its own spatial kernel (kernel_radius_r/g/b) to convolve the full
//   RGB field, then applies its own growth function (mu_r/g/b, sigma_r/g/b).
//
// PER-CHANNEL CONVOLUTION:
//
//   conv_kr = weighted_avg(field_rgb, kernel_r)  → feeds R output
//   conv_kg = weighted_avg(field_rgb, kernel_g)  → feeds G output
//   conv_kb = weighted_avg(field_rgb, kernel_b)  → feeds B output
//
//   The loop runs once to max(R_r, R_g, R_b).  All three kernel weights are
//   evaluated as a branchless vec3 per tap — zero loop-count overhead when
//   radii differ, and SIMD-friendly when they are equal.
//
// KERNEL WEIGHT FORMULA (for channel c, tap distance d):
//
//   half_r   = kernel_radius_c * 0.5
//   inv_band = 5.0 / kernel_radius_c        (== 1 / (radius * 0.2))
//   w_c      = exp(−((d − half_r) * inv_band)²) * step(d, kernel_radius_c)
//
// CROSS-CHANNEL MIX:
//
//   Each output channel's neighbourhood signal is the dot of its own
//   normalised conv result with its row of the coefficient matrix:
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
// REMOVED FROM PREVIOUS VERSION:
//   • max_r/g/b  — per-channel upper clamp (hardcoded to 1.0)
//   • exp_r/g/b  — per-channel nonlinear growth exponent (always linear now)
//   • Single shared mu / sigma / kernel_radius (replaced by per-channel)
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ── Bindings ──────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D src_tex;   // REPEAT/NEAREST, toroidal
layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D dst_img;

// ── Push Constants ────────────────────────────────────────────────────────────
// 21 floats = 84 bytes.  Layout MUST match GDScript PackedFloat32Array exactly:
//
//  [0]  grid_w           [1]  grid_h
//  [2]  dt
//  [3]  mu_r             [4]  mu_g            [5]  mu_b
//  [6]  sigma_r          [7]  sigma_g         [8]  sigma_b
//  [9]  kernel_radius_r  [10] kernel_radius_g [11] kernel_radius_b
//  [12] coeff_rr        [13] coeff_rg        [14] coeff_rb
//  [15] coeff_gr        [16] coeff_gg        [17] coeff_gb
//  [18] coeff_br        [19] coeff_bg        [20] coeff_bb

layout(push_constant) uniform PC {
    float grid_w;
    float grid_h;
    float dt;

    float mu_r;     // Growth bell-curve centre, R channel
    float mu_g;     // Growth bell-curve centre, G channel
    float mu_b;     // Growth bell-curve centre, B channel

    float sigma_r;  // Growth bell-curve half-width, R channel
    float sigma_g;  // Growth bell-curve half-width, G channel
    float sigma_b;  // Growth bell-curve half-width, B channel

    float kernel_radius_r;  // Spatial kernel radius, R output
    float kernel_radius_g;  // Spatial kernel radius, G output
    float kernel_radius_b;  // Spatial kernel radius, B output

    float coeff_rr;  float coeff_rg;  float coeff_rb;  // R output row
    float coeff_gr;  float coeff_gg;  float coeff_gb;  // G output row
    float coeff_br;  float coeff_bg;  float coeff_bb;  // B output row
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

    if (coord.x >= dims.x || coord.y >= dims.y) return;

    // ── Per-invocation constants ─────────────────────────────────────────────
    vec2 inv_dims  = 1.0 / vec2(dims);
    vec2 center_uv = (vec2(coord) + 0.5) * inv_dims;

    // Pack the three channel radii into a vec3 for SIMD weight evaluation.
    // These are uniform values — the driver scalar-promotes them on Adreno.
    vec3 kr       = vec3(pc.kernel_radius_r, pc.kernel_radius_g, pc.kernel_radius_b);
    vec3 half_kr  = kr * 0.5;           // ring peak distance per channel
    vec3 inv_band = 5.0 / kr;           // reciprocal ring width (= 1 / (kr * 0.2))

    // One unified loop bound.  When all radii are equal this is exact.
    // When they differ, the larger kernel drives the bound and the smaller
    // channels' step() returns 0 naturally for the extra outer taps.
    int R_max = int(ceil(max(kr.x, max(kr.y, kr.z))));

    // ── Convolution ──────────────────────────────────────────────────────────
    // Each channel accumulates the full RGB field weighted by its own kernel.
    //   conv_kr[c] = Σ field_rgb[c] * w_r(tap)
    //   conv_kg[c] = Σ field_rgb[c] * w_g(tap)
    //   conv_kb[c] = Σ field_rgb[c] * w_b(tap)
    //
    // All three scalar weights w = (w_r, w_g, w_b) are computed as a single
    // vec3 per tap with no branches.

    vec3 conv_kr = vec3(0.0);
    vec3 conv_kg = vec3(0.0);
    vec3 conv_kb = vec3(0.0);
    vec3 w_sums  = vec3(0.0);  // per-channel weight sums for normalisation

    for (int dy = -R_max; dy <= R_max; ++dy) {
        float fdy = float(dy);

        for (int dx = -R_max; dx <= R_max; ++dx) {
            float fdx = float(dx);
            float d2  = fdx * fdx + fdy * fdy;
            float d   = sqrt(d2);

            // Branchless disk cutoff: compare squared distances to squared radii.
            vec3 in_disk = step(vec3(d2), kr * kr);

            // Gaussian ring weight, evaluated simultaneously for all 3 kernels.
            vec3 band = (d - half_kr) * inv_band;
            vec3 w    = exp(-band * band) * in_disk;

            vec3 smpl = texture(src_tex, center_uv + vec2(fdx, fdy) * inv_dims).rgb;

            conv_kr += smpl * w.r;
            conv_kg += smpl * w.g;
            conv_kb += smpl * w.b;
            w_sums  += w;
        }
    }

    // ── Normalisation ────────────────────────────────────────────────────────
    // Multiply by reciprocal; max() prevents NaN if a kernel is degenerate
    // (kernel_radius < 0.5 yields w_sum ≈ 0).
    conv_kr *= 1.0 / max(w_sums.r, 1.0e-7);
    conv_kg *= 1.0 / max(w_sums.g, 1.0e-7);
    conv_kb *= 1.0 / max(w_sums.b, 1.0e-7);

    // ── Cross-channel linear mix ─────────────────────────────────────────────
    // Each output channel uses its own conv result (not a shared one), then
    // takes a linear combination across input colours.
    //
    //   u_r = conv_kr · [coeff_rr, coeff_rg, coeff_rb]
    //   u_g = conv_kg · [coeff_gr, coeff_gg, coeff_gb]
    //   u_b = conv_kb · [coeff_br, coeff_bg, coeff_bb]
    float u_r = dot(conv_kr, vec3(pc.coeff_rr, pc.coeff_rg, pc.coeff_rb));
    float u_g = dot(conv_kg, vec3(pc.coeff_gr, pc.coeff_gg, pc.coeff_gb));
    float u_b = dot(conv_kb, vec3(pc.coeff_br, pc.coeff_bg, pc.coeff_bb));

    // ── Per-channel growth ───────────────────────────────────────────────────
    vec3 g_vec = vec3(
        growth(u_r, pc.mu_r, pc.sigma_r),
        growth(u_g, pc.mu_g, pc.sigma_g),
        growth(u_b, pc.mu_b, pc.sigma_b)
    );

    // ── Euler integration ────────────────────────────────────────────────────
    vec3 cur  = texture(src_tex, center_uv).rgb;
    vec3 next = clamp(cur + pc.dt * g_vec, 0.0, 1.0);

    imageStore(dst_img, coord, vec4(next, 1.0));
}
