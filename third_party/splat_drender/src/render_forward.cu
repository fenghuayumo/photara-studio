// Forward render kernels: fused per-Gaussian preprocessing, double-sort
// instance emission, and the single-pass tile blender producing color,
// alpha, normal and median depth.
#include "device/geometry.cuh"
#include "device/tiles.cuh"
#include "kernels.h"
#include "splat_drender/buffers.h"
#include "splat_drender/camera.h"

#include <cooperative_groups.h>
#include <type_traits>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

namespace cg = cooperative_groups;
using namespace splat_drender;

namespace splat_drender::kernels {

// Ordered 32-bit key for positive floats: monotonic map depth -> uint.
SD_D2 inline unsigned depth_bits(float d) { return __float_as_uint(d); }

// Fuses EWA projection, conic, SH color, tile counting, depth key, screen
// bounds and the visibility scan flag into one launch.
__global__ void preprocess_gaussians(
    const int count, const int sh_degree, const int sh_bases,
    const float* __restrict__ means, const float* __restrict__ sh,
    const float* __restrict__ colors, const float* __restrict__ opacities,
    const float* __restrict__ scales, const float* __restrict__ rotations,
    const float* __restrict__ cov6, const float* __restrict__ view,
    const float* __restrict__ camera_center, CameraIntrinsics K, int width,
    int height, float kernel_size, float scale_modifier, int grid_x,
    int grid_y, int wrap_width, ws::GaussianState st,
    int* __restrict__ radii) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    st.tiles_touched[i] = 0;
    st.visible_flag[i] = 0;
    st.radius[i] = 0;
    radii[i] = 0;

    const float3 mean =
        make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    const float3 t = mat::xform_point(mean, view);
    if (!camera_visible(t, K.mode)) return;

    geo::Splat splat;
    const float3* scale_ptr =
        scales ? reinterpret_cast<const float3*>(scales) + i : nullptr;
    const float4* rot_ptr =
        rotations ? reinterpret_cast<const float4*>(rotations) + i : nullptr;
    const float* cov_ptr = cov6 ? cov6 + 6 * i : nullptr;
    if (!geo::project_splat(mean, view, K, width, height, kernel_size,
                            scale_modifier, scale_ptr, rot_ptr, cov_ptr,
                            splat))
        return;

    const Projection p = project(t, K, width, height);
    if (!p.valid) return;

    const float det =
        splat.cov2d[0] * splat.cov2d[2] - splat.cov2d[1] * splat.cov2d[1];
    if (det == 0.f) return;
    const float det_inv = 1.f / det;
    const float4 conic_opacity =
        make_float4(splat.cov2d[2] * det_inv, -splat.cov2d[1] * det_inv,
                    splat.cov2d[0] * det_inv, opacities[i] * splat.coef);

    const float mid = 0.5f * (splat.cov2d[0] + splat.cov2d[2]);
    const float root = sqrtf(fmaxf(0.1f, mid * mid - det));
    const float radius = ceilf(3.f * sqrtf(fmaxf(mid + root, mid - root)));

    const unsigned touched =
        tiles::enumerate(make_float2(p.pixel_x, p.pixel_y), conic_opacity,
                         grid_x, grid_y, unsigned(i), 0, nullptr, nullptr,
                         wrap_width);
    if (touched == 0) return;

    // Point-query preprocessing passes neither sh nor colors; blended color
    // is never read on that path, so skip the per-view SH evaluation exactly
    // like the reference sampleSDF preprocessing (it feeds a dummy color).
    if (sh != nullptr) {
        const float3 rgb = sh::evaluate(
            i, sh_degree, sh_bases, reinterpret_cast<const float3*>(means),
            make_float3(camera_center[0], camera_center[1], camera_center[2]),
            sh, st.clamped);
        st.rgb[i] = rgb;
    }

    st.mean2d[i] = make_float2(p.pixel_x, p.pixel_y);
    st.depth_key[i] = depth_bits(norm3df(t.x, t.y, t.z));
    st.ray_plane[i] = splat.ray_plane;
    st.normal[i] = splat.normal;
    st.conic_opacity[i] = conic_opacity;
    st.tiles_touched[i] = touched;
    st.visible_flag[i] = 1;
    // All surviving lanes contribute once; avoid serializing every visible
    // Gaussian on the same two global counters. Visibility compaction still
    // uses the stable scan below, so this cannot reorder equal-depth splats.
    const unsigned mask = __activemask();
#if __CUDA_ARCH__ >= 800
    const unsigned warp_touched = __reduce_add_sync(mask, touched);
    if (int(threadIdx.x & 31) == __ffs(mask) - 1) {
        atomicAdd(st.n_visible, unsigned(__popc(mask)));
        atomicAdd(st.n_instances, warp_touched);
    }
#else
    atomicAdd(st.n_visible, 1u);
    atomicAdd(st.n_instances, touched);
#endif
    const int r = int(radius);
    st.radius[i] = r;
    radii[i] = r;
    // Match the opacity-aware tile support, which can extend past 3 sigma.
    // Store outward-rounded half-open integer bounds for warp rejection.
    const float threshold = 2.f * logf(conic_opacity.w / cfg::kAlphaFloor);
    const float ex = sqrtf(fmaxf(0.f, threshold * splat.cov2d[0]));
    const float ey = sqrtf(fmaxf(0.f, threshold * splat.cov2d[2]));
    st.screen_bounds[i] = make_ushort4(
        unsigned short(fmaxf(0.f, fminf(65535.f, floorf(p.pixel_x - ex)))),
        unsigned short(fmaxf(0.f, fminf(65535.f, floorf(p.pixel_x + ex) + 1.f))),
        unsigned short(fmaxf(0.f, fminf(65535.f, floorf(p.pixel_y - ey)))),
        unsigned short(fmaxf(0.f, fminf(65535.f, floorf(p.pixel_y + ey) + 1.f))));
}

// Emits (depth key, gaussian id) pairs in gaussian-index order; the stable
// radix sort then yields the exact depth order of the reference 64-bit key
// sort (ties keep index order).
__global__ void emit_depth_entries(
    const int count, const unsigned* __restrict__ visible_offset,
    const unsigned* __restrict__ visible_flag,
    const unsigned* __restrict__ depth_key, unsigned* __restrict__ key_out,
    unsigned* __restrict__ value_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || visible_flag[i] == 0) return;
    const unsigned pos = visible_offset[i] - 1u;
    key_out[pos] = depth_key[i];
    value_out[pos] = unsigned(i);
}

__global__ void gather_touched(
    const int visible_count, const unsigned* __restrict__ sorted_ids,
    const unsigned* __restrict__ tiles_touched,
    unsigned* __restrict__ compact_n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= visible_count) return;
    compact_n[i] = tiles_touched[sorted_ids[i]];
}

// Emits (tile id, gaussian id) instances in depth order.
__global__ void emit_instances(
    const int visible_count, const unsigned* __restrict__ depth_sorted_ids,
    const float2* __restrict__ mean2d,
    const float4* __restrict__ conic_opacity,
    const unsigned* __restrict__ compact_offset, int grid_x, int grid_y,
    int wrap_width, unsigned* __restrict__ tile_key,
    unsigned* __restrict__ instance_value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= visible_count) return;
    const unsigned g = depth_sorted_ids[i];
    const unsigned off = (i == 0) ? 0u : compact_offset[i - 1];
    tiles::enumerate(mean2d[g], conic_opacity[g], grid_x, grid_y, g, off,
                     tile_key, instance_value, wrap_width);
}

// Small-list path: emit in Gaussian index order to preserve depth ties.
__global__ void emit_packed_instances(int count, ws::GaussianState st,
    int grid_x, int grid_y, int wrap_width, unsigned long long* keys, unsigned* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || st.tiles_touched[i] == 0) return;
    const unsigned offset = i == 0 ? 0 : st.visible_offset[i - 1];
    tiles::enumerate(st.mean2d[i], st.conic_opacity[i], grid_x, grid_y,
        unsigned(i), offset, nullptr, values, wrap_width, keys, st.depth_key[i]);
}

__global__ void extract_packed_ranges(int count, const unsigned long long* keys, uint2* range) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned tile = unsigned(keys[i] >> 32);
    if (i == 0) range[tile].x = 0;
    else {
        const unsigned previous = unsigned(keys[i - 1] >> 32);
        if (tile != previous) { range[previous].y = unsigned(i); range[tile].x = unsigned(i); }
    }
    if (i == count - 1) range[tile].y = unsigned(count);
}

// Extracts per-tile instance ranges from the tile-sorted keys.
__global__ void extract_tile_ranges(
    const int instance_count, const unsigned* __restrict__ tile_key,
    uint2* __restrict__ range) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= instance_count) return;
    const unsigned tile = tile_key[i];
    if (i == 0) {
        range[tile].x = 0;
    } else {
        const unsigned prev = tile_key[i - 1];
        if (tile != prev) {
            range[prev].y = unsigned(i);
            range[tile].x = unsigned(i);
        }
    }
    if (i == instance_count - 1) range[tile].y = unsigned(instance_count);
}

// Single block: per-tile bucket counts, their inclusive scan, and the
// bucket->tile mapping. Entries beyond the exact bucket count map to tile 0
// so the upper-bound backward grid safely early-returns on them.
__global__ void bucket_offsets_kernel(int tiles, int buckets,
                                      const uint2* __restrict__ range,
                                      unsigned* __restrict__ bucket_count,
                                      unsigned* __restrict__ bucket_offset,
                                      unsigned* __restrict__ bucket_tile) {
    using Scan = cub::BlockScan<unsigned, cfg::kTileThreads>;
    __shared__ Scan::TempStorage scan;
    const int tid = threadIdx.x;
    unsigned carry = 0;
    for (int base = 0; base < tiles; base += cfg::kTileThreads) {
        const int t = base + tid;
        unsigned v = 0;
        if (t < tiles) {
            v = (range[t].y - range[t].x + 31u) >> 5;
            bucket_count[t] = v;
        }
        // CUB preserves the inclusive integer offsets without the eight
        // read/barrier/write/barrier stages of a Hillis-Steele scan.
        unsigned prefix, aggregate;
        Scan(scan).InclusiveSum(v, prefix, aggregate);
        if (t < tiles) bucket_offset[t] = carry + prefix;
        carry += aggregate;
        // All threads must finish using TempStorage before the next chunk.
        __syncthreads();
    }
    // blend_tile fills bucket_tile for real buckets; park the upper-bound
    // tail at tile 0 where the backward's range test rejects them.
    for (int b = carry + tid; b < buckets; b += cfg::kTileThreads)
        bucket_tile[b] = 0;
}

// Single-pass blender: color, alpha, accumulated normal and the
// median-depth bisection refine, with numbering-preserving warp culling.
template <bool GEOMETRY>
__global__ void __launch_bounds__(cfg::kTileThreads)
blend_tile(const uint2* __restrict__ tile_range,
           const unsigned* __restrict__ instance_value, int width, int height,
           CameraIntrinsics K, int wrap_width,
           const float2* __restrict__ mean2d,
           const float4* __restrict__ conic_opacity,
           const float3* __restrict__ rgb, const float* __restrict__ colors,
           const float4* __restrict__ ray_plane,
           const float3* __restrict__ normal,
           const ushort4* __restrict__ screen_bounds,
           unsigned* __restrict__ n_contrib,
           unsigned* __restrict__ max_contributor,
           const unsigned* __restrict__ bucket_offset,
           unsigned* __restrict__ bucket_tile, ws::PixelState pst,
           float3 background,
           float* __restrict__ out_color,
           float* __restrict__ out_alpha, float* __restrict__ out_normal,
           float* __restrict__ out_median_depth,
           float* __restrict__ visibility) {
    auto block = cg::this_thread_block();
    const unsigned tiles_x = (width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const unsigned tile_id =
        block.group_index().y * tiles_x + block.group_index().x;
    const int pix_min_x = block.group_index().x * cfg::kTileWidth;
    const int pix_min_y = block.group_index().y * cfg::kTileHeight;
    const int pix_x = pix_min_x + block.thread_index().x;
    const int pix_y = pix_min_y + block.thread_index().y;
    const bool inside = pix_x < width && pix_y < height;
    const unsigned pix_id = unsigned(width) * pix_y + pix_x;
    const float2 pixf = make_float2(float(pix_x), float(pix_y));

    const uint2 range = tile_range[tile_id];
    const int todo = range.y - range.x;
    const int rounds = (todo + cfg::kTileThreads - 1) / cfg::kTileThreads;
    // Global index of this tile's first bucket; every 32 instances of the
    // sorted list form one bucket whose 256-pixel state snapshot the
    // backward pass restarts from.
    const unsigned bucket_base =
        tile_id == 0 ? 0u : bucket_offset[tile_id - 1];
    for (int b = block.thread_rank(); b < (todo + 31) / 32;
         b += cfg::kTileThreads)
        bucket_tile[bucket_base + b] = tile_id;

    __shared__ unsigned sid[cfg::kTileThreads];
    __shared__ float2 sxy[cfg::kTileThreads];
    __shared__ float4 sconic[cfg::kTileThreads];
    __shared__ float3 srgb[cfg::kTileThreads];
    __shared__ float4 splane[cfg::kTileThreads];
    __shared__ float3 snormal[cfg::kTileThreads];

    float transmittance = 1.f;
    float color[3] = {0.f, 0.f, 0.f};
    float normal_acc[3] = {0.f, 0.f, 0.f};
    float depth_seed = 0.f;
    unsigned contributor = 0;
    unsigned last_contributor = 0;
    bool done = !inside;

    // ---- pass 1: color / alpha / normal / depth seed ----
    int remaining = todo;
    for (int round = 0; round < rounds; ++round, remaining -= cfg::kTileThreads) {
        if (__syncthreads_and(done)) break;
        const int progress = round * cfg::kTileThreads + block.thread_rank();
        if (range.x + progress < int(range.y)) {
            const unsigned g = instance_value[range.x + progress];
            sid[block.thread_rank()] = g;
            sxy[block.thread_rank()] = mean2d[g];
            sconic[block.thread_rank()] = conic_opacity[g];
            srgb[block.thread_rank()] =
                colors ? reinterpret_cast<const float3*>(colors)[g] : rgb[g];
            if constexpr (GEOMETRY) {
                splane[block.thread_rank()] = ray_plane[g];
                snormal[block.thread_rank()] = normal[g];
            }
        }
        __syncthreads();
        const int batch = min(cfg::kTileThreads, remaining);
        const int n_sub = (batch + 31) >> 5;
        for (int sub = 0; sub < n_sub; ++sub) {
            // Snapshot live pixel state at every 32-instance boundary; done
            // pixels terminated earlier never read their snapshot back.
            if (!done) {
                const std::size_t slot =
                    (std::size_t(bucket_base + round * 8 + sub) << 8) +
                    block.thread_rank();
                pst.snap_ct[slot] =
                    make_float4(color[0], color[1], color[2], transmittance);
                if constexpr (GEOMETRY)
                    pst.snap_normal[slot] = make_float4(
                        normal_acc[0], normal_acc[1], normal_acc[2], 0.f);
            }
            const int j_end = min(batch, (sub + 1) << 5);
            for (int j = sub << 5; !done && j < j_end; ++j) {
                contributor++;
                const bool active = !done;
                if (!active) continue;

                const float2 d =
                    make_float2(wrap_dx(sxy[j].x - pixf.x, wrap_width, K.mode),
                                sxy[j].y - pixf.y);
                const float4 co = sconic[j];
                const float power =
                    geo::gaussian_power(co, d.x, d.y);
                if (power > 0.f) continue;
                const float alpha = fminf(cfg::kAlphaClip, co.w * expf(power));
                if (alpha < cfg::kAlphaFloor) continue;
                const float test_t = transmittance * (1.f - alpha);
                if (test_t < cfg::kTransmittanceFloor) {
                    done = true;
                    continue;
                }

                const float a_t = alpha * transmittance;
                visibility[sid[j]] = 1.f;  // ADC+ observed flag, benign repeats
                color[0] += srgb[j].x * a_t;
                color[1] += srgb[j].y * a_t;
                color[2] += srgb[j].z * a_t;
                if constexpr (GEOMETRY) {
                    const float4 rp = splane[j];
                    const float3 n = snormal[j];
                    const float t = rp.x * d.x + rp.y * d.y + rp.z;
                    normal_acc[0] += n.x * a_t;
                    normal_acc[1] += n.y * a_t;
                    normal_acc[2] += n.z * a_t;
                    depth_seed = transmittance > 0.5f ? t : depth_seed;
                }
                transmittance = test_t;
                last_contributor = contributor;
            }
        }
    }

    __shared__ unsigned warp_scratch[cfg::kTileThreads / 32];
    const unsigned block_max =
        mat::block_max(last_contributor, warp_scratch);

    // ---- pass 2: median-depth bisection ----
    float median_depth = 0.f;
    if constexpr (GEOMETRY) {
        float depth_min = fmaxf(depth_seed - cfg::kDepthSeedWindow, 0.f);
        float depth_max = fmaxf(depth_seed + cfg::kDepthSeedWindow, 0.f);
        bool in_range = transmittance <= cfg::kDepthMinTransmittance;
        constexpr int S = cfg::kDepthSplit;
        float T_p[S + 1];

        auto refine = [&](auto first_tag) {
            constexpr bool first = decltype(first_tag)::value;
            constexpr int start = first ? 0 : 1;
            constexpr int end = first ? S + 1 : S;
#pragma unroll
            for (int s = start; s < end; ++s) T_p[s] = 1.f;
            const float interval = (depth_max - depth_min) * (1.f / float(S));
            bool local_done = !in_range;
            int todo2 = int(block_max);
            unsigned contributor2 = 0;
            const int depth_rounds = (int(block_max) + cfg::kTileThreads - 1) / cfg::kTileThreads;
            for (int round = 0; round < depth_rounds;
                 ++round, todo2 -= cfg::kTileThreads) {
                if (__syncthreads_and(local_done)) break;
                const int progress =
                    round * cfg::kTileThreads + block.thread_rank();
                if (progress < int(block_max)) {
                    const unsigned g = instance_value[range.x + progress];
                    sxy[block.thread_rank()] = mean2d[g];
                    sconic[block.thread_rank()] = conic_opacity[g];
                    splane[block.thread_rank()] = ray_plane[g];
                }
                __syncthreads();
                const int batch = min(cfg::kTileThreads, todo2);
                for (int j = 0; !local_done && j < batch; ++j) {
                    contributor2++;
                    const bool active = !local_done;
                    if (!active) continue;
                    local_done = contributor2 >= last_contributor;

                    const float2 d = make_float2(
                        wrap_dx(sxy[j].x - pixf.x, wrap_width, K.mode),
                        sxy[j].y - pixf.y);
                    const float4 co = sconic[j];
                    const float power =
                        geo::gaussian_power(co, d.x, d.y);
                    if (power > 0.f) continue;
                    const float alpha = fminf(cfg::kAlphaClip, co.w * expf(power));
                    if (alpha < cfg::kAlphaFloor) continue;

                    const float4 rp = splane[j];
                    const float t_peak = rp.x * d.x + rp.y * d.y + rp.z;
                    const float rsigma = rp.w;
                    const bool ball = rsigma > 0.f;
#pragma unroll
                    for (int s = start; s < end; ++s) {
                        const float ts = depth_min + interval * float(s);
                        const float delta = (ts - t_peak) * rsigma;
                        const float g = ball ? expf(-0.5f * delta * delta) : 0.f;
                        const float omg = 1.f - alpha * g;
                        const float rv = rsqrtf(omg);
                        T_p[s] *= (ts > t_peak ? (1.f - alpha) : omg) * rv;
                    }
                }
            }
            if constexpr (first) {
                in_range = (T_p[0] >= 0.5f) && (T_p[S] <= 0.5f) && in_range;
            }
            int start_id = 0;
#pragma unroll
            for (int p = 1; p < S; ++p) start_id = T_p[p] >= 0.5f ? p : start_id;
            depth_max = depth_min + (start_id + 1) * interval;
            depth_min = depth_min + (start_id + 0) * interval;
            T_p[0] = T_p[start_id];
            T_p[S] = T_p[start_id + 1];
        };

        refine(std::true_type{});
        for (int it = 0; it < cfg::kDepthRefinements - 1; ++it) refine(std::false_type{});

        const float w_max = __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[S]));
        const float w_min = 1.f - w_max;
        median_depth = in_range ? w_max * depth_max + w_min * depth_min : 0.f;
    }

    if (inside) {
        n_contrib[pix_id] = last_contributor;
        // Keep the no-background accumulated color so the bucket backward
        // can rebuild its "color after" state without the rendered image.
#pragma unroll
        for (int ch = 0; ch < 3; ++ch)
            pst.total_color[ch * width * height + pix_id] = color[ch];
#pragma unroll
        for (int ch = 0; ch < 3; ++ch)
            out_color[ch * width * height + pix_id] =
                color[ch] + transmittance *
                    (ch == 0 ? background.x : ch == 1 ? background.y : background.z);
        out_alpha[pix_id] = 1.f - transmittance;
        if constexpr (GEOMETRY) {
            const float rln = pixel_ray_z(pixf.x, pixf.y, K);
            out_median_depth[pix_id] = median_depth * rln;
            const float len_normal = 1.f - transmittance;
#pragma unroll
            for (int ch = 0; ch < 3; ++ch)
                out_normal[ch * width * height + pix_id] =
                    last_contributor ? normal_acc[ch] / len_normal : 0.f;
        }
    }
    if (block.thread_rank() == 0) max_contributor[tile_id] = block_max;
}

}  // namespace splat_drender::kernels

namespace splat_drender::launch {

void preprocess_gaussians(
    int count, int sh_degree, int sh_bases, const float* means, const float* sh,
    const float* colors, const float* opacities, const float* scales,
    const float* rotations, const float* cov6, const float* view,
    const float* camera_center, CameraIntrinsics K, int width, int height,
    float kernel_size, float scale_modifier, int grid_x, int grid_y,
    int wrap_width, ws::GaussianState st, int* radii) {
    kernels::preprocess_gaussians<<<(count + cfg::kGaussianBlock - 1) /
                                        cfg::kGaussianBlock,
                                    cfg::kGaussianBlock>>>(
        count, sh_degree, sh_bases, means, sh, colors, opacities, scales,
        rotations, cov6, view, camera_center, K, width, height, kernel_size,
        scale_modifier, grid_x, grid_y, wrap_width, st, radii);
}

void emit_depth_entries(int count, const unsigned* visible_offset,
                        const unsigned* visible_flag,
                        const unsigned* depth_key, unsigned* key_out,
                        unsigned* value_out) {
    kernels::emit_depth_entries<<<(count + cfg::kGaussianBlock - 1) /
                                      cfg::kGaussianBlock,
                                  cfg::kGaussianBlock>>>(
        count, visible_offset, visible_flag, depth_key, key_out, value_out);
}

void gather_touched(int visible_count, const unsigned* sorted_ids,
                    const unsigned* tiles_touched, unsigned* compact_n) {
    kernels::gather_touched<<<(visible_count + cfg::kGaussianBlock - 1) /
                                  cfg::kGaussianBlock,
                              cfg::kGaussianBlock>>>(
        visible_count, sorted_ids, tiles_touched, compact_n);
}

void emit_instances(int visible_count, const unsigned* depth_sorted_ids,
                    const float2* mean2d, const float4* conic_opacity,
                    const unsigned* compact_offset, int grid_x, int grid_y,
                    int wrap_width, unsigned* tile_key,
                    unsigned* instance_value) {
    kernels::emit_instances<<<(visible_count + cfg::kGaussianBlock - 1) /
                                  cfg::kGaussianBlock,
                              cfg::kGaussianBlock>>>(
        visible_count, depth_sorted_ids, mean2d, conic_opacity, compact_offset,
        grid_x, grid_y, wrap_width, tile_key, instance_value);
}

void emit_packed_instances(int count, ws::GaussianState st, int grid_x, int grid_y,
                           int wrap_width, unsigned long long* keys, unsigned* values) {
    kernels::emit_packed_instances<<<(count + cfg::kGaussianBlock - 1) / cfg::kGaussianBlock,
        cfg::kGaussianBlock>>>(count, st, grid_x, grid_y, wrap_width, keys, values);
}
void extract_packed_ranges(int count, const unsigned long long* keys, uint2* range) {
    kernels::extract_packed_ranges<<<(count + cfg::kGaussianBlock - 1) / cfg::kGaussianBlock,
        cfg::kGaussianBlock>>>(count, keys, range);
}

void extract_ranges(int instance_count, const unsigned* tile_key,
                    uint2* range) {
    kernels::extract_tile_ranges<<<(instance_count + cfg::kGaussianBlock - 1) /
                                       cfg::kGaussianBlock,
                                   cfg::kGaussianBlock>>>(instance_count,
                                                          tile_key, range);
}

void bucket_offsets(int tiles, int buckets, const uint2* range,
                    unsigned* bucket_count, unsigned* bucket_offset,
                    unsigned* bucket_tile) {
    kernels::bucket_offsets_kernel<<<1, cfg::kTileThreads>>>(
        tiles, buckets, range, bucket_count, bucket_offset, bucket_tile);
}

void blend(bool need_depth, const uint2* tile_range,
           const unsigned* instance_value, int width, int height,
           CameraIntrinsics K, int wrap_width, const float2* mean2d,
           const float4* conic_opacity, const float3* rgb, const float* colors,
           const float4* ray_plane, const float3* normal,
           const ushort4* screen_bounds, unsigned* n_contrib,
           unsigned* max_contributor, const unsigned* bucket_offset,
           unsigned* bucket_tile, ws::PixelState pst, float3 background,
           float* out_color, float* out_alpha, float* out_normal,
           float* out_median_depth, float* visibility, dim3 grid) {
    dim3 block(cfg::kTileWidth, cfg::kTileHeight);
    if (need_depth) {
        kernels::blend_tile<true><<<grid, block>>>(
            tile_range, instance_value, width, height, K, wrap_width, mean2d,
            conic_opacity, rgb, colors, ray_plane, normal, screen_bounds,
            n_contrib, max_contributor, bucket_offset, bucket_tile, pst,
            background, out_color, out_alpha, out_normal, out_median_depth,
            visibility);
    } else {
        kernels::blend_tile<false><<<grid, block>>>(
            tile_range, instance_value, width, height, K, wrap_width, mean2d,
            conic_opacity, rgb, colors, ray_plane, normal, screen_bounds,
            n_contrib, max_contributor, bucket_offset, bucket_tile, pst,
            background, out_color, out_alpha, out_normal, out_median_depth,
            visibility);
    }
}

}  // namespace splat_drender::launch
