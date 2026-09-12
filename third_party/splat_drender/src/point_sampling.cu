// Point-query kernels: occupancy (1 - transmittance along a pixel ray) and
// median-depth sampling of world-space points, plus their backward pass.
//
// One CTA per tile; points of the tile are processed in fixed-size rounds
// inside the kernel (no materialized tile duplication).
#include "device/geometry.cuh"
#include "kernels.h"
#include "splat_drender/buffers.h"

#include <cooperative_groups.h>
#include <type_traits>
#include <cub/block/block_reduce.cuh>

namespace cg = cooperative_groups;
using namespace splat_drender;

namespace splat_drender::kernels {

__global__ void preprocess_points(
    const int count, const float* __restrict__ points,
    const float* __restrict__ view, CameraIntrinsics K, int width, int height,
    ws::PointState ps, bool fixed_list) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int grid_x = (width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    if (fixed_list) {
        ps.point_key[0][i] = unsigned(grid_x * grid_y);
        ps.point_value[0][i] = unsigned(i);
    }
    ps.touched[i] = 0;
    const float3 p = make_float3(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    const float3 t = mat::xform_point(p, view);
    if (!camera_visible(t, K.mode)) return;
    const Projection pr = project(t, K, width, height);
    if (!pr.valid) return;
    if (pr.pixel_x < 0.f || pr.pixel_x > width - 1.f || pr.pixel_y < 0.f ||
        pr.pixel_y > height - 1.f)
        return;
    ps.point2d[i] = make_float2(pr.pixel_x, pr.pixel_y);
    ps.point_t[i] = norm3df(t.x, t.y, t.z);
    ps.touched[i] = 1;
    if (fixed_list) {
        const int x = min(grid_x - 1, max(0, int((pr.pixel_x + 0.5f) / cfg::kTileWidth)));
        const int y = min(grid_y - 1, max(0, int((pr.pixel_y + 0.5f) / cfg::kTileHeight)));
        ps.point_key[0][i] = unsigned(y * grid_x + x);
    }

}

// Sentinel key == tiles is sorted after every real tile. Invalid points must
// neither index ranges[tiles] nor leave the final valid range open.
__global__ void extract_point_ranges(int count, const unsigned* keys,
    int tiles, uint2* ranges) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned tile = keys[i];
    if (tile >= unsigned(tiles)) return;
    if (i == 0 || keys[i - 1] != tile) ranges[tile].x = unsigned(i);
    if (i == count - 1 || keys[i + 1] != tile) ranges[tile].y = unsigned(i + 1);
}

__global__ void emit_point_instances(
    const int count, const float2* __restrict__ point2d,
    const unsigned* __restrict__ tile_offset,
    const unsigned* __restrict__ touched, int grid_x, int grid_y,
    unsigned* __restrict__ key_out, unsigned* __restrict__ value_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || touched[i] == 0) return;
    const float2 p = point2d[i];
    const int x = min(grid_x - 1, max(0, int((p.x + 0.5f) / cfg::kTileWidth)));
    const int y = min(grid_y - 1, max(0, int((p.y + 0.5f) / cfg::kTileHeight)));
    const unsigned off = tile_offset[i] - 1u;
    key_out[off] = unsigned(y) * unsigned(grid_x) + unsigned(x);
    value_out[off] = unsigned(i);
}

namespace {

constexpr int kPts = cfg::kPointsPerRound;

// Per-point accumulators shared by both query modes.
struct PointRegs {
    unsigned idx = 0;
    float2 xy{};
    float t = 0.f;
    float T_point = 1.f;
    float T_gauss = 1.f;
    float depth_seed = 0.f;
    unsigned last_contributor = 0;
    bool active = false;
    bool done = false;
};

// Shared gaussian batch fetched by the whole CTA.
struct GaussianBatch {
    float2 xy[cfg::kTileThreads];
    float4 conic[cfg::kTileThreads];
    float4 plane[cfg::kTileThreads];
};

__device__ inline void fetch_gaussians(
    const uint2& grange, int round, int thread_rank,
    const unsigned* __restrict__ gauss_value,
    const float2* __restrict__ mean2d,
    const float4* __restrict__ conic_opacity,
    const float4* __restrict__ ray_plane, GaussianBatch& batch) {
    const int progress = round * cfg::kTileThreads + thread_rank;
    if (grange.x + progress < int(grange.y)) {
        const unsigned g = gauss_value[grange.x + progress];
        batch.xy[thread_rank] = mean2d[g];
        batch.conic[thread_rank] = conic_opacity[g];
        batch.plane[thread_rank] = ray_plane[g];
    }
}

__device__ inline void evaluate_gaussian(const float2 gxy, const float4 co,
                                         const float4 rp, PointRegs& p) {
    const float2 d = make_float2(gxy.x - p.xy.x, gxy.y - p.xy.y);
    const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                        co.y * d.x * d.y;
    if (power > 0.f) return;
    const float alpha = fminf(cfg::kAlphaClip, co.w * expf(power));
    if (alpha < cfg::kAlphaFloor) return;
    const float test_t = p.T_gauss * (1.f - alpha);
    if (test_t < cfg::kTransmittanceFloor) {
        p.done = true;
        return;
    }
    const float t_peak = rp.x * d.x + rp.y * d.y + rp.z;
    const float rsigma = rp.w;
    const float delta = (t_peak - p.t) * rsigma;
    const float g = rsigma > 0.f ? expf(-0.5f * delta * delta) : 0.f;
    const float omg = 1.f - alpha * g;
    const float rv = rsqrtf(omg);
    p.T_point *= (p.t > t_peak ? (1.f - alpha) : omg) * rv;
    p.T_gauss = test_t;
}

}  // namespace

// MODE 0: occupancy only (single traversal).
// MODE 1: median depth (seed pass + bisection, like the pixel blender).
template <int MODE>
__global__ void __launch_bounds__(cfg::kTileThreads)
evaluate_points(const uint2* __restrict__ tile_range,
                 const unsigned* __restrict__ gauss_value,
                 const uint2* __restrict__ point_range,
                 const unsigned* __restrict__ point_value, int width,
                 int height, CameraIntrinsics K,
                 const float2* __restrict__ point2d,
                 const float* __restrict__ point_t,
                 const float2* __restrict__ mean2d,
                 const float4* __restrict__ conic_opacity,
                 const float4* __restrict__ ray_plane,
                 float* __restrict__ out_occupancy,
                 float3* __restrict__ out_ray_point,
                 float* __restrict__ out_median_depth,
                 unsigned* __restrict__ out_n_contrib, bool* __restrict__ inside) {
    auto block = cg::this_thread_block();
    const int tile = blockIdx.x;
    const uint2 grange = tile_range[tile];
    const uint2 prange = point_range[tile];
    const int gauss_todo = grange.y - grange.x;
    const int gauss_rounds =
        (gauss_todo + cfg::kTileThreads - 1) / cfg::kTileThreads;
    const int pts_per_round = cfg::kTileThreads * kPts;
    const int point_rounds =
        (int(prange.y - prange.x) + pts_per_round - 1) / pts_per_round;

    __shared__ GaussianBatch batch;

    for (int rp = 0; rp < point_rounds; ++rp) {
        PointRegs pts[kPts];
        int n_pts = 0;
#pragma unroll
        for (int p = 0; p < kPts; ++p) {
            const int progress =
                (rp * pts_per_round + p * cfg::kTileThreads) + block.thread_rank();
            if (prange.x + progress < int(prange.y)) {
                const unsigned pid = point_value[prange.x + progress];
                pts[p].idx = pid;
                pts[p].xy = point2d[pid];
                pts[p].t = point_t[pid];
                pts[p].active = true;
                ++n_pts;
            }
        }
        // Threads without points in this round must still reach the
        // __syncthreads() inside the gaussian walk; their work is a no-op.

        unsigned contributor = 0;
        int remaining = gauss_todo;
        for (int round = 0; round < gauss_rounds;
             ++round, remaining -= cfg::kTileThreads) {
            __syncthreads();
            fetch_gaussians(grange, round, block.thread_rank(), gauss_value,
                            mean2d, conic_opacity, ray_plane, batch);
            __syncthreads();
            const int n = min(cfg::kTileThreads, remaining);
            for (int j = 0; j < n; ++j) {
                contributor++;
#pragma unroll
                for (int p = 0; p < kPts; ++p) {
                    if (!pts[p].active || pts[p].done) continue;
                    if (MODE == 1) {
                        // Seed pass: track T, depth seed, last contributor.
                        const float2 d =
                            make_float2(batch.xy[j].x - pts[p].xy.x,
                                        batch.xy[j].y - pts[p].xy.y);
                        const float4 co = batch.conic[j];
                        const float power = -0.5f * (co.x * d.x * d.x +
                                                     co.z * d.y * d.y) -
                                            co.y * d.x * d.y;
                        if (power > 0.f) continue;
                        const float alpha =
                            fminf(cfg::kAlphaClip, co.w * expf(power));
                        if (alpha < cfg::kAlphaFloor) continue;
                        const float test_t = pts[p].T_gauss * (1.f - alpha);
                        if (test_t < cfg::kTransmittanceFloor) {
                            pts[p].done = true;
                            continue;
                        }
                        const float4 rpl = batch.plane[j];
                        const float t =
                            rpl.x * d.x + rpl.y * d.y + rpl.z;
                        pts[p].depth_seed =
                            pts[p].T_gauss > 0.5f ? t : pts[p].depth_seed;
                        pts[p].T_gauss = test_t;
                        pts[p].last_contributor = contributor;
                    } else {
                        evaluate_gaussian(batch.xy[j], batch.conic[j],
                                          batch.plane[j], pts[p]);
                    }
                }
            }
        }

        if (MODE == 1) {
            // Median-depth bisection per point, bounded by the block max
            // of last_contributor over this round's points.
            __shared__ unsigned warp_scratch[cfg::kTileThreads / 32];
            unsigned local_max = 0;
#pragma unroll
            for (int p = 0; p < kPts; ++p)
                if (pts[p].active) local_max = max(local_max, pts[p].last_contributor);
            const unsigned block_max =
                mat::block_max(local_max, warp_scratch);

            float depth_min[kPts], depth_max[kPts];
            float T_p[kPts][cfg::kDepthSplit + 1];
            bool in_range[kPts];
#pragma unroll
            for (int p = 0; p < kPts; ++p) {
                depth_min[p] = fmaxf(pts[p].depth_seed - cfg::kDepthSeedWindowTesting, 0.f);
                depth_max[p] = fmaxf(pts[p].depth_seed + cfg::kDepthSeedWindowTesting, 0.f);
                in_range[p] = pts[p].T_gauss <= cfg::kDepthMinTransmittance;
            }

            auto refine = [&](auto first_tag) {
                constexpr bool first = decltype(first_tag)::value;
                constexpr int s0 = first ? 0 : 1;
                constexpr int s1 = first ? cfg::kDepthSplit + 1 : cfg::kDepthSplit;
#pragma unroll
                for (int p = 0; p < kPts; ++p)
#pragma unroll
                    for (int s = s0; s < s1; ++s) T_p[p][s] = 1.f;
                float interval[kPts];
#pragma unroll
                for (int p = 0; p < kPts; ++p)
                    interval[p] = (depth_max[p] - depth_min[p]) *
                                  (1.f / float(cfg::kDepthSplit));

                bool done[kPts];
                int n_done = 0;
#pragma unroll
                for (int p = 0; p < kPts; ++p) {
                    done[p] = !pts[p].active || !in_range[p] ||
                              pts[p].last_contributor == 0;
                    n_done += int(done[p]);
                }
                unsigned contributor2 = 0;
                int todo = int(block_max);
                int remaining = gauss_todo;
                // Uniform round bound: __syncthreads() below requires every
                // thread; the per-thread n_done early-exit is unsafe.
                for (int round = 0; round < gauss_rounds;
                     ++round, remaining -= cfg::kTileThreads, todo -= cfg::kTileThreads) {
                    __syncthreads();
                    fetch_gaussians(grange, round, block.thread_rank(),
                                    gauss_value, mean2d, conic_opacity,
                                    ray_plane, batch);
                    __syncthreads();
                    const int n = min(cfg::kTileThreads, min(remaining, todo));
                    for (int j = 0; j < n; ++j) {
                        contributor2++;
#pragma unroll
                        for (int p = 0; p < kPts; ++p) {
                            if (done[p]) continue;
                            const float2 d =
                                make_float2(batch.xy[j].x - pts[p].xy.x,
                                            batch.xy[j].y - pts[p].xy.y);
                            const float4 co = batch.conic[j];
                            const float power =
                                -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                                co.y * d.x * d.y;
                            if (power > 0.f) continue;
                            const float alpha =
                                fminf(cfg::kAlphaClip, co.w * expf(power));
                            if (alpha < cfg::kAlphaFloor) continue;
                            const float4 rpl = batch.plane[j];
                            const float t_peak = rpl.x * d.x + rpl.y * d.y + rpl.z;
                            const float rsigma = rpl.w;
                            const bool ball = rsigma > 0.f;
#pragma unroll
                            for (int s = s0; s < s1; ++s) {
                                const float ts = depth_min[p] + interval[p] * float(s);
                                const float delta = (ts - t_peak) * rsigma;
                                const float g =
                                    ball ? expf(-0.5f * delta * delta) : 0.f;
                                const float omg = 1.f - alpha * g;
                                const float rv = rsqrtf(omg);
                                T_p[p][s] *=
                                    (ts > t_peak ? (1.f - alpha) : omg) * rv;
                            }
                            if (contributor2 >= pts[p].last_contributor) {
                                done[p] = true;
                                ++n_done;
                            }
                        }
                    }
                }
#pragma unroll
                for (int p = 0; p < kPts; ++p) {
                    if (first)
                        in_range[p] = (T_p[p][0] >= 0.5f) &&
                                      (T_p[p][cfg::kDepthSplit] <= 0.5f) &&
                                      in_range[p];
                    int sid = 0;
#pragma unroll
                    for (int s = 1; s < cfg::kDepthSplit; ++s)
                        sid = T_p[p][s] >= 0.5f ? s : sid;
                    depth_max[p] = depth_min[p] + (sid + 1) * interval[p];
                    depth_min[p] = depth_min[p] + (sid + 0) * interval[p];
                    T_p[p][0] = T_p[p][sid];
                    T_p[p][cfg::kDepthSplit] = T_p[p][sid + 1];
                }
            };

            refine(std::true_type{});
            for (int it = 0; it < cfg::kDepthRefinementsTesting - 1; ++it)
                refine(std::false_type{});

#pragma unroll
            for (int p = 0; p < kPts; ++p) {
                if (!pts[p].active) continue;
                const float w_max = __saturatef(
                    (T_p[p][0] - 0.5f) / (T_p[p][0] - T_p[p][cfg::kDepthSplit]));
                const float w_min = 1.f - w_max;
                const float md = in_range[p]
                                     ? w_max * depth_max[p] + w_min * depth_min[p]
                                     : 0.f;
                const float3 ray = pixel_unit_ray(pts[p].xy.x, pts[p].xy.y, K);
                out_ray_point[pts[p].idx] =
                    make_float3(ray.x * md, ray.y * md, ray.z * md);
                out_median_depth[pts[p].idx] = md;
                out_n_contrib[pts[p].idx] = pts[p].last_contributor;
                inside[pts[p].idx] = in_range[p];
            }
        } else {
#pragma unroll
            for (int p = 0; p < kPts; ++p) {
                if (!pts[p].active) continue;
                out_occupancy[pts[p].idx] = 1.f - pts[p].T_point;
                inside[pts[p].idx] = true;
            }
        }
    }
}

// Backward of evaluate_points<1>: per-point median-depth gradients
// propagated to the Gaussian geometry and to the query points themselves.
__global__ void __launch_bounds__(cfg::kTileThreads)
sample_depth_backward(const uint2* __restrict__ tile_range,
                      const unsigned* __restrict__ gauss_value,
                      const uint2* __restrict__ point_range,
                      const unsigned* __restrict__ point_value,
                      CameraIntrinsics K, const float2* __restrict__ point2d,
                      const float2* __restrict__ mean2d,
                      const float4* __restrict__ conic_opacity,
                      const float4* __restrict__ ray_plane,
                      const unsigned* __restrict__ n_contrib,
                      const float* __restrict__ median_depth,
                      const bool* __restrict__ inside,
                      const float3* __restrict__ dL_dray_points,
                      ws::GradState gs,
                      float2* __restrict__ dL_dpoint2d) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int tile = blockIdx.x;
    const uint2 grange = tile_range[tile];
    const uint2 prange = point_range[tile];
    const int gauss_todo = grange.y - grange.x;
    const int gauss_rounds =
        (gauss_todo + cfg::kTileThreads - 1) / cfg::kTileThreads;
    const int pts_per_round = cfg::kTileThreads * kPts;
    const int point_rounds =
        (int(prange.y - prange.x) + pts_per_round - 1) / pts_per_round;

    __shared__ GaussianBatch batch;

    for (int rp = 0; rp < point_rounds; ++rp) {
        unsigned idx[kPts];
        float2 xy[kPts]{};
        float mdepth[kPts]{};
        unsigned last_contrib[kPts]{};
        float dL_dDepth[kPts]{};
        float2 dL_dxy[kPts]{};
        bool done[kPts];
        bool any_active = false;
#pragma unroll
        for (int p = 0; p < kPts; ++p) {
            const int progress = (rp * pts_per_round + p * cfg::kTileThreads) +
                                 block.thread_rank();
            if (prange.x + progress < int(prange.y)) {
                const unsigned pid = point_value[prange.x + progress];
                idx[p] = pid;
                xy[p] = point2d[pid];
                mdepth[p] = median_depth[pid];
                last_contrib[p] = n_contrib[pid];
                const bool in = inside[pid];
                done[p] = (last_contrib[p] == 0) || !in;
                any_active = true;

                const float3 dl = dL_dray_points[pid];
                if (is_pinhole(K.mode) || is_equirect(K.mode)) {
                    const float2 nf = make_float2((xy[p].x - K.cx) / K.fx,
                                                  (xy[p].y - K.cy) / K.fy);
                    const float rln = rnorm3df(nf.x, nf.y, 1.f);
                    const float rln2 = 1.f / (nf.x * nf.x + nf.y * nf.y + 1.f);
                    const float depth = mdepth[p] * rln;
                    const float dl_depth = dl.x * nf.x + dl.y * nf.y + dl.z;
                    dL_dDepth[p] = rln * dl_depth;
                    const float aux = dl_depth * rln2;
                    const float2 dnf = make_float2(
                        (dl.x - aux * nf.x) * depth, (dl.y - aux * nf.y) * depth);
                    dL_dxy[p] = make_float2(dnf.x / K.fx, dnf.y / K.fy);
                }
                if (is_fisheye(K.mode)) {
                    const float3 ray = pixel_unit_ray(xy[p].x, xy[p].y, K);
                    dL_dDepth[p] =
                        ray.x * dl.x + ray.y * dl.y + ray.z * dl.z;
                    const Projection pj = project_fisheye(ray, K);
                    const float3 ju = make_float3(pj.J.du[0], pj.J.du[1], pj.J.du[2]);
                    const float3 jv = make_float3(pj.J.dv[0], pj.J.dv[1], pj.J.dv[2]);
                    const float aa = mat::dot3(ju, ju);
                    const float ab = mat::dot3(ju, jv);
                    const float bb = mat::dot3(jv, jv);
                    const float det = aa * bb - ab * ab;
                    dL_dxy[p] = make_float2(
                        mdepth[p] * mat::dot3(dl, make_float3(
                            (bb * ju.x - ab * jv.x) / det,
                            (bb * ju.y - ab * jv.y) / det,
                            (bb * ju.z - ab * jv.z) / det)),
                        mdepth[p] * mat::dot3(dl, make_float3(
                            (aa * jv.x - ab * ju.x) / det,
                            (aa * jv.y - ab * ju.y) / det,
                            (aa * jv.z - ab * ju.z) / det)));
                }
                if (!is_pinhole(K.mode) && !is_equirect(K.mode) &&
                    !is_fisheye(K.mode)) {
                    dL_dDepth[p] = 0.f;
                    dL_dxy[p] = make_float2(0.f, 0.f);
                }
            } else {
                done[p] = true;
            }
        }
        // Threads without active points must still reach the
        // __syncthreads() below; their per-point work is a no-op.

        // Per-round max contributor bounds both traversals.
        __shared__ unsigned warp_scratch[cfg::kTileThreads / 32];
        unsigned local_max = 0;
#pragma unroll
        for (int p = 0; p < kPts; ++p)
            if (!done[p]) local_max = max(local_max, last_contrib[p]);
        const unsigned block_max =
            mat::block_max(local_max, warp_scratch);
        const int rounds =
            (int(block_max) + cfg::kTileThreads - 1) / cfg::kTileThreads;

        // ---- dT/dtm accumulation ----
        float dL_dmt_dT_dtm[kPts];
        {
            float dT_dtm[kPts];
#pragma unroll
            for (int p = 0; p < kPts; ++p) dT_dtm[p] = 0.f;
            int todo = int(block_max);
            unsigned contributor = 0;
            for (int round = 0; round < rounds;
                 ++round, todo -= cfg::kTileThreads) {
                __syncthreads();
                fetch_gaussians(grange, round, block.thread_rank(), gauss_value,
                                mean2d, conic_opacity, ray_plane, batch);
                __syncthreads();
                const int n = min(cfg::kTileThreads, todo);
                for (int j = 0; j < n; ++j) {
                    contributor++;
#pragma unroll
                    for (int p = 0; p < kPts; ++p) {
                        // Forward stores a one-based inclusive contributor count.
                        if (done[p] || contributor > last_contrib[p]) continue;
                        const float2 d = make_float2(batch.xy[j].x - xy[p].x,
                                                     batch.xy[j].y - xy[p].y);
                        const float4 co = batch.conic[j];
                        const float power =
                            -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                            co.y * d.x * d.y;
                        if (power > 0.f) continue;
                        const float alpha =
                            fminf(cfg::kAlphaClip, co.w * expf(power));
                        if (alpha < cfg::kAlphaFloor) continue;
                        const float4 rpl = batch.plane[j];
                        const float t_peak = rpl.x * d.x + rpl.y * d.y + rpl.z;
                        const float rsigma = rpl.w;
                        const float t_delta = (mdepth[p] - t_peak) * rsigma;
                        const float G_exp = expf(-0.5f * t_delta * t_delta);
                        const float Gt = alpha * G_exp;
                        dT_dtm[p] +=
                            -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < kPts; ++p)
                dL_dmt_dT_dtm[p] = dL_dDepth[p] / fmaxf(-dT_dtm[p], 1.0e-7f);
        }

        // ---- main backward traversal ----
        int todo = int(block_max);
        unsigned contributor = block_max;
        for (int round = 0; round < rounds;
             ++round, todo -= cfg::kTileThreads) {
            __syncthreads();
            const int progress = round * cfg::kTileThreads + block.thread_rank();
            if (progress < int(block_max)) {
                // Reverse fetch: slot j of this batch holds list position
                // block_max - (round * kTileThreads + j) - 1.
                const unsigned g =
                    gauss_value[grange.x + int(block_max) - progress - 1];
                batch.xy[block.thread_rank()] = mean2d[g];
                batch.conic[block.thread_rank()] = conic_opacity[g];
                batch.plane[block.thread_rank()] = ray_plane[g];
            }
            __syncthreads();
            const int n = min(cfg::kTileThreads, todo);
            for (int j = 0; j < n; ++j) {
                contributor--;
                bool any_valid = false;
                float G[kPts];
                float alpha_raw[kPts];
#pragma unroll
                for (int p = 0; p < kPts; ++p) {
                    const float2 d = make_float2(batch.xy[j].x - xy[p].x,
                                                 batch.xy[j].y - xy[p].y);
                    const float4 co = batch.conic[j];
                    const float power =
                        -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                        co.y * d.x * d.y;
                    G[p] = expf(power);
                    alpha_raw[p] = co.w * G[p];
                    const bool valid = !(done[p] ||
                                         (contributor >= last_contrib[p]) ||
                                         (power > 0.f) ||
                                         (alpha_raw[p] < cfg::kAlphaFloor));
                    any_valid = any_valid || valid;
                }
                if (!warp.any(any_valid)) continue;

                float dL_dt_local = 0.f;
                float2 dL_dplane_local = make_float2(0.f, 0.f);
                float2 dL_dmean2d_local = make_float2(0.f, 0.f);
                float4 dL_dconic_local = make_float4(0.f, 0.f, 0.f, 0.f);
                float dL_drsigma_local = 0.f;
#pragma unroll
                for (int p = 0; p < kPts; ++p) {
                    const float2 d = make_float2(batch.xy[j].x - xy[p].x,
                                                 batch.xy[j].y - xy[p].y);
                    const float4 co = batch.conic[j];
                    const float power =
                        -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                        co.y * d.x * d.y;
                    const bool valid = !(done[p] ||
                                         (contributor >= last_contrib[p]) ||
                                         (power > 0.f) ||
                                         (alpha_raw[p] < cfg::kAlphaFloor));
                    if (!valid) continue;
                    const float alpha = fminf(cfg::kAlphaClip, alpha_raw[p]);

                    const float4 rpl = batch.plane[j];
                    const float t_peak = rpl.x * d.x + rpl.y * d.y + rpl.z;
                    const float rsigma = rpl.w;
                    const float t_delta = (mdepth[p] - t_peak) * rsigma;
                    const float G_exp = expf(-0.5f * t_delta * t_delta);
                    const float Gt = alpha * G_exp;

                    float dL_dGt = dL_dmt_dT_dtm[p] * 0.25f / (1.f - Gt);
                    dL_dGt = mdepth[p] > t_peak ? dL_dGt : -dL_dGt;
                    dL_dGt = rsigma > 0.f ? dL_dGt : 0.f;
                    float dL_dopa =
                        dL_dGt * G_exp -
                        dL_dmt_dT_dtm[p] *
                            (t_delta > 0.f ? 0.5f / (1.f - alpha) : 0.f);
                    const float dL_ddelta = -dL_dGt * Gt * t_delta;
                    dL_drsigma_local += dL_ddelta * (mdepth[p] - t_peak);
                    const float dL_dt_peak = -dL_ddelta * rsigma;

                    dL_dt_local += dL_dt_peak;
                    dL_dplane_local.x += dL_dt_peak * d.x;
                    dL_dplane_local.y += dL_dt_peak * d.y;

                    if (is_fisheye(K.mode) && alpha_raw[p] >= cfg::kAlphaClip)
                        dL_dopa = 0.f;

                    const float dL_dG = co.w * dL_dopa;
                    const float gdx = G[p] * d.x;
                    const float gdy = G[p] * d.y;
                    const float dG_ddelx = -gdx * co.x - gdy * co.y;
                    const float dG_ddely = -gdy * co.z - gdx * co.y;

                    const float dL_ddelx = dL_dG * dG_ddelx + dL_dt_peak * rpl.x;
                    const float dL_ddely = dL_dG * dG_ddely + dL_dt_peak * rpl.y;
                    dL_dmean2d_local.x += dL_ddelx;
                    dL_dmean2d_local.y += dL_ddely;
                    dL_dxy[p].x -= dL_ddelx;
                    dL_dxy[p].y -= dL_ddely;

                    dL_dconic_local.x += -0.5f * gdx * d.x * dL_dG;
                    dL_dconic_local.y += -0.5f * gdx * d.y * dL_dG;
                    dL_dconic_local.z += -0.5f * gdy * d.y * dL_dG;
                    dL_dconic_local.w += G[p] * dL_dopa;
                    done[p] = contributor >= last_contrib[p];
                }
                mat::warp_sum(dL_dt_local);
                mat::warp_sum(dL_dplane_local);
                mat::warp_sum(dL_dmean2d_local);
                mat::warp_sum(dL_dconic_local);
                mat::warp_sum(dL_drsigma_local);
                if (warp.thread_rank() == 0) {
                    const int progress2 =
                        round * cfg::kTileThreads + j;
                    const unsigned g = gauss_value
                        [grange.x + int(block_max) - progress2 - 1];
                    atomicAdd(&gs.d_ray_plane[g].x, dL_dplane_local.x);
                    atomicAdd(&gs.d_ray_plane[g].y, dL_dplane_local.y);
                    atomicAdd(&gs.d_ray_plane[g].z, dL_dt_local);
                    atomicAdd(&gs.d_ray_plane[g].w, dL_drsigma_local);
                    atomicAdd(&gs.d_mean2d[g].x, dL_dmean2d_local.x);
                    atomicAdd(&gs.d_mean2d[g].y, dL_dmean2d_local.y);
                    atomicAdd(&gs.d_conic[g].x, dL_dconic_local.x);
                    atomicAdd(&gs.d_conic[g].y, dL_dconic_local.y);
                    atomicAdd(&gs.d_conic[g].z, dL_dconic_local.z);
                    atomicAdd(&gs.d_conic[g].w, dL_dconic_local.w);
                }
            }
        }

#pragma unroll
        for (int p = 0; p < kPts; ++p) {
            const int progress = (rp * pts_per_round + p * cfg::kTileThreads) +
                                 block.thread_rank();
            if (prange.x + progress < int(prange.y))
                dL_dpoint2d[idx[p]] = dL_dxy[p];
        }
    }
}

// Unprojects dL/d(pixel) back to dL/d(world point).
__global__ void point_2d_backward(
    const int count, const float* __restrict__ points,
    const float* __restrict__ view, CameraIntrinsics K, int width, int height,
    const unsigned* __restrict__ touched, const float2* __restrict__ dL_dpoint2d,
    float3* __restrict__ dL_dpoint3d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || touched[i] == 0) return;
    const float3 m = make_float3(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    const float3 t = mat::xform_point(m, view);
    float3 grad;
    if (is_pinhole(K.mode)) {
        const float rz = 1.f / (t.z + 1.0e-7f);
        const float sx = t.x * rz, sy = t.y * rz;
        const float2 g = dL_dpoint2d[i];
        grad.x = (K.fx * (view[0] - sx * view[2]) * g.x +
                  K.fy * (view[1] - sy * view[2]) * g.y) * rz;
        grad.y = (K.fx * (view[4] - sx * view[6]) * g.x +
                  K.fy * (view[5] - sy * view[6]) * g.y) * rz;
        grad.z = (K.fx * (view[8] - sx * view[10]) * g.x +
                  K.fy * (view[9] - sy * view[10]) * g.y) * rz;
    } else {
        const Projection pj = project(t, K, width, height);
        if (pj.valid) {
            const float2 g = dL_dpoint2d[i];
            const float gcx = pj.J.du[0] * g.x + pj.J.dv[0] * g.y;
            const float gcy = pj.J.du[1] * g.x + pj.J.dv[1] * g.y;
            const float gcz = pj.J.du[2] * g.x + pj.J.dv[2] * g.y;
            grad.x = view[0] * gcx + view[1] * gcy + view[2] * gcz;
            grad.y = view[4] * gcx + view[5] * gcy + view[6] * gcz;
            grad.z = view[8] * gcx + view[9] * gcy + view[10] * gcz;
        } else {
            grad = make_float3(0.f, 0.f, 0.f);
        }
    }
    dL_dpoint3d[i] = grad;
}

}  // namespace splat_drender::kernels

namespace splat_drender::launch {

void preprocess_points(int count, const float* points, const float* view,
                       CameraIntrinsics K, int width, int height,
                       ws::PointState ps, bool fixed_list) {
    kernels::preprocess_points<<<(count + cfg::kGaussianBlock - 1) /
                                     cfg::kGaussianBlock,
                                 cfg::kGaussianBlock>>>(count, points, view, K,
                                                        width, height, ps, fixed_list);
}

void extract_point_ranges(int count, const unsigned* keys, int tiles, uint2* ranges) {
    kernels::extract_point_ranges<<<(count + cfg::kGaussianBlock - 1) / cfg::kGaussianBlock,
        cfg::kGaussianBlock>>>(count, keys, tiles, ranges);
}

void emit_point_instances(int count, const float2* point2d,
                          const unsigned* tile_offset,
                          const unsigned* touched, int grid_x, int grid_y,
                          unsigned* key_out, unsigned* value_out) {
    kernels::emit_point_instances<<<(count + cfg::kGaussianBlock - 1) /
                                        cfg::kGaussianBlock,
                                    cfg::kGaussianBlock>>>(
        count, point2d, tile_offset, touched, grid_x, grid_y, key_out,
        value_out);
}

void evaluate_points(bool median_mode, const uint2* tile_range,
                     const unsigned* gauss_value, const uint2* point_range,
                     const unsigned* point_value, int width, int height,
                     CameraIntrinsics K, const float2* point2d,
                     const float* point_t, const float2* mean2d,
                     const float4* conic_opacity, const float4* ray_plane,
                     float* out_occupancy, float3* out_ray_point,
                     float* out_median_depth, unsigned* out_n_contrib,
                     bool* inside, int tiles) {
    if (median_mode) {
        kernels::evaluate_points<1><<<tiles, cfg::kTileThreads>>>(
            tile_range, gauss_value, point_range, point_value, width, height,
            K, point2d, point_t, mean2d, conic_opacity, ray_plane,
            out_occupancy, out_ray_point, out_median_depth, out_n_contrib,
            inside);
    } else {
        kernels::evaluate_points<0><<<tiles, cfg::kTileThreads>>>(
            tile_range, gauss_value, point_range, point_value, width, height,
            K, point2d, point_t, mean2d, conic_opacity, ray_plane,
            out_occupancy, out_ray_point, out_median_depth, out_n_contrib,
            inside);
    }
}

void sample_depth_backward(const uint2* tile_range,
                           const unsigned* gauss_value,
                           const uint2* point_range,
                           const unsigned* point_value, CameraIntrinsics K,
                           const float2* point2d, const float2* mean2d,
                           const float4* conic_opacity,
                           const float4* ray_plane, const unsigned* n_contrib,
                           const float* median_depth, const bool* inside,
                           const float3* dL_dray_points, ws::GradState gs,
                           float2* dL_dpoint2d, int tiles) {
    kernels::sample_depth_backward<<<tiles, cfg::kTileThreads>>>(
        tile_range, gauss_value, point_range, point_value, K, point2d, mean2d,
        conic_opacity, ray_plane, n_contrib, median_depth, inside,
        dL_dray_points, gs, dL_dpoint2d);
}

void point_2d_backward(int count, const float* points, const float* view,
                       CameraIntrinsics K, int width, int height,
                       const unsigned* touched, const float2* dL_dpoint2d,
                       float3* dL_dpoint3d) {
    kernels::point_2d_backward<<<(count + cfg::kGaussianBlock - 1) /
                                     cfg::kGaussianBlock,
                                 cfg::kGaussianBlock>>>(
        count, points, view, K, width, height, touched, dL_dpoint2d,
        dL_dpoint3d);
}

}  // namespace splat_drender::launch
