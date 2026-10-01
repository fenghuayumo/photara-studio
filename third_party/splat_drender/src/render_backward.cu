// Backward render kernels.
//
// The blending backward uses the FasterGS bucket-parallel layout: one warp
// owns 32 consecutive instances (a bucket) of one tile; each lane evaluates
// its own instance against all 256 tile pixels while the per-pixel state
// (transmittance, accumulated "color after" and, for geometry, "normal
// after") enters at lane 0 from the forward bucket snapshot and flows
// diagonally through warp shuffles. Per-lane gradients live in registers
// for the whole tile and commit with a single atomic set, replacing the
// per-instance warp reductions of the classic thread-per-pixel walk.
//
// median_scale_backward first turns the median-depth loss into the per-pixel
// scale dL_dmedian*ray_z / max(-dT_dtm, 1e-7); its formerly sequential
// contributor walk is split across the eight tile warps.
// gaussian_backward fuses the EWA geometry backward, the mean2d->mean3d
// chain and the SH backward into one launch.
#include "device/geometry.cuh"
#include "kernels.h"
#include "splat_drender/buffers.h"
#include "splat_drender/sh_adam_quant.cuh"

#include <cooperative_groups.h>
#include <type_traits>

namespace cg = cooperative_groups;
using namespace splat_drender;

namespace splat_drender::kernels {

namespace {

struct WarpBounds {
    float x0, x1, y0, y1;
};

SD_D2 inline bool overlaps(ushort4 b, const WarpBounds& w) {
    return b.x < w.x1 && b.y > w.x0 && b.z < w.y1 && b.w > w.y0;
}

}  // namespace

// ---------------------------------------------------------------------------
// dT/dtm accumulation for the median-depth gradient (thread-per-pixel walk,
// same contributor numbering as the forward pass).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(cfg::kTileThreads)
median_scale_walk(const uint2* __restrict__ tile_range,
                  const unsigned* __restrict__ instance_value, int width,
                  int height, CameraIntrinsics K, int wrap_width,
                  const float2* __restrict__ mean2d,
                  const float4* __restrict__ conic_opacity,
                  const float4* __restrict__ ray_plane,
                  const ushort4* __restrict__ screen_bounds,
                  const unsigned* __restrict__ n_contrib,
                  const unsigned* __restrict__ max_contributor,
                  const float* __restrict__ median_depth_out,
                  const float* __restrict__ dL_dmedian_in,
                  float* __restrict__ dL_dmt_out) {
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
    const unsigned max_contrib = max_contributor[tile_id];

    float mDepth = 0.f;
    float dL_dpixel_mt = 0.f;
    unsigned last_contributor = 0;
    bool done = !inside;
    if (inside) {
        const float rln = pixel_ray_z(pixf.x, pixf.y, K);
        mDepth = median_depth_out[pix_id] / fmaxf(rln, 1.0e-8f);
        dL_dpixel_mt = dL_dmedian_in[pix_id] * rln;
        last_contributor = n_contrib[pix_id];
        // A zero upstream gradient keeps dL_dmt at zero whatever the walk
        // accumulates, so the walk itself can stop immediately.
        done = (mDepth == 0.f) || (last_contributor == 0) ||
               (dL_dpixel_mt == 0.f);
    }
    if (max_contrib == 0) {
        if (inside) dL_dmt_out[pix_id] = 0.f;
        return;
    }

    WarpBounds wb;
    if (wrap_width == 0) {
        const int warp_row = (block.thread_index().y >> 1) * 2;
        wb.x0 = float(pix_min_x);
        wb.x1 = float(pix_min_x + cfg::kTileWidth);
        wb.y0 = float(pix_min_y + warp_row);
        wb.y1 = float(pix_min_y + warp_row + 2);
    } else {
        wb.x0 = -3.0e38f;
        wb.x1 = 3.0e38f;
        wb.y0 = -3.0e38f;
        wb.y1 = 3.0e38f;
    }

    __shared__ float2 sxy[cfg::kTileThreads];
    __shared__ float4 sconic[cfg::kTileThreads];
    __shared__ float4 splane[cfg::kTileThreads];
    __shared__ ushort4 sbounds[cfg::kTileThreads];

    float dT_dtm = 0.f;
    const int rounds =
        (int(max_contrib) + cfg::kTileThreads - 1) / cfg::kTileThreads;
    int todo = int(max_contrib);
    unsigned contributor = 0;
    for (int round = 0; round < rounds;
         ++round, todo -= cfg::kTileThreads) {
        if (__syncthreads_and(done)) break;
        const int progress = round * cfg::kTileThreads + block.thread_rank();
        if (progress < int(max_contrib)) {
            const unsigned g = instance_value[range.x + progress];
            sxy[block.thread_rank()] = mean2d[g];
            sconic[block.thread_rank()] = conic_opacity[g];
            splane[block.thread_rank()] = ray_plane[g];
            sbounds[block.thread_rank()] = screen_bounds[g];
        }
        __syncthreads();
        const int batch = min(cfg::kTileThreads, todo);
        for (int j = 0; !done && j < batch; ++j) {
            contributor++;
            done = contributor >= last_contributor;
            if (!overlaps(sbounds[j], wb)) continue;

            const float2 d =
                make_float2(wrap_dx(sxy[j].x - pixf.x, wrap_width, K.mode),
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
            const float t_delta = (mDepth - t_peak) * rsigma;
            const float G_exp = expf(-0.5f * t_delta * t_delta);
            const float Gt = alpha * G_exp;
            dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
        }
    }

    if (inside)
        dL_dmt_out[pix_id] = dL_dpixel_mt / fmaxf(-dT_dtm, 1.0e-7f);
}

// ---------------------------------------------------------------------------
// Bucket-parallel blending backward (FasterGS transposition).
// ---------------------------------------------------------------------------
template <bool GEOMETRY>
__global__ void __launch_bounds__(256)
blend_bucket_backward(const uint2* __restrict__ tile_range,
                      const unsigned* __restrict__ instance_value, int width,
                      int height, CameraIntrinsics K, int wrap_width,
                      float3 background, const float2* __restrict__ mean2d,
                      const float4* __restrict__ conic_opacity,
                      const float3* __restrict__ rgb,
                      const float* __restrict__ colors,
                      const float4* __restrict__ ray_plane,
                      const float3* __restrict__ normal,
                      const ushort4* __restrict__ screen_bounds,
                      const float* __restrict__ alphas,
                      const float* __restrict__ normal_map,
                      const float* __restrict__ median_depth_out,
                      const unsigned* __restrict__ n_contrib,
                      const unsigned* __restrict__ max_contributor,
                      const unsigned* __restrict__ bucket_offset,
                      const unsigned* __restrict__ bucket_tile,
                      const ws::PixelState pst, const float* __restrict__ dL_dpixel_in,
                      const float* __restrict__ dL_dmedian_in,
                      const float* __restrict__ dL_dalpha_in,
                      const float* __restrict__ dL_dnormal_in,
                      ws::GradState gs, float* __restrict__ dL_dcolors,
                      float* __restrict__ refine_weight,
                      const float* __restrict__ densify_map,
                      float* __restrict__ densify_weight,
                      float* __restrict__ densify_weight_den, int buckets) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int bucket_idx = blockIdx.x * 8 + warp_id;
    if (bucket_idx >= buckets) return;

    const unsigned tile_id = bucket_tile[bucket_idx];
    const uint2 range = tile_range[tile_id];
    const unsigned tile_first = tile_id == 0 ? 0u : bucket_offset[tile_id - 1];
    const int tile_bucket = bucket_idx - int(tile_first);
    const unsigned tile_max = max_contributor[tile_id];
    // Upper-bound tail buckets park at tile 0 far beyond its bucket count.
    if (unsigned(tile_bucket) * 32u >= tile_max) return;

    const int tile_n = range.y - range.x;
    const int pos = tile_bucket * 32 + lane;
    const bool valid_instance = pos < tile_n;

    // Per-lane instance data, loaded once for the whole tile.
    unsigned g = 0;
    float2 xy = make_float2(0.f, 0.f);
    float4 co = make_float4(0.f, 0.f, 0.f, 0.f);
    float3 col = make_float3(0.f, 0.f, 0.f);
    float4 rp = make_float4(0.f, 0.f, 0.f, 0.f);
    float3 nrm = make_float3(0.f, 0.f, 0.f);
    ushort4 bounds = make_ushort4(0, 0, 0, 0);
    if (valid_instance) {
        g = instance_value[range.x + pos];
        xy = mean2d[g];
        co = conic_opacity[g];
        col = colors ? reinterpret_cast<const float3*>(colors)[g] : rgb[g];
        bounds = screen_bounds[g];
        if constexpr (GEOMETRY) {
            rp = ray_plane[g];
            nrm = normal[g];
        }
    }

    const unsigned tiles_x = (width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int pix_min_x = int(tile_id % tiles_x) * cfg::kTileWidth;
    const int pix_min_y = int(tile_id / tiles_x) * cfg::kTileHeight;
    const std::size_t pixels = std::size_t(width) * height;
    const float4* __restrict__ snap_ct =
        pst.snap_ct + (std::size_t(bucket_idx) << 8);
    const float4* __restrict__ snap_normal =
        GEOMETRY ? pst.snap_normal + (std::size_t(bucket_idx) << 8) : nullptr;

    __shared__ unsigned s_last[8][32];
    __shared__ float4 s_state[8][32];  // color-after init xyz, T init
    __shared__ float4 s_c0[8][32];     // dL_dpixel xyz, dL_dfinalT
    __shared__ float4 s_c1[8][32];     // T_final, mDepth, dL_dmt, dL_dfinalT_render
    __shared__ float4 s_gnorm[8][32];  // geometry: dL_dpixel_normal xyz
    __shared__ float4 s_nstate[8][32]; // geometry: normal-after init xyz

    // Register accumulators, committed once per lane.
    float acc_color[3] = {0.f, 0.f, 0.f};
    float3 acc_mean2d = make_float3(0.f, 0.f, 0.f);
    float4 acc_conic = make_float4(0.f, 0.f, 0.f, 0.f);
    float acc_refine = 0.f;
    float acc_densify = 0.f;
    float acc_densify_den = 0.f;
    float3 acc_normal = make_float3(0.f, 0.f, 0.f);
    float4 acc_ray_plane = make_float4(0.f, 0.f, 0.f, 0.f);

    // Mutable per-pixel state flowing diagonally through the warp.
    float T = 0.f;
    float cax = 0.f, cay = 0.f, caz = 0.f;
    float nax = 0.f, nay = 0.f, naz = 0.f;
    // Per-pixel constants ride the same wavefront: a 32-slot staging buffer
    // alone cannot serve lanes 1..31 because their reads happen after the
    // next staging refresh.
    float gpx = 0.f, gpy = 0.f, gpz = 0.f;
    float dL_fin = 0.f, dL_fin_r = 0.f, T_final = 0.f;
    unsigned last_contrib = 0;
    float gnx = 0.f, gny = 0.f, gnz = 0.f;
    float mDepth_reg = 0.f, dL_dmt_reg = 0.f;

    for (int i = 0; i < cfg::kTileThreads + 31; ++i) {
        if ((i & 31) == 0) {
            // Stage constants and snapshot-derived initial state for the
            // next 32 pixels. Done pixels keep stale snapshot values; every
            // lane of this bucket rejects them via last_contributor.
            const int local = i + lane;
            if (local < cfg::kTileThreads) {
                const int px = pix_min_x + (local & 15);
                const int py = pix_min_y + (local >> 4);
                const bool pin = px < width && py < height;
                const unsigned pid = unsigned(width) * py + px;
                const float4 sct = pin ? snap_ct[local]
                                       : make_float4(0.f, 0.f, 0.f, 1.f);
                float3 ctot = make_float3(0.f, 0.f, 0.f);
                float3 grad = make_float3(0.f, 0.f, 0.f);
                float w_final = 0.f;
                float dL_dfinalT = 0.f;
                float dL_dfinalT_render = 0.f;
                float mDepth = 0.f;
                float dL_dmt = 0.f;
                float3 gnorm = make_float3(0.f, 0.f, 0.f);
                float3 ntot = make_float3(0.f, 0.f, 0.f);
                unsigned last = 0;
                if (pin) {
                    w_final = alphas[pid];
                    ctot = make_float3(pst.total_color[pid],
                                       pst.total_color[pixels + pid],
                                       pst.total_color[2 * pixels + pid]);
                    grad = make_float3(dL_dpixel_in[pid],
                                       dL_dpixel_in[pixels + pid],
                                       dL_dpixel_in[2 * pixels + pid]);
                    // The reference snapshots the render branch BEFORE the
                    // geometry additions: it keeps the -dL_dalpha term.
                    dL_dfinalT_render = background.x * grad.x +
                                        background.y * grad.y +
                                        background.z * grad.z -
                                        dL_dalpha_in[pid];
                    dL_dfinalT = dL_dfinalT_render;
                    last = n_contrib[pid];
                    if constexpr (GEOMETRY) {
                        // Reference builds unnormalized accumulated normals
                        // (out_normal = Normal / (1 - T)).
                        const float inv_w = w_final > 0.f ? 1.f / w_final : 0.f;
                        const float nm[3] = {
                            normal_map[pid], normal_map[pixels + pid],
                            normal_map[2 * pixels + pid]};
                        gnorm = make_float3(
                            dL_dnormal_in[pid] * inv_w,
                            dL_dnormal_in[pixels + pid] * inv_w,
                            dL_dnormal_in[2 * pixels + pid] * inv_w);
                        dL_dfinalT += gnorm.x * nm[0] + gnorm.y * nm[1] +
                                      gnorm.z * nm[2];
                        ntot = make_float3(nm[0] * w_final, nm[1] * w_final,
                                           nm[2] * w_final);
                        const float rln = pixel_ray_z(float(px), float(py), K);
                        mDepth = median_depth_out[pid] / fmaxf(rln, 1.0e-8f);
                        dL_dmt = pst.dL_dmt[pid];
                    }
                }
                s_state[warp_id][lane] = make_float4(ctot.x - sct.x,
                                                     ctot.y - sct.y,
                                                     ctot.z - sct.z, sct.w);
                s_c0[warp_id][lane] =
                    make_float4(grad.x, grad.y, grad.z, dL_dfinalT);
                s_c1[warp_id][lane] = make_float4(1.f - w_final, mDepth,
                                                  dL_dmt, dL_dfinalT_render);
                s_last[warp_id][lane] = last;
                if constexpr (GEOMETRY) {
                    const float4 sn = snap_normal[local];
                    s_nstate[warp_id][lane] =
                        make_float4(ntot.x - sn.x, ntot.y - sn.y,
                                    ntot.z - sn.z, 0.f);
                    s_gnorm[warp_id][lane] =
                        make_float4(gnorm.x, gnorm.y, gnorm.z, 0.f);
                }
            }
            __syncwarp();
        }

        if (i > 0) {
            // The state of pixel (i-lane) was produced one iteration earlier
            // by lane-1 and moves down the warp.
            T = __shfl_up_sync(0xffffffffu, T, 1);
            cax = __shfl_up_sync(0xffffffffu, cax, 1);
            cay = __shfl_up_sync(0xffffffffu, cay, 1);
            caz = __shfl_up_sync(0xffffffffu, caz, 1);
            gpx = __shfl_up_sync(0xffffffffu, gpx, 1);
            gpy = __shfl_up_sync(0xffffffffu, gpy, 1);
            gpz = __shfl_up_sync(0xffffffffu, gpz, 1);
            dL_fin = __shfl_up_sync(0xffffffffu, dL_fin, 1);
            dL_fin_r = __shfl_up_sync(0xffffffffu, dL_fin_r, 1);
            T_final = __shfl_up_sync(0xffffffffu, T_final, 1);
            last_contrib = __shfl_up_sync(0xffffffffu, last_contrib, 1);
            if constexpr (GEOMETRY) {
                nax = __shfl_up_sync(0xffffffffu, nax, 1);
                nay = __shfl_up_sync(0xffffffffu, nay, 1);
                naz = __shfl_up_sync(0xffffffffu, naz, 1);
                gnx = __shfl_up_sync(0xffffffffu, gnx, 1);
                gny = __shfl_up_sync(0xffffffffu, gny, 1);
                gnz = __shfl_up_sync(0xffffffffu, gnz, 1);
                mDepth_reg = __shfl_up_sync(0xffffffffu, mDepth_reg, 1);
                dL_dmt_reg = __shfl_up_sync(0xffffffffu, dL_dmt_reg, 1);
            }
        }

        const int idx = i - lane;
        // Lane 0 is where each pixel's state enters the wavefront.
        if (lane == 0 && idx >= 0 && idx < cfg::kTileThreads) {
            const float4 st = s_state[warp_id][idx & 31];
            cax = st.x;
            cay = st.y;
            caz = st.z;
            T = st.w;
            const float4 c0 = s_c0[warp_id][idx & 31];
            gpx = c0.x;
            gpy = c0.y;
            gpz = c0.z;
            dL_fin = c0.w;
            const float4 c1 = s_c1[warp_id][idx & 31];
            T_final = c1.x;
            mDepth_reg = c1.y;
            dL_dmt_reg = c1.z;
            dL_fin_r = c1.w;
            last_contrib = s_last[warp_id][idx & 31];
            if constexpr (GEOMETRY) {
                const float4 ns = s_nstate[warp_id][idx & 31];
                nax = ns.x;
                nay = ns.y;
                naz = ns.z;
                const float4 gg = s_gnorm[warp_id][idx & 31];
                gnx = gg.x;
                gny = gg.y;
                gnz = gg.z;
            }
        }
        if (idx < 0 || idx >= cfg::kTileThreads) continue;

        const int px = pix_min_x + (idx & 15);
        const int py = pix_min_y + (idx >> 4);
        const bool pin = px < width && py < height;
        if (!valid_instance || !pin || unsigned(pos) >= last_contrib) continue;
        // The screen bounds are the exact axis-aligned box of the
        // alpha >= floor ellipse, so pixels outside were never blended.
        if (wrap_width == 0 && (px < bounds.x || px >= bounds.y ||
                                py < bounds.z || py >= bounds.w))
            continue;

        const float dx = wrap_dx(xy.x - float(px), wrap_width, K.mode);
        const float dy = xy.y - float(py);
        const float power =
            geo::gaussian_power(co, dx, dy);
        if (power > 0.f) continue;
        const float G = expf(power);
        const float alpha = fminf(cfg::kAlphaClip, co.w * G);
        if (alpha < cfg::kAlphaFloor) continue;

        const float blend_weight = alpha * T;
        const float inv_1ma = 1.f / (1.f - alpha);
        // Advance the state past this instance first: the accum chains
        // need "after" = contributions strictly beyond position pos.
        cax -= blend_weight * col.x;
        cay -= blend_weight * col.y;
        caz -= blend_weight * col.z;
        if constexpr (GEOMETRY) {
            nax -= blend_weight * nrm.x;
            nay -= blend_weight * nrm.y;
            naz -= blend_weight * nrm.z;
        }

        // Reference chains equal (state-after / T_next) dots; folding T
        // into both terms (FasterGS form) avoids the trailing *= T.
        const float c_dot = col.x * gpx + col.y * gpy + col.z * gpz;
        const float accum_color_dot =
            (cax * gpx + cay * gpy + caz * gpz) * inv_1ma;
        float dL_dopa_render = T * c_dot - accum_color_dot;
        float dL_dopa = dL_dopa_render;
        float dL_dopa_sigma = 0.f;
        float dL_dt = 0.f;

        if constexpr (GEOMETRY) {
            const float n_dot =
                nrm.x * gnx + nrm.y * gny + nrm.z * gnz;
            const float accum_normal_dot =
                (nax * gnx + nay * gny + naz * gnz) * inv_1ma;
            dL_dopa += T * n_dot - accum_normal_dot;
            acc_normal.x += blend_weight * gnx;
            acc_normal.y += blend_weight * gny;
            acc_normal.z += blend_weight * gnz;

            const float t_peak = rp.x * dx + rp.y * dy + rp.z;
            const float rsigma = rp.w;
            const float mDepth = mDepth_reg;
            const float dL_dmt = dL_dmt_reg;
            const float t_delta = (mDepth - t_peak) * rsigma;
            const float G_exp = expf(-0.5f * t_delta * t_delta);
            const float Gt = alpha * G_exp;
            float dL_dGt = dL_dmt * 0.25f / (1.f - Gt);
            dL_dGt = mDepth > t_peak ? dL_dGt : -dL_dGt;
            dL_dGt = rsigma > 0.f ? dL_dGt : 0.f;
            dL_dopa_sigma =
                dL_dGt * G_exp -
                dL_dmt * (t_delta > 0.f ? 0.5f * inv_1ma : 0.f);
            const float dL_ddelta = -dL_dGt * Gt * t_delta;
            acc_ray_plane.w += dL_ddelta * (mDepth - t_peak);
            dL_dt = -dL_ddelta * rsigma;
            acc_ray_plane.x += dL_dt * dx;
            acc_ray_plane.y += dL_dt * dy;
            acc_ray_plane.z += dL_dt;
        }

        if constexpr (GEOMETRY) dL_dopa += dL_dopa_sigma;
        dL_dopa_render += -T_final * inv_1ma * dL_fin_r;
        dL_dopa += -T_final * inv_1ma * dL_fin;

        // The native camera path differentiates the alpha clamp.
        if (is_fisheye(K.mode) && co.w * G >= cfg::kAlphaClip) {
            dL_dopa = 0.f;
            dL_dopa_render = 0.f;
        }

        acc_color[0] += blend_weight * gpx;
        acc_color[1] += blend_weight * gpy;
        acc_color[2] += blend_weight * gpz;

        const float dL_dG = co.w * dL_dopa;
        const float dL_dG_render = co.w * dL_dopa_render;
        const float gdx = G * dx;
        const float gdy = G * dy;
        const float dG_ddelx = -gdx * co.x - gdy * co.y;
        const float dG_ddely = -gdy * co.z - gdx * co.y;
        const float dL_ddelx_render = dL_dG_render * dG_ddelx;
        const float dL_ddely_render = dL_dG_render * dG_ddely;
        float dL_ddelx = dL_dG * dG_ddelx;
        float dL_ddely = dL_dG * dG_ddely;
        if constexpr (GEOMETRY) {
            dL_ddelx += dL_dt * rp.x;
            dL_ddely += dL_dt * rp.y;
        }
        acc_mean2d.x += dL_ddelx;
        acc_mean2d.y += dL_ddely;
        acc_mean2d.z += fabsf(dL_ddelx) + fabsf(dL_ddely);
        acc_conic.x += -0.5f * gdx * dx * dL_dG;
        acc_conic.y += -0.5f * gdx * dy * dL_dG;
        acc_conic.z += -0.5f * gdy * dy * dL_dG;
        acc_conic.w += G * dL_dopa;
        if (refine_weight != nullptr) {
            const float gx = dL_ddelx_render * float(width);
            const float gy = dL_ddely_render * float(height);
            acc_refine += sqrtf(gx * gx + gy * gy) / fmaxf(1.f - T_final, 1.0e-5f);
        }
        if (densify_map != nullptr) {
            const unsigned pid = unsigned(width) * py + px;
            acc_densify += densify_map[pid] * blend_weight;
            acc_densify_den += blend_weight;
        }

        T *= 1.f - alpha;
    }

    if (valid_instance) {
        atomicAdd(&dL_dcolors[3 * g + 0], acc_color[0]);
        atomicAdd(&dL_dcolors[3 * g + 1], acc_color[1]);
        atomicAdd(&dL_dcolors[3 * g + 2], acc_color[2]);
        atomicAdd(&gs.d_mean2d[g].x, acc_mean2d.x);
        atomicAdd(&gs.d_mean2d[g].y, acc_mean2d.y);
        atomicAdd(&gs.d_mean2d[g].z, acc_mean2d.z);
        atomicAdd(&gs.d_conic[g].x, acc_conic.x);
        atomicAdd(&gs.d_conic[g].y, acc_conic.y);
        atomicAdd(&gs.d_conic[g].z, acc_conic.z);
        atomicAdd(&gs.d_conic[g].w, acc_conic.w);
        if (refine_weight != nullptr) atomicAdd(&refine_weight[g], acc_refine);
        if (densify_weight != nullptr) atomicAdd(&densify_weight[g], acc_densify);
        if (densify_weight_den != nullptr)
            atomicAdd(&densify_weight_den[g], acc_densify_den);
        if constexpr (GEOMETRY) {
            atomicAdd(&gs.d_normal[g].x, acc_normal.x);
            atomicAdd(&gs.d_normal[g].y, acc_normal.y);
            atomicAdd(&gs.d_normal[g].z, acc_normal.z);
            atomicAdd(&gs.d_ray_plane[g].x, acc_ray_plane.x);
            atomicAdd(&gs.d_ray_plane[g].y, acc_ray_plane.y);
            atomicAdd(&gs.d_ray_plane[g].z, acc_ray_plane.z);
            atomicAdd(&gs.d_ray_plane[g].w, acc_ray_plane.w);
        }
    }
}

__device__ inline float structure_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ inline void adam_scalar(
    float* parameter, int index, float grad, float* first, float* second,
    float lr, float beta1, float beta2, float correction1, float correction2,
    float epsilon, float clamp_min, float clamp_max) {
    const float previous = parameter[index];
    if (!isfinite(previous) || !isfinite(grad)) {
        first[index] = 0.f;
        second[index] = 0.f;
        parameter[index] = isfinite(previous)
            ? fminf(fmaxf(previous, clamp_min), clamp_max)
            : fminf(fmaxf(0.f, clamp_min), clamp_max);
        return;
    }
    const float m = beta1 * first[index] + (1.f - beta1) * grad;
    const float v = beta2 * second[index] + (1.f - beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        first[index] = 0.f;
        second[index] = 0.f;
        parameter[index] = fminf(fmaxf(previous, clamp_min), clamp_max);
        return;
    }
    first[index] = m;
    second[index] = v;
    const float candidate =
        previous - lr * (m / correction1) / (sqrtf(v / correction2) + epsilon);
    const float updated = isfinite(candidate) ? candidate : previous;
    parameter[index] = fminf(fmaxf(updated, clamp_min), clamp_max);
}

__device__ inline void add_shape_regularizer_grads(
    const float* log_scales, const float* quaternions, int i,
    float* grad_log, float* grad_quat, const StructureAdam& a) {
    if (a.shape_scale_reg > 0.f)
        for (int axis = 0; axis < 3; ++axis)
            if (log_scales[3 * i + axis] > -40.f)
                grad_log[axis] += a.shape_scale_reg / 3.f;
    if (a.shape_quat_norm_reg > 0.f) {
        const float ns =
            quaternions[4 * i] * quaternions[4 * i] +
            quaternions[4 * i + 1] * quaternions[4 * i + 1] +
            quaternions[4 * i + 2] * quaternions[4 * i + 2] +
            quaternions[4 * i + 3] * quaternions[4 * i + 3];
        if (ns > 1e-20f) {
            const float n = sqrtf(ns);
            const float factor = a.shape_quat_norm_reg * (n - 1.f) / ns;
            for (int c = 0; c < 4; ++c)
                grad_quat[c] += factor * quaternions[4 * i + c];
        }
    }
    if (a.shape_erank_reg <= 0.f && a.shape_erank_s3_reg <= 0.f) return;
    float log_scale[3]{};
    int largest = 0;
    for (int axis = 0; axis < 3; ++axis) {
        log_scale[axis] = log_scales[3 * i + axis];
        if (!(log_scale[axis] > -1e4f)) log_scale[axis] = -1e4f;
        if (axis > 0 && log_scale[axis] > log_scale[largest]) largest = axis;
    }
    float x[3]{};
    for (int axis = 0; axis < 3; ++axis)
        x[axis] = expf(2.f * (log_scale[axis] - log_scale[largest]));
    const float sum = x[0] + x[1] + x[2];
    if (!(sum > 0.f)) return;
    int smallest = 0;
    for (int axis = 1; axis < 3; ++axis)
        if (x[axis] < x[smallest]) smallest = axis;
    const float share_largest = x[largest] / sum;
    const float raw_share_smallest = x[smallest] / sum;
    const bool smallest_clamped = !(raw_share_smallest > 1e-30f);
    const float share_smallest =
        smallest_clamped ? 1e-30f : raw_share_smallest;
    const float middle_complement = 1.f - share_largest - share_smallest;
    const bool middle_clamped = !(middle_complement > 1e-30f);
    const float share_middle = middle_clamped ? 1e-30f : middle_complement;
    const float entropy = -(
        share_largest * logf(share_largest) +
        share_middle * logf(share_middle) +
        share_smallest * logf(share_smallest));
    const float rank = expf(entropy);
    float dreg_dshare[3]{};
    if (a.shape_erank_reg > 0.f && rank - 0.99999f < 1.f) {
        const float shares[3]{share_largest, share_middle, share_smallest};
        for (int k = 0; k < 3; ++k)
            dreg_dshare[k] = rank * (logf(shares[k]) + 1.f) / (rank - 0.99999f);
    }
    const float dloss_dshare_smallest = dreg_dshare[2] + a.shape_erank_s3_reg;
    float dloss_dx[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        float d_largest = axis == largest ? (1.f - share_largest) / sum
                                          : -share_largest / sum;
        float d_smallest = 0.f;
        if (axis == smallest) {
            if (!smallest_clamped)
                d_smallest = (1.f - share_smallest) / sum;
        } else {
            d_smallest = -share_smallest / sum;
        }
        const float d_middle =
            middle_clamped ? 0.f : -(d_largest + d_smallest);
        dloss_dx[axis] = dreg_dshare[0] * d_largest +
                         dreg_dshare[1] * d_middle +
                         dloss_dshare_smallest * d_smallest;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (axis == largest) {
            float pull = 0.f;
            for (int other = 0; other < 3; ++other)
                if (other != largest) pull += dloss_dx[other] * x[other];
            grad_log[axis] += a.shape_erank_reg * (-2.f * pull);
        } else {
            grad_log[axis] += a.shape_erank_reg * dloss_dx[axis] * 2.f * x[axis];
        }
    }
}

__device__ inline void update_structure_adam_row(
    int i, float3 d_mean, float3 d_scale, float4 d_rot, float d_opacity,
    const float* __restrict__ activated_scales,
    const float* __restrict__ activated_opacities, const StructureAdam& a) {
    const float filter_squared =
        a.filter_3d != nullptr ? a.filter_3d[i] * a.filter_3d[i] : 0.f;
    const float activated_opacity = activated_opacities[i];
    float grad_log[3];
    for (int axis = 0; axis < 3; ++axis) {
        const int offset = 3 * i + axis;
        const float raw_scale = expf(a.log_scales[offset]);
        const float filtered_scale = activated_scales[offset];
        const float scale_grad =
            axis == 0 ? d_scale.x : axis == 1 ? d_scale.y : d_scale.z;
        grad_log[axis] =
            scale_grad * raw_scale * raw_scale / filtered_scale +
            d_opacity * activated_opacity * filter_squared /
                (filtered_scale * filtered_scale);
    }
    const float rw = a.quaternions[4 * i];
    const float rx = a.quaternions[4 * i + 1];
    const float ry = a.quaternions[4 * i + 2];
    const float rz = a.quaternions[4 * i + 3];
    const float inverse_norm =
        rsqrtf(fmaxf(rw * rw + rx * rx + ry * ry + rz * rz, 1e-20f));
    const float q[4] = {rw * inverse_norm, rx * inverse_norm, ry * inverse_norm,
                        rz * inverse_norm};
    const float rot[4] = {d_rot.x, d_rot.y, d_rot.z, d_rot.w};
    const float qdot = q[0] * rot[0] + q[1] * rot[1] + q[2] * rot[2] + q[3] * rot[3];
    float grad_quat[4];
    for (int c = 0; c < 4; ++c)
        grad_quat[c] = inverse_norm * (rot[c] - q[c] * qdot);
    const float opacity = structure_sigmoid(a.opacity_logits[i]);
    float grad_logit = d_opacity * activated_opacity * (1.f - opacity);
    if (isfinite(a.opacity_logits[i]) && a.opacity_reg > 0.f)
        grad_logit += a.opacity_reg * opacity * (1.f - opacity);
    if (a.log_scale_reg > 0.f)
        for (int axis = 0; axis < 3; ++axis) {
            const float log_scale = a.log_scales[3 * i + axis];
            if (isfinite(log_scale) && log_scale > -40.f)
                grad_log[axis] += a.log_scale_reg * expf(log_scale);
        }
    add_shape_regularizer_grads(
        a.log_scales, a.quaternions, i, grad_log, grad_quat, a);
    if (a.opacity_gradient) a.opacity_gradient[i] = grad_logit;
    const float inf = 1.e30f;
    for (int c = 0; c < 3; ++c) {
        const int mi = 3 * i + c;
        const float mg = c == 0 ? d_mean.x : c == 1 ? d_mean.y : d_mean.z;
        adam_scalar(a.means, mi, mg, a.means_first, a.means_second, a.means_lr,
                    a.beta1, a.beta2, a.correction1, a.correction2, a.epsilon,
                    -inf, inf);
        adam_scalar(a.log_scales, mi, grad_log[c], a.scales_first, a.scales_second,
                    a.scales_lr, a.beta1, a.beta2, a.correction1, a.correction2,
                    a.epsilon, a.minimum_log_scale, a.maximum_log_scale);
    }
    if (a.max_log_scale_ratio > 0.f) {
        float* v = a.log_scales + 3 * i;
        const float minimum = fminf(v[0], fminf(v[1], v[2]));
        const float maximum = fmaxf(v[0], fmaxf(v[1], v[2]));
        if (maximum - minimum > a.max_log_scale_ratio) {
            const float midpoint = 0.5f * (minimum + maximum);
            const float half_range = 0.5f * a.max_log_scale_ratio;
            for (int c = 0; c < 3; ++c)
                v[c] = fminf(fmaxf(v[c], midpoint - half_range),
                             midpoint + half_range);
        }
    }
    for (int c = 0; c < 4; ++c)
        adam_scalar(a.quaternions, 4 * i + c, grad_quat[c], a.rotations_first,
                    a.rotations_second, a.quaternions_lr, a.beta1, a.beta2,
                    a.correction1, a.correction2, a.epsilon, -inf, inf);
    adam_scalar(a.opacity_logits, i, grad_logit, a.opacity_first, a.opacity_second,
                a.opacities_lr, a.beta1, a.beta2, a.correction1, a.correction2,
                a.epsilon, -12.f, 12.f);
}

template <bool HasSH, bool HasCov, bool FusedSH>
__global__ void __launch_bounds__(cfg::kGaussianBlock, 1)
gaussian_backward(
    const int count, const int sh_degree, const int sh_bases,
    const float* __restrict__ means,
    const float* __restrict__ sh, const float* __restrict__ opacities,
    const float* __restrict__ scales, const float* __restrict__ rotations,
    const float* __restrict__ cov6, const float* __restrict__ view,
    const float* __restrict__ camera_center, CameraIntrinsics K, int width,
    int height, float kernel_size, float scale_modifier,
    const int* __restrict__ radius, const bool* __restrict__ clamped,
    ws::GaussianState st, ws::GradState gs, float* __restrict__ grad_mean,
    float* __restrict__ grad_sh, float* __restrict__ grad_colors,
    float* __restrict__ grad_opacity, float* __restrict__ grad_scale,
    float* __restrict__ grad_rotation, float* __restrict__ grad_cov, SHAdam sh_adam,
    StructureAdam structure_adam) {
    // Fused SH keeps the coefficients of this thread's Gaussian in a shared
    // tile instead of a 48-register row: the warp then updates all 32 rows it
    // owns with one lane per column, which is the coalesced pattern the
    // standalone optimizer used.
    __shared__ float4 sh_tile[FusedSH
        ? cfg::kGaussianBlock * cfg::kShRowWords
        : 1];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float* shared_row = nullptr;
    if constexpr (FusedSH) {
        shared_row = reinterpret_cast<float*>(
            &sh_tile[threadIdx.x * cfg::kShRowWords]);
#pragma unroll
        for (int k = 0; k < cfg::kShRowWords; ++k)
            sh_tile[threadIdx.x * cfg::kShRowWords + k] =
                make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float3 d_mean = make_float3(0.f, 0.f, 0.f);
    float3 d_scale = make_float3(0.f, 0.f, 0.f);
    float4 d_rot = make_float4(0.f, 0.f, 0.f, 0.f);
    float d_opacity = 0.f;
    // Culled rows keep zero grads: Adam moments still decay, matching the
    // standalone optimizer on a zeroed gradient buffer.
    if (i < count && radius[i] > 0) {

        geo::SplatBackward io;
        io.d_conic = gs.d_conic[i];
        // Color-only pools leave the geometry scratch unallocated; zero inputs
        // make splat_backward() take its unused-branch early-out, exactly like a
        // memset-to-zero scratch buffer did.
        io.d_ray_plane =
            gs.d_ray_plane ? gs.d_ray_plane[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        io.d_normal =
            gs.d_normal ? gs.d_normal[i] : make_float3(0.f, 0.f, 0.f);
        io.d_mean2d = make_float2(gs.d_mean2d[i].x, gs.d_mean2d[i].y);
        io.mean = make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
        io.view = view;
        io.K = K;
        io.width = width;
        io.height = height;
        io.kernel_size = kernel_size;
        io.scale_modifier = scale_modifier;
        io.scale = scales ? reinterpret_cast<const float3*>(scales) + i : nullptr;
        io.rotation =
            rotations ? reinterpret_cast<const float4*>(rotations) + i : nullptr;
        io.cov6 = HasCov ? cov6 + 6 * i : nullptr;
        io.opacity = opacities[i];
        geo::splat_backward(io);
        d_mean = io.grad_mean;
        d_opacity = io.grad_opacity;
        if (HasCov) {
            if (grad_cov) {
#pragma unroll
                for (int k = 0; k < 6; ++k) grad_cov[6 * i + k] += io.grad_cov6[k];
            }
        } else {
            d_scale = io.grad_scale;
            d_rot = io.grad_rotation;
        }

        if (HasSH) {
            sh::BackwardIO bwd;
            bwd.grad_mean = make_float3(0.f, 0.f, 0.f);
            bwd.grad_sh = reinterpret_cast<float3*>(grad_sh);
            if constexpr (FusedSH)
                bwd.grad_sh_row = reinterpret_cast<float3*>(shared_row);
            sh::backward(i, sh_degree, sh_bases,
                         reinterpret_cast<const float3*>(means),
                         make_float3(camera_center[0], camera_center[1],
                                     camera_center[2]),
                         sh, st.clamped, gs.d_color, bwd);
            d_mean.x += bwd.grad_mean.x;
            d_mean.y += bwd.grad_mean.y;
            d_mean.z += bwd.grad_mean.z;
        } else if (grad_colors) {
            grad_colors[3 * i + 0] += gs.d_color[i].x;
            grad_colors[3 * i + 1] += gs.d_color[i].y;
            grad_colors[3 * i + 2] += gs.d_color[i].z;
        }
        if (!structure_adam.means) {
            if (grad_mean) {
                grad_mean[3 * i + 0] += d_mean.x;
                grad_mean[3 * i + 1] += d_mean.y;
                grad_mean[3 * i + 2] += d_mean.z;
            }
            if (grad_opacity) grad_opacity[i] += d_opacity;
            if constexpr (!HasCov) {
                if (grad_scale) {
                    grad_scale[3 * i + 0] += d_scale.x;
                    grad_scale[3 * i + 1] += d_scale.y;
                    grad_scale[3 * i + 2] += d_scale.z;
                }
                if (grad_rotation) {
                    grad_rotation[4 * i + 0] += d_rot.x;
                    grad_rotation[4 * i + 1] += d_rot.y;
                    grad_rotation[4 * i + 2] += d_rot.z;
                    grad_rotation[4 * i + 3] += d_rot.w;
                }
            }
        }
    }
    if constexpr (FusedSH) {
        __syncthreads();
        const int warp_thread0 = (int(threadIdx.x) >> 5) * 32;
        const int warp_row0 = int(blockIdx.x) * int(blockDim.x) + warp_thread0;
        float* warp_tile = reinterpret_cast<float*>(sh_tile) +
            std::size_t(warp_thread0) * cfg::kShRowStride;
        const int active = (sh_degree + 1) * (sh_degree + 1) * 3;
        for (int r = 0; r < 32; ++r) {
            const int row = warp_row0 + r;
            if (row >= count) break;
            float* parameter = sh_adam.parameter +
                std::size_t(row) * sh_adam.stride;
            __half* first = reinterpret_cast<__half*>(sh_adam.first) +
                std::size_t(row) * sh_adam.stride;
            sh_adam_quant_warp_row(
                sh_adam.stride, active, parameter,
                warp_tile + r * cfg::kShRowStride, first,
                sh_adam.packed + std::size_t(row) * sh_adam.stride,
                sh_adam.bounds + std::size_t(row) * 4, sh_adam.lr,
                sh_adam.rest_lr, sh_adam.beta1, sh_adam.beta2,
                sh_adam.correction1, sh_adam.correction2,
                sh_adam.epsilon, sh_adam.regularization_factor);
        }
    }
    if constexpr (!HasCov) {
        if (structure_adam.means && i < count)
            update_structure_adam_row(
                i, d_mean, d_scale, d_rot, d_opacity, scales, opacities,
                structure_adam);
    }
}

}  // namespace splat_drender::kernels

namespace splat_drender::launch {

void median_scale_backward(const uint2* tile_range,
                           const unsigned* instance_value, int width,
                           int height, CameraIntrinsics K, int wrap_width,
                           const float2* mean2d, const float4* conic_opacity,
                           const float4* ray_plane,
                           const ushort4* screen_bounds,
                           const unsigned* n_contrib,
                           const unsigned* max_contributor,
                           const float* median_depth,
                          const float* dL_dmedian, float* dL_dmt,
                          dim3 grid) {
    const dim3 block(cfg::kTileWidth, cfg::kTileHeight);
    kernels::median_scale_walk<<<grid, block>>>(
        tile_range, instance_value, width, height, K, wrap_width, mean2d,
        conic_opacity, ray_plane, screen_bounds, n_contrib, max_contributor,
        median_depth, dL_dmedian, dL_dmt);
}

void blend_bucket_backward(bool need_depth, const uint2* tile_range,
                           const unsigned* instance_value, int width,
                           int height, CameraIntrinsics K, int wrap_width,
                           float3 background, const float2* mean2d,
                           const float4* conic_opacity, const float3* rgb,
                           const float* colors, const float4* ray_plane,
                           const float3* normal,
                           const ushort4* screen_bounds, const float* alphas,
                           const float* normal_map, const float* median_depth,
                           const unsigned* n_contrib,
                           const unsigned* max_contributor,
                           const unsigned* bucket_offset,
                           const unsigned* bucket_tile,
                           const ws::PixelState pst, const float* dL_color,
                           const float* dL_median, const float* dL_alpha,
                           const float* dL_normal, ws::GradState gs,
                           float* dL_colors, float* refine_weight,
                           const float* densify_map, float* densify_weight,
                           float* densify_weight_den, int buckets) {
    if (buckets <= 0) return;
    const dim3 grid((buckets + 7) / 8, 1, 1);
    if (need_depth) {
        kernels::blend_bucket_backward<true><<<grid, 256>>>(
            tile_range, instance_value, width, height, K, wrap_width,
            background, mean2d, conic_opacity, rgb, colors, ray_plane, normal,
            screen_bounds, alphas, normal_map, median_depth,
            n_contrib, max_contributor, bucket_offset, bucket_tile, pst,
            dL_color, dL_median, dL_alpha, dL_normal, gs, dL_colors,
            refine_weight, densify_map, densify_weight, densify_weight_den,
            buckets);
    } else {
        kernels::blend_bucket_backward<false><<<grid, 256>>>(
            tile_range, instance_value, width, height, K, wrap_width,
            background, mean2d, conic_opacity, rgb, colors, ray_plane, normal,
            screen_bounds, alphas, normal_map, median_depth,
            n_contrib, max_contributor, bucket_offset, bucket_tile, pst,
            dL_color, dL_median, dL_alpha, dL_normal, gs, dL_colors,
            refine_weight, densify_map, densify_weight, densify_weight_den,
            buckets);
    }
}

void gaussian_backward(bool has_sh, bool has_cov, int count, int sh_degree,
                       int sh_bases, const float* means, const float* sh,
                       const float* opacities, const float* scales,
                       const float* rotations, const float* cov6,
                       const float* view, const float* camera_center,
                       CameraIntrinsics K, int width, int height,
                       float kernel_size, float scale_modifier,
                       const int* radius, const bool* clamped,
                       ws::GaussianState st, ws::GradState gs, float* grad_mean,
                       float* grad_sh, float* grad_colors, float* grad_opacity,
                       float* grad_scale, float* grad_rotation,
                       float* grad_cov, SHAdam sh_adam,
                       StructureAdam structure_adam) {
    auto dispatch = [&](auto sh_c, auto cov_c, auto fused_c) {
        constexpr bool kSH = decltype(sh_c)::value;
        constexpr bool kCov = decltype(cov_c)::value;
        constexpr bool kFused = decltype(fused_c)::value;
        kernels::gaussian_backward<kSH, kCov, kFused>
            <<<(count + cfg::kGaussianBlock - 1) / cfg::kGaussianBlock,
                cfg::kGaussianBlock>>>(
                count, sh_degree, sh_bases, means, sh, opacities, scales,
                rotations, cov6, view, camera_center, K, width, height,
                kernel_size, scale_modifier, radius, clamped, st, gs, grad_mean,
                grad_sh, grad_colors, grad_opacity, grad_scale, grad_rotation,
                grad_cov, sh_adam, structure_adam);
    };
    const auto off = std::false_type{};
    const auto on = std::true_type{};
    if (has_sh) {
        if (sh_adam.parameter) {
            if (has_cov) dispatch(on, on, on);
            else dispatch(on, off, on);
        } else {
            if (has_cov) dispatch(on, on, off);
            else dispatch(on, off, off);
        }
    } else {
        if (has_cov) dispatch(off, on, off);
        else dispatch(off, off, off);
    }
}

}  // namespace splat_drender::launch
