// Backward render kernels.
//
// blend_tile_backward walks the sorted instances exactly like the forward
// pass (same contributor numbering) and accumulates per-Gaussian blending
// gradients with warp reduction. gaussian_backward fuses the EWA geometry
// backward, the mean2d->mean3d chain and the SH backward into one launch.
#include "device/geometry.cuh"
#include "kernels.h"
#include "splat_drender/buffers.h"

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

template <bool GEOMETRY>
__global__ void __launch_bounds__(cfg::kTileThreads)
blend_tile_backward(const uint2* __restrict__ tile_range,
                    const unsigned* __restrict__ instance_value, int width,
                    int height, CameraIntrinsics K, int wrap_width,
                    const float2* __restrict__ mean2d,
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
                    const float* __restrict__ dL_dpixel_in,
                    const float* __restrict__ dL_dmedian_in,
                    const float* __restrict__ dL_dalpha_in,
                    const float* __restrict__ dL_dnormal_in, ws::GradState gs,
                    float* __restrict__ dL_dcolors, float3 background,
                    float* __restrict__ refine_weight) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
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
    const int rounds =
        (int(max_contrib) + cfg::kTileThreads - 1) / cfg::kTileThreads;

    __shared__ unsigned sid[cfg::kTileThreads];
    __shared__ float2 sxy[cfg::kTileThreads];
    __shared__ float4 sconic[cfg::kTileThreads];
    __shared__ float3 scolor[cfg::kTileThreads];
    __shared__ float4 splane[cfg::kTileThreads];
    __shared__ float3 snormal[cfg::kTileThreads];
    __shared__ ushort4 sbounds[cfg::kTileThreads];

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

    const float w_final = inside ? alphas[pix_id] : 0.f;
    const float T_final = 1.f - w_final;
    float T = T_final;
    float mDepth = 0.f;
    const unsigned last_contributor = inside ? n_contrib[pix_id] : 0;

    float accum_color_dot = 0.f;
    float dL_dpixel[3];
    float dL_dfinalT = 0.f;
    float dL_dfinalT_render = 0.f;
    float dL_dpixel_mt = 0.f;
    float accum_normal_dot = 0.f;
    float dL_dpixel_normal[3] = {0.f, 0.f, 0.f};

    if (inside) {
#pragma unroll
        for (int ch = 0; ch < 3; ++ch)
            dL_dpixel[ch] = dL_dpixel_in[ch * width * height + pix_id];

        dL_dfinalT = -dL_dalpha_in[pix_id];
#pragma unroll
        for (int ch = 0; ch < 3; ++ch)
            dL_dfinalT += (ch == 0 ? background.x : ch == 1 ? background.y : background.z) * dL_dpixel[ch];
        dL_dfinalT_render = dL_dfinalT;

        if constexpr (GEOMETRY) {
            const float inv_w = w_final > 0.f ? 1.f / w_final : 0.f;
            const float rln = pixel_ray_z(pixf.x, pixf.y, K);
            dL_dpixel_mt = dL_dmedian_in[pix_id] * rln;

            const float gn[3] = {
                dL_dnormal_in[pix_id],
                dL_dnormal_in[width * height + pix_id],
                dL_dnormal_in[2 * width * height + pix_id]};
            const float nm[3] = {
                normal_map[pix_id],
                normal_map[width * height + pix_id],
                normal_map[2 * width * height + pix_id]};
            // Reference builds unnormalized accumulated normals
            // (out_normal = Normal / (1 - T)).
            dL_dpixel_normal[0] = gn[0] * inv_w;
            dL_dpixel_normal[1] = gn[1] * inv_w;
            dL_dpixel_normal[2] = gn[2] * inv_w;
            dL_dfinalT += dL_dpixel_normal[0] * nm[0] +
                          dL_dpixel_normal[1] * nm[1] +
                          dL_dpixel_normal[2] * nm[2];
            mDepth = median_depth_out[pix_id] / fmaxf(rln, 1.0e-8f);
            // A failed median bracket keeps its upstream loss gradient: the
            // dT_dtm walk stays empty and dL_dmt_dT_dtm inherits the huge
            // 1/1e-7 penalty, matching the reference trainer exactly (it is
            // what pushes pixels back inside the median bracket).
        }
    }

    // ---- dT/dtm accumulation for the median-depth gradient ----
    float dL_dmt_dT_dtm = 0.f;
    if constexpr (GEOMETRY) {
        float dT_dtm = 0.f;
        int todo = int(max_contrib);
        unsigned contributor = 0;
        bool done = (mDepth == 0.f) || (last_contributor == 0) || !inside;
        for (int round = 0; round < rounds;
             ++round, todo -= cfg::kTileThreads) {
            __syncthreads();
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
            for (int j = 0; j < batch; ++j) {
                contributor++;
                const bool active = !done && overlaps(sbounds[j], wb);
                if (__ballot_sync(0xffffffffu, active) == 0u) continue;
                if (!active) continue;
                done = contributor >= last_contributor;

                const float2 d =
                    make_float2(wrap_dx(sxy[j].x - pixf.x, wrap_width, K.mode),
                                sxy[j].y - pixf.y);
                const float4 co = sconic[j];
                const float power =
                    -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                    co.y * d.x * d.y;
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
        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1.0e-7f);
    }

    // ---- main backward traversal (back to front) ----
    bool done = !inside;
    int todo = int(max_contrib);
    unsigned contributor = max_contrib;
    float last_alpha = 0.f;
    float last_color_dot = 0.f;
    float last_normal_dot = 0.f;

    for (int round = 0; round < rounds; ++round, todo -= cfg::kTileThreads) {
        __syncthreads();
        const int progress = round * cfg::kTileThreads + block.thread_rank();
        if (progress < int(max_contrib)) {
            // Fetch in reverse order: batch slot j holds list position
            // max_contrib - (round * kTileThreads + j) - 1.
            const unsigned g =
                instance_value[range.x + max_contrib - progress - 1];
            sid[block.thread_rank()] = g;
            sxy[block.thread_rank()] = mean2d[g];
            sconic[block.thread_rank()] = conic_opacity[g];
            scolor[block.thread_rank()] =
                colors ? reinterpret_cast<const float3*>(colors)[g] : rgb[g];
            if constexpr (GEOMETRY) {
                splane[block.thread_rank()] = ray_plane[g];
                snormal[block.thread_rank()] = normal[g];
            }
            sbounds[block.thread_rank()] = screen_bounds[g];
        }
        __syncthreads();

        const int batch = min(cfg::kTileThreads, todo);
        for (int j = 0; j < batch; ++j) {
            contributor--;

            const float2 d =
                make_float2(wrap_dx(sxy[j].x - pixf.x, wrap_width, K.mode),
                            sxy[j].y - pixf.y);
            const float4 co = sconic[j];
            const float power =
                -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) -
                co.y * d.x * d.y;
            const float G = expf(power);
            const float alpha = fminf(cfg::kAlphaClip, co.w * G);
            bool valid = !(done || (contributor >= last_contributor) ||
                           (power > 0.f) || (alpha < cfg::kAlphaFloor));
            valid = valid && overlaps(sbounds[j], wb);

            if (!warp.any(valid)) continue;

            float dL_dcolors_local[3] = {0.f, 0.f, 0.f};
            float3 dL_dnormals_local = make_float3(0.f, 0.f, 0.f);
            float2 dL_dray_planes_local = make_float2(0.f, 0.f);
            float dL_dtc_local = 0.f;
            float dL_drsigma_local = 0.f;
            float3 dL_dmean2d_local = make_float3(0.f, 0.f, 0.f);
            float4 dL_dconic_local = make_float4(0.f, 0.f, 0.f, 0.f);
            float dL_drefine_local = 0.f;

            if (valid) {
                T = T / (1.f - alpha);
                const float blend_weight = alpha * T;

                float dL_dopa_render = 0.f;
                float dL_dopa = 0.f;
                accum_color_dot = last_alpha * last_color_dot +
                                  (1.f - last_alpha) * accum_color_dot;

                float c_dot = 0.f;
#pragma unroll
                for (int ch = 0; ch < 3; ++ch) {
                    const float c = ch == 0 ? scolor[j].x
                                   : ch == 1 ? scolor[j].y : scolor[j].z;
                    c_dot = fmaf(c, dL_dpixel[ch], c_dot);
                    dL_dcolors_local[ch] = blend_weight * dL_dpixel[ch];
                }
                dL_dopa_render = (c_dot - accum_color_dot);
                dL_dopa = dL_dopa_render;
                last_color_dot = c_dot;

                float dL_dt = 0.f;
                float4 rp = make_float4(0.f, 0.f, 0.f, 0.f);
                float dL_dopa_sigma = 0.f;
                if constexpr (GEOMETRY) {
                    const float3 n = snormal[j];
                    accum_normal_dot = last_alpha * last_normal_dot +
                                       (1.f - last_alpha) * accum_normal_dot;
                    const float n_dot = fmaf(n.x, dL_dpixel_normal[0],
                                             fmaf(n.y, dL_dpixel_normal[1],
                                                  n.z * dL_dpixel_normal[2]));
                    dL_dopa += (n_dot - accum_normal_dot);
                    last_normal_dot = n_dot;
                    dL_dnormals_local.x = blend_weight * dL_dpixel_normal[0];
                    dL_dnormals_local.y = blend_weight * dL_dpixel_normal[1];
                    dL_dnormals_local.z = blend_weight * dL_dpixel_normal[2];

                    rp = splane[j];
                    const float t_peak = rp.x * d.x + rp.y * d.y + rp.z;
                    const float rsigma = rp.w;
                    const float t_delta = (mDepth - t_peak) * rsigma;
                    const float G_exp = expf(-0.5f * t_delta * t_delta);
                    const float Gt = alpha * G_exp;

                    float dL_dGt = dL_dmt_dT_dtm * 0.25f / (1.f - Gt);
                    dL_dGt = mDepth > t_peak ? dL_dGt : -dL_dGt;
                    dL_dGt = rsigma > 0.f ? dL_dGt : 0.f;
                    dL_dopa_sigma =
                        dL_dGt * G_exp -
                        dL_dmt_dT_dtm * (t_delta > 0.f ? 0.5f / (1.f - alpha) : 0.f);
                    const float dL_ddelta = -dL_dGt * Gt * t_delta;
                    dL_drsigma_local = dL_ddelta * (mDepth - t_peak);
                    const float dL_dt_peak = -dL_ddelta * rsigma;

                    dL_dt = dL_dt_peak;
                    dL_dtc_local = dL_dt;
                    dL_dray_planes_local.x = dL_dt * d.x;
                    dL_dray_planes_local.y = dL_dt * d.y;
                }

                dL_dopa_render *= T;
                dL_dopa *= T;
                if constexpr (GEOMETRY) dL_dopa += dL_dopa_sigma;
                dL_dopa_render += -T_final / (1.f - alpha) * dL_dfinalT_render;
                dL_dopa += -T_final / (1.f - alpha) * dL_dfinalT;
                last_alpha = alpha;

                // The native camera path differentiates the alpha clamp.
                if (is_fisheye(K.mode) && co.w * G >= cfg::kAlphaClip) {
                    dL_dopa = 0.f;
                    dL_dopa_render = 0.f;
                }

                const float dL_dG = co.w * dL_dopa;
                const float dL_dG_render = co.w * dL_dopa_render;
                const float gdx = G * d.x;
                const float gdy = G * d.y;
                const float dG_ddelx = -gdx * co.x - gdy * co.y;
                const float dG_ddely = -gdy * co.z - gdx * co.y;

                // Refine signal keeps only the render branch so geometry
                // gradients do not leak into the densification weight.
                const float dL_ddelx_render = dL_dG_render * dG_ddelx;
                const float dL_ddely_render = dL_dG_render * dG_ddely;
                float dL_ddelx = dL_dG * dG_ddelx;
                float dL_ddely = dL_dG * dG_ddely;
                if constexpr (GEOMETRY) {
                    dL_ddelx += dL_dt * rp.x;
                    dL_ddely += dL_dt * rp.y;
                }
                dL_dmean2d_local = make_float3(
                    dL_ddelx, dL_ddely, fabsf(dL_ddelx) + fabsf(dL_ddely));

                dL_dconic_local = make_float4(-0.5f * gdx * d.x * dL_dG,
                                               -0.5f * gdx * d.y * dL_dG,
                                               -0.5f * gdy * d.y * dL_dG,
                                               G * dL_dopa);

                if (refine_weight != nullptr) {
                    const float gx = dL_ddelx_render * float(width);
                    const float gy = dL_ddely_render * float(height);
                    const float final_a = fmaxf(w_final, 1.0e-5f);
                    dL_drefine_local = sqrtf(gx * gx + gy * gy) / final_a;
                }
            }
            mat::warp_sum(dL_dcolors_local);
            mat::warp_sum(dL_dmean2d_local);
            mat::warp_sum(dL_dconic_local);
            if (refine_weight != nullptr) mat::warp_sum(dL_drefine_local);
            if constexpr (GEOMETRY) {
                mat::warp_sum(dL_dnormals_local);
                mat::warp_sum(dL_dray_planes_local);
                mat::warp_sum(dL_dtc_local);
                mat::warp_sum(dL_drsigma_local);
            }
            if (warp.thread_rank() == 0) {
                const unsigned g = sid[j];
                atomicAdd(&dL_dcolors[3 * g + 0], dL_dcolors_local[0]);
                atomicAdd(&dL_dcolors[3 * g + 1], dL_dcolors_local[1]);
                atomicAdd(&dL_dcolors[3 * g + 2], dL_dcolors_local[2]);
                atomicAdd(&gs.d_mean2d[g].x, dL_dmean2d_local.x);
                atomicAdd(&gs.d_mean2d[g].y, dL_dmean2d_local.y);
                atomicAdd(&gs.d_mean2d[g].z, dL_dmean2d_local.z);
                atomicAdd(&gs.d_conic[g].x, dL_dconic_local.x);
                atomicAdd(&gs.d_conic[g].y, dL_dconic_local.y);
                atomicAdd(&gs.d_conic[g].z, dL_dconic_local.z);
                atomicAdd(&gs.d_conic[g].w, dL_dconic_local.w);
                if (refine_weight != nullptr)
                    atomicAdd(&refine_weight[g], dL_drefine_local);
                if constexpr (GEOMETRY) {
                    atomicAdd(&gs.d_normal[g].x, dL_dnormals_local.x);
                    atomicAdd(&gs.d_normal[g].y, dL_dnormals_local.y);
                    atomicAdd(&gs.d_normal[g].z, dL_dnormals_local.z);
                    atomicAdd(&gs.d_ray_plane[g].x, dL_dray_planes_local.x);
                    atomicAdd(&gs.d_ray_plane[g].y, dL_dray_planes_local.y);
                    atomicAdd(&gs.d_ray_plane[g].z, dL_dtc_local);
                    atomicAdd(&gs.d_ray_plane[g].w, dL_drsigma_local);
                }
            }
        }
    }
}

// Fused per-Gaussian backward: EWA geometry + mean2d + SH.
template <bool HasSH, bool HasCov>
__global__ void gaussian_backward(
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
    float* __restrict__ grad_rotation, float* __restrict__ grad_cov) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || !(radius[i] > 0)) return;

    geo::SplatBackward io;
    io.d_conic = gs.d_conic[i];
    io.d_ray_plane = gs.d_ray_plane[i];
    io.d_normal = gs.d_normal[i];
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
    grad_mean[3 * i + 0] += io.grad_mean.x;
    grad_mean[3 * i + 1] += io.grad_mean.y;
    grad_mean[3 * i + 2] += io.grad_mean.z;
    grad_opacity[i] += io.grad_opacity;
    if (HasCov) {
#pragma unroll
        for (int k = 0; k < 6; ++k) grad_cov[6 * i + k] += io.grad_cov6[k];
    } else {
        grad_scale[3 * i + 0] += io.grad_scale.x;
        grad_scale[3 * i + 1] += io.grad_scale.y;
        grad_scale[3 * i + 2] += io.grad_scale.z;
        grad_rotation[4 * i + 0] += io.grad_rotation.x;
        grad_rotation[4 * i + 1] += io.grad_rotation.y;
        grad_rotation[4 * i + 2] += io.grad_rotation.z;
        grad_rotation[4 * i + 3] += io.grad_rotation.w;
    }

    if (HasSH) {
        sh::BackwardIO bwd;
        bwd.grad_mean = make_float3(0.f, 0.f, 0.f);
        bwd.grad_sh = reinterpret_cast<float3*>(grad_sh);
        sh::backward(i, sh_degree, sh_bases,
                     reinterpret_cast<const float3*>(means),
                     make_float3(camera_center[0], camera_center[1],
                                 camera_center[2]),
                     sh, st.clamped, gs.d_color, bwd);
        grad_mean[3 * i + 0] += bwd.grad_mean.x;
        grad_mean[3 * i + 1] += bwd.grad_mean.y;
        grad_mean[3 * i + 2] += bwd.grad_mean.z;
    } else if (grad_colors) {
        grad_colors[3 * i + 0] += gs.d_color[i].x;
        grad_colors[3 * i + 1] += gs.d_color[i].y;
        grad_colors[3 * i + 2] += gs.d_color[i].z;
    }
}

}  // namespace splat_drender::kernels

namespace splat_drender::launch {

void blend_backward(bool need_depth, const uint2* tile_range,
                    const unsigned* instance_value, int width, int height,
                    CameraIntrinsics K, int wrap_width, float3 background,
                    const float2* mean2d, const float4* conic_opacity,
                    const float3* rgb, const float* colors,
                    const float4* ray_plane, const float3* normal,
                    const ushort4* screen_bounds, const float* alphas,
                    const float* normal_map, const float* median_depth,
                    const unsigned* n_contrib, const unsigned* max_contributor,
                    const float* dL_color, const float* dL_median,
                    const float* dL_alpha, const float* dL_normal,
                    ws::GradState gs, float* dL_colors, float* refine_weight,
                    dim3 grid) {
    dim3 block(cfg::kTileWidth, cfg::kTileHeight);
    if (need_depth) {
        kernels::blend_tile_backward<true><<<grid, block>>>(
            tile_range, instance_value, width, height, K, wrap_width, mean2d,
            conic_opacity, rgb, colors, ray_plane, normal, screen_bounds,
            alphas, normal_map, median_depth, n_contrib, max_contributor,
            dL_color, dL_median, dL_alpha, dL_normal, gs, dL_colors, background,
            refine_weight);
    } else {
        kernels::blend_tile_backward<false><<<grid, block>>>(
            tile_range, instance_value, width, height, K, wrap_width, mean2d,
            conic_opacity, rgb, colors, ray_plane, normal, screen_bounds,
            alphas, normal_map, median_depth, n_contrib, max_contributor,
            dL_color, dL_median, dL_alpha, dL_normal, gs, dL_colors, background,
            refine_weight);
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
                       float* grad_cov) {
    auto dispatch = [&](auto sh_c, auto cov_c) {
        constexpr bool kSH = decltype(sh_c)::value;
        constexpr bool kCov = decltype(cov_c)::value;
        kernels::gaussian_backward<kSH, kCov>
            <<<(count + cfg::kGaussianBlock - 1) / cfg::kGaussianBlock,
                cfg::kGaussianBlock>>>(
                count, sh_degree, sh_bases, means, sh, opacities, scales,
                rotations, cov6, view, camera_center, K, width, height,
                kernel_size, scale_modifier, radius, clamped, st, gs, grad_mean,
                grad_sh, grad_colors, grad_opacity, grad_scale, grad_rotation,
                grad_cov);
    };
    if (has_sh) {
        if (has_cov) dispatch(std::true_type{}, std::true_type{});
        else dispatch(std::true_type{}, std::false_type{});
    } else {
        if (has_cov) dispatch(std::false_type{}, std::true_type{});
        else dispatch(std::false_type{}, std::false_type{});
    }
}

}  // namespace splat_drender::launch
