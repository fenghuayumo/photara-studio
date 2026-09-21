#include "auxiliary.h"
#include "sample_backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// refer to gsplat https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/cuda/include/Utils.cuh#L94
template <uint32_t DIM, class WarpT>
__forceinline__ __device__ void warpSum(float* val, WarpT& warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float& val, WarpT& warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float2& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float3& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float4& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}

__global__ void preprocessPointsCUDA(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const RasterIntrinsics K,
    const uint32_t* tiles_touched,
    const float2* dL_dpoints2D,
    float3* dL_dpoints3D) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || tiles_touched[idx] == 0)
        return;

    float3 m = points3D[idx];

    float3 p_view = transformPoint4x3(m, viewmatrix);
    float3 dL_dpoints;
    if (raster_is_pinhole(K.model)) {
        float rz = 1.0f / (p_view.z + 0.0000001f);
        float sx = p_view.x * rz;
        float sy = p_view.y * rz;
        dL_dpoints.x = (K.focal_x * (viewmatrix[0] - sx * viewmatrix[2]) * dL_dpoints2D[idx].x
                      + K.focal_y * (viewmatrix[1] - sy * viewmatrix[2]) * dL_dpoints2D[idx].y) * rz;
        dL_dpoints.y = (K.focal_x * (viewmatrix[4] - sx * viewmatrix[6]) * dL_dpoints2D[idx].x
                      + K.focal_y * (viewmatrix[5] - sy * viewmatrix[6]) * dL_dpoints2D[idx].y) * rz;
        dL_dpoints.z = (K.focal_x * (viewmatrix[8] - sx * viewmatrix[10]) * dL_dpoints2D[idx].x
                      + K.focal_y * (viewmatrix[9] - sy * viewmatrix[10]) * dL_dpoints2D[idx].y) * rz;
    } else {
        const RasterProjection projected = project_raster_camera(p_view, K, W, H);
        const glm::vec3 dL_dcam(
            projected.J[0][0] * dL_dpoints2D[idx].x + projected.J[1][0] * dL_dpoints2D[idx].y,
            projected.J[0][1] * dL_dpoints2D[idx].x + projected.J[1][1] * dL_dpoints2D[idx].y,
            projected.J[0][2] * dL_dpoints2D[idx].x + projected.J[1][2] * dL_dpoints2D[idx].y);
        dL_dpoints.x = viewmatrix[0] * dL_dcam.x + viewmatrix[1] * dL_dcam.y + viewmatrix[2] * dL_dcam.z;
        dL_dpoints.y = viewmatrix[4] * dL_dcam.x + viewmatrix[5] * dL_dcam.y + viewmatrix[6] * dL_dcam.z;
        dL_dpoints.z = viewmatrix[8] * dL_dcam.x + viewmatrix[9] * dL_dcam.y + viewmatrix[10] * dL_dcam.z;
        if (!projected.valid) {
            dL_dpoints = {0.f, 0.f, 0.f};
        }
    }

    dL_dpoints3D[idx] = dL_dpoints;
}

template <int SAMPLES_PRE_ROUND>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    sampleDepthCUDA(
        const uint32_t* __restrict__ max_contributors,
        const uint32_t* __restrict__ tile_ids,
        const uint32_t* __restrict__ tile_offsets,
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        float center_x, float center_y, const RasterIntrinsics pixel_K,
        const float2* __restrict__ points2D,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ conic_opacity,
        const float4* __restrict__ ray_planes,
        const float3* __restrict__ normals,
        const uint32_t* __restrict__ n_contrib,
        const float* __restrict__ median_depth,
        const bool* __restrict__ inside,
        const float3* __restrict__ dL_doutputs,
        float2* __restrict__ dL_dpoints2D,
        float3* __restrict__ dL_dgaussians2D,
        float4* __restrict__ dL_dconic2D,
        float4* __restrict__ dL_dray_planes,
        float3* __restrict__ dL_dnormals) {
    // We rasterize again. Compute necessary block info.
    auto block                     = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t block_id        = block.group_index().x;
    const int tile_id              = tile_ids[block_id];
    const uint32_t max_contributor = max_contributors[block_id];
    const uint32_t tile_offset     = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
    const int p_round              = block_id - tile_offset;
    const uint2 p_range            = point_ranges[tile_id];

    const uint2 range = gaussian_ranges[tile_id];

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_ray_planes[BLOCK_SIZE];

    uint32_t point_idx[SAMPLES_PRE_ROUND];
    float2 point_xy[SAMPLES_PRE_ROUND];
    float dL_dDepth[SAMPLES_PRE_ROUND];
    float2 dL_dpoint_xy[SAMPLES_PRE_ROUND];
    uint32_t last_contributor[SAMPLES_PRE_ROUND] = {0};
    float mDepth[SAMPLES_PRE_ROUND];
    bool inside_range[SAMPLES_PRE_ROUND] = {false};
    bool done[SAMPLES_PRE_ROUND]         = {false};
    int point_done                       = 0;

    int point_num_round = 0;
    for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
        int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
        if (p_range.x + progress < p_range.y) {
            int pid      = point_list[p_range.x + progress];
            point_idx[p] = pid;

            // We start from the back. The ID of the last contributing
            // Gaussian is known from each pixel from the forward.
            last_contributor[p] = n_contrib[pid];
            float3 dL_doutput   = dL_doutputs[pid];
            point_xy[p]         = points2D[pid];
            mDepth[p]           = median_depth[pid];
            inside_range[p]     = inside[pid];

            done[p] = (last_contributor[p] == 0) || !inside_range[p];
            point_done += static_cast<int>(done[p]);
            float2 pixnf      = {(point_xy[p].x - center_x) / focal_x,
                                 (point_xy[p].y - center_y) / focal_y};
            const float rln   = rnorm3df(pixnf.x, pixnf.y, 1.f);
            const float rln2  = 1.f / (pixnf.x * pixnf.x + pixnf.y * pixnf.y + 1.f);
            const float depth = mDepth[p] * rln;

            float dL_ddepth  = dL_doutput.x * pixnf.x + dL_doutput.y * pixnf.y + dL_doutput.z;
            dL_dDepth[p]     = rln * dL_ddepth;
            float aux        = (dL_doutput.x * pixnf.x + dL_doutput.y * pixnf.y + dL_doutput.z) * rln2;
            float2 dL_dpixnf = {(dL_doutput.x - aux * pixnf.x) * depth,
                                (dL_doutput.y - aux * pixnf.y) * depth};
            dL_dpoint_xy[p]  = {dL_dpixnf.x / focal_x, dL_dpixnf.y / focal_y};
            if(raster_is_fisheye(pixel_K.model)) {
                const float3 ray=pixel_unit_ray(point_xy[p],pixel_K);
                dL_dDepth[p]=ray.x*dL_doutput.x+ray.y*dL_doutput.y+ray.z*dL_doutput.z;
                const auto projected=project_fisheye_camera(ray,pixel_K);
                const glm::vec3 ju=projected.J[0], jv=projected.J[1];
                const float aa=glm::dot(ju,ju),bb=glm::dot(ju,jv),cc=glm::dot(jv,jv);
                const float det=aa*cc-bb*bb;
                const glm::vec3 grad(dL_doutput.x,dL_doutput.y,dL_doutput.z);
                dL_dpoint_xy[p]={mDepth[p]*glm::dot(grad,(cc*ju-bb*jv)/det),
                                mDepth[p]*glm::dot(grad,(aa*jv-bb*ju)/det)};
            }
            point_num_round++;
        }
    }

    bool all_done                          = point_done == point_num_round;
    float dT_dtm[SAMPLES_PRE_ROUND]        = {0.f};
    float dL_dmt_dT_dtm[SAMPLES_PRE_ROUND] = {0.f};
    int toDo                               = max_contributor;
    const int rounds                       = (max_contributor + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t contributor                   = 0;
    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (progress < max_contributor) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
        }
        block.sync();

        // Iterate over Gaussians
        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
            // refer to gsplat that uses warp-wise reduction before atomicAdd
            // Keep track of current Gaussian ID. Skip, if this one
            // is behind the last contributor for this pixel.

            contributor++;

            // Compute blending values, as before.
            const float2 xy        = collected_xy[j];
            const float4 con_o     = collected_conic_opacity[j];
            const float4 ray_plane = collected_ray_planes[j];
            for (int p = 0; p < point_num_round; p++) {
                if (done[p])
                    continue;

                float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                if (power > 0.0f) {
                    continue;
                }
                float alpha = fminf(0.99f, con_o.w * expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;
                const float t_peak  = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                const float rsigma  = ray_plane.w;
                const float t_delta = (mDepth[p] - t_peak) * rsigma;
                const float G_exp   = expf(-0.5f * t_delta * t_delta);
                const float Gt      = alpha * G_exp;
                dT_dtm[p] += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
                done[p] = contributor >= last_contributor[p];
                point_done += static_cast<int>(contributor == last_contributor[p]);
            }
            all_done = point_done == point_num_round;
        }
    }

    for (int p = 0; p < point_num_round; p++) {
        dL_dmt_dT_dtm[p] = dL_dDepth[p] / fmaxf(-dT_dtm[p], 1e-7f);
    }

    toDo        = max_contributor;
    contributor = 0;
    point_done  = 0;
    for (int p = 0; p < point_num_round; p++) {
        done[p] = (last_contributor[p] == 0) || !inside_range[p];
        point_done += static_cast<int>(done[p]);
    }
    all_done = point_done == point_num_round;

    // Traverse all Gaussians
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (progress < max_contributor) {
            int coll_id                                  = gaussian_list[range.x + progress];
            collected_id[block.thread_rank()]            = coll_id;
            collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
        }
        block.sync();

        // Iterate over Gaussians
        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
            // refer to gsplat that uses warp-wise reduction before atomicAdd
            // Keep track of current Gaussian ID. Skip, if this one
            // is behind the last contributor for this pixel.

            contributor++;

            // Compute blending values, as before.
            const float2 xy    = collected_xy[j];
            const float4 con_o = collected_conic_opacity[j];
            float G[SAMPLES_PRE_ROUND];
            bool valid_sample[SAMPLES_PRE_ROUND];
            int valid_num = 0;
#pragma unroll
            for (int p = 0; p < point_num_round; p++) {
                const float2 d   = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float power      = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                G[p]             = expf(power);
                float alpha      = con_o.w * G[p];
                bool valid       = !(done[p] || (power > 0.0f) || (alpha < 1.0f / 255.0f));
                valid_sample[p]  = valid;
                valid_num += static_cast<int>(valid);
            }
            if (!warp.any(valid_num))
                continue;

            const float4 ray_plane      = collected_ray_planes[j];
            float dL_dt_local           = 0.f;
            float2 dL_dray_planes_local = {0};
            float2 dL_dmean2D_local     = {0};
            float4 dL_dconic2D_local    = {0};
            float dL_drsigma_local      = 0.f;
#pragma unroll
            for (int p = 0; p < point_num_round; p++) {
                if (!valid_sample[p])
                    continue;
                const float2 d = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                float alpha    = fminf(0.99f, con_o.w * G[p]);

                // Propagate gradients to per-Gaussian colors and keep
                // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
                // pair).
                float t_peak       = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                const float rsigma = ray_plane.w;
                float t_delta      = (mDepth[p] - t_peak) * rsigma;
                float G_exp        = expf(-0.5f * t_delta * t_delta);
                float Gt           = alpha * G_exp;

                float dL_dGt    = dL_dmt_dT_dtm[p] * 0.25f / (1.f - Gt);
                dL_dGt          = mDepth[p] > t_peak ? dL_dGt : -dL_dGt;
                dL_dGt          = rsigma > 0 ? dL_dGt : 0.f;
                float dL_dopa   = dL_dGt * G_exp - dL_dmt_dT_dtm[p] * (t_delta > 0 ? 0.5f / (1.f - alpha) : 0.f);
                float dL_ddelta = -dL_dGt * Gt * t_delta;
                dL_drsigma_local += dL_ddelta * (mDepth[p] - t_peak);
                const float dL_dt_peak = -dL_ddelta * rsigma;

                dL_dt_local += dL_dt_peak;
                dL_dray_planes_local.x += dL_dt_peak * d.x;
                dL_dray_planes_local.y += dL_dt_peak * d.y;

                if(raster_is_fisheye(pixel_K.model) && con_o.w*G[p]>=0.99f)
                    dL_dopa=0.f;

                // Helpful reusable temporary variables
                const float dL_dG    = con_o.w * dL_dopa;
                const float gdx      = G[p] * d.x;
                const float gdy      = G[p] * d.y;
                const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
                const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

                // Update gradients w.r.t. 2D mean position of the Gaussian
                float dL_ddelx = dL_dG * dG_ddelx + dL_dt_peak * ray_plane.x;
                float dL_ddely = dL_dG * dG_ddely + dL_dt_peak * ray_plane.y;

                dL_dmean2D_local.x += dL_ddelx;
                dL_dmean2D_local.y += dL_ddely;

                dL_dpoint_xy[p].x -= dL_ddelx;
                dL_dpoint_xy[p].y -= dL_ddely;

                dL_dconic2D_local.x += -0.5f * gdx * d.x * dL_dG;
                dL_dconic2D_local.y += -0.5f * gdx * d.y * dL_dG;
                dL_dconic2D_local.z += -0.5f * gdy * d.y * dL_dG;
                dL_dconic2D_local.w += G[p] * dL_dopa;
                done[p] = contributor >= last_contributor[p];
                point_done += static_cast<int>(contributor == last_contributor[p]);
            }
            all_done = point_done == point_num_round;
            warpSum(dL_dt_local, warp);
            warpSum(dL_dray_planes_local, warp);
            warpSum(dL_dmean2D_local, warp);
            warpSum(dL_dconic2D_local, warp);
            warpSum(dL_drsigma_local, warp);
            if (warp.thread_rank() == 0) {
                const int global_id = collected_id[j];
                atomicAdd(&dL_dray_planes[global_id].x, dL_dray_planes_local.x);
                atomicAdd(&dL_dray_planes[global_id].y, dL_dray_planes_local.y);
                atomicAdd(&dL_dray_planes[global_id].z, dL_dt_local);
                atomicAdd(&dL_dray_planes[global_id].w, dL_drsigma_local);
                atomicAdd(&dL_dgaussians2D[global_id].x, dL_dmean2D_local.x);
                atomicAdd(&dL_dgaussians2D[global_id].y, dL_dmean2D_local.y);
                atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_local.x);
                atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_local.y);
                atomicAdd(&dL_dconic2D[global_id].z, dL_dconic2D_local.z);
                atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_local.w);
            }
        }
    }
    for (int p = 0; p < point_num_round; p++) {
        dL_dpoints2D[point_idx[p]] = dL_dpoint_xy[p];
    }
}

void BACKWARD::preprocess_points(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const RasterIntrinsics K,
    const uint32_t* tiles_touched,
    const float2* dL_dpoints2D,
    float3* dL_dpoints3D) {
    preprocessPointsCUDA<<<(P + 255) / 256, 256>>>(
        P,
        points3D,
        viewmatrix,
        cam_pos,
        W, H,
        K,
        tiles_touched,
        dL_dpoints2D,
        dL_dpoints3D);
}

void BACKWARD::sampleDepth(
    const int num_duplicated_tiles,
    const uint32_t* max_contributors,
    const uint32_t* tile_ids,
    const uint32_t* tile_offsets,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    float center_x, float center_y, const RasterIntrinsics pixel_K,
    const float2* points2D,
    const float2* gaussians2D,
    const float4* conic_opacity,
    const float4* ray_planes,
    const float3* normals,
    const uint32_t* n_contrib,
    const float* median_depth,
    const bool* inside,
    const float3* dL_doutputs,
    float2* dL_dpoints2D,
    float3* dL_dgaussians2D,
    float4* dL_dconic2D,
    float4* dL_dray_planes,
    float3* dL_dnormals) {
    if (num_duplicated_tiles == 0)
        return;
    sampleDepthCUDA<SAMPLE_BATCH_SIZE><<<num_duplicated_tiles, BLOCK_SIZE>>>(
        max_contributors,
        tile_ids,
        tile_offsets,
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        center_x, center_y, pixel_K,
        points2D,
        gaussians2D,
        conic_opacity,
        ray_planes,
        normals,
        n_contrib,
        median_depth,
        inside,
        dL_doutputs,
        dL_dpoints2D,
        dL_dgaussians2D,
        dL_dconic2D,
        dL_dray_planes,
        dL_dnormals);
}
