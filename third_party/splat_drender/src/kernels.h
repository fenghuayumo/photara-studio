// Internal host launchers. Each .cu file owns its kernels; the pipeline
// only sees these functions. Runtime flags dispatch template variants.
#pragma once

#include "splat_drender/api.h"
#include "splat_drender/buffers.h"
#include "splat_drender/camera.h"
#include "splat_drender/config.h"

namespace splat_drender::launch {

// `geometry` selects the ray-plane/footprint-normal variant of the same
// launch; the false variant leaves st.ray_plane/st.normal untouched (they are
// not allocated by GaussianState::from_pool in that mode).
void preprocess_gaussians(int count, int sh_degree, int sh_bases,
                          const float* means, const float* sh,
                          const float* colors, const float* opacities,
                          const float* scales, const float* rotations,
                          const float* cov6, const float* view,
                          const float* camera_center, CameraIntrinsics K,
                          int width, int height, float kernel_size,
                          float scale_modifier, int grid_x, int grid_y,
                          int wrap_width, bool geometry, ws::GaussianState st,
                          int* radii);

void emit_depth_entries(int count, const unsigned* visible_offset,
                        const unsigned* visible_flag, const unsigned* depth_key,
                        unsigned* key_out, unsigned* value_out);

void gather_touched(int visible_count, const unsigned* sorted_ids,
                    const unsigned* tiles_touched, unsigned* compact_n);

void emit_instances(int visible_count, const unsigned* depth_sorted_ids,
                    const float2* mean2d, const float4* conic_opacity,
                    const unsigned* compact_offset, int grid_x, int grid_y,
                    int wrap_width, unsigned* tile_key,
                    unsigned* instance_value);

void emit_packed_instances(int count, ws::GaussianState st, int grid_x, int grid_y,
                           int wrap_width, unsigned long long* keys, unsigned* values);
void extract_packed_ranges(int count, const unsigned long long* keys, uint2* range);

void extract_ranges(int instance_count, const unsigned* tile_key,
                    uint2* range);

// Fills per-tile bucket counts, their inclusive-scan offsets and the
// bucket->tile mapping (entries beyond the exact bucket count map to tile 0
// so an upper-bound backward grid can safely early-return).
void bucket_offsets(int tiles, int buckets, const uint2* range,
                    unsigned* bucket_count, unsigned* bucket_offset,
                    unsigned* bucket_tile);

void blend(bool need_depth, const uint2* tile_range,
           const unsigned* instance_value, int width, int height,
           CameraIntrinsics K, int wrap_width, const float2* mean2d,
           const float4* conic_opacity, const float3* rgb, const float* colors,
           const float4* ray_plane, const float3* normal,
           const ushort4* screen_bounds, unsigned* n_contrib,
           unsigned* max_contributor, const unsigned* bucket_offset,
           unsigned* bucket_tile, ws::PixelState pst, float3 background,
           float* out_color, float* out_alpha, float* out_normal,
           float* out_median_depth, float* visibility, dim3 grid);

// Geometry pre-pass: computes dL_dmedian * ray_z / max(-dT_dtm, 1e-7) per
// pixel. Warps split the tile's instance list so the formerly sequential
// contributor walk runs eight-way parallel per pixel.
void median_scale_backward(const uint2* tile_range,
                           const unsigned* instance_value, int width,
                           int height, CameraIntrinsics K, int wrap_width,
                           const float2* mean2d,
                           const float4* conic_opacity,
                           const float4* ray_plane,
                           const ushort4* screen_bounds,
                           const unsigned* n_contrib,
                           const unsigned* max_contributor,
                           const float* median_depth,
                           const float* dL_dmedian, float* dL_dmt,
                           dim3 grid);

// FasterGS-style bucket-parallel blending backward: one warp owns 32
// instances of one tile bucket; each lane walks all 256 tile pixels while
// the per-pixel (T, color-after[, normal-after]) state flows diagonally
// through warp shuffles. Gradients accumulate in registers and commit once.
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
                           float* densify_weight_den, int buckets);

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
                       float* grad_cov, SHAdam sh_adam = {});

void preprocess_points(int count, const float* points, const float* view,
                       CameraIntrinsics K, int width, int height,
                       ws::PointState ps, bool fixed_list);

void extract_point_ranges(int count, const unsigned* keys, int tiles, uint2* ranges);

void emit_point_instances(int count, const float2* point2d,
                          const unsigned* tile_offset, const unsigned* touched,
                          int grid_x, int grid_y, unsigned* key_out,
                          unsigned* value_out);

void evaluate_points(bool median_mode, const uint2* tile_range,
                     const unsigned* gauss_value, const uint2* point_range,
                     const unsigned* point_value, int width, int height,
                     CameraIntrinsics K, const float2* point2d,
                     const float* point_t, const float2* mean2d,
                     const float4* conic_opacity, const float4* ray_plane,
                     float depth_bracket, float depth_tolerance,
                     float* out_occupancy, float3* out_ray_point,
                     float* out_median_depth, unsigned* out_n_contrib,
                     bool* inside, int tiles);

void sample_depth_backward(const uint2* tile_range,
                           const unsigned* gauss_value,
                           const uint2* point_range,
                           const unsigned* point_value, CameraIntrinsics K,
                           const float2* point2d, const float2* mean2d,
                           const float4* conic_opacity,
                           const float4* ray_plane, const unsigned* n_contrib,
                          const float* median_depth, const bool* inside,
                          const float3* dL_dray_points, ws::GradState gs,
                          float2* dL_dpoint2d, int tiles, int width, int height);

void point_2d_backward(int count, const float* points, const float* view,
                       CameraIntrinsics K, int width, int height,
                       const unsigned* touched, const float2* dL_dpoint2d,
                       float3* dL_dpoint3d);

}  // namespace splat_drender::launch
