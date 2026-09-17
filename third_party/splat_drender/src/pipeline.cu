// Host pipeline: workspace sizing, launcher dispatch and the
// order-preserving double radix sort (depth pre-sort + stable tile sort).
#include "kernels.h"
#include "splat_drender/api.h"
#include "splat_drender/buffers.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace splat_drender {
namespace {

using ws::GaussianState;
using ws::GradState;
using ws::bucket_upper_bound;
using ws::InstanceState;
using ws::PixelState;
using ws::TileState;

CameraIntrinsics intrinsics_of(const CameraView& cam) {
    CameraIntrinsics k;
    k.fx = cam.fx;
    k.fy = cam.fy;
    k.cx = cam.cx;
    k.cy = cam.cy;
    k.mode = cam.mode;
    k.k1 = cam.k1;
    k.k2 = cam.k2;
    k.k3 = cam.k3;
    k.k4 = cam.k4;
    return k;
}

void check(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("splat_drender: ") + what);
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("splat_drender: ") + what +
                                 ": " + cudaGetErrorString(err));
}

int msb_bits(unsigned v) {
    int bits = 0;
    while (v) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

std::size_t sort_scratch_bytes(int n) {
    std::size_t bytes = 0;
    if (n > 0) {
        unsigned *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
        cub::DoubleBuffer<unsigned> keys(a, b);
        cub::DoubleBuffer<unsigned> vals(c, d);
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, vals, n),
                   "sort size query");
    }
    return bytes;
}

std::size_t instance_sort_bytes(int visible, int instances) {
    if (!InstanceState::single_sort(instances))
        return std::max(sort_scratch_bytes(visible), sort_scratch_bytes(instances));
    std::size_t bytes = 0;
    if (instances > 0) {
        cub::DoubleBuffer<unsigned long long> keys(nullptr, nullptr);
        cub::DoubleBuffer<unsigned> values(nullptr, nullptr);
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, values, instances),
                   "packed sort size query");
    }
    return bytes;
}

unsigned read_u32(const unsigned* ptr) {
    unsigned v = 0;
    check_cuda(cudaMemcpy(&v, ptr, sizeof(unsigned), cudaMemcpyDeviceToHost),
               "read u32");
    return v;
}

struct SortCounts {
    int visible = 0;
    int instances = 0;
    int instance_selector = 0;
};

// Per-Gaussian geometry scratch (ray plane + footprint normal, and their
// gradient counterparts) is allocated when the render produces depth or
// normals; RenderSettings::force_geometry_workspace restores the older
// "always allocate" layout for measurement.
bool gaussian_workspace_geometry(const RenderSettings& s, bool geometry) {
    return geometry || s.force_geometry_workspace;
}

// Preprocessing + FasterGS double sort shared by render and point-query
// passes. Instances are emitted in depth order so the subsequent stable
// tile sort preserves the GGGS 64-bit (tile | depth) order, including ties.
SortCounts build_draw_lists(const WorkspacePools& pools, const Gaussians& g,
                            const CameraView& cam, const RenderSettings& s,
                            int grid_x, int grid_y, int tiles, int wrap_width,
                            bool geometry, GaussianState& gst, TileState& tst,
                            InstanceState& ist, int* radii) {
    const CameraIntrinsics K = intrinsics_of(cam);
    geometry = gaussian_workspace_geometry(s, geometry);

    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
                                             (unsigned*)nullptr,
                                             (unsigned*)nullptr, g.count),
               "scan size query");
    char* gpool =
        pools.gaussian(GaussianState::bytes(g.count, scan_bytes, geometry));
    gst = GaussianState::from_pool(gpool, g.count, scan_bytes, geometry);
    if (radii == nullptr) radii = gst.radius;  // internal fallback (samples)
    check_cuda(cudaMemset(gst.n_visible, 0, 2 * sizeof(unsigned)),
               "counter memset");

    launch::preprocess_gaussians(
        g.count, g.sh_degree, g.sh_bases, g.means, g.sh, g.colors, g.opacities,
        g.scales, g.rotations, g.covariances, cam.world_to_camera, cam.center,
        K, cam.width, cam.height, s.kernel_size, s.scale_modifier, grid_x,
        grid_y, wrap_width, geometry, gst, radii);
    check_cuda(cudaGetLastError(), "preprocess_gaussians");

    SortCounts counts;
    // The counters are adjacent; one readback avoids a second host fence.
    unsigned host_counts[2];
    check_cuda(cudaMemcpy(host_counts, gst.n_visible, sizeof(host_counts),
                          cudaMemcpyDeviceToHost), "read draw counts");
    counts.visible = int(host_counts[0]);
    counts.instances = int(host_counts[1]);

    // Carve the tile pool only now: its bucket bookkeeping is sized from the
    // instance count, and the deterministic upper bound keeps forward() and
    // backward() layouts identical. The tile-range memset must stay ordered
    // with the previous frame's backward (which reads tst.range on the
    // default stream) and with extract_ranges below. A non-blocking side
    // stream let the memset race the in-flight backward and corrupt its tile
    // ranges.
    const std::size_t buckets_ub =
        bucket_upper_bound(counts.instances, tiles);
    char* tpool = pools.tile(TileState::bytes(tiles, buckets_ub));
    tst = TileState::from_pool(tpool, tiles, buckets_ub);
    check_cuda(cudaMemsetAsync(tst.range, 0, tiles * sizeof(uint2)),
               "range memset");

    const std::size_t sort_bytes =
        instance_sort_bytes(counts.visible, counts.instances);
    char* ipool = pools.instance(InstanceState::bytes(
        counts.visible, counts.instances, sort_bytes));
    ist = InstanceState::from_pool(ipool, counts.visible, counts.instances,
                                   sort_bytes);

    if (InstanceState::single_sort(counts.instances)) {
        if (counts.instances == 0) return counts;
        check_cuda(cub::DeviceScan::InclusiveSum(gst.scan_scratch, gst.scan_bytes,
                       gst.tiles_touched, gst.visible_offset, g.count), "tile scan");
        launch::emit_packed_instances(g.count, gst, grid_x, grid_y, wrap_width,
                                      ist.packed_key[0], ist.instance_value[0]);
        check_cuda(cudaGetLastError(), "emit packed instances");
        cub::DoubleBuffer<unsigned long long> keys(ist.packed_key[0], ist.packed_key[1]);
        cub::DoubleBuffer<unsigned> values(ist.instance_value[0], ist.instance_value[1]);
        check_cuda(cub::DeviceRadixSort::SortPairs(ist.sort_scratch, ist.sort_bytes,
                       keys, values, counts.instances, 0,
                       32 + msb_bits(unsigned(std::max(tiles - 1, 0)))), "packed sort");
        counts.instance_selector = values.selector;
        launch::extract_packed_ranges(counts.instances, keys.Current(), tst.range);
        check_cuda(cudaGetLastError(), "extract packed ranges");
        return counts;
    }

    if (g.count > 0) {
        check_cuda(cub::DeviceScan::InclusiveSum(
                       gst.scan_scratch, gst.scan_bytes, gst.visible_flag,
                       gst.visible_offset, g.count),
                   "visible scan");
    }

    launch::emit_depth_entries(g.count, gst.visible_offset, gst.visible_flag,
                               gst.depth_key, ist.depth_key[0],
                               ist.depth_value[0]);
    check_cuda(cudaGetLastError(), "emit_depth_entries");

    int depth_selector = 0;
    if (counts.visible > 0) {
        cub::DoubleBuffer<unsigned> keys(ist.depth_key[0], ist.depth_key[1]);
        cub::DoubleBuffer<unsigned> vals(ist.depth_value[0], ist.depth_value[1]);
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       ist.sort_scratch, ist.sort_bytes, keys, vals,
                       counts.visible),
                   "depth sort");
        depth_selector = vals.selector;
    }

    if (counts.visible > 0) {
        launch::gather_touched(counts.visible, ist.depth_value[depth_selector],
                               gst.tiles_touched, ist.compact_offset);
        check_cuda(cudaGetLastError(), "gather_touched");
        check_cuda(cub::DeviceScan::InclusiveSum(
                       gst.scan_scratch, gst.scan_bytes, ist.compact_offset,
                       ist.compact_offset, counts.visible),
                   "compact tile scan");
    }

    if (counts.visible > 0) launch::emit_instances(counts.visible, ist.depth_value[depth_selector],
                           gst.mean2d, gst.conic_opacity, ist.compact_offset,
                           grid_x, grid_y, wrap_width, ist.tile_key[0],
                           ist.instance_value[0]);
    check_cuda(cudaGetLastError(), "emit_instances");

    if (counts.instances > 0) {
        cub::DoubleBuffer<unsigned> keys(ist.tile_key[0], ist.tile_key[1]);
        cub::DoubleBuffer<unsigned> vals(ist.instance_value[0],
                                         ist.instance_value[1]);
        const int end_bit = msb_bits(unsigned(std::max(tiles - 1, 0)));
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       ist.sort_scratch, ist.sort_bytes, keys, vals,
                       counts.instances, 0, end_bit),
                   "tile sort");
        counts.instance_selector = vals.selector;
        launch::extract_ranges(counts.instances,
                               ist.tile_key[counts.instance_selector],
                               tst.range);
        check_cuda(cudaGetLastError(), "extract_ranges");
    } else {
    }
    return counts;
}

void rebuild_views(const WorkspacePools& pools, const Gaussians& g,
                   const CameraView& cam, int visible, int instances,
                   bool pixel_geometry, bool gaussian_geometry,
                   GaussianState& gst, InstanceState& ist, TileState& tst,
                   PixelState& pst) {
    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
                                             (unsigned*)nullptr,
                                             (unsigned*)nullptr, g.count),
               "scan size query");
    char* gpool = pools.gaussian(
        GaussianState::bytes(g.count, scan_bytes, gaussian_geometry));
    gst = GaussianState::from_pool(gpool, g.count, scan_bytes,
                                   gaussian_geometry);

    const std::size_t sort_bytes =
        instance_sort_bytes(visible, instances);
    char* ipool =
        pools.instance(InstanceState::bytes(visible, instances, sort_bytes));
    ist = InstanceState::from_pool(ipool, visible, instances, sort_bytes);

    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int tiles = grid_x * grid_y;
    const std::size_t buckets_ub = bucket_upper_bound(instances, tiles);
    char* tpool = pools.tile(TileState::bytes(tiles, buckets_ub));
    tst = TileState::from_pool(tpool, tiles, buckets_ub);

    const std::size_t pixels = std::size_t(cam.width) * cam.height;
    char* ppool =
        pools.pixel(PixelState::bytes(pixels, buckets_ub, pixel_geometry));
    pst = PixelState::from_pool(ppool, pixels, buckets_ub, pixel_geometry);
}

struct PointListCounts {
    int instances = 0;
    int selector = 0;
};

// Point queries (sample_depth / occupancy) never read blended color, so they
// drop the SH view: per-Gaussian preprocessing then skips the per-view SH
// evaluation and the fused backward skips sh::backward, matching the
// reference sampleSDF behavior (its preprocessing receives a dummy color and
// SHD=0).
Gaussians colorless(const Gaussians& g) {
    Gaussians c = g;
    c.sh = nullptr;
    c.colors = nullptr;
    c.sh_degree = 0;
    c.sh_bases = 0;
    return c;
}

PointListCounts build_point_lists(const WorkspacePools& pools, int point_count,
                      const float* world_points, const CameraView& cam,
                      int grid_x, int grid_y, int tiles, ws::PointState& ps, bool fixed_list) {
    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
                                             (unsigned*)nullptr,
                                             (unsigned*)nullptr, point_count),
               "point scan size query");
    const std::size_t sort_bytes = sort_scratch_bytes(point_count);
    const std::size_t cub_bytes = std::max(scan_bytes, sort_bytes);

    char* pool = pools.point(ws::PointState::bytes(point_count, tiles, cub_bytes));
    ps = ws::PointState::from_pool(pool, point_count, tiles, cub_bytes);

    launch::preprocess_points(point_count, world_points, cam.world_to_camera,
                              intrinsics_of(cam), cam.width, cam.height, ps, fixed_list);
    check_cuda(cudaGetLastError(), "preprocess_points");

    if (fixed_list) {
        check_cuda(cudaMemset(ps.point_range, 0, tiles * sizeof(uint2)), "point range memset");
        cub::DoubleBuffer<unsigned> keys(ps.point_key[0], ps.point_key[1]);
        cub::DoubleBuffer<unsigned> vals(ps.point_value[0], ps.point_value[1]);
        check_cuda(cub::DeviceRadixSort::SortPairs(ps.sort_scratch, ps.sort_bytes,
            keys, vals, point_count, 0, msb_bits(unsigned(tiles))), "fixed point sort");
        launch::extract_point_ranges(point_count, keys.Current(), tiles, ps.point_range);
        check_cuda(cudaGetLastError(), "extract fixed point ranges");
        return {-1, vals.selector};
    }

    unsigned point_instances = 0;
    if (point_count > 0) {
        check_cuda(cub::DeviceScan::InclusiveSum(ps.scan_scratch, ps.scan_bytes,
                                                 ps.touched, ps.tile_offset,
                                                 point_count),
                   "point scan");
        point_instances = read_u32(ps.tile_offset + point_count - 1);
    }

    launch::emit_point_instances(point_count, ps.point2d, ps.tile_offset,
                                 ps.touched, grid_x, grid_y, ps.point_key[0],
                                 ps.point_value[0]);
    check_cuda(cudaGetLastError(), "emit_point_instances");

    check_cuda(cudaMemset(ps.point_range, 0, tiles * sizeof(uint2)),
               "point range memset");
    PointListCounts out;
    out.instances = int(point_instances);
    if (point_instances > 0) {
        cub::DoubleBuffer<unsigned> keys(ps.point_key[0], ps.point_key[1]);
        cub::DoubleBuffer<unsigned> vals(ps.point_value[0], ps.point_value[1]);
        const int end_bit = msb_bits(unsigned(std::max(tiles - 1, 0)));
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       ps.sort_scratch, ps.sort_bytes, keys, vals,
                       int(point_instances), 0, end_bit),
                   "point sort");
        out.selector = vals.selector;
        launch::extract_ranges(int(point_instances), keys.Current(),
                               ps.point_range);
        check_cuda(cudaGetLastError(), "extract point ranges");
    }
    return out;
}

void run_gaussian_backward(const Gaussians& g, const CameraView& cam,
                           const RenderSettings& s, const GaussianState& gst,
                           const GradState& gs, const ModelGradients& grads,
                           const int* radius, bool want_sh) {
    launch::gaussian_backward(want_sh && g.sh != nullptr, g.covariances != nullptr,
                              g.count, g.sh_degree, g.sh_bases, g.means, g.sh,
                              g.opacities, g.scales, g.rotations,
                              g.covariances, cam.world_to_camera, cam.center,
                              intrinsics_of(cam), cam.width, cam.height,
                              s.kernel_size, s.scale_modifier, radius,
                              gst.clamped, gst, gs, grads.means, grads.sh,
                              grads.colors, grads.opacities, grads.scales,
                              grads.rotations, grads.covariances, grads.sh_adam);
    check_cuda(cudaGetLastError(), "gaussian_backward");
}

GradState zero_grad_state(const WorkspacePools& pools, int count,
                          bool geometry) {
    const std::size_t bytes = GradState::bytes(count, geometry);
    char* gradpool = pools.grad(bytes);
    check_cuda(cudaMemset(gradpool, 0, bytes),
               "grad scratch memset");
    return GradState::from_pool(gradpool, count, geometry);
}

}  // namespace

ForwardResult Rasterizer::forward(const WorkspacePools& pools,
                                  const Gaussians& g, const CameraView& cam,
                                  const RenderSettings& s,
                                  const RenderOutputs& out) {
    check(g.count > 0 && cam.width > 0 && cam.height > 0, "empty inputs");
    check((g.sh != nullptr) != (g.colors != nullptr),
          "exactly one color source required");
    check((g.scales != nullptr) != (g.covariances != nullptr),
          "exactly one covariance source required");
    check(out.color && out.alpha && out.visibility && out.radii,
          "forward outputs required");
    if (s.need_depth)
        check(out.median_depth && out.normal, "depth outputs required");

    const CameraIntrinsics K = intrinsics_of(cam);
    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int tiles = grid_x * grid_y;
    const int wrap_width = is_equirect(K.mode) ? cam.width : 0;

    GaussianState gst;
    TileState tst;
    InstanceState ist;
    const SortCounts counts = build_draw_lists(pools, g, cam, s, grid_x, grid_y,
                                               tiles, wrap_width, s.need_depth,
                                               gst, tst, ist, out.radii);

    // Bucket metadata for the warp-per-bucket backward. The upper bound
    // keeps the pool layout deterministic without a second host readback;
    // tail entries park at tile 0 and are rejected by the range test.
    const std::size_t buckets_ub =
        bucket_upper_bound(counts.instances, tiles);
    if (buckets_ub > 0) {
        launch::bucket_offsets(tiles, int(buckets_ub), tst.range,
                               tst.bucket_count, tst.bucket_offset,
                               tst.bucket_tile);
        check_cuda(cudaGetLastError(), "bucket_offsets");
    }

    const std::size_t pixels = std::size_t(cam.width) * cam.height;
    char* ppool =
        pools.pixel(PixelState::bytes(pixels, buckets_ub, s.need_depth));
    PixelState pst = PixelState::from_pool(ppool, pixels, buckets_ub,
                                           s.need_depth);

    launch::blend(
        s.need_depth, tst.range, ist.instance_value[counts.instance_selector],
        cam.width, cam.height, K, wrap_width, gst.mean2d, gst.conic_opacity,
        gst.rgb, g.colors, gst.ray_plane, gst.normal, gst.screen_bounds,
        pst.n_contrib, tst.max_contributor, tst.bucket_offset, tst.bucket_tile,
        pst,
        make_float3(s.background[0], s.background[1], s.background[2]),
        out.color, out.alpha, out.normal, out.median_depth, out.visibility,
        dim3(grid_x, grid_y));
    check_cuda(cudaGetLastError(), "blend");

    ForwardResult r;
    r.instance_count = counts.instances;
    r.visible_count = counts.visible;
    r.instance_selector = counts.instance_selector;
    return r;
}

void Rasterizer::backward(const WorkspacePools& pools, const Gaussians& g,
                          const CameraView& cam, const RenderSettings& s,
                          const ForwardResult& fwd,
                          const ForwardOutputsView& fo,
                          const LossGradients& dL, const ModelGradients& grads) {
    if (fwd.instance_count <= 0 || g.count <= 0) return;
    check(dL.color, "color gradient required");
    check(dL.alpha, "alpha gradient required");
    if (s.need_depth)
        check(dL.median_depth && dL.normal, "depth grads required");

    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int wrap_width = is_equirect(cam.mode) ? cam.width : 0;

    GaussianState gst;
    InstanceState ist;
    TileState tst;
    PixelState pst;
    const bool geometry_workspace =
        gaussian_workspace_geometry(s, s.need_depth);
    rebuild_views(pools, g, cam, fwd.visible_count, fwd.instance_count,
                  s.need_depth, geometry_workspace, gst, ist, tst, pst);

    GradState gs = zero_grad_state(pools, g.count, geometry_workspace);

    if (s.need_depth) {
        // Median-depth scale first: the bucket walk reads dL_dmt per pixel.
        launch::median_scale_backward(
            tst.range, ist.instance_value[fwd.instance_selector], cam.width,
            cam.height, intrinsics_of(cam), wrap_width, gst.mean2d,
            gst.conic_opacity, gst.ray_plane, gst.screen_bounds, pst.n_contrib,
            tst.max_contributor, fo.median_depth, dL.median_depth, pst.dL_dmt,
            dim3(grid_x, grid_y));
        check_cuda(cudaGetLastError(), "median_scale_backward");
    }

    launch::blend_bucket_backward(
        s.need_depth, tst.range, ist.instance_value[fwd.instance_selector],
        cam.width, cam.height, intrinsics_of(cam), wrap_width,
        make_float3(s.background[0], s.background[1], s.background[2]),
        gst.mean2d, gst.conic_opacity, gst.rgb, g.colors, gst.ray_plane,
        gst.normal, gst.screen_bounds, fo.alpha, fo.normal, fo.median_depth,
        pst.n_contrib, tst.max_contributor, tst.bucket_offset, tst.bucket_tile,
        pst, dL.color, dL.median_depth, dL.alpha, dL.normal, gs,
        reinterpret_cast<float*>(gs.d_color), grads.refine_weight,
        dL.densify_map, grads.densify_weight, grads.densify_weight_den,
        int(bucket_upper_bound(fwd.instance_count, grid_x * grid_y)));
    check_cuda(cudaGetLastError(), "blend_bucket_backward");

    run_gaussian_backward(g, cam, s, gst, gs, grads, fo.radii, g.sh != nullptr);
}

Rasterizer::SampleCounts Rasterizer::sample_depth(
    const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
    const RenderSettings& s, const float* world_points, int point_count,
    const SampleOutputs& out) {
    check(g.count > 0 && point_count > 0, "empty sample inputs");
    check(out.ray_points && out.median_depth && out.n_contrib && out.inside,
          "sample outputs required");

    check_cuda(cudaMemset(out.ray_points, 0, point_count * sizeof(float3)),
               "sample ray memset");
    check_cuda(cudaMemset(out.median_depth, 0, point_count * sizeof(float)),
               "sample depth memset");
    check_cuda(cudaMemset(out.n_contrib, 0, point_count * sizeof(unsigned)),
               "sample n_contrib memset");
    check_cuda(cudaMemset(out.inside, 0, point_count * sizeof(bool)),
               "sample inside memset");

    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int tiles = grid_x * grid_y;
    const int wrap_width = is_equirect(cam.mode) ? cam.width : 0;
    const Gaussians gq = colorless(g);

    GaussianState gst;
    TileState tst;
    InstanceState ist;
    const SortCounts counts = build_draw_lists(pools, gq, cam, s, grid_x, grid_y,
                                               tiles, wrap_width, true, gst,
                                               tst, ist, nullptr);
    ws::PointState ps;
    const PointListCounts point_counts =
        build_point_lists(pools, point_count, world_points, cam, grid_x, grid_y,
                          tiles, ps, s.device_point_lists);

    launch::evaluate_points(true, tst.range,
                            ist.instance_value[counts.instance_selector],
                            ps.point_range, ps.point_value[point_counts.selector],
                            cam.width,
                            cam.height, intrinsics_of(cam), ps.point2d,
                            ps.point_t, gst.mean2d, gst.conic_opacity,
                            gst.ray_plane, s.point_depth_bracket,
                            s.point_depth_tolerance, nullptr, out.ray_points,
                            out.median_depth, out.n_contrib, out.inside, tiles);
    check_cuda(cudaGetLastError(), "evaluate_points median");

    SampleCounts c;
    c.point_instances = point_counts.instances;
    c.gaussian_instances = counts.instances;
    c.visible_count = counts.visible;
    c.tile_blocks = tiles;
    c.instance_selector = counts.instance_selector;
    c.point_selector = point_counts.selector;
    return c;
}

void Rasterizer::sample_depth_backward(
    const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
    const RenderSettings& s, const float* world_points, int point_count,
    const SampleCounts& counts, const SampleOutputsView& fwd,
    const float3* dL_dray_points, const SampleGradients& point_grads,
    const ModelGradients& grads) {
    if (counts.gaussian_instances <= 0 || g.count <= 0) return;

    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int tiles = grid_x * grid_y;

    GaussianState gst;
    InstanceState ist;
    TileState tst;
    PixelState pst;
    rebuild_views(pools, g, cam, counts.visible_count, counts.gaussian_instances,
                  false, true, gst, ist, tst, pst);

    std::size_t scan_bytes = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
                                             (unsigned*)nullptr,
                                             (unsigned*)nullptr, point_count),
               "point scan size query");
    const std::size_t sort_bytes = sort_scratch_bytes(point_count);
    const std::size_t cub_bytes = std::max(scan_bytes, sort_bytes);
    char* ppool = pools.point(ws::PointState::bytes(point_count, tiles, cub_bytes));
    ws::PointState ps =
        ws::PointState::from_pool(ppool, point_count, tiles, cub_bytes);

    GradState gs = zero_grad_state(pools, g.count, true);
    check_cuda(cudaMemset(ps.grad_point2d, 0, point_count * sizeof(float2)),
               "point2d grad memset");
    if (point_grads.points)
        check_cuda(cudaMemset(point_grads.points, 0,
                              point_count * sizeof(float3)),
                   "point3d grad memset");

    launch::sample_depth_backward(
        tst.range, ist.instance_value[counts.instance_selector], ps.point_range,
        ps.point_value[counts.point_selector], intrinsics_of(cam), ps.point2d,
        gst.mean2d,
        gst.conic_opacity, gst.ray_plane, fwd.n_contrib, fwd.median_depth,
        fwd.inside, dL_dray_points, gs, ps.grad_point2d, tiles, cam.width,
        cam.height);
    check_cuda(cudaGetLastError(), "sample_depth_backward");

    launch::point_2d_backward(point_count, world_points, cam.world_to_camera,
                              intrinsics_of(cam), cam.width, cam.height,
                              ps.touched, ps.grad_point2d, point_grads.points);
    check_cuda(cudaGetLastError(), "point_2d_backward");

    run_gaussian_backward(colorless(g), cam, s, gst, gs, grads, gst.radius,
                          false);
}

void Rasterizer::evaluate_occupancy(
    const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
    const RenderSettings& s, const float* world_points, int point_count,
    const OccupancyOutputs& out) {
    check(g.count > 0 && point_count > 0, "empty occupancy inputs");
    check(out.occupancy && out.inside, "occupancy outputs required");

    check_cuda(cudaMemset(out.occupancy, 0, point_count * sizeof(float)),
               "occupancy memset");
    check_cuda(cudaMemset(out.inside, 0, point_count * sizeof(bool)),
               "occupancy inside memset");

    const int grid_x = (cam.width + cfg::kTileWidth - 1) / cfg::kTileWidth;
    const int grid_y = (cam.height + cfg::kTileHeight - 1) / cfg::kTileHeight;
    const int tiles = grid_x * grid_y;
    const int wrap_width = is_equirect(cam.mode) ? cam.width : 0;
    const Gaussians gq = colorless(g);

    GaussianState gst;
    TileState tst;
    InstanceState ist;
    const SortCounts counts = build_draw_lists(pools, gq, cam, s, grid_x, grid_y,
                                               tiles, wrap_width, true, gst,
                                               tst, ist, nullptr);
    ws::PointState ps;
    const PointListCounts point_counts =
        build_point_lists(pools, point_count, world_points, cam, grid_x, grid_y,
                          tiles, ps, s.device_point_lists);

    launch::evaluate_points(false, tst.range,
                            ist.instance_value[counts.instance_selector],
                            ps.point_range, ps.point_value[point_counts.selector],
                            cam.width,
                            cam.height, intrinsics_of(cam), ps.point2d,
                            ps.point_t, gst.mean2d, gst.conic_opacity,
                            gst.ray_plane, s.point_depth_bracket,
                            s.point_depth_tolerance, out.occupancy, nullptr, nullptr,
                            nullptr, out.inside, tiles);
    check_cuda(cudaGetLastError(), "evaluate_points occupancy");
}

}  // namespace splat_drender
