#pragma once

#include <cstdint>
#include <filesystem>

namespace photara::mvs {

enum class MeshMethod : std::uint8_t {
    // Delaunay tetrahedralization + visibility-weighted s-t graph cut.
    delaunay_cut = 0,
    // Open3D-compatible sparse block TSDF fusion followed by Marching Cubes.
    // This is the preferred backend for GGGS median-depth geometry.
    tsdf = 2,
    // Skip meshing.
    none = 3,
};

enum class DensifyQuality : std::uint8_t {
    preview = 0,
    default_quality = 1,
    high = 2,
};

struct DensifyOptions {
    // Optional per-image foreground masks. Files must have the same basename
    // as source images; white is reconstructed and black is ignored.
    std::filesystem::path mask_dir;
    // Erode foreground masks by this many working-resolution pixels. A small
    // guard band prevents uncertain segmentation/sampling at silhouettes from
    // turning into long grazing-angle depth sheets.
    unsigned mask_border_px{1};
    // Image downscale steps before densify (0 = full res, 1 ~= half, ...).
    unsigned resolution_level{1};
    // Minimum working image dimension after downscale.
    unsigned min_resolution{640};
    // Coarse-to-fine levels below working resolution (0 = single scale).
    // 1 => half then full; 2 => quarter, half, full.
    unsigned sub_resolution_levels{1};
    // PatchMatch iterations per pyramid level.
    unsigned estimation_iters{4};
    // Global geometric-consistency rounds (0 = off). Neighbor depth maps are
    // snapshotted again after every round, so information propagates between
    // views instead of repeatedly optimizing against a stale first estimate.
    unsigned geometric_iters{2};
    // Weight of geometric consistency term in score.
    float geometric_weight{0.1F};
    // Random refine trials per pixel per iteration.
    unsigned random_iters{6};
    // Max neighbor source views per reference.
    unsigned max_neighbors{12};
    // Minimum number of source images that must support a PatchMatch
    // hypothesis. The value is clamped to the number of available sources.
    unsigned min_patch_views{2};
    // Preferred triangulation angle (degrees).
    float optim_angle_deg{12.F};
    // Keep pixels with photometric cost (1-NCC) below this.
    // Maximum aggregated 1-ZNCC cost. 0.9 accepted essentially unrelated
    // patches and was the main source of floating geometry in the old path.
    float ncc_keep_threshold{0.45F};
    // Minimum shared sparse points for a neighbor candidate.
    unsigned min_shared_points{3};
    // Fusion: minimum agreeing views.
    unsigned min_views_fuse{2};
    // Remove depth-connected components smaller than this many pixels before
    // fusion. Connectivity also requires locally consistent depth.
    unsigned speckle_size{40};
    // Relative depth agreement at fusion.
    float depth_diff_threshold{0.01F};
    // Max forward-backward reprojection error in pixels at fusion.
    float reprojection_error_px{2.F};
    // Normal agreement at fusion (degrees).
    float normal_diff_threshold_deg{25.F};
    // Soft weight retained for a sample viewed at 90 degrees. Incidence is a
    // confidence term, not a hard cut: hard rejection caused missing circular
    // silhouettes even when another view could stabilize the same surface.
    float grazing_weight_floor{0.12F};
    // Filter/adjust each final depth map against reprojected neighbor maps.
    // This removes free-space conflicts and averages mutually consistent
    // estimates before fusion (OpenMVS AdjustConfidence-style filtering).
    bool filter_depth_maps{true};
    // Number of agreeing neighbor maps required by the final depth filter.
    // The reference map itself is not included in this count.
    unsigned min_views_filter{1};
    // Average consistent reprojected estimates instead of filter-only mode.
    bool adjust_filtered_depth{true};
    // Enable geometric-consistency pass after photometric densify.
    bool geometric_consistency{true};
    // Build mesh after fusion.
    bool build_mesh{true};
    // MVS uses the CGAL global visibility graph-cut backend. GGGS median-depth
    // geometry uses TSDF.
    MeshMethod mesh_method{MeshMethod::delaunay_cut};
    // Run topology cleanup after either meshing backend: remove duplicate,
    // degenerate and non-manifold faces, orient connected components, remove
    // small islands and close small boundary loops.
    bool mesh_clean{true};
    // Close boundary loops with at most this many edges (0 = disabled).
    unsigned mesh_close_hole_edges{16};
    // Optional boundary-preserving, shrink-resistant Taubin smoothing after
    // global Delaunay cleanup.
    unsigned mesh_smooth_iters{0};
    float mesh_smooth_lambda{0.15F};
    // Skip inserting a fused point if an existing Delaunay vertex projects
    // within this many pixels in every observing view (0 = insert all).
    float mesh_dist_insert_px{0.75F};
    // Cap fused points before meshing (0 = no cap); random subsample.
    std::uint64_t mesh_max_points{2'000'000};
    // Graph-cut / visibility weights (Jancosek-Pajdla style).
    float mesh_k_sigma{2.F};
    // Scale the visibility uncertainty by the median length of edges incident
    // to each sample. This follows OpenMVS's density-adaptive reconstruction
    // model while clamping pathological neighborhoods to the global scale.
    bool mesh_adaptive_sigma{true};
    float mesh_k_qual{1.F};
    // Visibility ray continuation behind a sample, in sigma units. One sigma
    // is the base surface-thickness model used by the graph-cut energy.
    float mesh_k_behind{1.F};
    // OpenMVS/Jancosek-Pajdla weak-surface reinforcement. A second local
    // traversal compares free-space support before (beta) and behind (gamma)
    // every observed sample. A strong beta/gamma discontinuity multiplies
    // the endpoint cell's sink t-edge, preserving weak but coherent sheets.
    bool mesh_use_free_space_support{true};
    float mesh_k_free_space_front{3.F};
    float mesh_k_free_space_back{4.F};
    float mesh_k_free_space_rel{0.1F};
    float mesh_k_free_space_abs{1000.F};
    float mesh_k_free_space_outlier{400.F};
    // Photara's fused visibility weights do not share OpenMVS's absolute
    // confidence scale. Map this quantile of ratio-qualified beta-gamma
    // samples onto mesh_k_free_space_abs before applying the OpenMVS test.
    // Set to zero to use the literal OpenMVS absolute scale.
    float mesh_k_free_space_calibration_quantile{0.95F};
    float mesh_k_inf{1.0e6F};
    // TSDF voxel size in world units (0 = infer from median pixel footprint).
    // GGGS overrides the automatic value with max_depth / 2048 to match
    // pygsplat's gs2mesh.py.
    float mesh_tsdf_voxel_size{0.F};
    // Padding multiplier for point-cloud-derived TSDF bounds. Foreground
    // clouds can miss weakly textured outer surfaces, so keep this
    // conservative while making it explicitly tunable.
    float mesh_tsdf_bounds_padding{2.F};
    // Multiplier applied only to the inferred voxel size. Values above one
    // extract a coarser, locally regular mesh directly from the TSDF instead
    // of relying on a topology-damaging post-decimation pass.
    float mesh_tsdf_voxel_scale{1.F};
    // Truncation half-width in voxels.
    float mesh_tsdf_truncation_voxels{4.F};
    // Pixel stride used only while allocating sparse TSDF blocks. Integration
    // still projects every voxel into the full-resolution depth map. Open3D's
    // default is four; use one for sub-pixel wires and other thin structures.
    unsigned mesh_tsdf_pixel_step{4};
    // Ignore field samples whose accumulated integration weight is lower.
    float mesh_tsdf_min_weight{0.25F};
    // Before Marching Cubes, fill a zero-weight voxel only when this many
    // coordinate axes have valid samples on both sides. Two closes isolated
    // one-voxel support gaps without growing open silhouettes; zero disables.
    unsigned mesh_tsdf_support_closing_axes{2};
    // Optional lossless export of the exact uint16-millimetre depth frames,
    // intrinsics, and world-to-camera poses consumed by TSDF. Intended for
    // backend A/B diagnostics; empty disables the roughly 2 bytes/pixel dump.
    std::filesystem::path mesh_tsdf_frame_export_dir;
    // Optional output directory for source-aligned multi-view depth
    // consistency and TSDF observation-weight heatmaps.
    std::filesystem::path mesh_tsdf_diagnostics_dir;
    // TSDF can contain tiny closed bubbles where depth maps disagree. Remove
    // components smaller than this fraction of the largest component.
    float mesh_tsdf_min_component_fraction{0.0005F};
    // Boundary-locked Taubin smoothing after Marching Cubes. The lambda/mu
    // pair removes sub-voxel stair-stepping without the shrinkage of repeated
    // positive-only Laplacian smoothing. Set iterations to zero for exact
    // gs2mesh/Open3D output.
    unsigned mesh_tsdf_smooth_iters{2};
    float mesh_tsdf_smooth_lambda{0.5F};
    float mesh_tsdf_smooth_mu{-0.53F};
    // Weld radius in units of the scene's median pixel footprint.
    float mesh_weld_pixel_fraction{0.65F};
    // Drop graph-cut facets whose longest edge exceeds this multiple of the
    // median longest edge over all cut facets. Delaunay vertices are observed
    // samples, so these outliers are unsupported webbing across occluded gaps.
    // The relative statistic is scene-scale invariant; zero disables.
    float mesh_max_edge_scale{4.F};
    // Reject a triangle spanning a larger relative depth discontinuity.
    float mesh_depth_diff_threshold{0.025F};
    // Drop tiny disconnected triangle islands after welding.
    unsigned mesh_min_component_faces{32};
    // OpenMVS-style scale-aware cleanup. Faces with an edge longer than the
    // 95th-percentile edge times this factor and components whose AABB is
    // smaller than the 55th-percentile edge times this factor are removed.
    // Set to 0 to disable the scale-aware pass.
    float mesh_spurious_factor{20.F};
    // Iteratively remove vertices incident to at most one face.
    bool mesh_remove_spikes{true};
    // Number of image rows in a PatchMatch scheduling tile. Propagation uses
    // red/black phases, so every tile in a phase can execute independently.
    unsigned patchmatch_tile_rows{8};
    // Reference views concurrently sharing the CPU budget. Each view still
    // runs row tiles; the final partial batch receives more threads per view.
    unsigned patchmatch_concurrent_views{8};
    unsigned thread_count{0};
};

// Product presets intentionally tune the whole pipeline rather than only the
// image resolution. Callers may override individual fields afterwards.
inline void apply_quality_preset(
    DensifyOptions& options, const DensifyQuality quality) {
    options = DensifyOptions{};
    switch (quality) {
    case DensifyQuality::preview:
        options.mesh_method = MeshMethod::delaunay_cut;
        options.resolution_level = 2;
        options.mask_border_px = 0;
        options.sub_resolution_levels = 1;
        options.estimation_iters = 3;
        options.geometric_iters = 1;
        options.random_iters = 4;
        options.max_neighbors = 8;
        options.min_patch_views = 2;
        options.ncc_keep_threshold = 0.50F;
        options.min_views_fuse = 2;
        options.min_views_filter = 1;
        options.grazing_weight_floor = 0.20F;
        options.speckle_size = 24;
        options.mesh_min_component_faces = 24;
        options.mesh_close_hole_edges = 8;
        break;
    case DensifyQuality::default_quality:
        options.mesh_method = MeshMethod::delaunay_cut;
        options.resolution_level = 1;
        options.mask_border_px = 1;
        options.sub_resolution_levels = 1;
        options.estimation_iters = 4;
        options.geometric_iters = 2;
        options.random_iters = 6;
        options.max_neighbors = 12;
        options.min_patch_views = 2;
        options.ncc_keep_threshold = 0.45F;
        options.min_views_fuse = 3;
        options.min_views_filter = 1;
        options.speckle_size = 40;
        options.mesh_min_component_faces = 32;
        options.mesh_close_hole_edges = 16;
        options.mesh_dist_insert_px = 0.75F;
        break;
    case DensifyQuality::high:
        options.mesh_method = MeshMethod::delaunay_cut;
        options.resolution_level = 0;
        options.mask_border_px = 1;
        options.sub_resolution_levels = 1;
        options.estimation_iters = 5;
        options.geometric_iters = 3;
        options.random_iters = 8;
        options.max_neighbors = 16;
        options.min_patch_views = 3;
        options.ncc_keep_threshold = 0.40F;
        options.min_views_fuse = 3;
        options.min_views_filter = 2;
        options.speckle_size = 80;
        options.depth_diff_threshold = 0.008F;
        options.reprojection_error_px = 1.5F;
        options.normal_diff_threshold_deg = 20.F;
        options.grazing_weight_floor = 0.08F;
        // Full-resolution depth is retained for fusion; sampling every other
        // pixel keeps the default high-quality mesh at a product-manageable
        // size. API callers can still set this to 1 for an ultra-dense mesh.
        options.mesh_weld_pixel_fraction = 0.55F;
        options.mesh_depth_diff_threshold = 0.018F;
        options.mesh_min_component_faces = 64;
        options.mesh_close_hole_edges = 24;
        options.mesh_dist_insert_px = 0.75F;
        break;
    }
}

}  // namespace photara::mvs
