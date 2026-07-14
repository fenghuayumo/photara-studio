#pragma once

#include <cstdint>
#include <filesystem>

namespace aetherscan::mvs {

enum class MeshMethod : std::uint8_t {
    // Delaunay tetrahedralization + visibility-weighted s-t graph cut.
    delaunay_cut = 0,
    // Projective depth-map triangulation (fast preview fallback).
    depth_projective = 1,
    // Skip meshing.
    none = 2,
};

struct DensifyOptions {
    // Optional per-image foreground masks. Files must have the same basename
    // as source images; white is reconstructed and black is ignored.
    std::filesystem::path mask_dir;
    // Image downscale steps before densify (0 = full res, 1 ~= half, ...).
    unsigned resolution_level{1};
    // Minimum working image dimension after downscale.
    unsigned min_resolution{640};
    // Coarse-to-fine levels below working resolution (0 = single scale).
    // 1 => half then full; 2 => quarter, half, full.
    unsigned sub_resolution_levels{1};
    // PatchMatch iterations per pyramid level.
    unsigned estimation_iters{4};
    // Geometric-consistency refinement iterations (0 = off).
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
    // Enable geometric-consistency pass after photometric densify.
    bool geometric_consistency{true};
    // Build mesh after fusion.
    bool build_mesh{true};
    // The scalable projective path preserves the full depth-map resolution.
    // The in-tree Delaunay implementation remains available for small clouds.
    MeshMethod mesh_method{MeshMethod::depth_projective};
    // Skip inserting a fused point if an existing Delaunay vertex projects
    // within this many pixels in every observing view (0 = insert all).
    float mesh_dist_insert_px{2.F};
    // Cap fused points before meshing (0 = no cap); random subsample.
    std::uint64_t mesh_max_points{2'000'000};
    // Graph-cut / visibility weights (Jancosek-Pajdla style).
    float mesh_k_sigma{2.F};
    float mesh_k_qual{1.F};
    float mesh_k_inf{1.0e6F};
    // Projective meshing samples every Nth depth pixel.
    unsigned mesh_pixel_step{2};
    // Weld radius and maximum triangle edge in units of the scene's median
    // pixel footprint. These are scale invariant unlike bbox fractions.
    float mesh_weld_pixel_fraction{0.65F};
    float mesh_max_edge_voxels{5.F};
    // Reject a triangle spanning a larger relative depth discontinuity.
    float mesh_depth_diff_threshold{0.025F};
    // Drop tiny disconnected triangle islands after welding.
    unsigned mesh_min_component_faces{32};
    unsigned thread_count{0};
};

}  // namespace aetherscan::mvs
