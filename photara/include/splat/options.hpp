#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace photara::splat {

enum class AlphaMode {
    masked,
    transparent,
};

enum class DensificationStrategy {
    adc_igs,
    adc_plus,
    dense_adaptive,
    // Error-map MCMC: the contribution-weighted image error is the
    // primary densification score, dead rows are recycled in place through a
    // long-axis split, and growth is a fixed per-refine multiplier of the
    // live count rather than a gradient-qualified fraction.
    emc,
};

// PPISP (per-pixel image signal processing) parameter layout. The default
// keeps exposure and a colour-homography white-balance; vignetting and the
// CRF are opt-in for captures that need them.
enum class PpispParamType {
    no_crf_no_vig,
    no_crf,
    original,
};

[[nodiscard]] constexpr int ppisp_parameter_count(PpispParamType type) {
    switch (type) {
    case PpispParamType::no_crf_no_vig:
        return 9;
    case PpispParamType::no_crf:
        return 24;
    case PpispParamType::original:
        return 36;
    }
    return 9;
}

[[nodiscard]] constexpr bool is_adc_strategy(DensificationStrategy strategy) {
    return strategy == DensificationStrategy::adc_plus ||
           strategy == DensificationStrategy::adc_igs;
}

struct TrainingOptions {
    unsigned iterations{10'000};
    unsigned sh_degree{3};
    unsigned sh_degree_interval{1'000};
    unsigned seed{42};
    // Console training stats every N steps. Zero logs only the first and last.
    unsigned log_interval{100};
    // Optional windowed CUDA-event timings for the GGGS training loop.
    // Disabled by default so production training does not create or record
    // profiling events. The interval is capped by the trainer to avoid an
    // accidentally unbounded CUDA event pool.
    bool profile_cuda{false};
    bool fuse_sh_adam{true};
    unsigned cuda_profile_interval{100};
    // Sparse COLMAP initialization enables dynamic Gaussian management. Dense
    // MVS initialization only enables it for the explicit dense_adaptive mode.
    bool input_is_dense{true};
    bool enable_densification{true};
    DensificationStrategy densification_strategy{
        DensificationStrategy::adc_igs};
    // Growth ceiling while densify is enabled. Initialization uses the full
    // source cloud unless initial_point_budget asks for a subset.
    std::size_t densification_cap{1'000'000};
    // Upper bound on the Gaussians built from the input point cloud. Sparse
    // SfM clouds regularly outnumber the growth ceiling, and an initialization
    // already sitting on it leaves IGS no room to refine anything: every later
    // step just swaps rows. 0 keeps every input point.
    std::size_t initial_point_budget{0};
    unsigned refine_start_iter{0};  // 0 selects the strategy preset
    unsigned refine_stop_iter{0};   // 0 selects the strategy preset
    // IGS stop is max(refine_stop_iter, iterations - refine_stop_num_iter).
    unsigned refine_stop_num_iter{2'500};
    // Iteration after which densification stops adding parents and only
    // prunes, replaces recovered slots, splits oversized splats and decays.
    // 0 selects the Brush proportion (half of `iterations`, resolved by
    // apply_strategy_defaults); an explicit value always wins.
    unsigned grow_stop_iter{0};
    unsigned refine_every{0};       // 0 selects the strategy preset
    unsigned opacity_reset_every{3'000};
    // Shared split/growth controls. Strategies may differ in how they build
    // the candidate set, but should not expose duplicate threshold knobs.
    float densify_gradient_threshold{0.0025F};
    float densify_select_fraction{0.25F};
    // Broad semi-transparent splats that own the rendered median depth wrap
    // meshes in a floater shell. Start size repair past a quarter frame.
    float densify_screen_threshold{0.25F};
    // Per-view exponent after image-to-splat reduction, before window
    // averaging. EMC is the only strategy that scores by image error.
    float densify_score_power{0.4F};
    // Per-pixel error-map exponent before raster scatter ( default 4).
    float densify_loss_map_power{4.F};
    // Extra Gaussians per refine as a multiplier of the live count. Values
    // <= 1 fall back to densify_select_fraction of above-threshold rows.
    float densify_growth_factor{0.F};
    // Strategy-neutral per-splat shape regularizers, added to the
    // scale/quaternion gradients when their weights are non-zero. The
    // scale weight follows the front-loaded (p+1)(1-t)^p schedule so its
    // integral over the run is fixed.
    float shape_scale_reg{0.F};
    float shape_scale_reg_decay_power{0.4F};
    float shape_erank_reg{0.F};
    float shape_erank_s3_reg{0.F};
    float shape_quat_norm_reg{0.F};
    // Strategy-neutral on-screen size control: every step spends a share
    // of one scale learning-rate step per octave over the limit, and each
    // refinement hard-clips the remainder (bounded by the hardness
    // factor).
    float oversize_screen_limit{0.F};
    float oversize_penalty{0.F};
    float oversize_clip_hardness{1.5F};
    // Share of the growth budget spent on oversized (screen-cap) parents.
    // 0 keeps the default "split every oversized row that fits" path.
    float densify_oversize_split_fraction{0.F};
    float densify_oversize_score_blend{1.F};
    bool densify_clip_screen_size{false};
    float densify_screen_clip_hardness{1.5F};
    bool densify_revised_noise{false};
    // Sample replacement parents by densify score rather than opacity.
    bool densify_relocate{false};
    bool densify_keep_parent_adam{false};
    // Dense MVS points already cover the surface. Recycle only a small part of
    // the budget per refinement and grow more conservatively than sparse ADC.
    float dense_recycle_fraction{0.01F};
    float dense_growth_fraction{0.005F};
    float prune_opacity{1.F / 255.F};
    float opacity_decay{0.004F};
    // Optional per-refine scale shrink, tapered by remaining training
    // progress. Disabled by default: the 2026-09-18 six-seed paired A/B
    // found no detectable quality effect on either ADC strategy
    // (artifacts/scale_decay_ab_20260918), and Brush does not decay scales.
    float scale_decay{0.F};
    float mean_noise_weight{50.F};
    // Uniform noise around a black background, clamped to [0,1].
    // Dense MVS input trains with a fixed background; both ADC strategies
    // run with Brush's 0.1 through the reconstruction CLI.
    float background_noise_strength{0.F};
    float initial_opacity{0.1F};
    float initial_scale{1.F};
    // Initialize isotropic scales from the RMS distance to the three nearest
    // neighbours in the selected input cloud.
    bool initialize_scale_from_knn{false};
    // Freeze dense MVS structure after its warm-up and keep optimizing the
    // progressively enabled SH bands. dense_adaptive may still recycle/split
    // rows after this point, without applying further structure Adam steps.
    unsigned dense_structure_freeze_iter{1'000};  // 0 disables the freeze
    // Optional all-input diagnostic: stop means/scale/quaternion/opacity Adam
    // after this iteration while continuing SH optimization.
    unsigned structure_freeze_iter{0};  // 0 disables the freeze
    float means_lr{1.6e-4F};
    float scales_lr{5e-3F};
    float opacities_lr{5e-2F};
    float quaternions_lr{1e-3F};
    float sh0_lr{2.5e-3F};
    float sh_rest_lr{1.25e-4F};
    // Mean-square prior on non-DC SH coefficients; zero disables.
    float sh_regularization_weight{0.F};
    // Optional per-Gaussian priors; independent of colour correction.
    float opacity_regularization_weight{0.F};
    float log_scale_regularization_weight{0.F};
    float beta1{0.9F};
    float beta2{0.999F};
    // The raster gradients are averaged over every image pixel; a value as
    // large as 1e-8 would suppress useful geometry updates.
    float adam_epsilon{1e-15F};
    float photometric_weight{1.F};
    float ssim_weight{0.2F};
    // Training-time colour correction for auto-exposure / auto-white-balance
    // drift in video captures. Both transforms are applied to the rendered
    // image the photometric loss sees; evaluation, preview and exported models
    // keep canonical appearance.
    bool use_bilateral_grid{false};
    // One grid shared by every view (the default) or one grid per view. A
    // per-view grid has more free parameters than the view has pixels and
    // absorbs that frame's appearance, which costs held-out quality even with
    // the mean projection below; the shared form can only learn variation that
    // is consistent across views (lens shading, vignetting, sensor response).
    bool bilateral_grid_shared{true};
    unsigned bilateral_grid_width{16};
    unsigned bilateral_grid_height{16};
    unsigned bilateral_grid_luma{8};
    // The grid is a low-resolution LUT with Adam; a large rate diverges and
    // produces non-finite renders, so the default is deliberately gentle.
    float bilateral_grid_lr{2e-4F};
    float bilateral_grid_tv_weight{10.F};
    // Maximum distance of any grid coefficient from its identity value. Zero
    // disables the bound (not recommended).
    float bilateral_grid_deviation_limit{0.5F};
    // Keep every view's grid mean on the identity affine after each update.
    // The matrix stays in the option struct (rather than being unconditional)
    // so the projection can be ablated; disabling it is not recommended, since
    // an unconstrained mean absorbs the global colour mapping.
    bool bilateral_grid_identity_projection{true};
    bool use_ppisp{false};
    PpispParamType ppisp_type{PpispParamType::no_crf_no_vig};
    float ppisp_lr{2e-3F};
    float ppisp_reg_exposure_mean{1.F};
    float ppisp_reg_color_mean{1.F};
    float ppisp_reg_vig_center{0.02F};
    float ppisp_reg_vig_non_pos{0.01F};
    float ppisp_reg_vig_channel_var{0.1F};
    float ppisp_reg_crf_channel_var{0.1F};
    bool ppisp_clamp_output{false};
    // When both PPISP and the bilateral grid are enabled, PPISP runs first
    // (exposure and white balance, then the spatially varying affine).
    bool ppisp_before_bilagrid{true};
    float depth_weight{0.05F};
    float normal_weight{0.01F};
    // GGGS meshing loss from the Python reference: cosine consistency between
    // the rasterized Gaussian normal and the normal differentiated from the
    // rendered median-depth map.
    bool use_depth_normal_loss{false};
    float depth_normal_weight{0.05F};
    unsigned depth_normal_from_iter{7'000};
    // GaussianWrapping normal field. Four features per Gaussian encode
    // normalize(xyz) * tanh(w), and are aligned to normals differentiated
    // from the rendered median-depth map.
    bool use_normal_field{false};
    float normal_field_weight{0.05F};
    float normal_field_depth_ratio{0.6F};
    unsigned normal_field_from_iter{8'001};
    float normal_features_lr{0.025F};
    // Depth-normal geometry training implicitly enables the reference
    // GGGS/Mip-Splatting low-pass filter. It is deliberately not an
    // independent switch: appearance-only 3DGS must keep canonical Gaussian
    // scale/opacity untouched, while the 3DGS -> mesh path needs stable
    // sub-pixel coverage.
    unsigned filter_3d_update_interval{100};
    // GGGS multi-view PatchMatch supervision. Geometry uses a differentiable
    // depth round trip through a nearby camera; NCC uses the reference
    // plane-induced homography and the original image pair.
    float multi_view_geo_weight{0.F};
    float multi_view_ncc_weight{0.F};
    // Multi-view point-query median-depth search. > 0 overrides the
    // scene-derived default, 0 derives it from the scene extent, < 0 keeps the
    // wide default search (+/-200 window, eight refinements).
    float multi_view_depth_bracket{0.F};
    float multi_view_depth_tolerance{0.F};
    unsigned multi_view_num{8};
    // Subsample the expensive multi-view objective after ADC growth stops.
    // Active-step weights are multiplied by this interval so the stochastic
    // objective remains unchanged in expectation.
    unsigned multi_view_tail_interval{1};
    // Optional quality-first replacement for the fixed grow-stop gate.  The
    // trainer observes topology churn, depth round-trip consistency, and the
    // opacity/scale distribution at refinement boundaries.  It only raises
    // the interval after several stable windows and immediately restores
    // every-step supervision when any metric drifts.
    bool multi_view_adaptive_frequency{false};
    unsigned multi_view_adaptive_max_interval{2};
    unsigned multi_view_adaptive_stable_refinements{5};
    float multi_view_adaptive_count_threshold{0.005F};
    float multi_view_adaptive_churn_threshold{0.01F};
    float multi_view_adaptive_depth_threshold{0.02F};
    float multi_view_adaptive_min_depth_consistency{0.5F};
    float multi_view_adaptive_distribution_threshold{0.025F};
    float multi_view_max_angle{30.F};
    float multi_view_min_distance{0.01F};
    float multi_view_max_distance{1.5F};
    float multi_view_pixel_noise_threshold{1.F};
    bool multi_view_robust_ncc{true};
    float multi_view_ncc_lambda_reference{0.45F};
    float multi_view_ncc_sharpness{12.F};
    float multi_view_ncc_min_weight{0.F};
    // Mask training. In masked mode RGB is supervised
    // only in the foreground and background alpha leakage is penalized. In
    // transparent mode the same foreground RGB loss is combined with full-image
    // BCE(predicted alpha, mask) * match_alpha_weight.
    bool use_mask{false};
    AlphaMode alpha_mode{AlphaMode::transparent};
    float match_alpha_weight{0.25F};
    // Weight of the masked-mode "no opacity outside the foreground" penalty.
    // A dynamic occluder hides static geometry that other views still need
    // opaque, so the penalty fights the multi-view-consistent surface behind
    // it and grows semi-transparent bubbles where the subject stands. Setting
    // 0 keeps the static-region RGB supervision and drops the penalty, which
    // leaves those rays to the other views instead of forcing them empty.
    float mask_alpha_leak_weight{1.F};
    std::filesystem::path mask_dir;
    float minimum_scale_fraction{1e-4F};
    float maximum_scale_fraction{0.002F};
    // Hard anisotropy clamp; 0 disables it. Flat "pancake" splats are the
    // correct model for thin surfaces, so the best-quality default does not
    // clamp them.
    float max_scale_ratio{0.F};
    bool constrain_scale_range{true};
    float geometry_epsilon{1e-3F};
    // Classic rasterization. A positive kernel enables antialiasing and must
    // be selected explicitly.
    float kernel_size{0.F};
    float scale_modifier{1.F};
    // Photometric training baseline. Geometry supervision is opt-in and can
    // be added after fixed-model L1+SSIM convergence is verified.
    bool use_mvs_depth{false};
    bool use_mvs_normals{false};
    // Train against source-resolution undistorted images rather than the MVS
    // working resolution.
    bool use_source_resolution{false};
    // In unmasked images, undistortion outside the source is missing data,
    // not black/transparent geometry. Ignore those rays in RGB supervision.
    bool ignore_undistortion_border{false};
    // When true, resample OpenCV fisheye views onto a pinhole working camera.
    // Native fisheye/equirectangular rasterization is used otherwise. Equirect
    // cannot be undistorted and always trains natively.
    bool undistort_to_pinhole{false};
    // brush-style dataset controls. Images are resized so their largest
    // dimension does not exceed this value (0 keeps the source resolution),
    // and packed training views are retained in bounded host/device LRUs.
    // Only current reference/neighbour views are expanded to float on CUDA.
    unsigned max_image_dimension{1'920};
    // Coarse-to-fine image schedule. The active linear resolution starts at
    // 1/4, doubles every 3k iterations, and caps at max_image_dimension.
    // Advancing a level clears the previous decoded RGB/mask cache.
    // Library callers opt in explicitly; the reconstruction CLI enables this
    // by default while unit-test configurations remain fixed-size.
    bool progressive_resolution{false};
    unsigned progressive_resolution_interval{3'000};
    float progressive_initial_scale{0.25F};
    std::size_t training_view_cache_bytes{
        std::size_t{6} * 1024 * 1024 * 1024};
    // Expand the host budget when possible. The device budget is a hard upper
    // bound even in adaptive mode. Zero disables the host-side decoded-image
    // cache only;
    // bounded device prefetch remains available independently.
    bool adaptive_training_cache{true};
    // Packed RGBA8 plus optional depth/normal CUDA cache. Zero disables it.
    // In adaptive mode this is the floor of the budget. Fixed mode retains the
    // earlier free-VRAM/8 guard and never exceeds the configured value.
    std::size_t training_device_cache_bytes{std::size_t{512} * 1024 * 1024};
    // Ceiling the adaptive budget may grow to, using the observed device-cache
    // hit rate as feedback and the idle VRAM as a guard. Zero keeps the budget
    // at training_device_cache_bytes: measured on the alameda and iPhone sets,
    // a larger packed-image cache only pays while the uploads are exposed on the
    // critical path, and at 2MP/1M Gaussians they are already hidden behind
    // compute, so the VRAM is better left to the training state.
    std::size_t training_device_cache_max_bytes{0};
    // Upload already-decoded future views on a non-blocking CUDA copy stream.
    // CUDA allocation and enqueue remain on the training thread; each in-flight
    // transfer owns its pinned staging storage until its completion event fires.
    // The copy engine only ever writes loader-owned device staging; the packed
    // view is hopped into the cache entry on the compute stream, because pool
    // memory must not be touched by a transfer from another stream (the pool
    // hands blocks between streams and threads without any ordering). Measured
    // -3% iteration time on the alameda and iPhone sets with the integrity check
    // and ten repeat runs clean.
    bool training_async_upload{true};
    // Decode and pack upcoming shuffled views on background threads while CUDA
    // processes the current iteration, so a host cache miss does not stall the
    // step on image I/O. Host-side only: all CUDA calls stay on the training
    // thread. Zero disables prefetching. The value is the minimum lookahead;
    // with training_prefetch_adaptive the loader grows it to cover one measured
    // host decode plus one iteration of slack.
    std::size_t training_prefetch_views{4};
    // Size the prefetch lookahead from the measured host-load and iteration
    // times instead of using training_prefetch_views as a fixed count. Large
    // datasets decode for far longer than one iteration, so a fixed count of
    // four views leaves most of the decode exposed on the critical path.
    // Bounded to 32 views and to 512MB of in-flight packed views.
    bool training_prefetch_adaptive{true};
    // Hold out every Nth source view from optimization (0 trains on all).
    // The caller may render these views through the evaluation callback.
    // Zero trains every view; the reconstruction CLI keeps that default and
    // only holds out views when the user asks for a stride. Held-out metrics
    // need an explicit --splat-eval-split-every N.
    unsigned evaluation_split_every{0};
    // Iterations at which the caller may render fixed-view comparison snapshots.
    std::vector<unsigned> evaluation_iterations;
    unsigned preview_interval{0};
    // Live preview camera. 0 is the first captured frame. An optional sidecar
    // file lets the editor change this while training without restarting.
    unsigned preview_view_index{0};
    std::filesystem::path preview_view_file;
    // Optional orbit-camera sidecar written by the editor (W2C + intrinsics).
    // When present and readable it overrides preview_view_index.
    std::filesystem::path preview_camera_file;
    // Editor visualization mode sidecar: splat / points / rings.
    std::filesystem::path preview_vis_file;
    // Optional editor acknowledgement sidecar: the number of preview frames it
    // has finished copying out of the shared image. While it is readable the
    // trainer drops a preview instead of waiting for the editor, so a busy
    // editor can never stall the optimizer.
    std::filesystem::path preview_ack_file;
};

// Strategy-specific defaults for shared densification knobs. The CLI applies
// this after selecting the strategy and before mapping user overrides, so the
// same option name can carry a different default value per densification
// strategy without duplicating parameters.
void apply_strategy_defaults(TrainingOptions& options);

// The oversize hinge is a strong early stabilizer but holds back the late
// Growth cutoff: 0 selects the Brush proportion (half of `iterations`, which
// is its 15000 of 30000); the dense MVS path predates that cadence and keeps
// its historical absolute start. Callers that pin a value always win.
[[nodiscard]] inline unsigned grow_stop_iteration(
    const TrainingOptions& options) {
    if (options.grow_stop_iter != 0) return options.grow_stop_iter;
    return options.densification_strategy ==
            DensificationStrategy::dense_adaptive
        ? 15'000U
        : std::max(1U, options.iterations / 2);
}

// fine-tuning phase, so its strength decays linearly to zero over the run
// (office, ADC-IGS: +2.2 dB at 1k with the full penalty but -0.2 dB at
// 30k; the decayed form keeps the early gain, artifacts/office_igs_*).
[[nodiscard]] inline float oversize_penalty_at(
    const TrainingOptions& options, const unsigned iteration) {
    if (options.oversize_penalty <= 0.F ||
        options.oversize_screen_limit <= 0.F)
        return 0.F;
    const float progress = std::min(
        static_cast<float>(iteration - 1) /
            static_cast<float>(std::max(options.iterations, 1U)),
        1.F);
    return options.oversize_penalty * std::max(1.F - progress, 0.F);
}

}  // namespace photara::splat
