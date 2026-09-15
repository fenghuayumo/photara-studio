#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "sfm/export_mvs.hpp"
#include "sfm/frontend.hpp"
#include "sfm/reconstruct.hpp"
#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

namespace mvs = aetherscan::mvs;
namespace sfm = aetherscan::sfm;
namespace splat = aetherscan::splat;

struct SfmSceneHandle {
    sfm::Scene scene;
    sfm::ReconstructionSummary summary;
};

struct MvsSceneHandle {
    mvs::MvsScene scene;
    splat::DatasetFormat dataset_format{splat::DatasetFormat::auto_detect};
    std::vector<std::string> warnings;
};

struct GaussianModelHandle {
    splat::GaussianModel model;
};

struct MeshResultHandle {
    mvs::DenseCloud surface_cloud;
    mvs::Mesh mesh;
    std::size_t valid_depth_pixels{};
    std::size_t tetrahedron_count{};
    std::size_t occupied_tetrahedron_count{};
};

[[nodiscard]] bool supported_image_extension(
    std::filesystem::path extension) {
    std::string value = extension.string();
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return value == ".jpg" || value == ".jpeg" || value == ".png" ||
           value == ".tif" || value == ".tiff" || value == ".bmp";
}

[[nodiscard]] std::vector<std::filesystem::path> discover_images(
    const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory))
        throw std::invalid_argument(
            "Image directory does not exist: " + directory.string());
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        if (entry.is_regular_file() &&
            supported_image_extension(entry.path().extension()))
            paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
        throw std::invalid_argument(
            "Image directory contains no supported images: " +
            directory.string());
    return paths;
}

[[nodiscard]] std::shared_ptr<SfmSceneHandle> run_sfm(
    const std::vector<std::filesystem::path>& image_paths,
    const sfm::ReconstructionConfig& options) {
    auto result = std::make_shared<SfmSceneHandle>();
    result->summary = sfm::reconstruct(
        result->scene, image_paths, options);
    return result;
}

[[nodiscard]] std::shared_ptr<SfmSceneHandle> run_sfm_directory(
    const std::filesystem::path& image_directory,
    const sfm::ReconstructionConfig& options) {
    return run_sfm(discover_images(image_directory), options);
}

[[nodiscard]] std::shared_ptr<SfmSceneHandle> run_sfm_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const sfm::FrontEndOptions& options) {
    sfm::FrontEndResult frontend = sfm::run_frontend(image_paths, options);
    auto result = std::make_shared<SfmSceneHandle>();
    result->scene = std::move(frontend.scene);
    return result;
}

sfm::ReconstructionSummary run_sfm_mapping(
    SfmSceneHandle& handle, const sfm::ReconstructionMode mode) {
    switch (mode) {
        case sfm::ReconstructionMode::incremental:
            handle.summary = sfm::run_incremental_mapping(handle.scene);
            break;
        case sfm::ReconstructionMode::hierarchical:
            handle.summary = sfm::run_hierarchical_mapping(handle.scene);
            break;
        case sfm::ReconstructionMode::global:
            handle.summary = sfm::run_global_mapping(handle.scene);
            break;
    }
    return handle.summary;
}

[[nodiscard]] std::shared_ptr<MvsSceneHandle> build_mvs_scene(
    const SfmSceneHandle& sfm_scene, const mvs::DensifyOptions& options) {
    auto result = std::make_shared<MvsSceneHandle>();
    result->scene = mvs::build_mvs_scene(sfm_scene.scene, options);
    return result;
}

[[nodiscard]] std::shared_ptr<MvsSceneHandle> run_mvs(
    const SfmSceneHandle& sfm_scene, const mvs::DensifyOptions& options) {
    auto result = std::make_shared<MvsSceneHandle>();
    result->scene = mvs::densify_from_sfm(sfm_scene.scene, options);
    return result;
}

[[nodiscard]] std::shared_ptr<MvsSceneHandle> load_dataset(
    const splat::DatasetLoadRequest& request) {
    splat::DatasetLoadResult loaded = splat::load_splat_dataset(request);
    auto result = std::make_shared<MvsSceneHandle>();
    result->scene = std::move(loaded.scene);
    result->dataset_format = loaded.format;
    result->warnings = std::move(loaded.warnings);
    return result;
}

[[nodiscard]] std::shared_ptr<GaussianModelHandle> train_3dgs(
    const MvsSceneHandle& scene, const splat::TrainingOptions& options) {
    auto result = std::make_shared<GaussianModelHandle>();
    result->model = splat::Trainer(options).train(scene.scene);
    return result;
}

[[nodiscard]] std::shared_ptr<GaussianModelHandle> load_3dgs(
    const std::filesystem::path& path) {
    auto result = std::make_shared<GaussianModelHandle>();
    result->model = splat::load_gaussians(path);
    return result;
}

[[nodiscard]] std::shared_ptr<MeshResultHandle> extract_tsdf(
    const GaussianModelHandle& model, const MvsSceneHandle& scene,
    const splat::TrainingOptions& training_options,
    splat::SplatMeshOptions mesh_options) {
    mesh_options.fusion.mesh_method = mvs::MeshMethod::tsdf;
    mesh_options.fusion.build_mesh = true;
    splat::SplatMeshResult extracted = splat::extract_splat_mesh(
        model.model, scene.scene, training_options, mesh_options);
    auto result = std::make_shared<MeshResultHandle>();
    result->surface_cloud = std::move(extracted.surface_cloud);
    result->mesh = std::move(extracted.mesh);
    result->valid_depth_pixels = extracted.valid_depth_pixels;
    return result;
}

[[nodiscard]] std::shared_ptr<MeshResultHandle> extract_pam(
    const GaussianModelHandle& model, const MvsSceneHandle& scene,
    const MeshResultHandle& seed,
    const splat::TrainingOptions& training_options,
    const splat::PamMeshOptions& pam_options) {
    splat::PamMeshResult extracted = splat::extract_pam_mesh(
        model.model, scene.scene, seed.mesh, training_options, pam_options);
    auto result = std::make_shared<MeshResultHandle>();
    result->surface_cloud = std::move(extracted.candidate_cloud);
    result->mesh = std::move(extracted.mesh);
    result->tetrahedron_count = extracted.tetrahedron_count;
    result->occupied_tetrahedron_count =
        extracted.occupied_tetrahedron_count;
    return result;
}

}  // namespace

NB_MODULE(aetherscan_native, module) {
    module.doc() =
        "Nanobind experiment API for AetherScan SfM, MVS, 3DGS and TSDF";

    nb::enum_<sfm::ReconstructionMode>(module, "ReconstructionMode")
        .value("INCREMENTAL", sfm::ReconstructionMode::incremental)
        .value("HIERARCHICAL", sfm::ReconstructionMode::hierarchical)
        .value("GLOBAL", sfm::ReconstructionMode::global);

    nb::enum_<mvs::MeshMethod>(module, "MeshMethod")
        .value("DELAUNAY", mvs::MeshMethod::delaunay_cut)
        .value("TSDF", mvs::MeshMethod::tsdf)
        .value("NONE", mvs::MeshMethod::none);

    nb::enum_<mvs::DensifyQuality>(module, "DensifyQuality")
        .value("PREVIEW", mvs::DensifyQuality::preview)
        .value("DEFAULT", mvs::DensifyQuality::default_quality)
        .value("HIGH", mvs::DensifyQuality::high);

    nb::enum_<splat::DatasetFormat>(module, "DatasetFormat")
        .value("AUTO", splat::DatasetFormat::auto_detect)
        .value("COLMAP", splat::DatasetFormat::colmap)
        .value("REALITY_CAPTURE", splat::DatasetFormat::reality_capture)
        .value("OPENMVS", splat::DatasetFormat::openmvs);

    nb::enum_<splat::AlphaMode>(module, "AlphaMode")
        .value("MASKED", splat::AlphaMode::masked)
        .value("TRANSPARENT", splat::AlphaMode::transparent);

    nb::enum_<splat::PpispParamType>(module, "PpispParamType")
        .value("NO_CRF_NO_VIG", splat::PpispParamType::no_crf_no_vig)
        .value("NO_CRF", splat::PpispParamType::no_crf)
        .value("ORIGINAL", splat::PpispParamType::original);

    nb::enum_<splat::DensificationStrategy>(
        module, "DensificationStrategy")
        .value("ADC_PLUS", splat::DensificationStrategy::adc_plus)
        .value("ADC_IGS", splat::DensificationStrategy::adc_igs)

    nb::class_<sfm::FrontEndOptions>(module, "FrontEndOptions")
        .def(nb::init<>())
        .def_rw("focal_pixels", &sfm::FrontEndOptions::focal_pixels)
        .def_rw(
            "trust_focal_pixels",
            &sfm::FrontEndOptions::trust_focal_pixels)
        .def_rw(
            "neighbor_window", &sfm::FrontEndOptions::neighbor_window)
        .def_rw("thread_count", &sfm::FrontEndOptions::thread_count)
        .def_rw("extractor", &sfm::FrontEndOptions::extractor)
        .def_rw("matcher", &sfm::FrontEndOptions::matcher)
        .def_rw("pipeline", &sfm::FrontEndOptions::pipeline)
        .def_rw("match_ratio", &sfm::FrontEndOptions::match_ratio)
        .def_rw("mutual_check", &sfm::FrontEndOptions::mutual_check)
        .def_rw("max_features", &sfm::FrontEndOptions::max_features)
        .def_rw(
            "sift_contrast_threshold",
            &sfm::FrontEndOptions::sift_contrast_threshold)
        .def_rw(
            "extractor_model_path",
            &sfm::FrontEndOptions::extractor_model_path)
        .def_rw(
            "extractor_input_width",
            &sfm::FrontEndOptions::extractor_input_width)
        .def_rw(
            "extractor_input_height",
            &sfm::FrontEndOptions::extractor_input_height)
        .def_rw(
            "extractor_min_score",
            &sfm::FrontEndOptions::extractor_min_score)
        .def_rw(
            "extractor_use_cuda",
            &sfm::FrontEndOptions::extractor_use_cuda)
        .def_rw(
            "lightglue_model_path",
            &sfm::FrontEndOptions::lightglue_model_path)
        .def_rw(
            "lightglue_extractor",
            &sfm::FrontEndOptions::lightglue_extractor)
        .def_rw(
            "lightglue_input_width",
            &sfm::FrontEndOptions::lightglue_input_width)
        .def_rw(
            "lightglue_input_height",
            &sfm::FrontEndOptions::lightglue_input_height)
        .def_rw(
            "lightglue_min_score",
            &sfm::FrontEndOptions::lightglue_min_score)
        .def_rw(
            "lightglue_use_cuda",
            &sfm::FrontEndOptions::lightglue_use_cuda)
        .def_rw(
            "progressive_pair_expansion",
            &sfm::FrontEndOptions::progressive_pair_expansion)
        .def_rw(
            "progressive_min_verified_degree",
            &sfm::FrontEndOptions::progressive_min_verified_degree)
        .def_rw(
            "progressive_rescue_match_ratio",
            &sfm::FrontEndOptions::progressive_rescue_match_ratio)
        .def_rw(
            "progressive_rescue_min_inliers",
            &sfm::FrontEndOptions::progressive_rescue_min_inliers)
        .def_rw(
            "progressive_rescue_neighbor_window",
            &sfm::FrontEndOptions::progressive_rescue_neighbor_window)
        .def_rw(
            "progressive_rescue_retrieval_top_k",
            &sfm::FrontEndOptions::progressive_rescue_retrieval_top_k)
        .def_rw(
            "progressive_rescue_max_pairs_per_image",
            &sfm::FrontEndOptions::progressive_rescue_max_pairs_per_image)
        .def_rw(
            "progressive_rescue_max_features",
            &sfm::FrontEndOptions::progressive_rescue_max_features);

    nb::class_<sfm::ReconstructionConfig>(module, "SfmOptions")
        .def(nb::init<>())
        .def_rw("mode", &sfm::ReconstructionConfig::mode)
        .def_rw("frontend", &sfm::ReconstructionConfig::frontend)
        .def_rw(
            "incremental_hierarchical_rescue",
            &sfm::ReconstructionConfig::incremental_hierarchical_rescue)
        .def_rw(
            "incremental_hierarchical_rescue_min_missing",
            &sfm::ReconstructionConfig::
                incremental_hierarchical_rescue_min_missing)
        .def_rw(
            "incremental_hierarchical_rescue_min_missing_ratio",
            &sfm::ReconstructionConfig::
                incremental_hierarchical_rescue_min_missing_ratio);

    nb::class_<sfm::ReconstructionSummary>(module, "SfmSummary")
        .def_ro("valid", &sfm::ReconstructionSummary::valid)
        .def_ro(
            "registered_views",
            &sfm::ReconstructionSummary::registered_views)
        .def_ro(
            "alignment_reliable_views",
            &sfm::ReconstructionSummary::alignment_reliable_views)
        .def_ro(
            "alignment_unreliable_views",
            &sfm::ReconstructionSummary::alignment_unreliable_views)
        .def_ro("landmarks", &sfm::ReconstructionSummary::landmarks)
        .def_ro(
            "failed_views", &sfm::ReconstructionSummary::failed_views)
        .def_ro(
            "reprojection_observations",
            &sfm::ReconstructionSummary::reprojection_observations)
        .def_ro(
            "mean_reprojection_error_pixels",
            &sfm::ReconstructionSummary::mean_reprojection_error_pixels)
        .def_ro(
            "rms_reprojection_error_pixels",
            &sfm::ReconstructionSummary::rms_reprojection_error_pixels);

    nb::class_<mvs::DensifyOptions>(module, "MvsOptions")
        .def(nb::init<>())
        .def_rw("mask_dir", &mvs::DensifyOptions::mask_dir)
        .def_rw("mask_border_px", &mvs::DensifyOptions::mask_border_px)
        .def_rw(
            "resolution_level", &mvs::DensifyOptions::resolution_level)
        .def_rw("min_resolution", &mvs::DensifyOptions::min_resolution)
        .def_rw(
            "sub_resolution_levels",
            &mvs::DensifyOptions::sub_resolution_levels)
        .def_rw(
            "estimation_iters", &mvs::DensifyOptions::estimation_iters)
        .def_rw(
            "geometric_iters", &mvs::DensifyOptions::geometric_iters)
        .def_rw(
            "geometric_weight", &mvs::DensifyOptions::geometric_weight)
        .def_rw("random_iters", &mvs::DensifyOptions::random_iters)
        .def_rw("max_neighbors", &mvs::DensifyOptions::max_neighbors)
        .def_rw(
            "min_patch_views", &mvs::DensifyOptions::min_patch_views)
        .def_rw(
            "optim_angle_deg", &mvs::DensifyOptions::optim_angle_deg)
        .def_rw(
            "ncc_keep_threshold",
            &mvs::DensifyOptions::ncc_keep_threshold)
        .def_rw(
            "min_shared_points", &mvs::DensifyOptions::min_shared_points)
        .def_rw(
            "min_views_fuse", &mvs::DensifyOptions::min_views_fuse)
        .def_rw("speckle_size", &mvs::DensifyOptions::speckle_size)
        .def_rw(
            "depth_diff_threshold",
            &mvs::DensifyOptions::depth_diff_threshold)
        .def_rw(
            "reprojection_error_px",
            &mvs::DensifyOptions::reprojection_error_px)
        .def_rw(
            "normal_diff_threshold_deg",
            &mvs::DensifyOptions::normal_diff_threshold_deg)
        .def_rw(
            "filter_depth_maps",
            &mvs::DensifyOptions::filter_depth_maps)
        .def_rw(
            "min_views_filter", &mvs::DensifyOptions::min_views_filter)
        .def_rw(
            "adjust_filtered_depth",
            &mvs::DensifyOptions::adjust_filtered_depth)
        .def_rw(
            "geometric_consistency",
            &mvs::DensifyOptions::geometric_consistency)
        .def_rw("build_mesh", &mvs::DensifyOptions::build_mesh)
        .def_rw("mesh_method", &mvs::DensifyOptions::mesh_method)
        .def_rw("mesh_clean", &mvs::DensifyOptions::mesh_clean)
        .def_rw(
            "mesh_close_hole_edges",
            &mvs::DensifyOptions::mesh_close_hole_edges)
        .def_rw(
            "mesh_max_points", &mvs::DensifyOptions::mesh_max_points)
        .def_rw(
            "mesh_tsdf_voxel_size",
            &mvs::DensifyOptions::mesh_tsdf_voxel_size)
        .def_rw(
            "mesh_tsdf_bounds_padding",
            &mvs::DensifyOptions::mesh_tsdf_bounds_padding)
        .def_rw(
            "mesh_tsdf_voxel_scale",
            &mvs::DensifyOptions::mesh_tsdf_voxel_scale)
        .def_rw(
            "mesh_tsdf_truncation_voxels",
            &mvs::DensifyOptions::mesh_tsdf_truncation_voxels)
        .def_rw(
            "mesh_tsdf_pixel_step",
            &mvs::DensifyOptions::mesh_tsdf_pixel_step)
        .def_rw(
            "mesh_tsdf_min_weight",
            &mvs::DensifyOptions::mesh_tsdf_min_weight)
        .def_rw(
            "mesh_tsdf_support_closing_axes",
            &mvs::DensifyOptions::mesh_tsdf_support_closing_axes)
        .def_rw(
            "mesh_tsdf_frame_export_dir",
            &mvs::DensifyOptions::mesh_tsdf_frame_export_dir)
        .def_rw(
            "mesh_tsdf_diagnostics_dir",
            &mvs::DensifyOptions::mesh_tsdf_diagnostics_dir)
        .def_rw(
            "mesh_tsdf_min_component_fraction",
            &mvs::DensifyOptions::mesh_tsdf_min_component_fraction)
        .def_rw(
            "mesh_tsdf_smooth_iters",
            &mvs::DensifyOptions::mesh_tsdf_smooth_iters)
        .def_rw(
            "mesh_tsdf_smooth_lambda",
            &mvs::DensifyOptions::mesh_tsdf_smooth_lambda)
        .def_rw(
            "mesh_tsdf_smooth_mu",
            &mvs::DensifyOptions::mesh_tsdf_smooth_mu)
        .def_rw("thread_count", &mvs::DensifyOptions::thread_count);

    nb::class_<splat::TrainingOptions>(module, "TrainingOptions")
        .def(nb::init<>())
        .def_rw("iterations", &splat::TrainingOptions::iterations)
        .def_rw("sh_degree", &splat::TrainingOptions::sh_degree)
        .def_rw(
            "sh_degree_interval",
            &splat::TrainingOptions::sh_degree_interval)
        .def_rw("seed", &splat::TrainingOptions::seed)
        .def_rw("log_interval", &splat::TrainingOptions::log_interval)
        .def_rw("profile_cuda", &splat::TrainingOptions::profile_cuda)
        .def_rw(
            "cuda_profile_interval",
            &splat::TrainingOptions::cuda_profile_interval)
        .def_rw("input_is_dense", &splat::TrainingOptions::input_is_dense)
        .def_rw(
            "enable_densification",
            &splat::TrainingOptions::enable_densification)
        .def_rw(
            "densification_strategy",
            &splat::TrainingOptions::densification_strategy)
        .def_rw(
            "densification_cap",
            &splat::TrainingOptions::densification_cap)
        .def_rw(
            "refine_start_iter",
            &splat::TrainingOptions::refine_start_iter)
        .def_rw(
            "refine_stop_iter",
            &splat::TrainingOptions::refine_stop_iter)
        .def_rw(
            "grow_stop_iter", &splat::TrainingOptions::grow_stop_iter)
        .def_rw("refine_every", &splat::TrainingOptions::refine_every)
        .def_rw(
            "opacity_reset_every",
            &splat::TrainingOptions::opacity_reset_every)
        .def_rw(
            "densify_gradient_threshold",
            &splat::TrainingOptions::densify_gradient_threshold)
        .def_rw(
            "densify_select_fraction",
            &splat::TrainingOptions::densify_select_fraction)
        .def_rw(
            "densify_screen_threshold",
            &splat::TrainingOptions::densify_screen_threshold)
        .def_rw("prune_opacity", &splat::TrainingOptions::prune_opacity)
        .def_rw("opacity_decay", &splat::TrainingOptions::opacity_decay)
        .def_rw("scale_decay", &splat::TrainingOptions::scale_decay)
        .def_rw(
            "background_noise_strength",
            &splat::TrainingOptions::background_noise_strength)
        .def_rw(
            "initialize_scale_from_knn",
            &splat::TrainingOptions::initialize_scale_from_knn)
        .def_rw("means_lr", &splat::TrainingOptions::means_lr)
        .def_rw("scales_lr", &splat::TrainingOptions::scales_lr)
        .def_rw("opacities_lr", &splat::TrainingOptions::opacities_lr)
        .def_rw("quaternions_lr", &splat::TrainingOptions::quaternions_lr)
        .def_rw("sh0_lr", &splat::TrainingOptions::sh0_lr)
        .def_rw("sh_rest_lr", &splat::TrainingOptions::sh_rest_lr)
        .def_rw(
            "photometric_weight",
            &splat::TrainingOptions::photometric_weight)
        .def_rw("ssim_weight", &splat::TrainingOptions::ssim_weight)
        .def_rw(
            "use_bilateral_grid",
            &splat::TrainingOptions::use_bilateral_grid)
        .def_rw(
            "bilateral_grid_width",
            &splat::TrainingOptions::bilateral_grid_width)
        .def_rw(
            "bilateral_grid_height",
            &splat::TrainingOptions::bilateral_grid_height)
        .def_rw(
            "bilateral_grid_luma",
            &splat::TrainingOptions::bilateral_grid_luma)
        .def_rw(
            "bilateral_grid_lr",
            &splat::TrainingOptions::bilateral_grid_lr)
        .def_rw(
            "bilateral_grid_tv_weight",
            &splat::TrainingOptions::bilateral_grid_tv_weight)
        .def_rw("use_ppisp", &splat::TrainingOptions::use_ppisp)
        .def_rw("ppisp_type", &splat::TrainingOptions::ppisp_type)
        .def_rw("ppisp_lr", &splat::TrainingOptions::ppisp_lr)
        .def_rw(
            "ppisp_clamp_output",
            &splat::TrainingOptions::ppisp_clamp_output)
        .def_rw(
            "ppisp_reg_exposure_mean",
            &splat::TrainingOptions::ppisp_reg_exposure_mean)
        .def_rw(
            "ppisp_reg_color_mean",
            &splat::TrainingOptions::ppisp_reg_color_mean)
        .def_rw(
            "ppisp_before_bilagrid",
            &splat::TrainingOptions::ppisp_before_bilagrid)
        .def_rw(
            "use_depth_normal_loss",
            &splat::TrainingOptions::use_depth_normal_loss)
        .def_rw(
            "depth_normal_weight",
            &splat::TrainingOptions::depth_normal_weight)
        .def_rw(
            "depth_normal_from_iter",
            &splat::TrainingOptions::depth_normal_from_iter)
        .def_rw(
            "multi_view_geo_weight",
            &splat::TrainingOptions::multi_view_geo_weight)
        .def_rw(
            "multi_view_ncc_weight",
            &splat::TrainingOptions::multi_view_ncc_weight)
        .def_rw(
            "multi_view_num", &splat::TrainingOptions::multi_view_num)
        .def_rw(
            "multi_view_tail_interval",
            &splat::TrainingOptions::multi_view_tail_interval)
        .def_rw(
            "multi_view_adaptive_frequency",
            &splat::TrainingOptions::multi_view_adaptive_frequency)
        .def_rw(
            "multi_view_adaptive_max_interval",
            &splat::TrainingOptions::multi_view_adaptive_max_interval)
        .def_rw(
            "multi_view_adaptive_stable_refinements",
            &splat::TrainingOptions::multi_view_adaptive_stable_refinements)
        .def_rw(
            "multi_view_adaptive_count_threshold",
            &splat::TrainingOptions::multi_view_adaptive_count_threshold)
        .def_rw(
            "multi_view_adaptive_churn_threshold",
            &splat::TrainingOptions::multi_view_adaptive_churn_threshold)
        .def_rw(
            "multi_view_adaptive_depth_threshold",
            &splat::TrainingOptions::multi_view_adaptive_depth_threshold)
        .def_rw(
            "multi_view_adaptive_min_depth_consistency",
            &splat::TrainingOptions::multi_view_adaptive_min_depth_consistency)
        .def_rw(
            "multi_view_adaptive_distribution_threshold",
            &splat::TrainingOptions::multi_view_adaptive_distribution_threshold)
        .def_rw(
            "multi_view_pixel_noise_threshold",
            &splat::TrainingOptions::multi_view_pixel_noise_threshold)
        .def_rw("use_mask", &splat::TrainingOptions::use_mask)
        .def_rw("alpha_mode", &splat::TrainingOptions::alpha_mode)
        .def_rw(
            "match_alpha_weight",
            &splat::TrainingOptions::match_alpha_weight)
        .def_rw("mask_dir", &splat::TrainingOptions::mask_dir)
        .def_rw(
            "minimum_scale_fraction",
            &splat::TrainingOptions::minimum_scale_fraction)
        .def_rw(
            "maximum_scale_fraction",
            &splat::TrainingOptions::maximum_scale_fraction)
        .def_rw(
            "max_scale_ratio", &splat::TrainingOptions::max_scale_ratio)
        .def_rw(
            "constrain_scale_range",
            &splat::TrainingOptions::constrain_scale_range)
        .def_rw(
            "use_source_resolution",
            &splat::TrainingOptions::use_source_resolution)
        .def_rw(
            "undistort_to_pinhole",
            &splat::TrainingOptions::undistort_to_pinhole)
        .def_rw(
            "max_image_dimension",
            &splat::TrainingOptions::max_image_dimension)
        .def_rw(
            "progressive_resolution",
            &splat::TrainingOptions::progressive_resolution)
        .def_rw(
            "progressive_resolution_interval",
            &splat::TrainingOptions::progressive_resolution_interval)
        .def_rw(
            "progressive_initial_scale",
            &splat::TrainingOptions::progressive_initial_scale)
        .def_rw(
            "training_view_cache_bytes",
            &splat::TrainingOptions::training_view_cache_bytes)
        .def_rw(
            "training_prefetch_views",
            &splat::TrainingOptions::training_prefetch_views)
        .def_rw(
            "evaluation_split_every",
            &splat::TrainingOptions::evaluation_split_every)
        .def_rw(
            "use_normal_field",
            &splat::TrainingOptions::use_normal_field)
        .def_rw(
            "normal_field_weight",
            &splat::TrainingOptions::normal_field_weight)
        .def_rw(
            "normal_field_depth_ratio",
            &splat::TrainingOptions::normal_field_depth_ratio)
        .def_rw(
            "normal_field_from_iter",
            &splat::TrainingOptions::normal_field_from_iter)
        .def_rw(
            "normal_features_lr",
            &splat::TrainingOptions::normal_features_lr);

    nb::class_<splat::DatasetLoadRequest>(module, "DatasetRequest")
        .def(nb::init<>())
        .def_rw("source", &splat::DatasetLoadRequest::source)
        .def_rw(
            "image_directory",
            &splat::DatasetLoadRequest::image_directory)
        .def_rw(
            "initial_point_cloud",
            &splat::DatasetLoadRequest::initial_point_cloud)
        .def_rw("format", &splat::DatasetLoadRequest::format)
        .def_rw(
            "random_initial_point_count",
            &splat::DatasetLoadRequest::random_initial_point_count)
        .def_rw("seed", &splat::DatasetLoadRequest::seed);

    nb::class_<splat::SplatMeshOptions>(module, "TsdfOptions")
        .def(nb::init<>())
        .def_rw(
            "alpha_threshold",
            &splat::SplatMeshOptions::alpha_threshold)
        .def_rw("max_depth", &splat::SplatMeshOptions::max_depth)
        .def_rw(
            "diagnostics_dir",
            &splat::SplatMeshOptions::diagnostics_dir)
        .def_rw(
            "min_depth_normal_cosine",
            &splat::SplatMeshOptions::min_depth_normal_cosine)
        .def_rw("fusion", &splat::SplatMeshOptions::fusion);

    nb::class_<splat::PamMeshOptions>(module, "PamOptions")
        .def(nb::init<>())
        .def_rw("max_points", &splat::PamMeshOptions::max_points)
        .def_rw(
            "pivot_max_points",
            &splat::PamMeshOptions::pivot_max_points)
        .def_rw(
            "pivot_std_factor",
            &splat::PamMeshOptions::pivot_std_factor)
        .def_rw(
            "gaussian_seed_fraction",
            &splat::PamMeshOptions::gaussian_seed_fraction)
        .def_rw(
            "oversampling_factor",
            &splat::PamMeshOptions::oversampling_factor)
        .def_rw(
            "max_resample_rounds",
            &splat::PamMeshOptions::max_resample_rounds)
        .def_rw(
            "refinement_steps",
            &splat::PamMeshOptions::refinement_steps)
        .def_rw(
            "vector_field_neighbors",
            &splat::PamMeshOptions::vector_field_neighbors)
        .def_rw(
            "points_per_tetrahedron",
            &splat::PamMeshOptions::points_per_tetrahedron)
        .def_rw(
            "occupancy_iso_value",
            &splat::PamMeshOptions::occupancy_iso_value)
        .def_rw(
            "vacancy_threshold",
            &splat::PamMeshOptions::vacancy_threshold)
        .def_rw(
            "minimum_gradient_norm_squared",
            &splat::PamMeshOptions::minimum_gradient_norm_squared)
        .def_rw(
            "refinement_step",
            &splat::PamMeshOptions::refinement_step)
        .def_rw(
            "mask_background_threshold",
            &splat::PamMeshOptions::mask_background_threshold)
        .def_rw(
            "occupancy_chunk_size",
            &splat::PamMeshOptions::occupancy_chunk_size)
        .def_rw("seed", &splat::PamMeshOptions::seed);

    nb::class_<SfmSceneHandle>(module, "SfmScene")
        .def_prop_ro(
            "summary",
            [](const SfmSceneHandle& self) { return self.summary; })
        .def_prop_ro(
            "camera_count",
            [](const SfmSceneHandle& self) {
                return self.scene.cameras.size();
            })
        .def_prop_ro(
            "image_count",
            [](const SfmSceneHandle& self) {
                return self.scene.images.size();
            })
        .def_prop_ro(
            "registered_count",
            [](const SfmSceneHandle& self) {
                return self.scene.registered_count();
            })
        .def_prop_ro(
            "track_count",
            [](const SfmSceneHandle& self) {
                return self.scene.tracks.size();
            })
        .def(
            "save_openmvs",
            [](const SfmSceneHandle& self,
               const std::filesystem::path& path) {
                sfm::export_openmvs_interface(self.scene, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<MvsSceneHandle>(module, "MvsScene")
        .def_prop_ro(
            "view_count",
            [](const MvsSceneHandle& self) {
                return self.scene.views.size();
            })
        .def_prop_ro(
            "dense_point_count",
            [](const MvsSceneHandle& self) {
                return self.scene.dense_cloud.points.size();
            })
        .def_prop_ro(
            "mesh_vertex_count",
            [](const MvsSceneHandle& self) {
                return self.scene.mesh.vertices.size();
            })
        .def_prop_ro(
            "mesh_face_count",
            [](const MvsSceneHandle& self) {
                return self.scene.mesh.faces.size();
            })
        .def_ro("warnings", &MvsSceneHandle::warnings)
        .def(
            "load_dense_cloud",
            [](MvsSceneHandle& self, const std::filesystem::path& path) {
                self.scene.dense_cloud = mvs::load_dense_ply(path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_dense_cloud",
            [](const MvsSceneHandle& self,
               const std::filesystem::path& path) {
                mvs::save_dense_ply(self.scene.dense_cloud, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_mesh",
            [](const MvsSceneHandle& self,
               const std::filesystem::path& path) {
                mvs::save_mesh_ply(self.scene.mesh, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<GaussianModelHandle>(module, "GaussianModel")
        .def_prop_ro(
            "size",
            [](const GaussianModelHandle& self) {
                return self.model.size();
            })
        .def_prop_ro(
            "sh_degree",
            [](const GaussianModelHandle& self) {
                return self.model.sh_degree;
            })
        .def(
            "save",
            [](const GaussianModelHandle& self,
               const std::filesystem::path& path) {
                splat::save_gaussians(self.model, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<MeshResultHandle>(module, "MeshResult")
        .def_prop_ro(
            "vertex_count",
            [](const MeshResultHandle& self) {
                return self.mesh.vertices.size();
            })
        .def_prop_ro(
            "face_count",
            [](const MeshResultHandle& self) {
                return self.mesh.faces.size();
            })
        .def_prop_ro(
            "surface_point_count",
            [](const MeshResultHandle& self) {
                return self.surface_cloud.points.size();
            })
        .def_ro(
            "valid_depth_pixels",
            &MeshResultHandle::valid_depth_pixels)
        .def_ro(
            "tetrahedron_count",
            &MeshResultHandle::tetrahedron_count)
        .def_ro(
            "occupied_tetrahedron_count",
            &MeshResultHandle::occupied_tetrahedron_count)
        .def(
            "save",
            [](const MeshResultHandle& self,
               const std::filesystem::path& path) {
                mvs::save_mesh_ply(self.mesh, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_surface_cloud",
            [](const MeshResultHandle& self,
               const std::filesystem::path& path) {
                mvs::save_dense_ply(self.surface_cloud, path);
            },
            nb::arg("path"),
            nb::call_guard<nb::gil_scoped_release>());

    module.def(
        "discover_images", &discover_images, nb::arg("directory"),
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "run_sfm", &run_sfm, nb::arg("image_paths"),
        nb::arg("options") = sfm::ReconstructionConfig{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "run_sfm_directory", &run_sfm_directory,
        nb::arg("image_directory"),
        nb::arg("options") = sfm::ReconstructionConfig{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "run_sfm_frontend", &run_sfm_frontend, nb::arg("image_paths"),
        nb::arg("options") = sfm::FrontEndOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "run_sfm_mapping", &run_sfm_mapping, nb::arg("scene"),
        nb::arg("mode") = sfm::ReconstructionMode::global,
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "build_mvs_scene", &build_mvs_scene, nb::arg("sfm_scene"),
        nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "run_mvs", &run_mvs, nb::arg("sfm_scene"),
        nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "mvs_select_neighbors",
        [](MvsSceneHandle& scene, const mvs::DensifyOptions& options) {
            mvs::select_neighbors(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "mvs_estimate_depth_maps",
        [](MvsSceneHandle& scene, const mvs::DensifyOptions& options) {
            mvs::estimate_depth_maps(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "mvs_fuse_depth_maps",
        [](MvsSceneHandle& scene, const mvs::DensifyOptions& options) {
            mvs::fuse_depth_maps(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "mvs_reconstruct_mesh",
        [](MvsSceneHandle& scene, const mvs::DensifyOptions& options) {
            mvs::reconstruct_mesh(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = mvs::DensifyOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "load_dataset", &load_dataset, nb::arg("request"),
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "train_3dgs", &train_3dgs, nb::arg("scene"),
        nb::arg("options") = splat::TrainingOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "load_3dgs", &load_3dgs, nb::arg("path"),
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "extract_tsdf", &extract_tsdf, nb::arg("model"),
        nb::arg("scene"),
        nb::arg("training_options") = splat::TrainingOptions{},
        nb::arg("tsdf_options") = splat::SplatMeshOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "extract_pam", &extract_pam, nb::arg("model"),
        nb::arg("scene"), nb::arg("seed_mesh"),
        nb::arg("training_options") = splat::TrainingOptions{},
        nb::arg("pam_options") = splat::PamMeshOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "apply_mvs_quality_preset",
        [](mvs::DensifyOptions& options, const mvs::DensifyQuality quality) {
            mvs::apply_quality_preset(options, quality);
        },
        nb::arg("options"), nb::arg("quality"));
}
