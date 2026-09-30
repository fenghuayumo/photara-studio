#include "core/version.hpp"
#include "features/registry.hpp"
#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "sfm/asfm.hpp"
#include "sfm/bundle.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/export_mvs.hpp"
#include "sfm/frontend.hpp"
#include "sfm/reconstruct.hpp"
#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
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

namespace mvs = photara::mvs;
namespace sfm = photara::sfm;
namespace splat = photara::splat;
namespace features = photara::features;
namespace ba = photara::ba;

struct SfmSceneHandle {
    sfm::Scene scene;
    sfm::ReconstructionSummary summary;
    sfm::FrontEndTiming frontend_timing;
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
    result->frontend_timing = frontend.timing;
    return result;
}

sfm::ReconstructionSummary run_sfm_mapping(
    SfmSceneHandle& handle, const sfm::ReconstructionConfig& options) {
    switch (options.mode) {
        case sfm::ReconstructionMode::incremental:
            handle.summary = sfm::run_incremental_mapping(
                handle.scene, options.star, options.resection);
            break;
        case sfm::ReconstructionMode::hierarchical:
            handle.summary = sfm::run_hierarchical_mapping(
                handle.scene, options.hierarchical);
            break;
        case sfm::ReconstructionMode::global:
            handle.summary = sfm::run_global_mapping(
                handle.scene, options.global_rotation,
                options.global_positioning, options.resection);
            break;
    }
    return handle.summary;
}

[[nodiscard]] std::shared_ptr<SfmSceneHandle> load_sfm_scene(
    const std::filesystem::path& path, const sfm::AsfmOptions& options) {
    auto result = std::make_shared<SfmSceneHandle>();
    result->scene = sfm::load_asfm(path, options);
    return result;
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

[[nodiscard]] splat::ProgressCallback make_progress_callback(
    const nb::object& progress) {
    if (progress.is_none())
        return {};
    return [progress](const splat::TrainingProgress& value) {
        nb::gil_scoped_acquire acquire;
        nb::object result = progress(value);
        if (result.is_none())
            return true;
        return nb::cast<bool>(result);
    };
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

NB_MODULE(photara, module) {
    module.doc() =
        "Native Python API for Photara SfM, MVS, 3DGS and meshing";
    module.attr("__version__") = PHOTARA_VERSION;

    nb::enum_<photara::CameraModel>(module, "CameraModel")
        .value("PINHOLE", photara::CameraModel::pinhole)
        .value("OPENCV_FISHEYE", photara::CameraModel::opencv_fisheye)
        .value("AUTOMATIC", photara::CameraModel::automatic)
        .value("EQUIRECTANGULAR", photara::CameraModel::equirectangular);

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

    nb::enum_<splat::GaussianFormat>(module, "GaussianFormat")
        .value("AUTO", splat::GaussianFormat::auto_detect)
        .value("PLY", splat::GaussianFormat::ply)
        .value("SOG", splat::GaussianFormat::sog)
        .value("SPZ", splat::GaussianFormat::spz)
        .value("GLB", splat::GaussianFormat::glb);

    nb::enum_<splat::AlphaMode>(module, "AlphaMode")
        .value("MASKED", splat::AlphaMode::masked)
        .value("TRANSPARENT", splat::AlphaMode::transparent);

    nb::enum_<splat::PpispParamType>(module, "PpispParamType")
        .value("NO_CRF_NO_VIG", splat::PpispParamType::no_crf_no_vig)
        .value("NO_CRF", splat::PpispParamType::no_crf)
        .value("ORIGINAL", splat::PpispParamType::original);

    nb::enum_<splat::TrainingBackend>(module, "TrainingBackend")
        .value("CUDA", splat::TrainingBackend::cuda)
        .value("VULKAN", splat::TrainingBackend::vulkan);

    nb::enum_<splat::DensificationStrategy>(
        module, "DensificationStrategy")
        .value("ADC_PLUS", splat::DensificationStrategy::adc_plus)
        .value("ADC_IGS", splat::DensificationStrategy::adc_igs)
        .value(
            "DENSE_ADAPTIVE",
            splat::DensificationStrategy::dense_adaptive)
        .value("EMC", splat::DensificationStrategy::emc);

    nb::enum_<sfm::PositioningBackend>(module, "PositioningBackend")
        .value("AUTOMATIC", sfm::PositioningBackend::automatic)
        .value("CPU", sfm::PositioningBackend::cpu)
        .value("CUDA", sfm::PositioningBackend::cuda)
        .value("VULKAN", sfm::PositioningBackend::vulkan);

    nb::enum_<sfm::BundleBackendPreference>(module, "BundleBackend")
        .value("AUTOMATIC", sfm::BundleBackendPreference::automatic)
        .value("CPU", sfm::BundleBackendPreference::cpu)
        .value("CUDA", sfm::BundleBackendPreference::cuda)
        .value("VULKAN", sfm::BundleBackendPreference::vulkan);

    nb::enum_<sfm::BundleBackend>(module, "BundleBackendUsed")
        .value("CPU", sfm::BundleBackend::cpu)
        .value("CUDA", sfm::BundleBackend::cuda)
        .value("VULKAN", sfm::BundleBackend::vulkan);

    nb::enum_<ba::TerminationReason>(module, "TerminationReason")
        .value("CONVERGED", ba::TerminationReason::converged)
        .value("MAXIMUM_ITERATIONS", ba::TerminationReason::maximum_iterations)
        .value("NUMERICAL_FAILURE", ba::TerminationReason::numerical_failure);

    nb::enum_<sfm::GlobalPositioningConstraint>(
        module, "GlobalPositioningConstraint")
        .value("ONLY_POINTS", sfm::GlobalPositioningConstraint::only_points)
        .value("ONLY_CAMERAS", sfm::GlobalPositioningConstraint::only_cameras)
        .value(
            "POINTS_AND_CAMERAS_BALANCED",
            sfm::GlobalPositioningConstraint::points_and_cameras_balanced)
        .value(
            "POINTS_AND_CAMERAS",
            sfm::GlobalPositioningConstraint::points_and_cameras);

    nb::enum_<sfm::GlobalRotationOptions::WeightType>(
        module, "GlobalRotationWeight")
        .value(
            "GEMAN_MCCLURE",
            sfm::GlobalRotationOptions::WeightType::geman_mcclure)
        .value("HALF_NORM", sfm::GlobalRotationOptions::WeightType::half_norm);

#define PHOTARA_BIND_RW(binding, type, field) \
    binding.def_rw(#field, &type::field)

    auto optimizer_options =
        nb::class_<ba::OptimizerOptions>(module, "OptimizerOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, maximum_iterations);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, maximum_pcg_iterations);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, huber_delta);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, minimum_depth);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, initial_damping);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, minimum_damping);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, maximum_damping);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, function_tolerance);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, step_tolerance);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, pcg_tolerance);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, fix_first_pose);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, fix_first_point);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_rotations);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_translations);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_points);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_focal);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_aspect_ratio);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_principal_point);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, optimize_distortion);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, focal_prior_weight);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, min_focal_ratio);
    PHOTARA_BIND_RW(optimizer_options, ba::OptimizerOptions, max_focal_ratio);

    nb::class_<ba::IterationSummary>(module, "OptimizerIterationSummary")
        .def_ro("iteration", &ba::IterationSummary::iteration)
        .def_ro("cost", &ba::IterationSummary::cost)
        .def_ro("damping", &ba::IterationSummary::damping)
        .def_ro("step_norm", &ba::IterationSummary::step_norm)
        .def_ro("pcg_iterations", &ba::IterationSummary::pcg_iterations)
        .def_ro("accepted", &ba::IterationSummary::accepted);

    nb::class_<ba::OptimizerSummary>(module, "OptimizerSummary")
        .def_ro("termination", &ba::OptimizerSummary::termination)
        .def_ro("initial_cost", &ba::OptimizerSummary::initial_cost)
        .def_ro("final_cost", &ba::OptimizerSummary::final_cost)
        .def_ro("total_time_ms", &ba::OptimizerSummary::total_time_ms)
        .def_ro("linearization_time_ms", &ba::OptimizerSummary::linearization_time_ms)
        .def_ro("assembly_time_ms", &ba::OptimizerSummary::assembly_time_ms)
        .def_ro("solve_time_ms", &ba::OptimizerSummary::solve_time_ms)
        .def_ro("update_and_cost_time_ms", &ba::OptimizerSummary::update_and_cost_time_ms)
        .def_ro("successful_steps", &ba::OptimizerSummary::successful_steps)
        .def_ro("unsuccessful_steps", &ba::OptimizerSummary::unsuccessful_steps)
        .def_ro("iterations", &ba::OptimizerSummary::iterations)
        .def("brief_report", &ba::OptimizerSummary::brief_report);

    auto bundle_options =
        nb::class_<sfm::BundleOptions>(module, "BundleOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, optimizer);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, optimize_points);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, write_intrinsics);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, free_image_ids);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, fixed_image_ids);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, optimize_all_registered);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, gate_intrinsics_by_observability);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, min_views_for_intrinsics);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, min_median_parallax_deg);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, prefer_cuda);
    PHOTARA_BIND_RW(bundle_options, sfm::BundleOptions, cuda_min_observations);

    nb::class_<sfm::BundleSummary>(module, "BundleSummary")
        .def_ro("success", &sfm::BundleSummary::success)
        .def_ro("optimizer", &sfm::BundleSummary::optimizer)
        .def_ro("num_cameras", &sfm::BundleSummary::num_cameras)
        .def_ro("num_points", &sfm::BundleSummary::num_points)
        .def_ro("num_observations", &sfm::BundleSummary::num_observations)
        .def_ro("backend", &sfm::BundleSummary::backend);

    auto relative_pose =
        nb::class_<sfm::RelativePoseOptions>(module, "RelativePoseOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, max_epipolar_error_px);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, max_reproj_error_px);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, min_ray_angle_deg);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, epipole_filter_px);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, confidence);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, max_iterations);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, min_iterations);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, min_inliers);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, force_fundamental);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, force_shared_focal);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, decompose_fundamental);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, estimate_homography);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, homography_degeneracy_ratio);
    PHOTARA_BIND_RW(relative_pose, sfm::RelativePoseOptions, degenerate_weight_scale);

    auto absolute_pose =
        nb::class_<sfm::AbsolutePoseOptions>(module, "AbsolutePoseOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(absolute_pose, sfm::AbsolutePoseOptions, max_reproj_error_px);
    PHOTARA_BIND_RW(absolute_pose, sfm::AbsolutePoseOptions, confidence);
    PHOTARA_BIND_RW(absolute_pose, sfm::AbsolutePoseOptions, max_iterations);
    PHOTARA_BIND_RW(absolute_pose, sfm::AbsolutePoseOptions, min_iterations);
    PHOTARA_BIND_RW(absolute_pose, sfm::AbsolutePoseOptions, min_inliers);

    auto pair_weighting =
        nb::class_<sfm::PairWeightingOptions>(module, "PairWeightingOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, min_inliers);
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, max_triplet_rotation_error_deg);
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, triplet_saturation);
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, min_triplets_for_penalty);
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, max_inconsistent_triplet_ratio);
    PHOTARA_BIND_RW(pair_weighting, sfm::PairWeightingOptions, inconsistent_triplet_scale);

    auto vocabulary =
        nb::class_<sfm::VocabularyConfig>(module, "VocabularyOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, branching);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, depth);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, max_iterations);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, seed);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, max_descriptors_per_image);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, max_training_descriptors);
    PHOTARA_BIND_RW(vocabulary, sfm::VocabularyConfig, sample_grid);

    auto retrieval =
        nb::class_<sfm::RetrievalOptions>(module, "RetrievalOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, top_k);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, max_descriptors_per_image);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, sample_grid);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, stop_word_ratio);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, max_posting_images);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, vocabulary);
    PHOTARA_BIND_RW(retrieval, sfm::RetrievalOptions, vocabulary_path);

    auto checkpoint =
        nb::class_<sfm::CheckpointOptions>(module, "CheckpointOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(checkpoint, sfm::CheckpointOptions, directory);
    PHOTARA_BIND_RW(checkpoint, sfm::CheckpointOptions, read);
    PHOTARA_BIND_RW(checkpoint, sfm::CheckpointOptions, write);
    PHOTARA_BIND_RW(checkpoint, sfm::CheckpointOptions, reconstruction_interval);
    PHOTARA_BIND_RW(checkpoint, sfm::CheckpointOptions, max_variants_per_stage);

    auto star = nb::class_<sfm::StarInitConfig>(module, "StarInitOptions")
                    .def(nb::init<>());
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, min_views);
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, max_views);
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, min_tracks_per_view);
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, max_reproj_error);
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, min_angle_deg);
    PHOTARA_BIND_RW(star, sfm::StarInitConfig, min_initial_tracks);

    auto resection =
        nb::class_<sfm::ResectionConfig>(module, "ResectionOptions")
            .def(nb::init<>());
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_correspondences);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_inliers);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, max_local_window);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, local_ba_every);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, max_pose_wave);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, full_ba_every);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, periodic_full_ba_probe_iterations);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, periodic_full_ba_tail_window);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, periodic_full_ba_tail_relative_improvement);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, final_ba_additional_iterations);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, final_ba_tail_window);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, final_ba_tail_relative_improvement);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_force_full_ba_samples);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_force_full_ba_interval);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, ratio_correspondences);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, avg_inliers_ratio_force_ba);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_inlier_ratio);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, inlier_grid_size);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_inlier_grid_cells);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, coverage_bypass_min_inliers);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, coverage_bypass_inlier_ratio);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, consistency_bypass_inlier_ratio);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_consistency_pair_weight);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_rotation_consistency_neighbors);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, max_median_rotation_error_deg);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_translation_consistency_neighbors);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, max_median_translation_error_deg);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, max_reproj_error);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, min_angle_deg);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, mult_depth_near);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, mult_depth_far);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, use_pair_match_correspondences);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, ransac);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, local_ba);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, full_ba);
    PHOTARA_BIND_RW(resection, sfm::ResectionConfig, checkpoint_interval);

    auto cluster = nb::class_<sfm::ClusterConfig>(module, "ClusterOptions")
                       .def(nb::init<>());
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, max_views_per_cluster);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, min_views_per_cluster);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, max_over_capacity);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, min_common_tracks);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, min_pair_weight);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, refine_weak_edges);
    PHOTARA_BIND_RW(cluster, sfm::ClusterConfig, edge_weight_percentile);

    auto alignment = nb::class_<sfm::GlobalAlignmentConfig>(
                         module, "GlobalAlignmentOptions")
                         .def(nb::init<>());
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, min_pair_weight);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, min_common_tracks);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, merge_track_inliers_only);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, ransac_relative_threshold);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, minimum_inlier_ratio);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, merge_proximity_relative_threshold);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, ransac_iterations);
    PHOTARA_BIND_RW(alignment, sfm::GlobalAlignmentConfig, random_seed);

    auto hierarchical = nb::class_<sfm::HierarchicalConfig>(
                            module, "HierarchicalOptions")
                            .def(nb::init<>());
    PHOTARA_BIND_RW(hierarchical, sfm::HierarchicalConfig, cluster);
    PHOTARA_BIND_RW(hierarchical, sfm::HierarchicalConfig, alignment);
    PHOTARA_BIND_RW(hierarchical, sfm::HierarchicalConfig, star);
    PHOTARA_BIND_RW(hierarchical, sfm::HierarchicalConfig, resection);
    PHOTARA_BIND_RW(hierarchical, sfm::HierarchicalConfig, final_bundle_adjustment);

    auto global_rotation = nb::class_<sfm::GlobalRotationOptions>(
                               module, "GlobalRotationOptions")
                               .def(nb::init<>());
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, max_l1_iterations);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, max_irls_iterations);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, step_convergence_threshold);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, irls_sigma_deg);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, max_relative_rotation_error_deg);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, use_pair_weights);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, reject_planar_pairs);
    PHOTARA_BIND_RW(global_rotation, sfm::GlobalRotationOptions, weight_type);

    auto global_positioning = nb::class_<sfm::GlobalPositioningOptions>(
                                  module, "GlobalPositioningOptions")
                                  .def(nb::init<>());
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, prefer_cuda);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, backend);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, min_views_per_track);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, min_tracks_for_positioning);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, tracks_per_registered_image);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, max_tracks_for_positioning);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, coverage_grid_size);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, max_irls_iterations);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, irls_inner_iterations);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, irls_tuning_constant);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, irls_min_weight);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, irls_quarantine_after);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, irls_weight_convergence);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, max_num_iterations);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, max_solver_time_sec);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, function_tolerance);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, huber_threshold);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, random_seed);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, generate_random_positions);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, generate_random_points);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, generate_scales);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, optimize_positions);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, optimize_points);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, optimize_scales);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, ray_initialize_points);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, camera_warm_start_first);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, camera_warm_start_min_images);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, constraint);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, constraint_reweight_scale);
    PHOTARA_BIND_RW(global_positioning, sfm::GlobalPositioningOptions, regularize_complete_orbits);

    auto asfm_options = nb::class_<sfm::AsfmOptions>(module, "AsfmOptions")
                            .def(nb::init<>());
    PHOTARA_BIND_RW(asfm_options, sfm::AsfmOptions, path_base);

    nb::class_<sfm::FrontEndOptions>(module, "FrontEndOptions")
        .def(nb::init<>())
        .def_rw("camera_model", &sfm::FrontEndOptions::camera_model)
        .def_rw("focal_pixels", &sfm::FrontEndOptions::focal_pixels)
        .def_rw(
            "trust_focal_pixels",
            &sfm::FrontEndOptions::trust_focal_pixels)
        .def_rw(
            "neighbor_window", &sfm::FrontEndOptions::neighbor_window)
        .def_rw("thread_count", &sfm::FrontEndOptions::thread_count)
        .def_rw("mask_dir", &sfm::FrontEndOptions::mask_dir)
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
            "hybrid_lightglue_max_features",
            &sfm::FrontEndOptions::hybrid_lightglue_max_features)
        .def_rw(
            "lightglue_use_cuda",
            &sfm::FrontEndOptions::lightglue_use_cuda)
        .def_rw("relative", &sfm::FrontEndOptions::relative)
        .def_rw("min_pair_weight", &sfm::FrontEndOptions::min_pair_weight)
        .def_rw("pair_weighting", &sfm::FrontEndOptions::pair_weighting)
        .def_rw(
            "retrieval_min_images",
            &sfm::FrontEndOptions::retrieval_min_images)
        .def_rw(
            "augment_sequential_with_retrieval",
            &sfm::FrontEndOptions::augment_sequential_with_retrieval)
        .def_rw(
            "progressive_pair_expansion",
            &sfm::FrontEndOptions::progressive_pair_expansion)
        .def_rw(
            "structural_pair_expansion",
            &sfm::FrontEndOptions::structural_pair_expansion)
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
            "progressive_max_images",
            &sfm::FrontEndOptions::progressive_max_images)
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
            &sfm::FrontEndOptions::progressive_rescue_max_features)
        .def_rw(
            "compress_descriptors_u8",
            &sfm::FrontEndOptions::compress_descriptors_u8)
        .def_rw("retrieval", &sfm::FrontEndOptions::retrieval)
        .def_rw("checkpoint", &sfm::FrontEndOptions::checkpoint);

    nb::class_<sfm::ReconstructionConfig>(module, "SfmOptions")
        .def(nb::init<>())
        .def_rw("mode", &sfm::ReconstructionConfig::mode)
        .def_rw("frontend", &sfm::ReconstructionConfig::frontend)
        .def_rw("star", &sfm::ReconstructionConfig::star)
        .def_rw("resection", &sfm::ReconstructionConfig::resection)
        .def_rw("hierarchical", &sfm::ReconstructionConfig::hierarchical)
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
                incremental_hierarchical_rescue_min_missing_ratio)
        .def_rw(
            "minimum_final_observations_per_image",
            &sfm::ReconstructionConfig::minimum_final_observations_per_image)
        .def_rw(
            "maximum_final_reprojection_error_pixels",
            &sfm::ReconstructionConfig::maximum_final_reprojection_error_pixels)
        .def_rw(
            "global_rotation", &sfm::ReconstructionConfig::global_rotation)
        .def_rw(
            "global_positioning",
            &sfm::ReconstructionConfig::global_positioning);

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

    nb::class_<sfm::FrontEndTiming>(module, "FrontEndTiming")
        .def_ro("threads_used", &sfm::FrontEndTiming::threads_used)
        .def_ro("extract_seconds", &sfm::FrontEndTiming::extract_seconds)
        .def_ro(
            "match_verify_seconds",
            &sfm::FrontEndTiming::match_verify_seconds)
        .def_ro("tracks_seconds", &sfm::FrontEndTiming::tracks_seconds);

    nb::class_<sfm::GlobalRotationSummary>(module, "GlobalRotationSummary")
        .def_ro("success", &sfm::GlobalRotationSummary::success)
        .def_ro("estimated_images", &sfm::GlobalRotationSummary::estimated_images)
        .def_ro("used_pairs", &sfm::GlobalRotationSummary::used_pairs)
        .def_ro("filtered_pairs", &sfm::GlobalRotationSummary::filtered_pairs)
        .def_ro("iterations", &sfm::GlobalRotationSummary::iterations)
        .def_ro("fixed_image", &sfm::GlobalRotationSummary::fixed_image);

    nb::class_<sfm::GlobalPositioningSummary>(
        module, "GlobalPositioningSummary")
        .def_ro("success", &sfm::GlobalPositioningSummary::success)
        .def_ro("positioned_images", &sfm::GlobalPositioningSummary::positioned_images)
        .def_ro("positioned_tracks", &sfm::GlobalPositioningSummary::positioned_tracks)
        .def_ro("observations", &sfm::GlobalPositioningSummary::observations)
        .def_ro("iterations", &sfm::GlobalPositioningSummary::iterations)
        .def_ro("irls_iterations", &sfm::GlobalPositioningSummary::irls_iterations)
        .def_ro("downweighted_constraints", &sfm::GlobalPositioningSummary::downweighted_constraints)
        .def_ro("quarantined_constraints", &sfm::GlobalPositioningSummary::quarantined_constraints)
        .def_ro("final_residual", &sfm::GlobalPositioningSummary::final_residual)
        .def_ro("median_residual", &sfm::GlobalPositioningSummary::median_residual)
        .def_ro("p90_residual", &sfm::GlobalPositioningSummary::p90_residual);

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
        .def_rw(
            "grazing_weight_floor",
            &mvs::DensifyOptions::grazing_weight_floor)
        .def_rw(
            "mesh_smooth_iters",
            &mvs::DensifyOptions::mesh_smooth_iters)
        .def_rw(
            "mesh_smooth_lambda",
            &mvs::DensifyOptions::mesh_smooth_lambda)
        .def_rw(
            "mesh_dist_insert_px",
            &mvs::DensifyOptions::mesh_dist_insert_px)
        .def_rw("mesh_k_sigma", &mvs::DensifyOptions::mesh_k_sigma)
        .def_rw(
            "mesh_adaptive_sigma",
            &mvs::DensifyOptions::mesh_adaptive_sigma)
        .def_rw("mesh_k_qual", &mvs::DensifyOptions::mesh_k_qual)
        .def_rw("mesh_k_behind", &mvs::DensifyOptions::mesh_k_behind)
        .def_rw(
            "mesh_use_free_space_support",
            &mvs::DensifyOptions::mesh_use_free_space_support)
        .def_rw(
            "mesh_k_free_space_front",
            &mvs::DensifyOptions::mesh_k_free_space_front)
        .def_rw(
            "mesh_k_free_space_back",
            &mvs::DensifyOptions::mesh_k_free_space_back)
        .def_rw(
            "mesh_k_free_space_rel",
            &mvs::DensifyOptions::mesh_k_free_space_rel)
        .def_rw(
            "mesh_k_free_space_abs",
            &mvs::DensifyOptions::mesh_k_free_space_abs)
        .def_rw(
            "mesh_k_free_space_outlier",
            &mvs::DensifyOptions::mesh_k_free_space_outlier)
        .def_rw(
            "mesh_k_free_space_calibration_quantile",
            &mvs::DensifyOptions::mesh_k_free_space_calibration_quantile)
        .def_rw("mesh_k_inf", &mvs::DensifyOptions::mesh_k_inf)
        .def_rw(
            "mesh_weld_pixel_fraction",
            &mvs::DensifyOptions::mesh_weld_pixel_fraction)
        .def_rw(
            "mesh_max_edge_scale",
            &mvs::DensifyOptions::mesh_max_edge_scale)
        .def_rw(
            "mesh_depth_diff_threshold",
            &mvs::DensifyOptions::mesh_depth_diff_threshold)
        .def_rw(
            "mesh_min_component_faces",
            &mvs::DensifyOptions::mesh_min_component_faces)
        .def_rw(
            "mesh_spurious_factor",
            &mvs::DensifyOptions::mesh_spurious_factor)
        .def_rw(
            "mesh_remove_spikes",
            &mvs::DensifyOptions::mesh_remove_spikes)
        .def_rw(
            "patchmatch_tile_rows",
            &mvs::DensifyOptions::patchmatch_tile_rows)
        .def_rw(
            "patchmatch_concurrent_views",
            &mvs::DensifyOptions::patchmatch_concurrent_views)
        .def_rw("thread_count", &mvs::DensifyOptions::thread_count);

    nb::class_<splat::TrainingOptions>(module, "TrainingOptions")
        .def(nb::init<>())
        .def_rw("backend", &splat::TrainingOptions::backend)
        .def_rw("iterations", &splat::TrainingOptions::iterations)
        .def_rw("sh_degree", &splat::TrainingOptions::sh_degree)
        .def_rw(
            "sh_degree_interval",
            &splat::TrainingOptions::sh_degree_interval)
        .def_rw("seed", &splat::TrainingOptions::seed)
        .def_rw("log_interval", &splat::TrainingOptions::log_interval)
        .def_rw("profile_cuda", &splat::TrainingOptions::profile_cuda)
        .def_rw("fuse_sh_adam", &splat::TrainingOptions::fuse_sh_adam)
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
            "training_prefetch_adaptive",
            &splat::TrainingOptions::training_prefetch_adaptive)
        .def_rw(
            "training_device_cache_max_bytes",
            &splat::TrainingOptions::training_device_cache_max_bytes)
        .def_rw(
            "training_async_upload",
            &splat::TrainingOptions::training_async_upload)
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
            &splat::TrainingOptions::normal_features_lr)
        .def_rw("initial_point_budget", &splat::TrainingOptions::initial_point_budget)
        .def_rw("refine_stop_num_iter", &splat::TrainingOptions::refine_stop_num_iter)
        .def_rw("densify_score_power", &splat::TrainingOptions::densify_score_power)
        .def_rw("densify_loss_map_power", &splat::TrainingOptions::densify_loss_map_power)
        .def_rw("densify_growth_factor", &splat::TrainingOptions::densify_growth_factor)
        .def_rw("shape_scale_reg", &splat::TrainingOptions::shape_scale_reg)
        .def_rw("shape_scale_reg_decay_power", &splat::TrainingOptions::shape_scale_reg_decay_power)
        .def_rw("shape_erank_reg", &splat::TrainingOptions::shape_erank_reg)
        .def_rw("shape_erank_s3_reg", &splat::TrainingOptions::shape_erank_s3_reg)
        .def_rw("shape_quat_norm_reg", &splat::TrainingOptions::shape_quat_norm_reg)
        .def_rw("oversize_screen_limit", &splat::TrainingOptions::oversize_screen_limit)
        .def_rw("oversize_penalty", &splat::TrainingOptions::oversize_penalty)
        .def_rw("oversize_clip_hardness", &splat::TrainingOptions::oversize_clip_hardness)
        .def_rw("densify_oversize_split_fraction", &splat::TrainingOptions::densify_oversize_split_fraction)
        .def_rw("densify_oversize_score_blend", &splat::TrainingOptions::densify_oversize_score_blend)
        .def_rw("densify_clip_screen_size", &splat::TrainingOptions::densify_clip_screen_size)
        .def_rw("densify_screen_clip_hardness", &splat::TrainingOptions::densify_screen_clip_hardness)
        .def_rw("densify_revised_noise", &splat::TrainingOptions::densify_revised_noise)
        .def_rw("densify_relocate", &splat::TrainingOptions::densify_relocate)
        .def_rw("densify_keep_parent_adam", &splat::TrainingOptions::densify_keep_parent_adam)
        .def_rw("dense_recycle_fraction", &splat::TrainingOptions::dense_recycle_fraction)
        .def_rw("dense_growth_fraction", &splat::TrainingOptions::dense_growth_fraction)
        .def_rw("mean_noise_weight", &splat::TrainingOptions::mean_noise_weight)
        .def_rw("initial_opacity", &splat::TrainingOptions::initial_opacity)
        .def_rw("initial_scale", &splat::TrainingOptions::initial_scale)
        .def_rw("dense_structure_freeze_iter", &splat::TrainingOptions::dense_structure_freeze_iter)
        .def_rw("structure_freeze_iter", &splat::TrainingOptions::structure_freeze_iter)
        .def_rw("sh_regularization_weight", &splat::TrainingOptions::sh_regularization_weight)
        .def_rw("opacity_regularization_weight", &splat::TrainingOptions::opacity_regularization_weight)
        .def_rw("log_scale_regularization_weight", &splat::TrainingOptions::log_scale_regularization_weight)
        .def_rw("beta1", &splat::TrainingOptions::beta1)
        .def_rw("beta2", &splat::TrainingOptions::beta2)
        .def_rw("adam_epsilon", &splat::TrainingOptions::adam_epsilon)
        .def_rw("bilateral_grid_shared", &splat::TrainingOptions::bilateral_grid_shared)
        .def_rw("bilateral_grid_deviation_limit", &splat::TrainingOptions::bilateral_grid_deviation_limit)
        .def_rw("bilateral_grid_identity_projection", &splat::TrainingOptions::bilateral_grid_identity_projection)
        .def_rw("ppisp_identity_projection", &splat::TrainingOptions::ppisp_identity_projection)
        .def_rw("ppisp_exposure_limit", &splat::TrainingOptions::ppisp_exposure_limit)
        .def_rw("ppisp_color_limit", &splat::TrainingOptions::ppisp_color_limit)
        .def_rw("ppisp_reg_vig_center", &splat::TrainingOptions::ppisp_reg_vig_center)
        .def_rw("ppisp_reg_vig_non_pos", &splat::TrainingOptions::ppisp_reg_vig_non_pos)
        .def_rw("ppisp_reg_vig_channel_var", &splat::TrainingOptions::ppisp_reg_vig_channel_var)
        .def_rw("ppisp_reg_crf_channel_var", &splat::TrainingOptions::ppisp_reg_crf_channel_var)
        .def_rw("depth_weight", &splat::TrainingOptions::depth_weight)
        .def_rw("normal_weight", &splat::TrainingOptions::normal_weight)
        .def_rw("filter_3d_update_interval", &splat::TrainingOptions::filter_3d_update_interval)
        .def_rw("multi_view_depth_bracket", &splat::TrainingOptions::multi_view_depth_bracket)
        .def_rw("multi_view_depth_tolerance", &splat::TrainingOptions::multi_view_depth_tolerance)
        .def_rw("multi_view_max_angle", &splat::TrainingOptions::multi_view_max_angle)
        .def_rw("multi_view_min_distance", &splat::TrainingOptions::multi_view_min_distance)
        .def_rw("multi_view_max_distance", &splat::TrainingOptions::multi_view_max_distance)
        .def_rw("multi_view_robust_ncc", &splat::TrainingOptions::multi_view_robust_ncc)
        .def_rw("multi_view_ncc_lambda_reference", &splat::TrainingOptions::multi_view_ncc_lambda_reference)
        .def_rw("multi_view_ncc_sharpness", &splat::TrainingOptions::multi_view_ncc_sharpness)
        .def_rw("multi_view_ncc_min_weight", &splat::TrainingOptions::multi_view_ncc_min_weight)
        .def_rw("geometry_epsilon", &splat::TrainingOptions::geometry_epsilon)
        .def_rw("kernel_size", &splat::TrainingOptions::kernel_size)
        .def_rw("scale_modifier", &splat::TrainingOptions::scale_modifier)
        .def_rw("use_mvs_depth", &splat::TrainingOptions::use_mvs_depth)
        .def_rw("use_mvs_normals", &splat::TrainingOptions::use_mvs_normals)
        .def_rw("ignore_undistortion_border", &splat::TrainingOptions::ignore_undistortion_border)
        .def_rw("adaptive_training_cache", &splat::TrainingOptions::adaptive_training_cache)
        .def_rw("training_device_cache_bytes", &splat::TrainingOptions::training_device_cache_bytes)
        .def_rw("evaluation_iterations", &splat::TrainingOptions::evaluation_iterations)
        .def_rw("preview_interval", &splat::TrainingOptions::preview_interval)
        .def_rw("preview_view_index", &splat::TrainingOptions::preview_view_index)
        .def_rw("preview_view_file", &splat::TrainingOptions::preview_view_file)
        .def_rw("preview_camera_file", &splat::TrainingOptions::preview_camera_file)
        .def_rw("preview_vis_file", &splat::TrainingOptions::preview_vis_file)
        .def_rw("preview_ack_file", &splat::TrainingOptions::preview_ack_file)
        .def(
            "apply_strategy_defaults",
            [](splat::TrainingOptions& self) {
                splat::apply_strategy_defaults(self);
                return self;
            },
            nb::rv_policy::reference_internal);

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
            "frontend_timing",
            [](const SfmSceneHandle& self) { return self.frontend_timing; })
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
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_asfm",
            [](const SfmSceneHandle& self,
               const std::filesystem::path& path,
               const sfm::AsfmOptions& options) {
                sfm::save_asfm(self.scene, path, options);
            },
            nb::arg("path"), nb::arg("options") = sfm::AsfmOptions{},
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_colmap",
            [](const SfmSceneHandle& self,
               const std::filesystem::path& directory,
               const std::filesystem::path& image_path_base,
               const bool write_points) {
                sfm::save_colmap_text(
                    self.scene, directory, image_path_base, write_points);
            },
            nb::arg("directory"), nb::arg("image_path_base") = "",
            nb::arg("write_points") = true,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "save_sparse_ply",
            [](const SfmSceneHandle& self,
               const std::filesystem::path& path) {
                sfm::save_sparse_ply(self.scene, path);
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
               const std::filesystem::path& path,
               const splat::GaussianFormat format) {
                splat::save_gaussians(self.model, path, format);
            },
            nb::arg("path"),
            nb::arg("format") = splat::GaussianFormat::auto_detect,
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

    module.def("available_extractors", [] {
        features::ensure_builtin_feature_backends();
        return features::list_extractors();
    });
    module.def("available_matchers", [] {
        features::ensure_builtin_feature_backends();
        return features::list_matchers();
    });
    module.def("available_pair_pipelines", [] {
        features::ensure_builtin_feature_backends();
        return features::list_pair_pipelines();
    });
    module.def("available_bundle_backends", [] {
        std::vector<std::string> backends{"cpu"};
#if defined(PHOTARA_HAS_CUDA)
        if (ba::CudaOptimizer::is_available()) backends.emplace_back("cuda");
#endif
#if defined(PHOTARA_HAS_VULKAN_BA)
        if (ba::VulkanOptimizer::is_available()) backends.emplace_back("vulkan");
#endif
        return backends;
    });
    module.def("compiled_training_backends", [] {
        std::vector<std::string> backends{"cuda"};
#if defined(TINYTENSOR_HAS_VULKAN)
        backends.emplace_back("vulkan");
#endif
        return backends;
    });
    module.def(
        "set_bundle_backend", &sfm::set_bundle_backend_preference,
        nb::arg("backend"));
    module.def("bundle_backend", &sfm::bundle_backend_preference);
    module.def(
        "load_sfm", &load_sfm_scene, nb::arg("path"),
        nb::arg("options") = sfm::AsfmOptions{},
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
        nb::arg("options") = sfm::ReconstructionConfig{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "estimate_global_rotations",
        [](SfmSceneHandle& scene, const sfm::GlobalRotationOptions& options) {
            return sfm::estimate_global_rotations(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = sfm::GlobalRotationOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "solve_global_positions",
        [](SfmSceneHandle& scene,
           const sfm::GlobalPositioningOptions& options) {
            return sfm::solve_global_positions(scene.scene, options);
        },
        nb::arg("scene"),
        nb::arg("options") = sfm::GlobalPositioningOptions{},
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "bundle_adjust",
        [](SfmSceneHandle& scene, const sfm::BundleOptions& options) {
            return sfm::run_bundle_adjustment(scene.scene, options);
        },
        nb::arg("scene"), nb::arg("options") = sfm::BundleOptions{},
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
    nb::class_<splat::TrainingProgress>(module, "TrainingProgress")
        .def_ro("iteration", &splat::TrainingProgress::iteration)
        .def_ro(
            "total_iterations",
            &splat::TrainingProgress::total_iterations)
        .def_ro("gaussian_count", &splat::TrainingProgress::gaussian_count)
        .def_ro(
            "rendered_instances",
            &splat::TrainingProgress::rendered_instances)
        .def_ro("view_index", &splat::TrainingProgress::view_index)
        .def_ro("grown_count", &splat::TrainingProgress::grown_count)
        .def_ro("pruned_count", &splat::TrainingProgress::pruned_count)
        .def_ro("loss", &splat::TrainingProgress::loss)
        .def_ro("rgb_loss", &splat::TrainingProgress::rgb_loss)
        .def_ro("alpha_loss", &splat::TrainingProgress::alpha_loss)
        .def_ro("depth_loss", &splat::TrainingProgress::depth_loss)
        .def_ro("normal_loss", &splat::TrainingProgress::normal_loss)
        .def_ro(
            "opacity_gradient_mean",
            &splat::TrainingProgress::opacity_gradient_mean)
        .def_ro(
            "opacity_gradient_positive_fraction",
            &splat::TrainingProgress::opacity_gradient_positive_fraction)
        .def_ro("opacity_mean", &splat::TrainingProgress::opacity_mean)
        .def_ro("milliseconds", &splat::TrainingProgress::milliseconds)
        .def_ro(
            "multi_view_geometry_loss",
            &splat::TrainingProgress::multi_view_geometry_loss)
        .def_ro(
            "multi_view_ncc_loss",
            &splat::TrainingProgress::multi_view_ncc_loss)
        .def_ro(
            "multi_view_geometry_pixels",
            &splat::TrainingProgress::multi_view_geometry_pixels)
        .def_ro(
            "multi_view_ncc_pixels",
            &splat::TrainingProgress::multi_view_ncc_pixels)
        .def_ro(
            "resolution_scale",
            &splat::TrainingProgress::resolution_scale)
        .def_ro("image_width", &splat::TrainingProgress::image_width)
        .def_ro("image_height", &splat::TrainingProgress::image_height)
        .def_ro(
            "active_sh_degree",
            &splat::TrainingProgress::active_sh_degree)
        .def_ro(
            "multi_view_interval",
            &splat::TrainingProgress::multi_view_interval)
        .def_ro(
            "multi_view_depth_consistency",
            &splat::TrainingProgress::multi_view_depth_consistency);

    module.def(
        "load_dataset", &load_dataset, nb::arg("request"),
        nb::call_guard<nb::gil_scoped_release>());
    module.def(
        "apply_strategy_defaults",
        [](splat::TrainingOptions& options) {
            splat::apply_strategy_defaults(options);
            return options;
        },
        nb::arg("options"),
        nb::rv_policy::reference_internal);
    module.def(
        "train_3dgs",
        [](const MvsSceneHandle& scene,
           const splat::TrainingOptions& options,
           nb::object progress) {
            auto result = std::make_shared<GaussianModelHandle>();
            if (progress.is_none()) {
                nb::gil_scoped_release release;
                result->model = splat::Trainer(options).train(scene.scene);
            } else {
                result->model = splat::Trainer(options).train(
                    scene.scene, make_progress_callback(progress));
            }
            return result;
        },
        nb::arg("scene"),
        nb::arg("options") = splat::TrainingOptions{},
        nb::arg("progress") = nb::none());
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

#undef PHOTARA_BIND_RW
}
