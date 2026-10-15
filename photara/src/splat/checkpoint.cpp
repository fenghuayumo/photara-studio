#include "checkpoint.hpp"
#include "splat/trainer.hpp"
#include "splat/formats.hpp"
#include "project/archive.hpp"
#include "sfm/checkpoint.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cmath>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace photara::splat {
namespace {
constexpr std::uint32_t version = 1;
using project::Archive;
using project::ChunkType;
using detail::CheckpointReader;
using detail::CheckpointWriter;
std::filesystem::path lock_path(std::filesystem::path path) { path += ".lock"; return path; }
void ensure_checkpoint_directory(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
#ifdef _WIN32
    // Hide only our own recovery root, never a caller's custom directory.
    for (auto directory = path.parent_path(); !directory.empty();) {
        if (directory.filename() == ".photara-recovery") {
            const DWORD attributes = GetFileAttributesW(directory.c_str());
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                !SetFileAttributesW(directory.c_str(), attributes | FILE_ATTRIBUTE_HIDDEN))
                throw std::system_error(GetLastError(), std::system_category(), "Cannot hide project recovery directory");
            break;
        }
        const auto parent = directory.parent_path();
        if (parent == directory) break;
        directory = parent;
    }
#endif
}

void sync_checkpoint(const std::filesystem::path& path) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::system_error(GetLastError(), std::system_category(), "Cannot flush checkpoint");
    const bool ok = FlushFileBuffers(file); const auto error = GetLastError(); CloseHandle(file);
    if (!ok) throw std::system_error(error, std::system_category(), "Cannot flush checkpoint");
#else
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file < 0) throw std::system_error(errno, std::generic_category(), "Cannot flush checkpoint");
    const int result = ::fsync(file); const int error = errno; ::close(file);
    if (result) throw std::system_error(error, std::generic_category(), "Cannot flush checkpoint");
    const auto directory = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int parent = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
    if (parent < 0) throw std::system_error(errno, std::generic_category(), "Cannot flush checkpoint directory");
    const int synced = ::fsync(parent); const int parent_error = errno; ::close(parent);
    if (synced) throw std::system_error(parent_error, std::generic_category(), "Cannot flush checkpoint directory");
#endif
}
TrainingCheckpointInfo decode_info(const std::vector<std::uint8_t>& bytes) {
    CheckpointReader reader(bytes);
    if (reader.pod<std::uint32_t>() != version)
        throw std::runtime_error("Unsupported training checkpoint version");
    TrainingCheckpointInfo info;
    info.iteration = reader.pod<unsigned>();
    info.total_iterations = reader.pod<unsigned>();
    info.gaussian_count = reader.pod<std::uint64_t>();
    info.fingerprint = reader.pod<std::uint64_t>();
    info.focus_bounds.valid = reader.pod<bool>();
    for (auto* data : {info.focus_bounds.center.data(), info.focus_bounds.half_extent.data()}) {
        auto bytes = reader.take(sizeof(float) * 3); std::memcpy(data, bytes.data(), bytes.size());
    }
    { auto bytes = reader.take(sizeof(float) * 9); std::memcpy(info.focus_bounds.axes.data(), bytes.data(), bytes.size()); }
    reader.finish();
    if (info.focus_bounds.valid && (!info.focus_bounds.center.allFinite() ||
        !info.focus_bounds.half_extent.allFinite() || !info.focus_bounds.axes.allFinite() ||
        (info.focus_bounds.half_extent.array() <= 0.F).any()))
        throw std::runtime_error("Invalid checkpoint focus bounds");
    if (!info.total_iterations || info.iteration > info.total_iterations ||
        !info.gaussian_count || info.gaussian_count > 50'000'000)
        throw std::runtime_error("Invalid training checkpoint metadata");
    return info;
}

constexpr std::string_view model_magic = "PHOTARA_PLY_CHECKPOINT\n";
std::optional<TrainingCheckpointInfo> read_model_checkpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string magic(model_magic.size(), '\0');
    if (!input.read(magic.data(), magic.size()) || magic != model_magic) return std::nullopt;
    TrainingCheckpointInfo info;
    std::string model_path;
    unsigned schema{};
    input >> schema >> info.iteration >> info.total_iterations >> info.gaussian_count >> info.fingerprint;
    input >> std::quoted(model_path) >> info.focus_bounds.valid;
    for (auto* values : {info.focus_bounds.center.data(), info.focus_bounds.half_extent.data()})
        for (int i = 0; i < 3; ++i) input >> values[i];
    for (int i = 0; i < 9; ++i) input >> info.focus_bounds.axes.data()[i];
    if (!input || schema != 1 || !info.total_iterations || info.iteration > info.total_iterations ||
        !info.gaussian_count || info.gaussian_count > 50'000'000 || model_path.empty())
        throw std::runtime_error("Invalid PLY checkpoint manifest");
    if (info.focus_bounds.valid && (!info.focus_bounds.center.allFinite() ||
        !info.focus_bounds.half_extent.allFinite() || !info.focus_bounds.axes.allFinite() ||
        (info.focus_bounds.half_extent.array() <= 0.F).any()))
        throw std::runtime_error("Invalid checkpoint focus bounds");
    info.model_path = std::filesystem::u8path(model_path);
    if (info.model_path.is_relative()) info.model_path = path.parent_path() / info.model_path;
    return info;
}

bool managed_checkpoint_model(const std::filesystem::path& checkpoint, const std::filesystem::path& model) {
    return model.parent_path().lexically_normal() == checkpoint.parent_path().lexically_normal() &&
        model.filename().string().starts_with(checkpoint.stem().string() + ".model.") &&
        model.extension() == ".ply";
}

void publish_checkpoint_file(const std::filesystem::path& temporary, const std::filesystem::path& path) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(GetLastError(), std::system_category(), "Cannot publish checkpoint");
#else
    std::filesystem::rename(temporary, path);
#endif
}
}
std::vector<std::uint8_t> training_checkpoint_info_bytes(const TrainingCheckpointInfo& info) {
    CheckpointWriter writer;
    writer.pod(version); writer.pod(info.iteration); writer.pod(info.total_iterations);
    writer.pod(info.gaussian_count); writer.pod(info.fingerprint); writer.pod(info.focus_bounds.valid);
    writer.raw(info.focus_bounds.center.data(), sizeof(float) * 3);
    writer.raw(info.focus_bounds.half_extent.data(), sizeof(float) * 3);
    writer.raw(info.focus_bounds.axes.data(), sizeof(float) * 9);
    return std::move(writer.bytes);
}
bool training_checkpoint_in_use(const std::filesystem::path& path) {
    if (path.empty()) return false;
    try { detail::TrainingCheckpointLease lease(path); return false; }
    catch (...) { return true; }
}
std::optional<TrainingCheckpointInfo> inspect_training_checkpoint(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path)) return std::nullopt;
    try {
        if (auto info = read_model_checkpoint(path)) {
            if (!std::filesystem::is_regular_file(info->model_path)) return std::nullopt;
            return info;
        }
        const auto archive = Archive::open(path);
        if (archive.has(ChunkType::training_model_info) && archive.has(ChunkType::gaussians)) {
            auto info = decode_info(archive.chunk(ChunkType::training_model_info));
            return info;
        }
    } catch (...) {}
    return std::nullopt;
}
std::vector<std::uint8_t> training_checkpoint_bytes(const std::filesystem::path& path) {
    if (auto info = read_model_checkpoint(path)) {
        auto portable = Archive::create();
        portable.set_chunk(ChunkType::training_model_info, training_checkpoint_info_bytes(*info));
        portable.set_chunk(ChunkType::gaussians, encode_gaussians(load_gaussians(info->model_path, tinytensor::Device::CPU)));
        return portable.encode();
    }
    const auto source = Archive::open(path);
    if (source.has(ChunkType::training_model_info) && source.has(ChunkType::gaussians)) {
        auto portable = Archive::create();
        portable.set_chunk(ChunkType::training_model_info, source.chunk(ChunkType::training_model_info));
        portable.set_chunk(ChunkType::gaussians, source.chunk(ChunkType::gaussians));
        return portable.encode();
    }
    throw std::runtime_error("No valid training checkpoint to save");
}

GaussianModel checkpoint_model(const std::filesystem::path& path) {
    if (auto info = read_model_checkpoint(path)) return load_gaussians(info->model_path, tinytensor::Device::CPU);
    const auto archive = Archive::open(path);
    if (archive.has(ChunkType::training_model_info))
        return decode_gaussians(archive.chunk(ChunkType::gaussians), tinytensor::Device::CPU);
    throw std::runtime_error("Unsupported training checkpoint format; start a new training run.");
}
void restore_training_checkpoint(std::span<const std::uint8_t> bytes, const std::filesystem::path& path) {
    detail::TrainingCheckpointLease lease(path);
    const auto portable = Archive::from_bytes(bytes);
    if (portable.has(ChunkType::training_model_info) && portable.has(ChunkType::gaussians)) {
        auto info = decode_info(portable.chunk(ChunkType::training_model_info));
        const auto model = decode_gaussians(portable.chunk(ChunkType::gaussians), tinytensor::Device::CPU);
        detail::save_model_checkpoint(path, info, model);
        return;
    }
    throw std::runtime_error("Unsupported training checkpoint format; start a new training run.");
}
void remove_training_checkpoint(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path)) return;
    detail::TrainingCheckpointLease lease(path);
    if (auto info = read_model_checkpoint(path)) {
        std::filesystem::remove(path);
        if (managed_checkpoint_model(path, info->model_path)) std::filesystem::remove(info->model_path);
        return;
    }
    auto archive = Archive::open(path);
    bool modified = false;
    for (auto type : {ChunkType::training_model_info}) {
        modified |= archive.has(type); archive.erase_chunk(type);
    }
    if (modified) { archive.save(path); sync_checkpoint(path); }
}
namespace detail {
void save_model_checkpoint(const std::filesystem::path& path,
    const TrainingCheckpointInfo& info, const GaussianModel& model) {
    ensure_checkpoint_directory(path);
    const auto previous = inspect_training_checkpoint(path);
    auto model_path = path.parent_path() / (path.stem().string() + ".model." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
    auto temporary = path; temporary += ".tmp";
    bool created = false;
    try {
        save_gaussians(model, model_path, GaussianFormat::ply);
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot write PLY checkpoint manifest");
        created = true;
        const auto filename = model_path.filename().generic_u8string();
        output << model_magic << "1 " << info.iteration << ' ' << info.total_iterations << ' '
            << info.gaussian_count << ' ' << info.fingerprint << '\n'
            << std::quoted(std::string(reinterpret_cast<const char*>(filename.data()), filename.size())) << '\n'
            << std::setprecision(9) << info.focus_bounds.valid;
        for (const auto* values : {info.focus_bounds.center.data(), info.focus_bounds.half_extent.data()})
            for (int i = 0; i < 3; ++i) output << ' ' << values[i];
        for (int i = 0; i < 9; ++i) output << ' ' << info.focus_bounds.axes.data()[i];
        output << '\n'; output.close();
        if (!output) throw std::runtime_error("Cannot finish PLY checkpoint manifest");
        sync_checkpoint(model_path);
        publish_checkpoint_file(temporary, path);
    } catch (...) {
        std::error_code ignored;
        if (created) std::filesystem::remove(temporary, ignored);
        std::filesystem::remove(model_path, ignored);
        throw;
    }
    if (previous && managed_checkpoint_model(path, previous->model_path)) {
        std::error_code ignored; std::filesystem::remove(previous->model_path, ignored);
    }
}

TrainingCheckpointLease::TrainingCheckpointLease(const std::filesystem::path& path) {
    if (path.empty()) return;
    ensure_checkpoint_directory(path);
    const auto lock = lock_path(path);
#ifdef _WIN32
    const HANDLE file = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_HIDDEN | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Training checkpoint is in use by another process or cannot be opened");
    handle_ = reinterpret_cast<std::intptr_t>(file);
#else
    handle_ = ::open(lock.c_str(), O_CREAT | O_RDWR, 0600);
    if (handle_ < 0 || ::flock(static_cast<int>(handle_), LOCK_EX | LOCK_NB)) {
        if (handle_ >= 0) ::close(static_cast<int>(handle_));
        handle_ = -1;
        throw std::runtime_error("Training checkpoint is in use by another process or cannot be opened");
    }
#endif
}
TrainingCheckpointLease::~TrainingCheckpointLease() {
    if (handle_ == -1) return;
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
    ::close(static_cast<int>(handle_));
#endif
}
std::uint64_t training_fingerprint(const mvs::MvsScene& scene, const TrainingOptions& options) {
    sfm::FingerprintBuilder hash;
    hash.append_string("photara-ply-checkpoint-v1");
    hash.append(options.backend);
    hash.append(options.iterations);
    hash.append(options.sh_degree);
    hash.append(options.sh_degree_interval);
    hash.append(options.seed);
    hash.append(options.speedy_pruning);
    hash.append(options.speedy_start);
    hash.append(options.speedy_every);
    hash.append(options.speedy_stop);
    hash.append(options.speedy_score_views);
    hash.append(options.speedy_soft_ratio);
    hash.append(options.speedy_hard_ratio);
    hash.append(options.input_is_dense);
    hash.append(options.enable_densification);
    hash.append(options.densification_strategy);
    hash.append(options.densification_cap);
    hash.append(options.initial_point_budget);
    hash.append(options.refine_start_iter);
    hash.append(options.refine_stop_iter);
    hash.append(options.refine_stop_num_iter);
    hash.append(options.grow_stop_iter);
    hash.append(options.refine_every);
    hash.append(options.opacity_reset_every);
    hash.append(options.densify_gradient_threshold);
    hash.append(options.densify_select_fraction);
    hash.append(options.densify_screen_threshold);
    hash.append(options.densify_score_power);
    hash.append(options.densify_loss_map_power);
    hash.append(options.densify_growth_factor);
    hash.append(options.shape_erank_reg);
    hash.append(options.shape_axis_ratio_reg);
    hash.append(options.shape_max_axis_ratio);
    hash.append(options.shape_erank_s3_reg);
    hash.append(options.shape_quat_norm_reg);
    hash.append(options.oversize_screen_limit);
    hash.append(options.oversize_penalty);
    hash.append(options.oversize_clip_hardness);
    hash.append(options.densify_oversize_split_fraction);
    hash.append(options.densify_oversize_score_blend);
    hash.append(options.densify_clip_screen_size);
    hash.append(options.densify_screen_clip_hardness);
    hash.append(options.densify_revised_noise);
    hash.append(options.densify_relocate);
    hash.append(options.densify_keep_parent_adam);
    hash.append(options.dense_recycle_fraction);
    hash.append(options.dense_growth_fraction);
    hash.append(options.prune_opacity);
    hash.append(options.opacity_decay);
    hash.append(options.scale_decay);
    hash.append(options.mean_noise_weight);
    hash.append(options.background_noise_strength);
    hash.append(options.initial_opacity);
    hash.append(options.initial_scale);
    hash.append(options.initialize_scale_from_knn);
    hash.append(options.dense_structure_freeze_iter);
    hash.append(options.structure_freeze_iter);
    hash.append(options.means_lr);
    hash.append(options.scales_lr);
    hash.append(options.opacities_lr);
    hash.append(options.quaternions_lr);
    hash.append(options.sh0_lr);
    hash.append(options.sh_rest_lr);
    hash.append(options.sh_regularization_weight);
    hash.append(options.opacity_regularization_weight);
    hash.append(options.scale_regularization_weight);
    hash.append(options.beta1);
    hash.append(options.beta2);
    hash.append(options.adam_epsilon);
    hash.append(options.photometric_weight);
    hash.append(options.ssim_weight);
    hash.append(options.use_bilateral_grid);
    hash.append(options.bilateral_grid_shared);
    hash.append(options.bilateral_grid_width);
    hash.append(options.bilateral_grid_height);
    hash.append(options.bilateral_grid_luma);
    hash.append(options.bilateral_grid_lr);
    hash.append(options.bilateral_grid_tv_weight);
    hash.append(options.bilateral_grid_deviation_limit);
    hash.append(options.bilateral_grid_identity_projection);
    hash.append(options.use_ppisp);
    hash.append(options.ppisp_type);
    hash.append(options.ppisp_lr);
    hash.append(options.ppisp_identity_projection);
    hash.append(options.ppisp_exposure_limit);
    hash.append(options.ppisp_color_limit);
    hash.append(options.ppisp_reg_exposure_mean);
    hash.append(options.ppisp_reg_color_mean);
    hash.append(options.ppisp_reg_vig_center);
    hash.append(options.ppisp_reg_vig_non_pos);
    hash.append(options.ppisp_reg_vig_channel_var);
    hash.append(options.ppisp_reg_crf_channel_var);
    hash.append(options.ppisp_clamp_output);
    hash.append(options.ppisp_before_bilagrid);
    hash.append(options.depth_weight);
    if (options.lambda_d > 0.F || options.lambda_n > 0.F) {
        hash.append(options.lambda_d);
        hash.append(options.lambda_n);
    }
    hash.append(options.normal_weight);
    hash.append(options.use_depth_normal_loss);
    hash.append(options.depth_normal_weight);
    hash.append(options.depth_normal_from_iter);
    hash.append(options.use_normal_field);
    hash.append(options.normal_field_weight);
    hash.append(options.normal_field_depth_ratio);
    hash.append(options.normal_field_from_iter);
    hash.append(options.normal_features_lr);
    hash.append(options.filter_3d_update_interval);
    hash.append(options.multi_view_geo_weight);
    hash.append(options.multi_view_ncc_weight);
    hash.append(options.multi_view_depth_bracket);
    hash.append(options.multi_view_depth_tolerance);
    hash.append(options.multi_view_num);
    hash.append(options.multi_view_tail_interval);
    hash.append(options.multi_view_adaptive_frequency);
    hash.append(options.multi_view_adaptive_max_interval);
    hash.append(options.multi_view_adaptive_stable_refinements);
    hash.append(options.multi_view_adaptive_count_threshold);
    hash.append(options.multi_view_adaptive_churn_threshold);
    hash.append(options.multi_view_adaptive_depth_threshold);
    hash.append(options.multi_view_adaptive_min_depth_consistency);
    hash.append(options.multi_view_adaptive_distribution_threshold);
    hash.append(options.multi_view_max_angle);
    hash.append(options.multi_view_min_distance);
    hash.append(options.multi_view_max_distance);
    hash.append(options.multi_view_pixel_noise_threshold);
    hash.append(options.multi_view_robust_ncc);
    hash.append(options.multi_view_ncc_lambda_reference);
    hash.append(options.multi_view_ncc_sharpness);
    hash.append(options.multi_view_ncc_min_weight);
    hash.append(options.use_mask);
    hash.append(options.alpha_mode);
    hash.append(options.match_alpha_weight);
    hash.append(options.minimum_scale_fraction);
    hash.append(options.maximum_scale_fraction);
    hash.append(options.constrain_scale_range);
    hash.append(options.geometry_epsilon);
    hash.append(options.kernel_size);
    hash.append(options.scale_modifier);
    hash.append(options.use_mvs_depth);
    hash.append(options.use_mvs_normals);
    hash.append(options.use_source_resolution);
    hash.append(options.ignore_undistortion_border);
    hash.append(options.undistort_to_pinhole);
    hash.append(options.max_image_dimension);
    hash.append(options.progressive_resolution);
    hash.append(options.progressive_resolution_interval);
    hash.append(options.progressive_initial_scale);
    hash.append(options.evaluation_split_every);
    hash.append(options.use_3d_filter.has_value());
    hash.append(options.use_3d_filter.value_or(false));
    std::vector<std::filesystem::path> paths;
    for (const auto& view : scene.views) {
        paths.push_back(view.path);
        const auto camera = camera_from_mvs_view(view);
        hash.append_bytes(camera.world_to_camera.data(), sizeof(float) * 16);
        hash.append_bytes(camera.position.data(), sizeof(float) * 3);
        hash.append(camera.fx); hash.append(camera.fy); hash.append(camera.cx); hash.append(camera.cy);
        hash.append(camera.width); hash.append(camera.height); hash.append(camera.model);
        hash.append(camera.k1); hash.append(camera.k2); hash.append(camera.k3); hash.append(camera.k4);
        hash.append(view.source_model);
        hash.append(view.src_fx); hash.append(view.src_fy); hash.append(view.src_cx); hash.append(view.src_cy);
        hash.append(view.src_width); hash.append(view.src_height);
        hash.append(view.k1); hash.append(view.k2); hash.append(view.p1); hash.append(view.p2);
        hash.append_bytes(view.foreground_mask.data(), view.foreground_mask.size());
        hash.append_bytes(view.depth_map.depth.data(), view.depth_map.depth.size() * sizeof(float));
        for (const auto& n : view.depth_map.normal) hash.append_bytes(n.data(), sizeof(float) * 3);
    }
    hash.append(sfm::fingerprint_images(paths));
    if (options.lambda_d > 0.F || options.lambda_n > 0.F) {
        paths.clear();
        for (std::size_t i = 0; i < scene.views.size(); ++i) {
            if (options.evaluation_split_every != 0 && i % options.evaluation_split_every == 0) continue;
            paths.push_back(options.prior_directory / ("view_" + std::to_string(scene.views[i].id) + ".prior"));
        }
        hash.append(sfm::fingerprint_images(paths));
    }
    if (options.use_mask && !options.mask_dir.empty()) {
        paths.clear();
        for (const auto& item : std::filesystem::directory_iterator(options.mask_dir))
            if (item.is_regular_file()) paths.push_back(item.path());
        std::sort(paths.begin(), paths.end());
        hash.append(sfm::fingerprint_images(paths));
    }
    for (const auto& point : scene.sparse_points) {
        hash.append_bytes(point.position.data(), sizeof(float) * 3);
        hash.append_bytes(point.color.data(), sizeof(float) * 3);
    }
    for (const auto& point : scene.dense_cloud.points) {
        hash.append_bytes(point.position.data(), sizeof(float) * 3);
        hash.append_bytes(point.color.data(), sizeof(float) * 3);
        hash.append_bytes(point.normal.data(), sizeof(float) * 3);
    }
    return hash.value();
}
} // namespace detail
} // namespace photara::splat
