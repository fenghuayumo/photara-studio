#include "checkpoint.hpp"
#include "splat/trainer.hpp"
#include "project/archive.hpp"
#include "sfm/checkpoint.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <memory>
#include <cctype>

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
constexpr std::uint32_t version = 2;
// Stable training-state ABI, retaining the initial v2 implementation identity.
// Storage/UI refactors must not invalidate saved optimizer state. Change this
// only when training semantics are incompatible; bump the container schema when
// the persistent state layout changes.
constexpr auto training_state_abi = "93595752154f44c098e54d8a28fb68b8420e79f2551810d4abf9a739b9b7cc90";
using project::Archive;
using project::ChunkType;
using detail::CheckpointReader;
using detail::CheckpointWriter;
std::filesystem::path previous_path(std::filesystem::path path) { path += ".previous"; return path; }
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
struct CheckpointRecord {
    std::vector<std::uint8_t> metadata;
    std::vector<std::uint8_t> payload;
};
CheckpointRecord read_record(const Archive& archive, bool previous = false) {
    const auto info_type = previous ? ChunkType::training_checkpoint_previous_info : ChunkType::training_checkpoint_info;
    const auto data_type = previous ? ChunkType::training_checkpoint_previous : ChunkType::training_checkpoint;
    if (archive.has(info_type)) {
        auto metadata = archive.chunk(info_type); decode_info(metadata);
        return {std::move(metadata), archive.chunk(data_type)};
    }
    if (previous) throw std::runtime_error("No previous checkpoint in this project");
    // Original sidecar format or the nested sidecar embedded by older Studio.
    if (archive.has(ChunkType::settings)) {
        try {
            auto metadata = archive.chunk(ChunkType::settings); decode_info(metadata);
            return {std::move(metadata), archive.chunk(ChunkType::training_checkpoint)};
        } catch (...) {}
    }
    return read_record(Archive::from_bytes(archive.chunk(ChunkType::training_checkpoint)));
}
TrainingCheckpointInfo read_info(const Archive& archive, bool previous = false) {
    const auto info_type = previous ? ChunkType::training_checkpoint_previous_info : ChunkType::training_checkpoint_info;
    if (archive.has(info_type)) return decode_info(archive.chunk(info_type));
    return decode_info(read_record(archive, previous).metadata);
}
void set_record(Archive& archive, CheckpointRecord record, bool previous = false) {
    archive.set_chunk(previous ? ChunkType::training_checkpoint_previous_info : ChunkType::training_checkpoint_info,
        std::move(record.metadata));
    archive.set_chunk(previous ? ChunkType::training_checkpoint_previous : ChunkType::training_checkpoint,
        std::move(record.payload));
}
Archive open_or_create_project(const std::filesystem::path& path) {
    return std::filesystem::exists(path) ? Archive::open(path) : Archive::create();
}
void write_tensor(CheckpointWriter& writer, const tinytensor::Tensor& tensor) {
    writer.pod(std::uint32_t(tensor.is_valid() ? tensor.shape().rank() : 0));
    if (!tensor.is_valid()) return;
    writer.pod(std::uint32_t(tensor.dtype()));
    for (std::size_t i = 0; i < tensor.shape().rank(); ++i)
        writer.pod(std::uint64_t(tensor.shape()[i]));
    const auto host = tensor.contiguous().cpu();
    writer.raw(host.data_ptr(), host.bytes());
}
tinytensor::Tensor read_tensor(CheckpointReader& reader, tinytensor::Device device) {
    const auto rank = reader.pod<std::uint32_t>();
    if (!rank) return {};
    if (rank > 8) throw std::runtime_error("Invalid checkpoint tensor rank");
    const auto type = reader.pod<std::uint32_t>();
    if (type > static_cast<std::uint32_t>(tinytensor::DataType::Float32))
        throw std::runtime_error("Invalid checkpoint tensor type");
    const auto dtype = static_cast<tinytensor::DataType>(type);
    std::vector<std::size_t> shape(rank);
    std::uint64_t elements = 1;
    for (auto& dim : shape) {
        dim = reader.pod<std::uint64_t>();
        if (dim > 1'000'000'000 || (dim && elements > 1'000'000'000 / dim))
            throw std::runtime_error("Invalid checkpoint tensor dimensions");
        elements *= dim;
    }
    auto bytes = reader.take(elements * tinytensor::dtype_size(dtype));
    auto host = tinytensor::Tensor::empty_unpinned(tinytensor::TensorShape(shape), dtype);
    if (!bytes.empty()) std::memcpy(host.data_ptr(), bytes.data(), bytes.size());
    return host.to(device);
}
}
std::filesystem::path training_checkpoint_path(const std::filesystem::path& project) {
    if (project.empty()) return {};
    auto native = project;
    std::string extension = native.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (extension != ".ascan") native.replace_extension(".ascan");
    return native;
}
void migrate_training_checkpoints(const std::filesystem::path& project) {
    if (project.empty()) return;
    auto visible = project; visible += ".training.ascan";
    const auto hidden = project.parent_path() / ".photara-recovery" / project.filename() / "training.ascan";
    std::vector<std::filesystem::path> sources;
    for (const auto& path : {hidden, previous_path(hidden), visible, previous_path(visible)})
        if (std::filesystem::exists(path)) sources.push_back(path);
    const auto native = training_checkpoint_path(project);
    const auto legacy_directory = hidden.parent_path() / "legacy";
    if (std::filesystem::is_directory(legacy_directory)) {
        for (const auto& item : std::filesystem::directory_iterator(legacy_directory))
            if (item.is_regular_file() && item.path().filename().native().starts_with(visible.filename().native()))
                sources.push_back(item.path());
    }
    auto archive = open_or_create_project(native);
    const bool nested = archive.has(ChunkType::training_checkpoint) && !archive.has(ChunkType::training_checkpoint_info);
    if (sources.empty() && !nested) return;
    detail::TrainingCheckpointLease project_lease(native);
    std::vector<std::unique_ptr<detail::TrainingCheckpointLease>> legacy_leases;
    if (!sources.empty()) {
        legacy_leases.push_back(std::make_unique<detail::TrainingCheckpointLease>(hidden));
        legacy_leases.push_back(std::make_unique<detail::TrainingCheckpointLease>(visible));
    }
    auto history = archive.has(ChunkType::training_checkpoint_history)
        ? Archive::from_bytes(archive.chunk(ChunkType::training_checkpoint_history)) : Archive::create();
    unsigned history_id = 1000;
    const auto retain_history = [&](CheckpointRecord record) {
        while (history.has(static_cast<ChunkType>(history_id))) ++history_id;
        auto portable = Archive::create();
        portable.set_chunk(ChunkType::settings, std::move(record.metadata));
        portable.set_chunk(ChunkType::training_checkpoint, std::move(record.payload));
        history.set_chunk(static_cast<ChunkType>(history_id++), portable.encode());
    };
    std::optional<CheckpointRecord> latest, previous;
    try { latest = read_record(archive); } catch (...) {}
    try { previous = read_record(archive, true); } catch (...) {}
    for (const auto& source : sources) {
        auto record = read_record(Archive::open(source)); // Check checksums before any cleanup.
        const auto info = decode_info(record.metadata);
        if (!latest) latest = std::move(record);
        else {
            const auto current = decode_info(latest->metadata);
            if (info.fingerprint != current.fingerprint) { retain_history(std::move(record)); continue; }
            if (info.iteration > current.iteration) {
                previous = std::move(latest); latest = std::move(record);
            } else if (info.iteration < current.iteration &&
                (!previous || info.iteration > decode_info(previous->metadata).iteration))
                previous = std::move(record);
        }
    }
    if (!latest) throw std::runtime_error("No valid checkpoint to merge into the project");
    set_record(archive, std::move(*latest));
    if (previous) set_record(archive, std::move(*previous), true);
    if (history_id > 1000) archive.set_chunk(ChunkType::training_checkpoint_history, history.encode());
    archive.save(native); sync_checkpoint(native);
    // Read the published generations back before removing the old copies.
    const auto verified = Archive::open(native); (void)read_record(verified);
    if (previous) (void)read_record(verified, true);
    for (const auto& source : sources) std::filesystem::remove(source);
    legacy_leases.clear(); // Release/delete old lock files before empty-directory cleanup.
    // Remove only empty directories belonging to this project. Other projects,
    // conflicting archived generations and unrelated files are never removed.
    std::error_code ignored;
    std::filesystem::remove(legacy_directory, ignored);
    std::filesystem::remove(hidden.parent_path(), ignored);
    std::filesystem::remove(hidden.parent_path().parent_path(), ignored);
}
bool training_checkpoint_in_use(const std::filesystem::path& path) {
    if (path.empty()) return false;
    try { detail::TrainingCheckpointLease lease(path); return false; }
    catch (...) { return true; }
}
std::optional<TrainingCheckpointInfo> inspect_training_checkpoint(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path)) return std::nullopt;
    try {
        const auto archive = Archive::open(path);
        for (bool previous : {false, true}) {
            try { return read_info(archive, previous); } catch (...) {}
        }
    } catch (...) {}
    return std::nullopt;
}
std::vector<std::uint8_t> training_checkpoint_bytes(const std::filesystem::path& path) {
    const auto source = Archive::open(path);
    for (bool previous : {false, true}) {
        try {
            auto record = read_record(source, previous);
            auto portable = Archive::create();
            portable.set_chunk(ChunkType::settings, std::move(record.metadata));
            portable.set_chunk(ChunkType::training_checkpoint, std::move(record.payload));
            return portable.encode();
        } catch (...) {}
    }
    throw std::runtime_error("No valid training checkpoint to save");
}
void restore_training_checkpoint(std::span<const std::uint8_t> bytes, const std::filesystem::path& path) {
    detail::TrainingCheckpointLease lease(path);
    auto record = read_record(Archive::from_bytes(bytes));
    auto archive = open_or_create_project(path);
    set_record(archive, std::move(record));
    archive.save(path); sync_checkpoint(path);
}
void remove_training_checkpoint(const std::filesystem::path& path) {
    if (path.empty() || !std::filesystem::exists(path)) return;
    detail::TrainingCheckpointLease lease(path);
    auto archive = Archive::open(path);
    bool modified = false;
    for (auto type : {ChunkType::training_checkpoint, ChunkType::training_checkpoint_info,
                     ChunkType::training_checkpoint_previous, ChunkType::training_checkpoint_previous_info, ChunkType::training_checkpoint_history}) {
        modified |= archive.has(type); archive.erase_chunk(type);
    }
    if (modified) { archive.save(path); sync_checkpoint(path); }
}
namespace detail {
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
    hash.append_string("photara-training-checkpoint-v2");
    hash.append_string(training_state_abi);
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
    hash.append(options.shape_scale_reg);
    hash.append(options.shape_scale_reg_decay_power);
    hash.append(options.shape_erank_reg);
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
    hash.append(options.log_scale_regularization_weight);
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
    hash.append(options.multi_view_interval);
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
    hash.append(options.max_scale_ratio);
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
TrainingCheckpoint load_training_checkpoint(const std::filesystem::path& path,
    std::uint64_t fingerprint, tinytensor::Device device) {
    std::string failure;
    const auto archive = Archive::open(path);
    for (bool previous : {false, true}) {
        try {
            auto record = read_record(archive, previous);
            TrainingCheckpoint result; result.info = decode_info(record.metadata);
            if (result.info.fingerprint != fingerprint)
                throw std::runtime_error("Training implementation, settings, cameras or input images changed. Choose Train 3DGS to train with the new inputs.");
            CheckpointReader reader(record.payload);
            auto runtime = reader.take(reader.pod<std::uint64_t>());
            result.runtime.assign(runtime.begin(), runtime.end());
            const auto count = reader.pod<std::uint32_t>();
            if (count > 128) throw std::runtime_error("Invalid checkpoint tensor count");
            for (unsigned i = 0; i < count; ++i) result.tensors.push_back(read_tensor(reader, device));
            reader.finish(); return result;
        } catch (const std::exception& error) { if (failure.empty()) failure = error.what(); }
    }
    throw std::runtime_error("Cannot resume training: " + failure);
}
void save_training_checkpoint(const std::filesystem::path& path,
    const TrainingCheckpointInfo& info, const std::vector<std::uint8_t>& runtime,
    const std::vector<tinytensor::Tensor*>& tensors,
    std::span<const std::uint8_t> project_settings,
    std::span<const std::uint8_t> project_sfm) {
    ensure_checkpoint_directory(path);
    CheckpointWriter metadata;
    metadata.pod(version); metadata.pod(info.iteration); metadata.pod(info.total_iterations);
    metadata.pod(info.gaussian_count); metadata.pod(info.fingerprint);
    metadata.pod(info.focus_bounds.valid);
    metadata.raw(info.focus_bounds.center.data(), sizeof(float) * 3);
    metadata.raw(info.focus_bounds.half_extent.data(), sizeof(float) * 3);
    metadata.raw(info.focus_bounds.axes.data(), sizeof(float) * 9);
    CheckpointWriter payload;
    payload.pod(std::uint64_t(runtime.size())); payload.raw(runtime.data(), runtime.size());
    payload.pod(std::uint32_t(tensors.size()));
    for (const auto* tensor : tensors) write_tensor(payload, *tensor);
    auto archive = open_or_create_project(path);
    try { set_record(archive, read_record(archive), true); }
    catch (...) {
        // If the current generation is corrupt, keep the existing previous
        // complete generation rather than copying damaged data over it.
    }
    set_record(archive, {std::move(metadata.bytes), std::move(payload.bytes)});
    if (!project_settings.empty()) archive.set_chunk(ChunkType::settings, {project_settings.begin(), project_settings.end()});
    if (!project_sfm.empty()) archive.set_chunk(ChunkType::sfm, {project_sfm.begin(), project_sfm.end()});
    archive.save(path); sync_checkpoint(path);
}

} // namespace detail
} // namespace photara::splat
