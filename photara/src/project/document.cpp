#include "project/document.hpp"

#include "io/format_version.hpp"
#include "io/video_frames.hpp"
#include "../io/binary_codec.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>
#include <fstream>
#include <array>

namespace photara::project {
namespace {

using io::binary::BufferReader;
using io::binary::BufferWriter;

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.generic_u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::filesystem::path path_from_utf8(const std::string& text) {
    return std::filesystem::path(
        std::u8string(
            reinterpret_cast<const char8_t*>(text.data()),
            reinterpret_cast<const char8_t*>(text.data() + text.size())));
}

std::string store_path(
    const std::filesystem::path& path, const std::filesystem::path& base) {
    if (path.empty()) return {};
    if (base.empty()) return path_utf8(path);
    std::error_code error;
    const auto relative = std::filesystem::relative(path, base, error);
    const std::string relative_text = relative.generic_string();
    if (!error && !relative_text.empty() &&
        relative_text.find("..") == std::string::npos)
        return relative_text;
    return path_utf8(std::filesystem::weakly_canonical(path, error));
}

std::filesystem::path load_path(
    const std::string& stored, const std::filesystem::path& base) {
    std::filesystem::path path = path_from_utf8(stored);
    if (path.empty() || path.is_absolute() || base.empty()) return path;
    std::error_code error;
    return std::filesystem::weakly_canonical(base / path, error);
}

std::filesystem::path project_directory(const std::filesystem::path& project_file) {
    const auto parent = project_file.parent_path();
    return parent.empty() ? std::filesystem::current_path() : parent;
}

}  // namespace

std::vector<std::uint8_t> encode_settings(
    const Settings& settings, const std::filesystem::path& project_file) {
    const auto base = project_directory(project_file);
    BufferWriter writer;
    writer.value(k_settings_version);
    const bool has_rotation = std::any_of(settings.gaussian_models.begin(), settings.gaussian_models.end(), [](const auto& model) {
        return model.rotation_degrees[0] != 0.F || model.rotation_degrees[1] != 0.F || model.rotation_degrees[2] != 0.F;
    });
    writer.value(settings.camera_effect.enabled ? (settings.camera_effect.kind == 1 ? 30U : 28U) : has_rotation ? 25U : !settings.camera_keyframes.empty() ? 24U : settings.filter_mode != 0 ? 22U : settings.render_method == 1 ? 21U : k_settings_min_reader);
    writer.string(settings.name);
    writer.string(store_path(settings.image_directory, base));
    writer.value(static_cast<std::int32_t>(settings.sfm_mode));
    writer.value(static_cast<std::uint8_t>(settings.reuse_cache));
    writer.value(settings.max_features);
    writer.value(static_cast<std::uint8_t>(settings.scene_mode));
    writer.value(static_cast<std::int32_t>(settings.iterations));
    writer.value(static_cast<std::int32_t>(settings.preview_interval));
    writer.value(static_cast<std::int32_t>(settings.strategy));
    writer.value(static_cast<std::int32_t>(settings.max_resolution));
    writer.value(static_cast<std::uint8_t>(settings.progressive_resolution));
    writer.value(static_cast<std::uint8_t>(settings.use_mask));
    writer.value(static_cast<std::uint8_t>(settings.build_mesh));
    writer.value(static_cast<std::int32_t>(settings.mesh_method));
    writer.value(settings.depth_normal_weight);
    writer.value(settings.multi_view_geo_weight);
    writer.value(settings.multi_view_ncc_weight);
    writer.value(static_cast<std::int32_t>(settings.geometry_from_iter));
    writer.value(static_cast<std::uint8_t>(settings.normal_field));
    writer.string(store_path(settings.dataset_source, base));
    writer.string(settings.dataset_format);
    writer.string(store_path(settings.dataset_initial_cloud, base));
    writer.string(settings.splat_output_format);
    writer.string(store_path(settings.splat_model_source, base));
    writer.value(static_cast<std::int32_t>(settings.camera_model));
    writer.string(store_path(settings.video_frames_dir, base));
    writer.value(settings.video_fps);
    writer.value(static_cast<std::int32_t>(settings.video_sharp_window));
    writer.value(static_cast<std::int32_t>(settings.video_max_frames));
    writer.value(static_cast<std::int32_t>(settings.video_quality));
    writer.value(settings.video_scale);
    writer.value(static_cast<std::int32_t>(settings.video_rotate));
    writer.value(static_cast<std::int32_t>(settings.mesh_source));
    writer.value(static_cast<std::int32_t>(settings.texture_quality));
    writer.value(static_cast<std::int32_t>(settings.atlas_resolution));
    writer.value(static_cast<std::uint8_t>(settings.texture_delight));
    writer.value(static_cast<std::uint8_t>(settings.texture_optimize));
    writer.value(static_cast<std::int32_t>(settings.densification_cap));
    writer.value(static_cast<std::int32_t>(settings.sh_degree));
    writer.value(static_cast<std::int32_t>(settings.ppisp_layout != 0 ? 1 : 0));
    writer.value(static_cast<std::uint8_t>(settings.bilateral_grid));
    writer.value(static_cast<std::uint8_t>(settings.sam_masks));
    writer.string(store_path(path_from_utf8(settings.sam_model), base));
    writer.string(settings.sam_text);
    writer.string(settings.sam_negative_text);
    writer.value(static_cast<std::uint8_t>(settings.sam_keep_prompted));
    writer.value(static_cast<std::uint8_t>(settings.sam_video));
    writer.value(static_cast<std::int32_t>(settings.sam_max_size));
    writer.value(static_cast<std::int32_t>(settings.mask_mode == 1 ? 1 : 0));
    writer.value(static_cast<std::uint8_t>(settings.texture_blend_linear));
    writer.value(0.F);  // Reserved legacy field for binary compatibility.
    writer.value(static_cast<std::uint8_t>(settings.sam_use_mask_for_sfm));
    writer.value(static_cast<std::int32_t>(settings.mesh_quality));
    writer.value(static_cast<std::int32_t>(settings.training_mode));
    writer.value(static_cast<std::uint8_t>(settings.speedy_pruning));
    writer.value(static_cast<std::int32_t>(settings.texture_photometric_steps));
    writer.value(static_cast<std::int32_t>(settings.texture_seam_steps));
    writer.value(static_cast<std::uint8_t>(settings.focus_region));
    writer.value(static_cast<std::int32_t>(settings.mesh_postprocess_quality));
    writer.value(static_cast<std::uint8_t>(settings.mesh_simplify));
    writer.value(settings.mesh_target_faces);
    writer.value(static_cast<std::uint8_t>(settings.mesh_remesh));
    writer.value(settings.mesh_remesh_target_faces);
    writer.value(static_cast<std::int32_t>(settings.mesh_remesh_smooth_iterations));
    writer.value_size(settings.gaussian_models.size());
    for (const auto& model : settings.gaussian_models) {
        writer.string(store_path(model.path, base));
        writer.string(model.name);
        writer.value(static_cast<std::uint8_t>(model.visible));
        for (float value : model.offset) writer.value(value);
        writer.value(model.scale);
    }
    writer.value(settings.shape_erank_reg);
    writer.value(settings.shape_axis_ratio_reg);
    writer.value(settings.shape_max_axis_ratio);
    writer.value(0.F);  // Retired v10 log-scale weight.
    writer.value(settings.scale_regularization_weight);
    for (const auto& field : splat::k_training_loss_fields)
        writer.value(settings.*field.value);
    writer.value(static_cast<std::uint8_t>(settings.geometry_training));
    writer.value(static_cast<std::int32_t>(settings.normal_field_from_iter));
    writer.value(static_cast<std::int32_t>(settings.ppisp_model));
    writer.value(static_cast<std::uint8_t>(settings.ai_priors));
    writer.value(static_cast<std::int32_t>(settings.ai_prior_source));
    writer.string(store_path(settings.ai_prior_model, base));
    writer.string(store_path(settings.ai_prior_directory, base));
    writer.value(settings.ai_depth_weight);
    writer.value(settings.ai_normal_weight);
    writer.value(static_cast<std::int32_t>(settings.ai_prior_tokens));
    writer.value(static_cast<std::uint8_t>(settings.refine_weak_cameras));
    writer.value(static_cast<std::int32_t>(settings.alignment_quality));
    writer.value(static_cast<std::int32_t>(settings.camera_sharing));
    writer.value(static_cast<std::int32_t>(settings.matching_strategy));
    writer.value(settings.initial_focal_pixels);
    writer.value(static_cast<std::uint8_t>(settings.trust_initial_focal));
    writer.value(static_cast<std::uint64_t>(settings.sequential_overlap));
    writer.value(static_cast<std::uint8_t>(settings.retrieval_augmentation));
    writer.string(settings.feature_extractor);
    writer.string(settings.feature_matcher);
    writer.string(store_path(settings.extractor_model, base));
    writer.string(store_path(settings.lightglue_model, base));
    writer.value(static_cast<std::int32_t>(settings.alignment_capture_type));
    writer.string(settings.video_segments);
    writer.value(settings.sam_threshold);
    writer.value(settings.sam_guided_threshold);
    writer.value(settings.sam_nms);
    writer.value(settings.sam_min_area);
    writer.value(settings.sam_max_area);
    writer.value(static_cast<std::int32_t>(settings.sam_close_kernel));
    writer.value(static_cast<std::uint8_t>(settings.sam_fill_holes));
    writer.value(static_cast<std::uint8_t>(settings.sam_sfm_guided));
    writer.value(static_cast<std::int32_t>(settings.sam_engine == 1 ? 1 : 0));
    writer.string(store_path(settings.sam_detector_model, base));
    writer.value(settings.sam_detector_threshold);
    writer.value(static_cast<std::int32_t>(settings.uv_parallel_partitions));
    writer.value(settings.uv_max_stretch);
    writer.value(static_cast<std::int32_t>(settings.render_method));
    writer.value(static_cast<std::int32_t>(settings.filter_mode));
    writer.value(static_cast<std::uint32_t>(settings.camera_keyframes.size()));
    for (const auto& key : settings.camera_keyframes) {
        writer.value(key.time);
        for (float v : key.target) writer.value(v);
        writer.value(key.yaw); writer.value(key.pitch); writer.value(key.distance);
        writer.value(key.fov_degrees); writer.value(key.ortho_height);
        writer.value(key.fisheye_k1); writer.value(static_cast<std::int32_t>(key.projection));
    }
    for (const auto& key : settings.camera_keyframes) {
        const auto expected = static_cast<std::uint64_t>(key.thumbnail_width) * key.thumbnail_height * 3;
        if (key.thumbnail_width > 256 || key.thumbnail_height > 256 ||
            expected != key.thumbnail_rgb.size())
            throw std::invalid_argument("Invalid camera thumbnail");
        writer.value(key.thumbnail_width); writer.value(key.thumbnail_height);
        writer.bytes(key.thumbnail_rgb.data(), key.thumbnail_rgb.size());
    }
    for (const auto& model : settings.gaussian_models)
        for (float value : model.rotation_degrees) {
            if (!std::isfinite(value)) throw std::invalid_argument("Invalid Gaussian rotation");
            writer.value(value);
        }
    for (const auto& key : settings.camera_keyframes) {
        if (key.interpolation < 0 || key.interpolation > 2)
            throw std::invalid_argument("Invalid camera interpolation");
        writer.value(static_cast<std::int32_t>(key.interpolation));
    }
    for (const auto& key : settings.camera_keyframes) {
        if (!std::isfinite(key.roll)) throw std::invalid_argument("Invalid camera roll");
        writer.value(key.roll);
    }
    settings.camera_effect.validate();
    writer.value(static_cast<std::uint8_t>(settings.camera_effect.enabled));
    writer.value(settings.camera_effect.start_time);
    writer.value(settings.camera_effect.rings_time);
    writer.value(settings.camera_effect.end_time);
    writer.value(settings.camera_effect.point_size);
    writer.value(settings.camera_effect.ring_scale);
    writer.value(static_cast<std::int32_t>(settings.camera_effect.easing));
    writer.value(static_cast<std::int32_t>(settings.camera_effect.kind));
    writer.value(static_cast<std::int32_t>(settings.camera_effect.scan_axis));
    writer.value(static_cast<std::uint8_t>(settings.camera_effect.scan_reverse));
    writer.value(settings.camera_effect.scan_softness);
    writer.value(settings.camera_effect.scan_glow);
    for (float v : settings.camera_effect.scan_direction) writer.value(v);
    writer.value(settings.camera_effect.scan_hold);
    writer.value(settings.camera_effect.scan_growth);
    writer.value(settings.camera_effect.scan_range);
    writer.value(settings.camera_effect.scan_density);
    return writer.take();
}

Settings decode_settings(
    const std::vector<std::uint8_t>& bytes,
    const std::filesystem::path& project_file) {
    BufferReader reader(bytes);
    const auto version = reader.value<std::uint32_t>();
    const auto min_reader = reader.value<std::uint32_t>();
    io::require_readable(
        version, min_reader, k_settings_version, ".ascan settings");
    const auto base = project_directory(project_file);
    Settings settings;
    settings.name = reader.string();
    settings.image_directory = load_path(reader.string(), base);
    settings.sfm_mode = reader.value<std::int32_t>();
    // Migrate the retired incremental mode to hierarchical; retain stored IDs.
    if (settings.sfm_mode == 1) settings.sfm_mode = 2;
    settings.reuse_cache = reader.value<std::uint8_t>() != 0;
    settings.max_features = reader.value<unsigned>();
    settings.scene_mode = reader.value<std::uint8_t>() != 0;
    settings.iterations = reader.value<std::int32_t>();
    settings.preview_interval = reader.value<std::int32_t>();
    settings.strategy = reader.value<std::int32_t>();
    if (settings.strategy < 0 || settings.strategy > 2)
        throw std::invalid_argument("Project splat strategy index is invalid");
    settings.max_resolution = reader.value<std::int32_t>();
    settings.progressive_resolution = reader.value<std::uint8_t>() != 0;
    settings.use_mask = reader.value<std::uint8_t>() != 0;
    settings.build_mesh = reader.value<std::uint8_t>() != 0;
    settings.mesh_method = reader.value<std::int32_t>();
    settings.depth_normal_weight = reader.value<float>();
    settings.multi_view_geo_weight = reader.value<float>();
    settings.multi_view_ncc_weight = reader.value<float>();
    settings.geometry_from_iter = reader.value<std::int32_t>();
    settings.normal_field = reader.value<std::uint8_t>() != 0;
    // Optional additive fields were appended after the original settings
    // prefix. Old .ascan files have no bytes remaining here.
    if (reader.remaining() > 0)
        settings.dataset_source = load_path(reader.string(), base);
    if (reader.remaining() > 0)
        settings.dataset_format = reader.string();
    if (reader.remaining() > 0)
        settings.dataset_initial_cloud = load_path(reader.string(), base);
    if (reader.remaining() > 0)
        settings.splat_output_format = reader.string();
    if (reader.remaining() > 0)
        settings.splat_model_source = load_path(reader.string(), base);
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.camera_model = reader.value<std::int32_t>();
    if (reader.remaining() > 0)
        settings.video_frames_dir = load_path(reader.string(), base);
    if (reader.remaining() >= sizeof(float))
        settings.video_fps = reader.value<float>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.video_sharp_window = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.video_max_frames = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.video_quality = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(float))
        settings.video_scale = reader.value<float>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.video_rotate = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.mesh_source = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.texture_quality = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.atlas_resolution = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.texture_delight = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.texture_optimize = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.densification_cap = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.sh_degree = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.ppisp_layout =
            reader.value<std::int32_t>() != 0 ? 1 : 0;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.bilateral_grid = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.sam_masks = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() > 0)
        settings.sam_model = path_utf8(load_path(reader.string(), base));
    if (reader.remaining() > 0) settings.sam_text = reader.string();
    if (reader.remaining() > 0) settings.sam_negative_text = reader.string();
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.sam_keep_prompted = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.sam_video = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.sam_max_size = reader.value<std::int32_t>();
    if (settings.sam_max_size < 0) settings.sam_max_size = 0;
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.mask_mode = reader.value<std::int32_t>() == 1 ? 1 : 0;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.texture_blend_linear = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(float)) {
        const float legacy_ratio = reader.value<float>();
        // Migrate an enabled legacy hard limit to the independent soft penalty.
        if (std::isfinite(legacy_ratio) && legacy_ratio >= 1.F) {
            settings.shape_max_axis_ratio = legacy_ratio;
            settings.shape_axis_ratio_reg = 0.001F;
        }
    }
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.sam_use_mask_for_sfm = reader.value<std::uint8_t>() != 0;
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.mesh_quality = reader.value<std::int32_t>();
    if (settings.mesh_quality < 0 || settings.mesh_quality > 2)
        settings.mesh_quality = 1;
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.training_mode = reader.value<std::int32_t>();
    if (settings.training_mode < 0 || settings.training_mode > 2)
        settings.training_mode = 1;
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.speedy_pruning = reader.value<std::uint8_t>() != 0;
    // Older projects stored only the quality preset, without custom step counts.
    settings.texture_photometric_steps = settings.texture_quality == 0 ? 1000
        : (settings.texture_quality == 2 ? 6000 : 3000);
    settings.texture_seam_steps = settings.texture_quality == 0 ? 30
        : (settings.texture_quality == 2 ? 120 : 60);
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.texture_photometric_steps = reader.value<std::int32_t>();
    if (reader.remaining() >= sizeof(std::int32_t))
        settings.texture_seam_steps = reader.value<std::int32_t>();
    if (settings.texture_photometric_steps < 1)
        settings.texture_photometric_steps = 3000;
    if (settings.texture_seam_steps < 0)
        settings.texture_seam_steps = 60;
    // Focus Region is independent of capture mode; projects saved before the
    // field existed default it off (Settings::focus_region).
    if (reader.remaining() >= sizeof(std::uint8_t))
        settings.focus_region = reader.value<std::uint8_t>() != 0;
    if (version >= 7) {
        settings.mesh_postprocess_quality = reader.value<std::int32_t>();
        settings.mesh_simplify = reader.value<std::uint8_t>() != 0;
        settings.mesh_target_faces = reader.value<std::uint64_t>();
        settings.mesh_remesh = reader.value<std::uint8_t>() != 0;
        settings.mesh_remesh_target_faces = reader.value<std::uint64_t>();
        settings.mesh_remesh_smooth_iterations = reader.value<std::int32_t>();
        if (settings.mesh_postprocess_quality < 0 || settings.mesh_postprocess_quality > 2)
            settings.mesh_postprocess_quality = 1;
        if (settings.mesh_target_faces < 4) settings.mesh_target_faces = 1'000'000;
        if (settings.mesh_remesh_target_faces < 4) settings.mesh_remesh_target_faces = 1'000'000;
        if (settings.mesh_remesh_smooth_iterations < 0 || settings.mesh_remesh_smooth_iterations > 10)
            settings.mesh_remesh_smooth_iterations = 1;
    }
    if (version >= 8) {
        const auto count = reader.size(33, 100'000);
        settings.gaussian_models.resize(count);
        for (auto& model : settings.gaussian_models) {
            model.path = load_path(reader.string(), base);
            model.name = reader.string();
            model.visible = reader.value<std::uint8_t>() != 0;
            for (float& value : model.offset) {
                value = reader.value<float>();
                if (!std::isfinite(value)) throw std::runtime_error("Invalid Gaussian placement");
            }
            model.scale = reader.value<float>();
            if (!std::isfinite(model.scale) || model.scale <= 0.F)
                throw std::runtime_error("Invalid Gaussian scale");
        }
    }
    if (version >= 9) {
        settings.shape_erank_reg = reader.value<float>();
        settings.shape_axis_ratio_reg = reader.value<float>();
        settings.shape_max_axis_ratio = reader.value<float>();
    } else {
        settings.shape_erank_reg = 0.F;
    }
    if (version >= 10) (void)reader.value<float>();  // Retired log-scale weight.
    settings.scale_regularization_weight = version >= 11 ? reader.value<float>() : 0.F;
    if (!std::isfinite(settings.scale_regularization_weight) || settings.scale_regularization_weight < 0.F)
        settings.scale_regularization_weight = 0.F;
    if (!std::isfinite(settings.shape_erank_reg) || settings.shape_erank_reg < 0.F)
        settings.shape_erank_reg = 0.F;
    if (!std::isfinite(settings.shape_axis_ratio_reg) || settings.shape_axis_ratio_reg < 0.F)
        settings.shape_axis_ratio_reg = 0.F;
    if (!std::isfinite(settings.shape_max_axis_ratio) || settings.shape_max_axis_ratio < 1.F)
        settings.shape_max_axis_ratio = 100.F;
    if (version >= 12) {
        for (const auto& field : splat::k_training_loss_fields)
            settings.*field.value = reader.value<float>();
        settings.geometry_training = reader.value<std::uint8_t>() != 0;
        settings.normal_field_from_iter = reader.value<std::int32_t>();
        settings.ppisp_model = reader.value<std::int32_t>();
    }
    if (version >= 13) {
        settings.ai_priors = reader.value<std::uint8_t>() != 0;
        settings.ai_prior_source = reader.value<std::int32_t>();
        settings.ai_prior_model = load_path(reader.string(), base);
        settings.ai_prior_directory = load_path(reader.string(), base);
        settings.ai_depth_weight = reader.value<float>();
        settings.ai_normal_weight = reader.value<float>();
        settings.ai_prior_tokens = reader.value<std::int32_t>();
        if ((settings.ai_prior_source != 0 && settings.ai_prior_source != 1) ||
            !std::isfinite(settings.ai_depth_weight) || settings.ai_depth_weight < 0.F ||
            !std::isfinite(settings.ai_normal_weight) || settings.ai_normal_weight < 0.F ||
            settings.ai_prior_tokens < 1 || settings.ai_prior_tokens > 10000)
            throw std::invalid_argument("Invalid AI prior project settings");
    }
    if (version >= 14)
        settings.refine_weak_cameras = reader.value<std::uint8_t>() != 0;
    if (version >= 15) {
        settings.alignment_quality = reader.value<std::int32_t>();
        settings.camera_sharing = reader.value<std::int32_t>();
        settings.matching_strategy = reader.value<std::int32_t>();
        settings.initial_focal_pixels = reader.value<double>();
        settings.trust_initial_focal = reader.value<std::uint8_t>() != 0;
        settings.sequential_overlap = reader.value<std::uint64_t>();
        settings.retrieval_augmentation = reader.value<std::uint8_t>() != 0;
        settings.feature_extractor = reader.string();
        settings.feature_matcher = reader.string();
        settings.extractor_model = load_path(reader.string(), base);
        settings.lightglue_model = load_path(reader.string(), base);
        if (settings.alignment_quality < 0 || settings.alignment_quality > 2)
            settings.alignment_quality = 1;
        if (settings.camera_sharing < 0 || settings.camera_sharing > 2)
            settings.camera_sharing = 0;
        if (settings.matching_strategy < 0 || settings.matching_strategy > 2)
            settings.matching_strategy = 0;
        if (!std::isfinite(settings.initial_focal_pixels) ||
            settings.initial_focal_pixels < 0.0)
            settings.initial_focal_pixels = 0.0;
        settings.sequential_overlap = std::clamp(
            settings.sequential_overlap, std::size_t{1}, std::size_t{100});
        if (settings.feature_extractor.empty()) settings.feature_extractor = "siftgpu";
        if (settings.feature_matcher.empty()) settings.feature_matcher = "gpu_mutual_ratio";
    }
    if (version >= 16) {
        settings.alignment_capture_type = reader.value<std::int32_t>() == 1 ? 1 : 0;
    } else {
        // Older projects inferred capture order from the input file type.
        settings.alignment_capture_type = io::is_video_path(settings.image_directory) ? 1 : 0;
    }
    if (version >= 17) {
        settings.video_segments = reader.string();
        settings.sam_threshold = reader.value<float>();
        settings.sam_guided_threshold = reader.value<float>();
        settings.sam_nms = reader.value<float>();
        settings.sam_min_area = reader.value<float>();
        settings.sam_max_area = reader.value<float>();
        settings.sam_close_kernel = reader.value<std::int32_t>();
        settings.sam_fill_holes = reader.value<std::uint8_t>() != 0;
        auto finite_range = [](const float value, const float low, const float high) {
            return std::isfinite(value) && value >= low && value <= high;
        };
        if (!finite_range(settings.sam_threshold, 0.F, 1.F))
            settings.sam_threshold = 0.5F;
        if (!finite_range(settings.sam_guided_threshold, 0.F, 1.F))
            settings.sam_guided_threshold = 0.2F;
        if (!finite_range(settings.sam_nms, 0.F, 1.F))
            settings.sam_nms = 0.1F;
        if (!finite_range(settings.sam_min_area, 0.F, 1.F))
            settings.sam_min_area = 0.001F;
        if (!finite_range(settings.sam_max_area, 0.F, 1.F) ||
            settings.sam_max_area <= settings.sam_min_area)
            settings.sam_max_area = 0.65F;
        if (settings.sam_close_kernel < 1 || settings.sam_close_kernel % 2 == 0)
            settings.sam_close_kernel = 5;
    }
    if (version >= 18)
        settings.sam_sfm_guided = reader.value<std::uint8_t>() != 0;
    if (version >= 19) {
        settings.sam_engine = reader.value<std::int32_t>() == 1 ? 1 : 0;
        settings.sam_detector_model = path_utf8(load_path(reader.string(), base));
        settings.sam_detector_threshold = reader.value<float>();
        if (!std::isfinite(settings.sam_detector_threshold) ||
            settings.sam_detector_threshold < 0.F ||
            settings.sam_detector_threshold > 1.F)
            settings.sam_detector_threshold = 0.35F;
    }
    if (version >= 20) {
        settings.uv_parallel_partitions = std::clamp(reader.value<std::int32_t>(), 1, 64);
        settings.uv_max_stretch = reader.value<float>();
        if (!std::isfinite(settings.uv_max_stretch) ||
            settings.uv_max_stretch < 0.F || settings.uv_max_stretch > 1.F)
            settings.uv_max_stretch = 1.F / 3.F;
    }
    if (version >= 21) {
        settings.render_method = reader.value<std::int32_t>();
        if (settings.render_method < 0 || settings.render_method > 1)
            throw std::invalid_argument("Invalid project rendering method");
    }
    if (version >= 22) {
        settings.filter_mode = reader.value<std::int32_t>();
        if (settings.filter_mode < 0 || settings.filter_mode > 2)
            throw std::invalid_argument("Invalid project filter mode");
    }
    if (version >= 23) {
        const auto count = reader.value<std::uint32_t>();
        if (count > 10000) throw std::invalid_argument("Too many camera keyframes");
        for (std::uint32_t i = 0; i < count; ++i) {
            CameraKeyframe key;
            key.time = reader.value<float>();
            for (float& v : key.target) v = reader.value<float>();
            key.yaw = reader.value<float>(); key.pitch = reader.value<float>();
            key.distance = reader.value<float>(); key.fov_degrees = reader.value<float>();
            key.ortho_height = reader.value<float>(); key.fisheye_k1 = reader.value<float>();
            key.projection = reader.value<std::int32_t>();
            if (!std::isfinite(key.time) || key.time < 0 || key.time > 36000.F || !std::isfinite(key.yaw) ||
                !std::isfinite(key.pitch) || !std::isfinite(key.distance) || key.distance <= 0 ||
                !std::isfinite(key.fov_degrees) || !std::isfinite(key.ortho_height) ||
                !std::isfinite(key.fisheye_k1) || key.projection < 0 || key.projection > 3 ||
                !std::all_of(std::begin(key.target), std::end(key.target), [](float v) { return std::isfinite(v); }))
                throw std::invalid_argument("Invalid camera keyframe");
            settings.camera_keyframes.push_back(key);
        }
        if (version >= 24) {
            for (auto& key : settings.camera_keyframes) {
                key.thumbnail_width = reader.value<std::uint32_t>();
                key.thumbnail_height = reader.value<std::uint32_t>();
                if (key.thumbnail_width > 256 || key.thumbnail_height > 256 ||
                    (key.thumbnail_width == 0) != (key.thumbnail_height == 0))
                    throw std::invalid_argument("Invalid camera thumbnail size");
                const auto size = static_cast<std::size_t>(key.thumbnail_width) * key.thumbnail_height * 3;
                key.thumbnail_rgb.resize(size);
                reader.bytes(key.thumbnail_rgb.data(), size);
            }
        }
    }
    if (version >= 25) {
        for (auto& model : settings.gaussian_models) for (float& value : model.rotation_degrees) {
            value = reader.value<float>();
            if (!std::isfinite(value)) throw std::invalid_argument("Invalid Gaussian rotation");
        }
    }
    if (version >= 26) {
        for (auto& key : settings.camera_keyframes) {
            key.interpolation = reader.value<std::int32_t>();
            if (key.interpolation < 0 || key.interpolation > 2)
                throw std::invalid_argument("Invalid camera interpolation");
        }
    }
    if (version >= 27) for (auto& key : settings.camera_keyframes) {
        key.roll = reader.value<float>();
        if (!std::isfinite(key.roll)) throw std::invalid_argument("Invalid camera roll");
    }
    if (version >= 28) {
        settings.camera_effect.enabled = reader.value<std::uint8_t>() != 0;
        settings.camera_effect.start_time = reader.value<float>();
        settings.camera_effect.rings_time = reader.value<float>();
        settings.camera_effect.end_time = reader.value<float>();
        settings.camera_effect.point_size = reader.value<float>();
        settings.camera_effect.ring_scale = reader.value<float>();
        settings.camera_effect.easing = reader.value<std::int32_t>();
        settings.camera_effect.validate();
    }
    if (version >= 29) {
        settings.camera_effect.kind = reader.value<std::int32_t>();
        settings.camera_effect.scan_axis = reader.value<std::int32_t>();
        settings.camera_effect.scan_reverse = reader.value<std::uint8_t>() != 0;
        settings.camera_effect.scan_softness = reader.value<float>();
        settings.camera_effect.scan_glow = reader.value<float>();
        for (float& v : settings.camera_effect.scan_direction) v = reader.value<float>();
        settings.camera_effect.validate();
    }
    if (version >= 30) {
        settings.camera_effect.scan_hold = reader.value<float>();
        settings.camera_effect.scan_growth = reader.value<float>();
        settings.camera_effect.scan_range = reader.value<float>();
        settings.camera_effect.scan_density = reader.value<float>();
        settings.camera_effect.validate();
    }
    std::stable_sort(settings.camera_keyframes.begin(), settings.camera_keyframes.end(),
        [](const auto& a, const auto& b) { return a.time < b.time; });
    settings.sanitize();
    return settings;
}

void write_gaussian_sources(Archive& archive,
    const std::vector<std::vector<std::uint8_t>>& sources) {
    if (sources.empty()) { archive.erase_chunk(ChunkType::gaussian_sources); return; }
    BufferWriter writer;
    writer.value(std::uint32_t{1});
    writer.value_size(sources.size());
    for (const auto& source : sources) {
        writer.value_size(source.size());
        writer.bytes(source.data(), source.size());
    }
    archive.set_chunk(ChunkType::gaussian_sources, writer.take());
}

std::vector<std::vector<std::uint8_t>> read_gaussian_sources(const Archive& archive) {
    if (!archive.has(ChunkType::gaussian_sources)) return {};
    const auto bytes = archive.chunk(ChunkType::gaussian_sources);
    BufferReader reader(bytes);
    if (reader.value<std::uint32_t>() != 1) throw std::runtime_error("Unsupported Gaussian sources version");
    std::vector<std::vector<std::uint8_t>> sources(reader.size(8, 100'000));
    for (auto& source : sources) {
        source.resize(reader.size(1, (std::numeric_limits<std::uint64_t>::max)()));
        reader.bytes(source.data(), source.size());
    }
    if (reader.remaining()) throw std::runtime_error("Invalid Gaussian sources payload");
    return sources;
}

void write_settings(
    Archive& archive,
    const Settings& settings,
    const std::filesystem::path& project_file) {
    archive.set_chunk(
        ChunkType::settings, encode_settings(settings, project_file));
}

Settings read_settings(const Archive& archive) {
    if (!archive.has(ChunkType::settings)) return {};
    return decode_settings(archive.chunk(ChunkType::settings), archive.source_path());
}

void write_sfm(
    Archive& archive,
    const sfm::Scene& scene,
    const std::filesystem::path& project_file) {
    sfm::AsfmOptions options;
    options.path_base = project_directory(project_file);
    options.compact = true;
    archive.set_chunk(ChunkType::sfm, sfm::encode_asfm(scene, options));
    archive.erase_chunk(ChunkType::sfm_source);
}

std::vector<std::uint8_t> sfm_source_signature(
    const std::filesystem::path& working_sfm,
    const std::filesystem::path& image_directory,
    const std::filesystem::path& project_file) {
    std::ifstream input(working_sfm, std::ios::binary);
    std::array<char, 32> header{};
    if (!input.read(header.data(), header.size()))
        throw std::runtime_error("Cannot read working SfM signature");
    BufferWriter writer;
    writer.value(std::uint32_t{2});
    writer.string(path_utf8(std::filesystem::weakly_canonical(working_sfm)));
    writer.string(path_utf8(std::filesystem::weakly_canonical(image_directory)));
    writer.string(path_utf8(std::filesystem::weakly_canonical(project_file.parent_path())));
    writer.value(std::filesystem::file_size(working_sfm));
    writer.value(std::filesystem::last_write_time(working_sfm).time_since_epoch().count());
    writer.bytes(header.data(), header.size());
    return writer.take();
}

std::optional<sfm::Scene> read_sfm(const Archive& archive) {
    if (!archive.has(ChunkType::sfm)) return std::nullopt;
    sfm::AsfmOptions options;
    options.path_base = project_directory(archive.source_path());
    return sfm::decode_asfm(archive.chunk(ChunkType::sfm), options);
}

void replace_sfm_stage(
    Archive& archive,
    const sfm::Scene& scene,
    const Settings& settings,
    const std::filesystem::path& project_file) {
    write_settings(archive, settings, project_file);
    write_sfm(archive, scene, project_file);
    archive.erase_chunk(ChunkType::training_checkpoint);
    archive.erase_chunk(ChunkType::training_checkpoint_info);
    archive.erase_chunk(ChunkType::training_checkpoint_previous);
    archive.erase_chunk(ChunkType::training_checkpoint_previous_info);
    archive.erase_chunk(ChunkType::training_checkpoint_history);
    archive.erase_chunk(ChunkType::gaussians);
    archive.erase_chunk(ChunkType::mesh);
    archive.erase_chunk(ChunkType::texture);
}

}  // namespace photara::project
