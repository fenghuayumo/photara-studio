#include "project/document.hpp"

#include "io/format_version.hpp"
#include "../io/binary_codec.hpp"

#include <stdexcept>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

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
    writer.value(k_settings_min_reader);
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
    writer.value(settings.max_scale_ratio);
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
    if (reader.remaining() >= sizeof(float))
        settings.max_scale_ratio = reader.value<float>();
    if (!(settings.max_scale_ratio == 0.F || settings.max_scale_ratio >= 1.F))
        settings.max_scale_ratio = 0.F;
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
    archive.set_chunk(ChunkType::sfm, sfm::encode_asfm(scene, options));
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
