#include "project/document.hpp"

#include "io/format_version.hpp"
#include "../io/binary_codec.hpp"

#include <stdexcept>
#include <system_error>
#include <utility>

namespace aetherscan::project {
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
    settings.reuse_cache = reader.value<std::uint8_t>() != 0;
    settings.max_features = reader.value<unsigned>();
    settings.scene_mode = reader.value<std::uint8_t>() != 0;
    settings.iterations = reader.value<std::int32_t>();
    settings.preview_interval = reader.value<std::int32_t>();
    settings.strategy = reader.value<std::int32_t>();
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
    return settings;
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
    archive.erase_chunk(ChunkType::gaussians);
    archive.erase_chunk(ChunkType::mesh);
    archive.erase_chunk(ChunkType::texture);
}

}  // namespace aetherscan::project
