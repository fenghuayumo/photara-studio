#include "app.hpp"
#include "cli_path.hpp"
#include "file_dialogs.hpp"
#include "model_import.hpp"

#include "i18n.hpp"

#include "io/image.hpp"
#include "io/mesh.hpp"
#include "io/video_frames.hpp"
#include "mvs/export.hpp"
#include "project/archive.hpp"
#include "project/document.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/export_mvs.hpp"
#include "sfm/export_nerfstudio.hpp"
#include "splat/formats.hpp"
#include "splat/simplify.hpp"
#include "splat/trainer.hpp"
#include "splat/visualize.hpp"
#if defined(PHOTARA_HAS_TEXTURE)
#include "texture/export.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace editor {
using i18n::tr;
photara::mvs::Mesh preview_mesh_to_mvs(const PreviewMesh& mesh);
std::string gaussian_source_cache_key(const GaussianModelEntry& entry);
std::shared_ptr<const photara::splat::GaussianModel> gaussian_source_model(const GaussianModelEntry& entry);

bool has_external_dataset(const App& app) {
    return app.settings.dataset_source[0] != '\0';
}

bool alignment_ready(const App& app) {
    return has_external_dataset(app) || app.has_sparse || app.scene.has_points();
}

bool waiting_for_train_preview(const App& app) {
    return app.job.running() && app.active_job == JobKind::train &&
           gpu::consumed_timeline_value() == 0;
}

bool reconstruction_available(const App& app) {
    return app.scene.has_points() || app.mesh.has() || app.has_model;
}

Vec3 reconstruction_local_centroid(const App& app) {
    if (app.view_mode == VisualizationMode::mesh && app.mesh.has())
        return app.mesh.centroid;
    if (app.scene.has_points()) return app.scene.centroid;
    if (app.mesh.has()) return app.mesh.centroid;
    return {};
}

float reconstruction_local_radius(const App& app) {
    if (app.view_mode == VisualizationMode::mesh && app.mesh.has())
        return std::max(1e-3F, app.mesh.radius);
    if (app.scene.has_points()) return std::max(1e-3F, app.scene.radius);
    if (app.mesh.has()) return std::max(1e-3F, app.mesh.radius);
    return 1.F;
}

void ensure_reconstruction_box(App& app) {
    ReconstructionBox& box = app.reconstruction_box;
    if (box.user_set) return;
    // Deriving the box walks every point with a spatial hash and a parallel
    // outlier pass (~0.4 s for a 640k cloud), and this is called from the
    // viewport draw path. It only depends on the loaded cloud/mesh, so cache
    // it and refit when a load replaces the source or the user asks for it.
    if (box.auto_fitted) return;
    if (app.scene.has_points()) {
        fit_reconstruction_box(box, app.scene.points);
        box.auto_fitted = true;
    } else if (app.mesh.has()) {
        fit_reconstruction_box(box, app.mesh.vertices);
        box.auto_fitted = true;
    }
}

void write_working_subject_bounds(App& app) {
    ensure_reconstruction_box(app);
    if (!app.reconstruction_box.valid) return;
    const ReconstructionBox& box = app.reconstruction_box;
    const ReconstructionBox& previous = app.published_reconstruction_box;
    std::error_code bounds_error;
    if (app.published_subject_bounds_path == app.layout.working_subject_bounds &&
        std::filesystem::exists(app.layout.working_subject_bounds, bounds_error) &&
        previous.valid && previous.min.x == box.min.x &&
        previous.min.y == box.min.y && previous.min.z == box.min.z &&
        previous.max.x == box.max.x && previous.max.y == box.max.y &&
        previous.max.z == box.max.z)
        return;
    if (write_reconstruction_box(box, app.layout.working_subject_bounds)) {
        app.published_reconstruction_box = box;
        app.published_subject_bounds_path = app.layout.working_subject_bounds;
    }
}

void frame_reconstruction(App& app) {
    app.camera.frame(
        reconstruction_local_centroid(app), reconstruction_local_radius(app));
}

std::filesystem::path existing_splat_model(const App& app) {
    const auto imported = path_from_utf8_field(app.settings.splat_model_source.data());
    std::error_code error;
    if (!imported.empty() && std::filesystem::exists(imported, error))
        return imported;
    const std::array<std::filesystem::path, 7> candidates = {
        app.layout.working_splat, app.layout.splat_model, app.layout.splat_ply,
        app.layout.splat_sog, app.layout.splat_spz, app.layout.splat_glb, app.layout.splat_usdc};
    for (const auto& candidate : candidates)
        if (std::filesystem::exists(candidate, error)) return candidate;
    return {};
}

bool live_preview_active(const App& app) {
    return app.job.running() && app.active_job == JobKind::train;
}


std::filesystem::path resolve_editor_ini() {
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA"))
        return std::filesystem::path(appdata) / "Photara" / "editor.ini";
#else
    if (const char* config = std::getenv("XDG_CONFIG_HOME");
        config != nullptr && config[0] != '\0')
        return std::filesystem::path(config) / "Photara" / "editor.ini";
    if (const char* home = std::getenv("HOME");
        home != nullptr && home[0] != '\0')
        return std::filesystem::path(home) / ".config" / "Photara" / "editor.ini";
#endif
    return std::filesystem::current_path() / "photara_studio.ini";
}

void set_message(App& app, std::string text, const ImVec4& colour) {
    app.message = std::move(text);
    app.message_colour = colour;
}

std::string lower_path_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension;
}

bool is_supported_image_extension(std::string extension) {
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".tif" ||
           extension == ".tiff" || extension == ".bmp";
}

bool directory_has_images(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return false;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        if (is_supported_image_extension(entry.path().extension().string()))
            return true;
    }
    return false;
}

std::filesystem::path path_from_drop(const std::string& text) {
#if defined(_WIN32)
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
#else
    return std::filesystem::path(text);
#endif
}

std::filesystem::path normalized_directory(const std::filesystem::path& path) {
    std::error_code error;
    auto canonical = std::filesystem::weakly_canonical(path, error);
    if (error) canonical = path.lexically_normal();
    return canonical;
}

bool same_directory(
    const std::filesystem::path& left, const std::filesystem::path& right) {
    std::error_code error;
    if (std::filesystem::equivalent(left, right, error)) return true;
    return normalized_directory(left) == normalized_directory(right);
}

bool is_supported_image_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
           is_supported_image_extension(path.extension().string());
}

std::optional<std::filesystem::path> resolve_dropped_image_directory(
    const std::vector<std::string>& dropped, std::string& error) {
    std::vector<std::filesystem::path> folders;
    std::vector<std::filesystem::path> image_folders;
    std::vector<std::filesystem::path> images;
    folders.reserve(dropped.size());
    images.reserve(dropped.size());
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        std::error_code status;
        if (std::filesystem::is_directory(path, status)) {
            const auto folder = normalized_directory(path);
            folders.push_back(folder);
            if (directory_has_images(folder)) image_folders.push_back(folder);
        } else if (is_supported_image_file(path)) {
            images.push_back(path);
        }
    }

    if (image_folders.size() > 1) {
        error = "Drop a single image folder";
        return std::nullopt;
    }
    if (image_folders.size() == 1) {
        const auto& folder = image_folders.front();
        for (const auto& image : images) {
            if (!same_directory(image.parent_path(), folder)) {
                error = "Dropped items must belong to one image folder";
                return std::nullopt;
            }
        }
        return folder;
    }
    if (!folders.empty() && images.empty()) {
        error = "Dropped folder contains no supported images";
        return std::nullopt;
    }
    if (images.empty()) {
        error = "Drop an image folder, photos, a video, .asfm, or .ascan project";
        return std::nullopt;
    }

    const auto parent = normalized_directory(images.front().parent_path());
    for (const auto& image : images) {
        if (!same_directory(image.parent_path(), parent)) {
            error = "Dropped photos must come from the same folder";
            return std::nullopt;
        }
    }
    return parent;
}

void refresh_artifacts(App& app) {
    app.training_checkpoint_info.reset();
    // A normal editor session starts without a project. Do not resolve an
    // empty path relative to the process working directory, otherwise a
    // stray sparse.ply next to the executable could be treated as the active
    // reconstruction.
    if (app.settings.project_dir[0] == '\0' && app.settings.images_dir[0] == '\0' && !app.settings.dataset_source[0] &&
        app.settings.runtime_dir[0] == '\0') {
        app.layout = {};
        app.has_sparse = false;
        app.has_asfm = false;
        app.has_mvs = false;
        app.has_model = gaussian_models_managed(app)
            ? visible_gaussian_model_count(app) > 0 : !existing_splat_model(app).empty();
        app.has_mesh = !app.imported_mesh_source.empty();
        app.has_texture = false;
        app.project_writer_version = 0;
        app.project_min_reader_version = 0;
        return;
    }
    app.layout = resolve_layout(app.settings);
    app.layout.use_saved_inputs = !app.saved_results_discarded;
    app.training_checkpoint_info = photara::splat::inspect_training_checkpoint(app.layout.training_checkpoint);
    if (!app.saved_results_discarded && !app.project_results_dirty && !app.training_checkpoint_info && !app.layout.project_file.empty())
        app.training_checkpoint_info = photara::splat::inspect_training_checkpoint(app.layout.project_file);
    std::error_code error;
    app.has_asfm = std::filesystem::exists(app.layout.sparse_asfm, error);
    app.has_mvs = std::filesystem::exists(app.layout.sparse_mvs, error);
    app.has_sparse =
        std::filesystem::exists(app.layout.working_sfm, error) ||
        std::filesystem::exists(app.layout.sparse_ply, error);
    app.has_model = gaussian_models_managed(app)
        ? visible_gaussian_model_count(app) > 0 : !existing_splat_model(app).empty();
    app.has_mesh =
        !app.imported_mesh_source.empty() ||
        std::filesystem::exists(app.layout.working_mesh, error) ||
        std::filesystem::exists(app.layout.mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_raw_mesh_ply, error);
    app.has_texture = textured_mesh_on_disk(app.layout.working_texture);
    app.project_writer_version = 0;
    app.project_min_reader_version = 0;
    if (app.layout.project_file.empty() || app.saved_results_discarded) return;
    try {
        if (!std::filesystem::exists(app.layout.project_file, error)) return;
        const auto archive =
            photara::project::Archive::open(app.layout.project_file);
        app.project_writer_version = archive.writer_version();
        app.project_min_reader_version = archive.min_reader_version();
        app.has_sparse =
            archive.has(photara::project::ChunkType::sfm) || app.has_sparse;
        if (!gaussian_models_managed(app))
            app.has_model = archive.has(photara::project::ChunkType::gaussians) || app.has_model;
        app.has_mesh =
            archive.has(photara::project::ChunkType::mesh) || app.has_mesh;
        app.has_texture =
            archive.has(photara::project::ChunkType::texture) ||
            app.has_texture;
    } catch (...) {
    }
}

std::filesystem::path reconstruction_images_path(const App& app) {
    return reconstruction_images_dir(app.settings);
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);

void ensure_task_cache(App& app) {
    if (app.settings.runtime_dir[0] != '\0') return;
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) return;
    const auto layout = resolve_layout(app.settings);
    store_utf8_path_field(app.settings.runtime_dir, layout.working_sfm.parent_path());
}

void store_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
    const std::string text = path_to_utf8(path);
    if (text.size() + 1 > field.size()) return;
    std::memcpy(field.data(), text.data(), text.size());
    field[text.size()] = '\0';
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
    store_path_field(field, path);
}

photara::sfm::Scene load_working_sfm(
    const std::filesystem::path& path,
    const std::filesystem::path& images_dir) {
    photara::sfm::AsfmOptions options;
    options.path_base = images_dir;
    return photara::sfm::load_asfm(path, options);
}

photara::project::Settings collect_project_settings(const App& app) {
    photara::project::Settings settings;
    settings.name = app.layout.project_file.empty()
        ? std::string("Untitled")
        : path_to_utf8(app.layout.project_file.stem());
    settings.image_directory =
        path_from_utf8_field(app.settings.images_dir.data());
    settings.dataset_source =
        path_from_utf8_field(app.settings.dataset_source.data());
    settings.dataset_format = "auto";
    settings.dataset_initial_cloud =
        path_from_utf8_field(app.settings.dataset_initial_cloud.data());
    settings.splat_model_source =
        path_from_utf8_field(app.settings.splat_model_source.data());
    if (gaussian_models_managed(app)) {
        // The embedded composition is the legacy reader's authoritative model.
        settings.splat_model_source.clear();
        if (app.splat_edit.dirty()) {
            photara::project::GaussianSource edited;
            edited.name = "Edited composition";
            settings.gaussian_models.push_back(std::move(edited));
        } else {
            for (const auto& entry : app.gaussian_models) {
                photara::project::GaussianSource source;
                source.path = entry.path; source.name = entry.name; source.visible = entry.visible;
                std::copy_n(entry.offset, 3, source.offset);
                source.scale = entry.scale;
                settings.gaussian_models.push_back(std::move(source));
            }
        }
    }
    settings.video_frames_dir =
        path_from_utf8_field(app.settings.video_frames_dir.data());
    settings.video_fps = app.settings.video_fps;
    settings.video_sharp_window = app.settings.video_sharp_window;
    settings.video_max_frames = app.settings.video_max_frames;
    settings.video_quality = app.settings.video_quality;
    settings.video_scale = app.settings.video_scale;
    settings.video_rotate = app.settings.video_rotate;
    settings.splat_output_format = app.settings.splat_format == 1
        ? "ply"
        : app.settings.splat_format == 2
            ? "sog"
            : app.settings.splat_format == 3 ? "spz"
            : app.settings.splat_format == 4 ? "glb"
            : app.settings.splat_format == 5 ? "usdc" : "auto";
    settings.camera_model = app.settings.camera_model;
    settings.sfm_mode = app.settings.sfm_mode;
    settings.reuse_cache = app.settings.reuse_cache;
    settings.max_features = static_cast<unsigned>(
        std::max(0, app.settings.max_features));
    settings.scene_mode = app.settings.scene_mode;
    settings.focus_region = app.settings.focus_region;
    settings.iterations = app.settings.iterations;
    settings.densification_cap = app.settings.densification_cap;
    settings.training_mode = app.settings.training_mode;
    settings.speedy_pruning = app.settings.speedy_pruning;
    settings.sh_degree = app.settings.sh_degree;
    settings.preview_interval = app.settings.preview_interval;
    settings.strategy = app.settings.strategy;
    settings.shape_erank_reg = app.settings.shape_erank_reg;
    settings.shape_axis_ratio_reg = app.settings.shape_axis_ratio_reg;
    settings.shape_max_axis_ratio = app.settings.shape_max_axis_ratio;
    settings.max_resolution = app.settings.max_resolution;
    settings.progressive_resolution = app.settings.progressive_resolution;
    settings.use_mask = app.settings.use_mask;
    settings.mask_mode = app.settings.mask_mode == 1 ? 1 : 0;
    settings.sam_masks = app.settings.sam_masks;
    settings.sam_use_mask_for_sfm = app.settings.sam_use_mask_for_sfm;
    settings.sam_model = app.settings.sam_model.data();
    settings.sam_text = app.settings.sam_text.data();
    settings.sam_negative_text = app.settings.sam_negative_text.data();
    settings.sam_keep_prompted = app.settings.sam_keep_prompted;
    settings.sam_video = app.settings.sam_video;
    settings.sam_max_size = app.settings.sam_max_size;
    settings.build_mesh = app.settings.build_mesh;
    settings.mesh_source = app.settings.mesh_source;
    settings.mesh_method = app.settings.mesh_method;
    settings.mesh_quality = app.settings.mesh_quality;
    settings.mesh_postprocess_quality = app.settings.mesh_postprocess_quality;
    settings.mesh_simplify = app.settings.mesh_simplify;
    settings.mesh_target_faces = app.settings.mesh_target_faces;
    settings.mesh_remesh = app.settings.mesh_remesh;
    settings.mesh_remesh_target_faces = app.settings.mesh_remesh_target_faces;
    settings.mesh_remesh_smooth_iterations = app.settings.mesh_remesh_smooth_iterations;
    settings.depth_normal_weight = app.settings.depth_normal_weight;
    settings.multi_view_geo_weight = app.settings.multi_view_geo_weight;
    settings.multi_view_ncc_weight = app.settings.multi_view_ncc_weight;
    settings.geometry_from_iter = app.settings.geometry_from_iter;
    settings.normal_field = app.settings.normal_field;
    settings.ppisp_layout = app.settings.ppisp_layout;
    settings.bilateral_grid = app.settings.bilateral_grid;
    settings.texture_quality = app.settings.texture_quality;
    settings.atlas_resolution = app.settings.atlas_resolution;
    settings.texture_delight = app.settings.texture_delight;
    settings.texture_optimize = app.settings.texture_optimize;
    settings.texture_photometric_steps = app.settings.texture_photometric_steps;
    settings.texture_seam_steps = app.settings.texture_seam_steps;
    settings.texture_blend_linear = app.settings.texture_blend_linear;
    return settings;
}

void apply_project_settings(
    App& app, const photara::project::Settings& settings) {
    if (!settings.image_directory.empty())
        store_path_field(app.settings.images_dir, settings.image_directory);
    store_path_field(app.settings.dataset_source, settings.dataset_source);
    store_path_field(
        app.settings.dataset_initial_cloud, settings.dataset_initial_cloud);
    store_path_field(app.settings.splat_model_source, settings.splat_model_source);
    app.gaussian_models.clear();
    for (const auto& source : settings.gaussian_models) {
        GaussianModelEntry entry;
        entry.path = source.path; entry.name = source.name; entry.visible = source.visible;
        std::copy_n(source.offset, 3, entry.offset);
        entry.scale = source.scale;
        app.gaussian_models.push_back(std::move(entry));
    }
    store_path_field(app.settings.video_frames_dir, settings.video_frames_dir);
    app.settings.video_fps = settings.video_fps;
    app.settings.video_sharp_window = settings.video_sharp_window;
    app.settings.video_max_frames = settings.video_max_frames;
    app.settings.video_quality = settings.video_quality;
    app.settings.video_scale = settings.video_scale;
    app.settings.video_rotate = settings.video_rotate;
    if (settings.splat_output_format == "ply")
        app.settings.splat_format = 1;
    else if (settings.splat_output_format == "sog")
        app.settings.splat_format = 2;
    else if (settings.splat_output_format == "spz")
        app.settings.splat_format = 3;
    else if (settings.splat_output_format == "glb")
        app.settings.splat_format = 4;
    else if (settings.splat_output_format == "usdc")
        app.settings.splat_format = 5;
    else
        app.settings.splat_format = 0;
    app.settings.camera_model = settings.camera_model;
    app.settings.sfm_mode = settings.sfm_mode;
    app.settings.reuse_cache = settings.reuse_cache;
    app.settings.max_features = static_cast<int>(settings.max_features);
    app.settings.scene_mode = settings.scene_mode;
    app.settings.focus_region = settings.focus_region;
    app.settings.iterations = settings.iterations;
    app.settings.densification_cap = settings.densification_cap;
    app.settings.training_mode = settings.training_mode;
    app.settings.speedy_pruning = settings.speedy_pruning;
    app.settings.sh_degree = settings.sh_degree;
    app.settings.preview_interval = settings.preview_interval;
    app.settings.strategy = settings.strategy;
    app.settings.shape_erank_reg = settings.shape_erank_reg;
    app.settings.shape_axis_ratio_reg = settings.shape_axis_ratio_reg;
    app.settings.shape_max_axis_ratio = settings.shape_max_axis_ratio;
    app.settings.max_resolution = settings.max_resolution;
    app.settings.progressive_resolution = settings.progressive_resolution;
    app.settings.use_mask = settings.use_mask;
    app.settings.mask_mode = settings.mask_mode == 1 ? 1 : 0;
    app.settings.sam_masks = settings.sam_masks;
    app.settings.sam_use_mask_for_sfm = settings.sam_use_mask_for_sfm;
    store_path_field(
        app.settings.sam_model,
        path_from_utf8_field(settings.sam_model.c_str()));
    const auto store_text = [](auto& field, const std::string& text) {
        if (text.size() + 1 > field.size()) return;
        std::memcpy(field.data(), text.data(), text.size());
        field[text.size()] = '\0';
    };
    store_text(app.settings.sam_text, settings.sam_text);
    store_text(app.settings.sam_negative_text, settings.sam_negative_text);
    app.settings.sam_keep_prompted = settings.sam_keep_prompted;
    app.settings.sam_video = settings.sam_video;
    app.settings.sam_max_size = std::max(0, settings.sam_max_size);
    app.settings.build_mesh = settings.build_mesh;
    app.settings.mesh_source = settings.mesh_source == 1 ? 1 : 0;
    app.settings.mesh_method = settings.mesh_method;
    app.settings.mesh_quality = settings.mesh_quality;
    app.settings.mesh_postprocess_quality = std::clamp(settings.mesh_postprocess_quality, 0, 2);
    app.settings.mesh_simplify = settings.mesh_simplify;
    app.settings.mesh_target_faces = std::max<std::uint64_t>(4, settings.mesh_target_faces);
    app.settings.mesh_remesh = settings.mesh_remesh;
    app.settings.mesh_remesh_target_faces = std::max<std::uint64_t>(4, settings.mesh_remesh_target_faces);
    app.settings.mesh_remesh_smooth_iterations = std::clamp(settings.mesh_remesh_smooth_iterations, 0, 10);
    app.settings.depth_normal_weight = settings.depth_normal_weight;
    app.settings.multi_view_geo_weight = settings.multi_view_geo_weight;
    app.settings.multi_view_ncc_weight = settings.multi_view_ncc_weight;
    app.settings.geometry_from_iter = settings.geometry_from_iter;
    app.settings.normal_field = settings.normal_field;
    app.settings.ppisp_layout = settings.ppisp_layout != 0 ? 1 : 0;
    app.settings.bilateral_grid = settings.bilateral_grid;
    app.settings.texture_quality = std::clamp(settings.texture_quality, 0, 2);
    app.settings.atlas_resolution = settings.atlas_resolution > 0
        ? settings.atlas_resolution
        : 2048;
    app.settings.texture_delight = settings.texture_delight;
    app.settings.texture_optimize = settings.texture_optimize;
    app.settings.texture_photometric_steps = std::max(1, settings.texture_photometric_steps);
    app.settings.texture_seam_steps = std::max(0, settings.texture_seam_steps);
    app.settings.texture_blend_linear = settings.texture_blend_linear;
}

void request_asfm_scene_load(
    App& app, const std::filesystem::path& asfm, std::string label) {
    if (app.loading_scene || asfm.empty()) return;
    app.suppress_scene_auto_load = false;
    const std::filesystem::path images = reconstruction_images_path(app);
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = std::move(label);
    app.pending_load = std::async(
        std::launch::async, [asfm, images] {
            SceneLoad loaded;
            try {
                return sparse_scene_from_sfm(load_working_sfm(asfm, images));
            } catch (const std::exception& failure) {
                loaded.error = failure.what();
                return loaded;
            }
        });
}

std::string external_dataset_signature(const App& app) {
    std::string key(app.settings.dataset_source.data());
    key += '|';
    key += app.settings.dataset_initial_cloud.data();
    key += '|';
    key += app.settings.images_dir.data();
    return key;
}

// Loads the imported camera alignment into the editor scene. Without those
// view poses the 2D QA workspace has no capture cameras to snap its compare
// render or feature overlay to, and the trainer keeps previewing from the
// orbit camera instead of the photo being inspected.
void request_dataset_scene_load(App& app) {
    if (app.loading_scene || !has_external_dataset(app)) return;
    const std::filesystem::path source(app.settings.dataset_source.data());
    const std::filesystem::path initial_cloud(
        app.settings.dataset_initial_cloud.data());
    const std::filesystem::path images = reconstruction_images_path(app);
    app.dataset_scene_key = external_dataset_signature(app);
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "External dataset";
    app.pending_load = std::async(
        std::launch::async, [source, initial_cloud, images] {
            return sparse_scene_from_dataset(
                source, initial_cloud, images);
        });
}

std::filesystem::path inferred_images_dir_near(const std::filesystem::path& asfm);
void clear_loaded_result(App& app);

void apply_external_dataset_selection(App& app) {
    clear_loaded_result(app);
    refresh_artifacts(app);
    if (!has_external_dataset(app)) return;
    const std::filesystem::path source(app.settings.dataset_source.data());
    if (app.settings.images_dir[0] == '\0') {
        auto inferred = inferred_images_dir_near(source);
        if (inferred.empty()) {
            std::error_code error;
            if (std::filesystem::is_directory(source, error)) {
                if (directory_has_images(source)) inferred = source;
                else if (directory_has_images(source / "images"))
                    inferred = source / "images";
            }
        }
        if (!inferred.empty())
            store_utf8_path_field(app.settings.images_dir, inferred);
    }
    ensure_task_cache(app);
    request_dataset_scene_load(app);
    set_message(
        app,
        "External cameras loaded. Align Photos is skipped; Train 3DGS or "
        "Extract Mesh can run next.",
        theme::accent);
}

void change_external_dataset(App& app, const std::filesystem::path& path) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app, path] { change_external_dataset(app, path); })) return;
    store_utf8_path_field(app.settings.dataset_source, path);
    app.settings.dataset_initial_cloud.fill('\0');
    app.settings.project_dir.fill('\0');
    app.settings.runtime_dir.fill('\0');
    app.saved_project_settings.clear();
    app.saved_results_discarded = false;
    app.project_results_dirty = true;
    apply_external_dataset_selection(app);
}

void ensure_dataset_scene_loaded(App& app) {
    if (app.suppress_scene_auto_load || !has_external_dataset(app) ||
        app.loading_scene)
        return;
    // A Gaussian-centre scene merges the imported views when no internal
    // poses CSV exists, so it also counts as showing the dataset cameras.
    const bool imported_views_visible =
        !app.scene.views.empty() &&
        (app.scene_source == "External dataset" ||
         app.scene_source == "Gaussian centres" ||
         app.scene_source == "Gaussian models");
    if (imported_views_visible) return;
    if (external_dataset_signature(app) == app.dataset_scene_key) return;
    request_dataset_scene_load(app);
}

void request_ascan_scene_load(App& app) {
    if (app.loading_scene) return;
    const auto ascan = app.layout.project_file;
    const auto working = app.layout.working_sfm;
    const auto asfm = app.layout.sparse_asfm;
    const std::filesystem::path images = reconstruction_images_path(app);
    if (ascan.empty() && working.empty() && asfm.empty()) return;
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "Project SfM";
    app.pending_load = std::async(
        std::launch::async, [ascan, working, asfm, images] {
            SceneLoad loaded;
            std::error_code error;
            try {
                // Live Align writes the cache working copy, not .ascan.
                // Relative photo paths in that file are stored against the
                // image folder and must be resolved with the same base.
                if (!working.empty() &&
                    std::filesystem::exists(working, error)) {
                    return sparse_scene_from_sfm(
                        load_working_sfm(working, images));
                }
                if (!ascan.empty() && std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        photara::project::Archive::open(ascan);
                    const auto scene = photara::project::read_sfm(archive);
                    if (scene) return sparse_scene_from_sfm(*scene);
                }
                if (!asfm.empty() && std::filesystem::exists(asfm, error))
                    return sparse_scene_from_sfm(load_working_sfm(asfm, images));
                loaded.error = "Project has no SfM stage yet";
            } catch (const std::exception& failure) {
                loaded.error = failure.what();
            }
            return loaded;
        });
}

// Composes visible imported models into one; defined with the export flow.
photara::splat::GaussianModel load_export_model(
    const App& app, tinytensor::Device device);

bool has_unsaved_project_changes(const App& app) {
    const bool has_content = app.settings.images_dir[0] || app.settings.dataset_source[0] ||
        has_reconstruction_result(app) || !app.gaussian_models.empty() || !app.imported_mesh_source.empty();
    if (app.saved_project_settings.empty()) return has_content;
    if (app.project_results_dirty || app.job.running() ||
        (app.splat_edit.dirty() && app.splat_edit.edit_revision() != app.saved_edit_revision)) return true;
    if (photara::project::encode_settings(collect_project_settings(app), app.layout.project_file) !=
        app.saved_project_settings) return true;
    std::error_code error;
    if (!app.layout.working_sfm.empty() && std::filesystem::exists(app.layout.working_sfm, error)) {
        const auto stamp = std::filesystem::last_write_time(app.layout.working_sfm, error);
        if (!error && app.saved_sfm_write_time && stamp != *app.saved_sfm_write_time) return true;
    }
    return false;
}

void mark_project_saved(App& app) {
    app.saved_project_settings = photara::project::encode_settings(
        collect_project_settings(app), app.layout.project_file);
    app.project_results_dirty = false;
    app.saved_results_discarded = false;
    app.saved_edit_revision = app.splat_edit.edit_revision();
    app.saved_sfm_write_time.reset();
    std::error_code error;
    if (!app.layout.working_sfm.empty() && std::filesystem::exists(app.layout.working_sfm, error)) {
        const auto stamp = std::filesystem::last_write_time(app.layout.working_sfm, error);
        if (!error) app.saved_sfm_write_time = stamp;
    }
}

bool defer_project_transition(App& app, std::function<void()> action) {
    if (app.project_transition_approved || !has_unsaved_project_changes(app)) return false;
    app.pending_project_action = std::move(action);
    app.show_project_switch_confirmation = true;
    app.checkpoint_save_error.clear();
    return true;
}

void complete_project_transition(App& app) {
    auto action = std::move(app.pending_project_action);
    app.pending_project_action = {};
    app.project_transition_approved = true;
    try { if (action) action(); }
    catch (const std::exception& error) { set_message(app, error.what(), theme::danger); }
    app.project_transition_approved = false;
}

bool save_project_to_path(App& app, const std::filesystem::path& path) {
    try {
        std::error_code error;
        photara::project::Archive archive =
            photara::project::Archive::create();
        const auto current = app.layout.project_file;
        if (!app.saved_results_discarded && !current.empty() && std::filesystem::exists(current, error))
            archive = photara::project::Archive::open(current);
        auto progress = photara::splat::inspect_training_checkpoint(app.layout.training_checkpoint);
        auto progress_source = app.layout.training_checkpoint;
        if (!progress && !app.project_results_dirty && !current.empty()) {
            progress = photara::splat::inspect_training_checkpoint(current);
            progress_source = current;
        }
        const bool save_progress_model = progress && (!app.splat_edit.dirty() ||
            (app.job.running() && app.active_job == JobKind::train));
        auto settings = collect_project_settings(app);
        settings.name = path_to_utf8(path.stem());
        if (save_progress_model) { settings.gaussian_models.clear(); settings.splat_model_source.clear(); }
        photara::project::write_settings(archive, settings, path);
        std::error_code working_error;
        if (!app.layout.working_sfm.empty() &&
            std::filesystem::exists(app.layout.working_sfm, working_error)) {
            const auto images = reconstruction_images_path(app);
            const auto signature = photara::project::sfm_source_signature(app.layout.working_sfm, images, path);
            if (!archive.has(photara::project::ChunkType::sfm) ||
                !archive.has(photara::project::ChunkType::sfm_source) ||
                archive.chunk(photara::project::ChunkType::sfm_source) != signature) {
                const auto scene = load_working_sfm(app.layout.working_sfm, images);
                photara::project::write_sfm(archive, scene, path);
                archive.set_chunk(photara::project::ChunkType::sfm_source, signature);
            }
        } else if (archive.has(photara::project::ChunkType::sfm)) {
            const auto bytes = archive.chunk(photara::project::ChunkType::sfm);
            std::uint32_t minimum_reader = 0;
            if (bytes.size() >= 16) std::memcpy(&minimum_reader, bytes.data() + 12, sizeof(minimum_reader));
            if (minimum_reader < 5 || current.parent_path() != path.parent_path()) {
                photara::sfm::AsfmOptions options;
                options.path_base = current.parent_path();
                photara::project::write_sfm(archive, photara::sfm::decode_asfm(bytes, options), path);
            }
        }
        const auto splat = progress && !progress->model_path.empty()
            ? progress->model_path : existing_splat_model(app);
        if (save_progress_model) {
            archive.set_chunk(photara::project::ChunkType::gaussians,
                photara::splat::encode_gaussians(photara::splat::checkpoint_model(progress_source)));
        } else if (app.splat_edit.dirty()) {
            photara::splat::GaussianModel edited;
            if (app.splat_edit.copy_model(edited)) {
                archive.set_chunk(
                    photara::project::ChunkType::gaussians,
                    photara::splat::encode_gaussians(edited));
                const bool project_owned =
                    !gaussian_models_managed(app) && edited.size() > 0 && !splat.empty() &&
                    (splat == app.layout.working_splat ||
                     splat == app.layout.splat_ply ||
                     splat == app.layout.splat_model ||
                     splat == app.layout.splat_sog ||
                     splat == app.layout.splat_spz ||
                     splat == app.layout.splat_glb || splat == app.layout.splat_usdc);
                if (project_owned)
                    photara::splat::save_gaussians(edited, splat);
            }
        } else if (gaussian_models_managed(app)) {
            if (visible_gaussian_model_count(app)) {
                archive.set_chunk(photara::project::ChunkType::gaussians,
                    photara::splat::encode_gaussians(load_export_model(app, tinytensor::Device::CPU)));
            } else archive.erase_chunk(photara::project::ChunkType::gaussians);
        } else if (!splat.empty()) {
            try {
                const photara::splat::GaussianModel model =
                    gaussian_models_managed(app)
                        ? load_export_model(app, tinytensor::Device::CUDA)
                        : photara::splat::load_gaussians(splat);
                archive.set_chunk(
                    photara::project::ChunkType::gaussians,
                    photara::splat::encode_gaussians(model));
            } catch (...) {
            }
        }
        std::vector<std::vector<std::uint8_t>> sources;
        if (!save_progress_model && gaussian_models_managed(app)) {
            if (app.splat_edit.dirty()) {
                photara::splat::GaussianModel edited;
                if (!app.splat_edit.copy_model(edited))
                    throw std::runtime_error("Cannot save edited composition");
                sources.push_back(photara::splat::encode_gaussians(edited));
            } else {
                for (const auto& entry : app.gaussian_models)
                    sources.push_back(photara::splat::encode_gaussians(*gaussian_source_model(entry)));
            }
        }
        photara::project::write_gaussian_sources(archive, sources);
        const std::array<std::filesystem::path, 4> mesh_candidates = {
            app.layout.working_mesh, app.layout.mesh_ply,
            app.layout.mvs_mesh_ply, app.layout.mvs_raw_mesh_ply};
        for (const auto& mesh_path : mesh_candidates) {
            if (!app.imported_mesh_source.empty()) break;
            if (mesh_path.empty() ||
                !std::filesystem::exists(mesh_path, working_error))
                continue;
            try {
                archive.set_chunk(
                    photara::project::ChunkType::mesh,
                    photara::mvs::encode_mesh(
                        photara::mvs::load_mesh_ply(mesh_path)));
            } catch (...) {
            }
            break;
        }
        if (!app.imported_mesh_source.empty() && app.mesh.has()) {
            const auto imported = preview_mesh_to_mvs(app.mesh);
            archive.set_chunk(photara::project::ChunkType::mesh,
                photara::mvs::encode_mesh(imported));
            // Keep the project-owned working geometry in sync: reopen prefers
            // this cache over the archive chunk.
            if (!app.layout.working_mesh.empty())
                photara::mvs::save_mesh_ply(imported, app.layout.working_mesh);
            archive.erase_chunk(photara::project::ChunkType::texture);
        }
#if defined(PHOTARA_HAS_TEXTURE)
        if (app.imported_mesh_source.empty() &&
            textured_mesh_on_disk(app.layout.working_texture)) {
            try {
                archive.set_chunk(
                    photara::project::ChunkType::texture,
                    photara::texture::encode_textured_obj(
                        app.layout.working_texture));
            } catch (...) {
            }
        }
#endif
        archive.erase_chunk(photara::project::ChunkType::training_model_info);
        for (auto type : {photara::project::ChunkType::training_checkpoint,
                 photara::project::ChunkType::training_checkpoint_info,
                 photara::project::ChunkType::training_checkpoint_previous,
                 photara::project::ChunkType::training_checkpoint_previous_info,
                 photara::project::ChunkType::training_checkpoint_history}) {
            archive.erase_chunk(type);
        }
        if (save_progress_model)
            archive.set_chunk(photara::project::ChunkType::training_model_info,
                photara::splat::training_checkpoint_info_bytes(*progress));
        archive.save(path);
        store_path_field(app.settings.project_dir, path);
        refresh_artifacts(app);
        mark_project_saved(app);
        refresh_artifacts(app);
        return true;
    } catch (const std::exception& error) {
        set_message(app, error.what(), theme::danger);
        return false;
    }
}

void clear_viewport_scene(App& app) {
    app.photos.clear();
    app.image_qa_session.clear();
    app.image_qa = {};
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    app.scene.clear();
    app.mesh.clear();
    app.imported_mesh_source.clear();
    app.mesh_load_failed = false;
    app.mesh_gpu_data.reset();
    app.mesh_gpu_uploaded = false;
    app.mesh_gpu_failed = false;
    app.mesh_renderer.set_mesh({});
    app.mesh_renderer.set_albedo({});
    app.splat_renderer.clear_model();
    app.splat_edit.clear();
    app.splat_load_failed_key.clear();
    app.atlas_preview.reset();
    app.atlas_preview_path.clear();
    app.scene_source.clear();
    app.camera = {};
    app.reconstruction_box.clear();
    app.gizmo.box = {};
    app.view_mode = VisualizationMode::points;
    ++app.scene_load_generation;
    app.loading_scene = false;
    app.gaussian_composition_reload = false;
}

void clear_loaded_result(App& app) {
    clear_viewport_scene(app);
    app.gaussian_models.clear();
    app.composed_model.reset();
    app.composed_model_key.clear();
    // A running merge cannot be interrupted without blocking the UI thread;
    // mark its result for discard and let poll_gaussian_merge finish it.
    if (app.gaussian_merge.running) {
        app.gaussian_merge.discard = true;
    } else {
        app.gaussian_merge = {};
    }
    app.monitor.reset();
    app.log.clear();
    app.console = {};
    app.fresh_lines.clear();
}

bool alignment_job_running(const App& app) {
    return app.job.running() && app.active_job == JobKind::align;
}

void new_project(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app] { new_project(app); })) return;
    clear_loaded_result(app);
    // The cache root, retention policy, and compute choices are editor-wide
    // settings, not part of the project.
    const auto cache_dir = app.settings.cache_dir;
    const int cache_retention_days = app.settings.cache_retention_days;
    const int training_backend = app.settings.training_backend;
    const int alignment_backend = app.settings.alignment_backend;
    const int sam_backend = app.settings.sam_backend;
    const int training_autosave_seconds = app.settings.training_autosave_seconds;
    app.settings = {};
    app.settings.cache_dir = cache_dir;
    app.settings.cache_retention_days = cache_retention_days;
    app.settings.training_backend = training_backend;
    app.settings.alignment_backend = alignment_backend;
    app.settings.sam_backend = sam_backend;
    app.settings.training_autosave_seconds = training_autosave_seconds;
    app.layout = {};
    app.has_sparse = false;
    app.has_asfm = false;
    app.has_mvs = false;
    app.has_model = false;
    app.has_mesh = false;
    app.has_texture = false;
    app.atlas_preview.reset();
    app.atlas_preview_path.clear();
    app.project_writer_version = 0;
    app.project_min_reader_version = 0;
    app.saved_project_settings.clear();
    app.project_results_dirty = false;
    app.saved_results_discarded = false;
    set_message(app, "New project", theme::text_muted);
}

void apply_image_directory_selection(App& app) {
    clear_loaded_result(app);
    app.settings.dataset_source.fill('\0');
    app.settings.dataset_initial_cloud.fill('\0');
    // An image-folder selection starts a different reconstruction. Retaining
    // an explicitly opened project here makes refresh_artifacts() immediately
    // reload that project's old SfM result into the new dataset.
    app.settings.project_dir.fill('\0');
    app.settings.runtime_dir.fill('\0');
    app.saved_project_settings.clear();
    app.project_results_dirty = true;
    app.saved_results_discarded = false;
    ensure_task_cache(app);
    refresh_artifacts(app);
    set_message(
        app, "Image dataset selected; previous viewport result cleared",
        theme::text_muted);
}

void select_image_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app] { select_image_folder(app); })) return;
    if (!pick_folder(
            L"Select the capture image folder", app.settings.images_dir))
        return;
    app.settings.video_frames_dir.fill('\0');
    apply_image_directory_selection(app);
}

void change_image_source(App& app, const std::filesystem::path& path) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app, path] { change_image_source(app, path); })) return;
    store_utf8_path_field(app.settings.images_dir, path);
    app.settings.dataset_source.fill('\0');
    app.settings.dataset_initial_cloud.fill('\0');
    app.settings.video_frames_dir.fill('\0');
    apply_image_directory_selection(app);
}

std::filesystem::path editor_cache_file() {
    return resolve_editor_ini().parent_path() / "editor.cache";
}

// Editor-level cache settings, stored next to editor.ini. They are deliberately
// not written into .ascan: the folder and the retention policy are machine
// specific, and opening a project elsewhere must use that machine's own choice.
//
// Layout: "dir=<utf8 path>" and "retention_days=<n>", one per line. The
// pre-existing single-line editor.cache_dir file is still read as a fallback.
void load_editor_cache_dir(App& app) {
    std::ifstream input(editor_cache_file(), std::ios::binary);
    if (!input) {
        std::ifstream legacy(
            resolve_editor_ini().parent_path() / "editor.cache_dir",
            std::ios::binary);
        if (!legacy) return;
        std::string text;
        std::getline(legacy, text);
        if (text.empty()) return;
        store_utf8_path_field(
            app.settings.cache_dir, path_from_utf8_field(text.c_str()));
        return;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        const std::string key = line.substr(0, equals);
        const std::string value = line.substr(equals + 1);
        if (key == "dir") {
            if (!value.empty())
                store_utf8_path_field(
                    app.settings.cache_dir, path_from_utf8_field(value.c_str()));
        } else if (key == "retention_days") {
            try {
                app.settings.cache_retention_days =
                    std::clamp(std::stoi(value), 0, 3650);
            } catch (...) {
            }
        }
    }
}

std::filesystem::path editor_prefs_file() {
    return resolve_editor_ini().parent_path() / "editor.prefs";
}

int preference_choice(
    const std::string& value, const char* const* tokens, const int count) {
    for (int index = 0; index < count; ++index)
        if (value == tokens[index]) return index;
    return -1;
}

void load_editor_preferences(App& app) {
    std::ifstream input(editor_prefs_file(), std::ios::binary);
    if (!input) return;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        const std::string key = line.substr(0, equals);
        const std::string value = line.substr(equals + 1);
        if (key == "training_autosave_seconds") {
            int seconds{};
            const auto result = std::from_chars(value.data(), value.data() + value.size(), seconds);
            if (result.ec == std::errc{} && result.ptr == value.data() + value.size() && seconds >= 0 && seconds <= 86400)
                app.settings.training_autosave_seconds = seconds;
        } else if (key == "training") {
            static const char* tokens[] = {"cuda", "vulkan"};
            const int choice = preference_choice(value, tokens, 2);
            if (choice >= 0) app.settings.training_backend = choice;
        } else if (key == "alignment") {
            static const char* tokens[] = {"automatic", "cpu", "cuda"};
            const int choice = preference_choice(value, tokens, 3);
            if (choice >= 0) app.settings.alignment_backend = choice;
        } else if (key == "sam") {
            static const char* tokens[] = {"auto", "cuda", "vulkan"};
            const int choice = preference_choice(value, tokens, 3);
            if (choice >= 0) app.settings.sam_backend = choice;
        }
    }
    // Keep the file so a later build can still honour a choice this binary
    // cannot run. The session itself uses the fallback.
    if (!studio_vulkan_training()) app.settings.training_backend = 0;
    if (!studio_ba_cuda() && app.settings.alignment_backend == 2)
        app.settings.alignment_backend = 0;
    if (!studio_sam_cuda() && app.settings.sam_backend == 1)
        app.settings.sam_backend = 0;
    if (!studio_sam_vulkan() && app.settings.sam_backend == 2)
        app.settings.sam_backend = 0;
}

void store_editor_preferences(const App& app) {
    const auto path = editor_prefs_file();
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return;
    const char* training = app.settings.training_backend == 1 ? "vulkan" : "cuda";
    const char* alignment = app.settings.alignment_backend == 1
        ? "cpu"
        : app.settings.alignment_backend == 2 ? "cuda" : "automatic";
    const char* sam = app.settings.sam_backend == 1
        ? "cuda"
        : app.settings.sam_backend == 2 ? "vulkan" : "auto";
    output << "training=" << training << '\n'
           << "alignment=" << alignment << '\n'
           << "sam=" << sam << '\n'
           << "training_autosave_seconds=" << app.settings.training_autosave_seconds << '\n';
}

void store_editor_cache_dir(const App& app) {
    const auto path = editor_cache_file();
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return;
    output << "dir=" << app.settings.cache_dir.data() << '\n'
           << "retention_days=" << app.settings.cache_retention_days << '\n';
}

void select_cache_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_folder(L"Select a cache folder", app.settings.cache_dir)) return;
    clear_loaded_result(app);
    store_editor_cache_dir(app);
    refresh_artifacts(app);
    app.cache_usage.valid = false;
    set_message(app, "Cache folder updated", theme::text_muted);
}

// A crashed editor leaves its session folder behind. Sweep the ones that are
// clearly stale, and leave recent folders alone so a second running instance
// keeps the handshake files its child processes are reading.
void sweep_stale_cache_sessions(const std::filesystem::path& root) {
    if (root.empty()) return;
    std::error_code error;
    std::filesystem::directory_iterator entries(root, error);
    if (error) return;
    const auto now = std::filesystem::file_time_type::clock::now();
    for (const auto& entry : entries) {
        std::error_code entry_error;
        if (!entry.is_directory(entry_error)) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("session-", 0) != 0) continue;
        std::error_code stamp_error;
        const auto stamp =
            std::filesystem::last_write_time(entry.path(), stamp_error);
        if (stamp_error) continue;
        if (now - stamp < std::chrono::hours(48)) continue;
        std::error_code remove_error;
        std::filesystem::remove_all(entry.path(), remove_error);
    }
}

void prepare_cache_session(const App& app) {
    const std::filesystem::path& dir = app.layout.session_dir;
    if (dir.empty()) return;
    std::error_code error;
    // Ours from a process that reused this pid: nothing can be reading it.
    std::filesystem::remove_all(dir, error);
    std::filesystem::create_directories(dir, error);
    sweep_stale_cache_sessions(dir.parent_path());
}

void cleanup_cache_session(const App& app) {
    if (app.layout.session_dir.empty()) return;
    std::error_code error;
    std::filesystem::remove_all(app.layout.session_dir, error);
}

void apply_video_selection(App& app) {
    clear_loaded_result(app);
    app.settings.dataset_source.fill('\0');
    app.settings.dataset_initial_cloud.fill('\0');
    app.settings.video_frames_dir.fill('\0');
    app.settings.project_dir.fill('\0');
    app.settings.runtime_dir.fill('\0');
    app.saved_project_settings.clear();
    app.project_results_dirty = true;
    app.saved_results_discarded = false;
    ensure_task_cache(app);
    refresh_artifacts(app);
    set_message(
        app,
        app.settings.sam_masks
            ? "Video selected. Align Photos extracts frames, generates masks, then aligns cameras"
            : "Video selected; Align Photos will extract sharp frames, then run SfM",
        theme::text_muted);
}

void select_video_file(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app] { select_video_file(app); })) return;
    if (!pick_video_file(
            L"Select a capture video", app.settings.images_dir))
        return;
    apply_video_selection(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path);
void open_asfm_from_path(App& app, const std::filesystem::path& asfm);
void request_gaussian_scene_load(App& app);

void apply_dropped_image_source(
    App& app, const std::vector<std::string>& dropped) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Stop the running job first to change images or video",
            theme::warning);
        return;
    }
    if (defer_project_transition(app, [&app, dropped] { apply_dropped_image_source(app, dropped); })) return;
    std::vector<std::filesystem::path> videos;
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        std::error_code status;
        if (std::filesystem::is_regular_file(path, status) &&
            photara::io::is_video_path(path))
            videos.push_back(path);
    }
    if (videos.size() > 1) {
        set_message(app, "Drop a single video file", theme::danger);
        return;
    }
    if (videos.size() == 1) {
        if (dropped.size() != 1) {
            set_message(
                app, "Drop a single video, or a folder of photos",
                theme::danger);
            return;
        }
        store_utf8_path_field(app.settings.images_dir, videos.front());
        apply_video_selection(app);
        return;
    }
    std::string error;
    const auto directory = resolve_dropped_image_directory(dropped, error);
    if (!directory) {
        set_message(app, error, theme::danger);
        return;
    }
    store_utf8_path_field(app.settings.images_dir, *directory);
    apply_image_directory_selection(app);
}

void apply_dropped_paths(App& app, const std::vector<std::string>& dropped) {
    // Validate the whole selection before touching the displayed models.
    std::vector<std::filesystem::path> gaussians;
    std::filesystem::path mesh;
    try {
        for (const auto& text : dropped) {
            if (text.empty()) continue;
            const auto path = path_from_drop(text);
            const auto candidate = model_import_kind(path);
            if (candidate == ModelImportKind::none) continue;
            std::error_code error;
            if (!std::filesystem::is_regular_file(path, error))
                throw std::runtime_error("Model file does not exist");
            if (candidate == ModelImportKind::gaussians)
                gaussians.push_back(path);
            else if (mesh.empty())
                mesh = path;
        }
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
        return;
    }

    if (!gaussians.empty() || !mesh.empty()) {
        const std::size_t model_items =
            gaussians.size() + (mesh.empty() ? 0U : 1U);
        // Model drops must be pure: no photos, projects, or meshes mixed in,
        // and a mesh stays a single file.
        if (model_items != dropped.size() ||
            (!mesh.empty() && model_items > 1)) {
            set_message(app, tr("Drop a single 3DGS or mesh model"), theme::warning);
            return;
        }
        if (app.job.running() || app.loading_scene ||
            app.splat_export.job.valid() || app.splat_simplify.running ||
            app.gaussian_merge.running) {
            set_message(app, tr("Wait for the current operation before importing a model"), theme::warning);
            return;
        }
        if (app.splat_edit.dirty()) {
            set_message(app, tr("Export Gaussian edits, then start a new project before importing another model"), theme::warning);
            return;
        }

        if (!gaussians.empty()) {
            // Dropping onto an existing list adds models so a merge can be
            // assembled incrementally; the first drop starts a fresh scene.
            if (app.gaussian_models.empty()) clear_loaded_result(app);
            import_gaussian_models(app, gaussians);
        } else {
            clear_loaded_result(app);
            app.imported_mesh_source = mesh;
            refresh_artifacts(app);
            app.suppress_scene_auto_load = true;
            show_mesh_view(app, true);
        }
        return;
    }
    std::vector<std::filesystem::path> projects;
    std::vector<std::filesystem::path> asfms;
    std::vector<std::string> remainder;
    projects.reserve(dropped.size());
    asfms.reserve(dropped.size());
    remainder.reserve(dropped.size());
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        const std::string extension = lower_path_extension(path);
        if (extension == ".ascan")
            projects.push_back(path);
        else if (extension == ".asfm")
            asfms.push_back(path);
        else
            remainder.push_back(text);
    }

    if (projects.size() > 1) {
        set_message(app, "Drop a single .ascan project file", theme::danger);
        return;
    }
    if (projects.size() == 1) {
        open_project_from_path(app, projects.front());
        return;
    }
    if (asfms.size() > 1) {
        set_message(app, "Drop a single .asfm file", theme::danger);
        return;
    }
    if (asfms.size() == 1) {
        if (!remainder.empty()) {
            std::string error;
            if (const auto directory =
                    resolve_dropped_image_directory(remainder, error))
                store_utf8_path_field(app.settings.images_dir, *directory);
        }
        open_asfm_from_path(app, asfms.front());
        return;
    }
    apply_dropped_image_source(app, remainder.empty() ? dropped : remainder);
}

bool mouse_over_viewport(const App& app) {
    if (!app.viewport_bounds_valid) return false;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    return mouse.x >= app.viewport_min.x && mouse.y >= app.viewport_min.y &&
           mouse.x < app.viewport_max.x && mouse.y < app.viewport_max.y;
}

void consume_dropped_paths(App& app) {
    if (app.dropped_paths.empty()) return;
    std::vector<std::string> dropped;
    dropped.swap(app.dropped_paths);
    if (!mouse_over_viewport(app)) {
        set_message(
            app,
            "Drop photos, a video, 3DGS, mesh, .asfm, or .ascan on the viewport",
            theme::warning);
        return;
    }
    apply_dropped_paths(app, dropped);
}

void request_scene_load(
    App& app, const std::filesystem::path& cloud,
    const std::filesystem::path& poses, std::string label) {
    if (app.loading_scene) return;
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = std::move(label);
    app.pending_load = std::async(
        std::launch::async,
        [cloud, poses] { return load_sparse_scene(cloud, poses); });
}

std::filesystem::path write_edited_splat_file(App& app) {
    if (!app.splat_edit.dirty()) return {};
    try {
        photara::splat::GaussianModel model;
        if (!app.splat_edit.copy_model(model) || model.size() == 0) return {};
        const std::filesystem::path out = app.layout.root.empty()
            ? std::filesystem::temp_directory_path() / "photara_edited_gaussians.ply"
            : app.layout.root / "edited_gaussians.ply";
        std::error_code error;
        if (out.has_parent_path())
            std::filesystem::create_directories(out.parent_path(), error);
        photara::splat::save_gaussians(
            model, out, photara::splat::GaussianFormat::ply);
        return out;
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
        return {};
    }
}

bool gaussian_models_managed(const App& app) {
    return !app.gaussian_models.empty();
}

std::size_t visible_gaussian_model_count(const App& app) {
    std::size_t visible = 0;
    for (const GaussianModelEntry& entry : app.gaussian_models)
        if (entry.visible) ++visible;
    return visible;
}

photara::splat::GaussianTransform gaussian_entry_transform(
    const GaussianModelEntry& entry) {
    photara::splat::GaussianTransform transform;
    transform.translation = {entry.offset[0], entry.offset[1], entry.offset[2]};
    transform.scale = std::max(entry.scale, 1e-6F);
    return transform;
}

std::filesystem::path normalized_model_key(const std::filesystem::path& path) {
    std::error_code error;
    auto key = std::filesystem::absolute(path, error);
    if (error) return path.lexically_normal();
    return key.lexically_normal();
}

std::string gaussian_source_cache_key(const GaussianModelEntry& entry) {
    if (entry.cached_model && !entry.cached_key.empty()) return entry.cached_key;
    if (entry.embedded)
        return "embedded:" + std::to_string(reinterpret_cast<std::uintptr_t>(entry.embedded.get()));
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(entry.path, error);
    if (error) throw std::runtime_error("Cannot read Gaussian source: " + path_to_utf8(entry.path));
    return path_to_utf8(entry.path) + ':' + std::to_string(stamp.time_since_epoch().count());
}

std::shared_ptr<const photara::splat::GaussianModel> gaussian_source_model(const GaussianModelEntry& entry) {
    if (entry.cached_model) return entry.cached_model;
    auto model = entry.embedded
        ? photara::splat::decode_gaussians(*entry.embedded, tinytensor::Device::CPU)
        : photara::splat::load_gaussians(entry.path, tinytensor::Device::CPU);
    return std::make_shared<const photara::splat::GaussianModel>(std::move(model));
}

// Cache key of the composed preview: every visible source file, its stamp,
// and its placement. Any change reloads the merged cloud.
// The legacy single-model field follows the first list entry so project
// settings, exports, and save keep working with one source of truth.
void sync_gaussian_model_settings(App& app) {
    if (app.gaussian_models.empty()) {
        app.settings.splat_model_source.fill('\0');
        return;
    }
    store_utf8_path_field(
        app.settings.splat_model_source, app.gaussian_models.front().path);
}

// Drops every preview derived from the composed model so the next ensure
// pass reloads the current model list.
void reset_gaussian_preview(App& app) {
    // Import/remove paths are rejected while a load is in flight, so the
    // pending future is either invalid or complete and can be dropped.
    app.pending_load = {};
    app.composed_model.reset();
    app.composed_model_key.clear();
    app.scene.clear();
    ++app.scene_load_generation;
    app.loading_scene = false;
    app.splat_renderer.clear_model();
    app.splat_edit.clear();
    app.splat_load_failed_key.clear();
    app.photos.clear();
    app.image_qa_session.clear();
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    invalidate_reconstruction_box(app.reconstruction_box);
}

void import_gaussian_models(
    App& app, const std::vector<std::filesystem::path>& paths,
    const bool announce) {
    if (paths.empty()) return;
    if (app.job.running() || app.loading_scene ||
        app.splat_export.job.valid() || app.splat_simplify.running ||
        app.gaussian_merge.running) {
        set_message(
            app, tr("Wait for the current operation before importing a model"),
            theme::warning);
        return;
    }
    if (app.splat_edit.dirty()) {
        set_message(
            app,
            tr("Export Gaussian edits, then start a new project before importing another model"),
            theme::warning);
        return;
    }
    std::size_t added = 0;
    std::string failure;
    for (const std::filesystem::path& path : paths) {
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) {
            failure = "Model file does not exist";
            continue;
        }
        try {
            if (model_import_kind(path) != ModelImportKind::gaussians) {
                continue;
            }
        } catch (const std::exception& reject) {
            failure = reject.what();
            continue;
        }
        const auto key = normalized_model_key(path);
        bool duplicate = false;
        for (const GaussianModelEntry& entry : app.gaussian_models)
            if (normalized_model_key(entry.path) == key) duplicate = true;
        if (duplicate) {
            continue;
        }
        if (app.gaussian_models.empty() &&
            path_to_utf8(path).size() >=
                app.settings.splat_model_source.size()) {
            failure = "Model path is too long";
            continue;
        }
        GaussianModelEntry entry;
        entry.path = path;
        entry.name = path_to_utf8(path.filename());
        app.gaussian_models.push_back(std::move(entry));
        ++added;
    }
    if (added == 0) {
        set_message(
            app,
            failure.empty() ? tr("Model is already loaded") : failure,
            theme::warning);
        return;
    }
    sync_gaussian_model_settings(app);
    refresh_artifacts(app);
    reset_gaussian_preview(app);
    app.workspace = ViewportWorkspace::scene_3d;
    app.view_mode = VisualizationMode::splat;
    request_gaussian_scene_load(app);
    if (announce) {
        const std::string caption =
            "Imported " + std::to_string(added) + " Gaussian model" +
            (added == 1 ? "" : "s") + " (" +
            std::to_string(app.gaussian_models.size()) + " total)";
        set_message(app, caption, theme::success);
    }
}

void remove_gaussian_model(App& app, const std::size_t index) {
    if (index >= app.gaussian_models.size()) return;
    if (app.gaussian_merge.running || app.job.running() || app.loading_scene ||
        app.splat_export.job.valid() || app.splat_simplify.running) {
        set_message(
            app, tr("Wait for the current operation before importing a model"),
            theme::warning);
        return;
    }
    if (app.splat_edit.dirty()) {
        set_message(
            app,
            tr("Export Gaussian edits, then start a new project before importing another model"),
            theme::warning);
        return;
    }
    app.gaussian_models.erase(
        app.gaussian_models.begin() + static_cast<std::ptrdiff_t>(index));
    sync_gaussian_model_settings(app);
    refresh_artifacts(app);
    reset_gaussian_preview(app);
    if (!app.gaussian_models.empty() || app.has_model) {
        // With an empty list this falls back to the project's trained
        // working copy, if one exists.
        request_gaussian_scene_load(app);
    } else {
        app.view_mode = VisualizationMode::points;
    }
}

// Rebuilds the composed preview after a visibility or placement change.
void refresh_gaussian_composition(App& app) {
    if (app.gaussian_models.empty()) return;
    if (app.splat_edit.dirty()) {
        set_message(
            app,
            tr("Export Gaussian edits, then start a new project before importing another model"),
            theme::warning);
        return;
    }
    app.splat_renderer.clear_model();
    app.splat_edit.clear();
    app.splat_load_failed_key.clear();
    app.composed_model.reset();
    app.composed_model_key.clear();
    ++app.scene_load_generation;
    if (app.loading_scene) {
        // Invalidate the in-flight result and retry once it lands; dropping
        // the future here would block the UI thread on file IO.
        app.gaussian_composition_reload = true;
        return;
    }
    request_gaussian_scene_load(app);
}

void request_gaussian_scene_load(App& app) {
    if (app.loading_scene) return;
    std::filesystem::path model_path = existing_splat_model(app);
    const auto edited = write_edited_splat_file(app);
    if (!edited.empty()) model_path = edited;
    const auto ascan = app.layout.project_file;
    const auto poses = app.layout.sparse_poses;
    if (model_path.empty() && ascan.empty() && !gaussian_models_managed(app)) return;
    const bool dataset = has_external_dataset(app);
    const std::filesystem::path dataset_source(app.settings.dataset_source.data());
    const std::filesystem::path dataset_initial_cloud(
        app.settings.dataset_initial_cloud.data());
    const std::filesystem::path dataset_images = reconstruction_images_path(app);
    const auto composition = edited.empty() ? app.gaussian_models : std::vector<GaussianModelEntry>{};
    if (!composition.empty()) app.has_model = visible_gaussian_model_count(app) > 0;
    const std::string composition_key = "composition:" + std::to_string(app.scene_load_generation);
    if (!composition.empty() && visible_gaussian_model_count(app) == 0) {
        app.scene.clear();
        app.composed_model.reset();
        invalidate_reconstruction_box(app.reconstruction_box);
        return;
    }
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = composition.empty() || composition.size() == 1
        ? "Gaussian centres"
        : "Gaussian models";
    app.pending_load = std::async(
        std::launch::async,
        [composition, composition_key, model_path, ascan, poses, dataset, dataset_source,
         dataset_initial_cloud, dataset_images] {
            const auto merge_dataset_views = [&](SceneLoad& loaded) {
                // External-dataset training uses the imported cameras, not
                // an internal poses CSV. Always replace so compare / feature
                // QA snap to the same views the trainer used.
                if (!dataset || !loaded.ok) return;
                SceneLoad imported = sparse_scene_from_dataset(
                    dataset_source, dataset_initial_cloud, dataset_images);
                if (!imported.ok) return;
                loaded.scene.views = std::move(imported.scene.views);
                loaded.scene.registered_views =
                    imported.scene.registered_views;
                loaded.scene.total_views = imported.scene.total_views;
            };
            try {
                if (!composition.empty()) {
                    SceneLoad loaded;
                    loaded.source_models.resize(composition.size());
                    loaded.source_keys.resize(composition.size());
                    std::vector<photara::splat::GaussianModel> parts;
                    for (std::size_t i = 0; i < composition.size(); ++i) {
                        const auto& entry = composition[i];
                        if (!entry.visible) continue;
                        loaded.source_keys[i] = gaussian_source_cache_key(entry);
                        loaded.source_models[i] = gaussian_source_model(entry);
                        parts.push_back(photara::splat::transform_gaussians(
                            *loaded.source_models[i], gaussian_entry_transform(entry)));
                    }
                    loaded.composed_model = std::make_shared<const photara::splat::GaussianModel>(
                        photara::splat::merge_gaussians(parts));
                    auto scene = gaussian_scene_from_model(*loaded.composed_model, poses);
                    loaded.scene = std::move(scene.scene);
                    loaded.ok = scene.ok; loaded.error = std::move(scene.error);
                    loaded.composed_key = composition_key;
                    merge_dataset_views(loaded);
                    return loaded;
                }
                if (!model_path.empty())
                {
                    SceneLoad loaded = load_gaussian_scene(model_path, poses);
                    merge_dataset_views(loaded);
                    return loaded;
                }
                std::error_code error;
                if (std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        photara::project::Archive::open(ascan);
                    if (archive.has(photara::project::ChunkType::gaussians))
                    {
                        SceneLoad loaded = gaussian_scene_from_model(
                            photara::splat::decode_gaussians(
                                archive.chunk(
                                    photara::project::ChunkType::gaussians)),
                            poses);
                        merge_dataset_views(loaded);
                        return loaded;
                    }
                }
                SceneLoad loaded;
                loaded.error = "Project has no trained Gaussian model yet";
                return loaded;
            } catch (const std::exception& failure) {
                SceneLoad loaded;
                loaded.error = failure.what();
                return loaded;
            }
        });
}

void ensure_sparse_loaded(App& app) {
    if (alignment_job_running(app) || app.suppress_scene_auto_load ||
        !app.has_sparse || app.scene.has_points() || app.loading_scene)
        return;
    if (!app.layout.project_file.empty() || !app.layout.working_sfm.empty())
        request_ascan_scene_load(app);
    else
        request_scene_load(
            app, app.layout.sparse_ply, app.layout.sparse_poses, "Sparse cloud");
}

void ensure_gaussian_scene(App& app) {
    if (app.scene.has_gaussians() || app.loading_scene || !app.has_model ||
        app.gaussian_merge.running)
        return;
    request_gaussian_scene_load(app);
}

std::filesystem::path existing_mesh_path(const App& app) {
    if (!app.imported_mesh_source.empty()) return app.imported_mesh_source;
    std::error_code error;
    const auto exists = [&](const std::filesystem::path& path) {
        return !path.empty() && std::filesystem::exists(path, error);
    };
    const std::array<std::filesystem::path, 4> preferred = {
        app.layout.working_mesh,
        mesh_from_mvs(app.settings) ? app.layout.mvs_mesh_ply
                                    : app.layout.mesh_ply,
        app.layout.mesh_ply, app.layout.mvs_mesh_ply};
    for (const auto& path : preferred)
        if (exists(path)) return path;
    if (exists(app.layout.mvs_raw_mesh_ply)) return app.layout.mvs_raw_mesh_ply;
    return {};
}

void request_mesh_load(App& app, const bool frame_when_ready) {
    if (app.loading_scene) return;
    const auto ply = existing_mesh_path(app);
    const auto ascan = app.layout.project_file;
    const auto textured = app.layout.working_texture;
    const bool have_textured = app.imported_mesh_source.empty() &&
        textured_mesh_on_disk(textured);
    if (ply.empty() && ascan.empty() && !have_textured) {
        app.mesh_load_failed = true;
        set_message(app, "No mesh found to preview", theme::warning);
        return;
    }
    app.loading_scene = true;
    app.mesh_load_failed = false;
    app.frame_mesh_on_load = frame_when_ready;
    app.pending_scene_load_generation = app.scene_load_generation;
    const bool fit_box = !app.scene.has_points() && !app.reconstruction_box.user_set;
    app.pending_mesh_load = std::async(
        std::launch::async,
        [ply, ascan, textured, have_textured, fit_box] {
            return prepare_mesh_preview(
                have_textured ? load_preview_textured_mesh(textured)
                              : load_preview_mesh(ply, ascan), fit_box);
        });
}

void ensure_mesh_loaded(App& app) {
    if (app.mesh.has() || app.loading_scene || app.mesh_load_failed ||
        (!app.has_mesh && !app.has_texture))
        return;
    request_mesh_load(app, true);
}

std::filesystem::path export_output_directory(const App& app) {
    if (!app.layout.project_file.empty())
        return app.layout.project_file.parent_path();
    if (!app.layout.root.empty()) return app.layout.root;
    return std::filesystem::current_path();
}

std::filesystem::path export_stem_path(const App& app) {
    if (!app.layout.project_file.empty()) return app.layout.project_file.stem();
    return std::filesystem::path("project");
}

std::wstring export_stem_wide(const App& app) {
    return export_stem_path(app).wstring();
}

bool can_export_model(const App& app);

photara::splat::GaussianFormat splat_export_format(const App& app) {
    switch (app.splat_export.format) {
        case 1: return photara::splat::GaussianFormat::sog;
        case 2: return photara::splat::GaussianFormat::spz;
        case 3: return photara::splat::GaussianFormat::glb;
        case 6: return photara::splat::GaussianFormat::usdc;
        default: return photara::splat::GaussianFormat::ply;
    }
}

const wchar_t* splat_export_extension_wide(const App& app) {
    switch (app.splat_export.format) {
        case 1: return L"sog";
        case 2: return L"spz";
        case 3: return L"glb";
        case 6: return L"usdc";
        default: return L"ply";
    }
}

int splat_export_format_from_settings(const int settings_format) {
    switch (settings_format) {
        case 2: return 1;
        case 3: return 2;
        case 4: return 3;
        case 5:
#ifdef PHOTARA_HAS_USDC
            return 6;
#else
            return 0;
#endif
        default: return 0;
    }
}

int splat_settings_format_from_export(const int export_format) {
    switch (export_format) {
        case 1: return 2;
        case 2: return 3;
        case 3: return 4;
        case 6: return 5;
        default: return 1;
    }
}

std::filesystem::path suggested_splat_export_path(
    const App& app, std::filesystem::path directory,
    std::filesystem::path stem) {
    if (directory.empty()) directory = export_output_directory(app);
    if (stem.empty()) stem = export_stem_path(app);
    auto path = directory / stem;
    path += "_splat";
    switch (app.splat_export.format) {
        case 1: path += ".sog"; break;
        case 2: path += ".spz"; break;
        case 3: path += ".glb"; break;
        case 4: path += "_lod"; break;
        case 5: path += "_collision"; break;
        case 6: path += ".usdc"; break;
        default: path += ".ply"; break;
    }
    return path;
}

void sync_splat_export_path(App& app, const bool force) {
    if (!force && app.splat_export.path_format == app.splat_export.format &&
        app.splat_export.path[0] != '\0')
        return;
    std::filesystem::path directory = export_output_directory(app);
    std::filesystem::path stem = export_stem_path(app);
    if (app.splat_export.path[0] != '\0') {
        const auto current = path_from_utf8_field(app.splat_export.path.data());
        auto parent = current.parent_path();
        if (!parent.empty()) directory = parent;
        auto current_stem = current.stem();
        if (app.splat_export.path_format == 4 || app.splat_export.path_format == 5) {
            auto name = current.filename().wstring();
            const std::wstring suffix = app.splat_export.path_format == 4 ? L"_lod" : L"_collision";
            if (name.ends_with(suffix)) name.resize(name.size() - suffix.size());
            current_stem = name;
        }
        const std::wstring text = current_stem.wstring();
        constexpr wchar_t suffix[] = L"_splat";
        const std::size_t suffix_n = 6;
        if (text.size() > suffix_n &&
            text.compare(text.size() - suffix_n, suffix_n, suffix) == 0)
            current_stem = text.substr(0, text.size() - suffix_n);
        if (!current_stem.empty()) stem = current_stem;
    }
    store_path_field(
        app.splat_export.path, suggested_splat_export_path(app, directory, stem));
    app.splat_export.path_format = app.splat_export.format;
}

bool browse_splat_export_path(App& app) {
    sync_splat_export_path(app, false);
    if (app.splat_export.format == 4 || app.splat_export.format == 5) {
        std::array<char, 1024> folder{};
        if (!pick_folder(L"Choose parent folder for export", folder)) return false;
        const auto current = path_from_utf8_field(app.splat_export.path.data());
        store_path_field(app.splat_export.path, path_from_utf8_field(folder.data()) / current.filename());
        return true;
    }
    const wchar_t* extension = splat_export_extension_wide(app);
    const auto current = path_from_utf8_field(app.splat_export.path.data());
    const std::wstring name = current.filename().empty()
        ? export_stem_wide(app) + L"_splat." + extension
        : current.filename().wstring();
    return pick_export_path(
        L"Export Splat", app.splat_export.path, FilePickKind::splat_model,
        name.c_str(), extension);
}

void start_splat_simplify(App& app) {
    if (app.splat_simplify.running || app.job.running()) return;
    if (!ensure_splat_renderer(app)) return;
    if (!app.splat_simplify.has_baseline) {
        if (!app.splat_edit.copy_model(app.splat_simplify.baseline) ||
            app.splat_simplify.baseline.size() == 0) {
            set_message(app, tr("Load a trained Gaussian model first."), theme::warning);
            return;
        }
        app.splat_simplify.has_baseline = true;
        app.splat_simplify.source_count =
            static_cast<std::uint32_t>(app.splat_simplify.baseline.size());
    }
    photara::splat::GaussianModel model = app.splat_simplify.baseline;
    auto progress = std::make_shared<std::atomic<float>>(0.F);
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    app.splat_simplify.progress_atom = progress;
    app.splat_simplify.cancel = cancel;
    app.splat_simplify.progress = 0.F;
    app.splat_simplify.running = true;
    app.splat_simplify.error.clear();
    photara::splat::SimplifyOptions options;
    options.keep_ratio = std::clamp(app.splat_simplify.keep_ratio, 0.05F, 1.F);
    options.min_opacity = std::max(app.splat_simplify.min_opacity, 0.F);
    options.progress = [progress](const float value) {
        progress->store(value, std::memory_order_relaxed);
    };
    options.cancelled = [cancel]() {
        return cancel->load(std::memory_order_relaxed);
    };
    app.splat_simplify.job = std::async(
        std::launch::async,
        [model = std::move(model), options = std::move(options)]() mutable {
            return photara::splat::simplify_gaussians(model, options);
        });
}

void poll_splat_simplify(App& app) {
    if (app.splat_simplify.progress_atom)
        app.splat_simplify.progress =
            app.splat_simplify.progress_atom->load(std::memory_order_relaxed);
    if (!app.splat_simplify.job.valid()) return;
    if (app.splat_simplify.job.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    app.splat_simplify.running = false;
    app.splat_simplify.progress_atom.reset();
    try {
        auto model = app.splat_simplify.job.get();
        if (app.splat_simplify.cancel &&
            app.splat_simplify.cancel->load(std::memory_order_relaxed)) {
            set_message(app, tr("Simplify cancelled"), theme::warning);
            return;
        }
        const bool record_undo = !app.splat_simplify.applied;
        app.splat_simplify.result_count =
            static_cast<std::uint32_t>(model.size());
        app.splat_edit.replace_from_model(app, model, record_undo);
        app.splat_simplify.applied = true;
    } catch (const std::exception& error) {
        app.splat_simplify.error = error.what();
        set_message(app, error.what(), theme::danger);
    }
}

void start_gaussian_merge(App& app, const std::filesystem::path& output) {
    if (app.gaussian_merge.running) return;
    if (app.job.running() || app.loading_scene ||
        app.splat_export.job.valid() || app.splat_simplify.running) {
        set_message(
            app, tr("Wait for the current operation before importing a model"),
            theme::warning);
        return;
    }
    if (app.splat_edit.dirty()) {
        set_message(
            app,
            tr("Export Gaussian edits, then start a new project before importing another model"),
            theme::warning);
        return;
    }
    if (output.empty()) return;
    std::vector<GaussianModelEntry> inputs;
    for (const GaussianModelEntry& entry : app.gaussian_models) {
        if (!entry.visible) continue;
        inputs.push_back(entry);
    }
    if (inputs.size() < 2) {
        set_message(
            app, tr("Select at least two visible models to merge"),
            theme::warning);
        return;
    }
    auto progress = std::make_shared<std::atomic<float>>(0.F);
    app.gaussian_merge.progress_atom = progress;
    app.gaussian_merge.running = true;
    app.gaussian_merge.discard = false;
    app.gaussian_merge.progress = 0.F;
    app.gaussian_merge.input_count =
        static_cast<std::uint32_t>(inputs.size());
    app.gaussian_merge.output = output;
    app.gaussian_merge.error.clear();
    app.gaussian_merge.job = std::async(
        std::launch::async, [inputs, output, progress] {
            GaussianMergeJob result;
            try {
                std::vector<photara::splat::GaussianModel> parts;
                parts.reserve(inputs.size());
                for (std::size_t index = 0; index < inputs.size(); ++index) {
                    progress->store(
                        static_cast<float>(index) /
                            static_cast<float>(inputs.size()),
                        std::memory_order_relaxed);
                    parts.push_back(photara::splat::transform_gaussians(
                        *gaussian_source_model(inputs[index]),
                        gaussian_entry_transform(inputs[index])));
                }
                const auto merged = photara::splat::merge_gaussians(parts);
                progress->store(0.92F, std::memory_order_relaxed);
                std::error_code error;
                if (output.has_parent_path())
                    std::filesystem::create_directories(
                        output.parent_path(), error);
                photara::splat::save_gaussians(merged, output);
                progress->store(1.F, std::memory_order_relaxed);
                result.splat_count = merged.size();
            } catch (const std::exception& failure) {
                result.error = failure.what();
            }
            return result;
        });
}

void apply_merged_result(
    App& app, const GaussianMergeJob& result,
    const std::filesystem::path& output, const std::uint32_t input_count) {
    // Replace the list with the merged file so exports and future edits see
    // one ordinary model.
    app.gaussian_models.clear();
    GaussianModelEntry entry;
    entry.path = output;
    entry.name = path_to_utf8(output.filename());
    entry.splat_count = result.splat_count;
    app.gaussian_models.push_back(std::move(entry));
    sync_gaussian_model_settings(app);
    refresh_artifacts(app);
    reset_gaussian_preview(app);
    app.workspace = ViewportWorkspace::scene_3d;
    app.view_mode = VisualizationMode::splat;
    request_gaussian_scene_load(app);
    set_message(
        app,
        "Merged " + std::to_string(input_count) + " models (" +
            format_count(result.splat_count) + " splats) into " +
            path_to_utf8(output.filename()),
        theme::success);
}

bool gaussian_merge_can_apply(const App& app) {
    return !app.job.running() && !app.loading_scene &&
           !app.splat_export.job.valid() && !app.splat_simplify.running &&
           !app.splat_edit.dirty();
}

void poll_gaussian_merge(App& app) {
    if (app.gaussian_merge.running) {
        if (app.gaussian_merge.progress_atom)
            app.gaussian_merge.progress = std::clamp(
                app.gaussian_merge.progress_atom->load(
                    std::memory_order_relaxed),
                0.F, 1.F);
        if (!app.gaussian_merge.job.valid() ||
            app.gaussian_merge.job.wait_for(std::chrono::seconds(0)) !=
                std::future_status::ready)
            return;
        const GaussianMergeJob result = app.gaussian_merge.job.get();
        const bool discard = app.gaussian_merge.discard;
        app.gaussian_merge.running = false;
        app.gaussian_merge.progress_atom.reset();
        if (discard) {
            app.gaussian_merge = {};
            return;
        }
        if (!result.error.empty()) {
            app.gaussian_merge.error = result.error;
            set_message(app, "Merge failed: " + result.error, theme::danger);
            return;
        }
        app.gaussian_merge.pending = result;
        app.gaussian_merge.apply_pending = true;
        if (!gaussian_merge_can_apply(app)) {
            set_message(
                app,
                tr("Merge written. Finish or export the current edits, then "
                "the merged model loads automatically."),
                theme::accent);
        }
    }
    if (app.gaussian_merge.apply_pending && gaussian_merge_can_apply(app)) {
        const GaussianMergeJob result = app.gaussian_merge.pending;
        const auto output = app.gaussian_merge.output;
        const auto input_count = app.gaussian_merge.input_count;
        app.gaussian_merge.pending = {};
        app.gaussian_merge.apply_pending = false;
        apply_merged_result(app, result, output, input_count);
    }
}

void open_splat_export_panel(App& app) {
    if (app.job.running() || !can_export_model(app)) return;
    if (!app.splat_export.initialized) {
        app.splat_export.format =
            splat_export_format_from_settings(app.settings.splat_format);
        app.splat_export.sh_degree = 3;
        app.splat_export.initialized = true;
    }
    app.splat_export.sh_degree =
        std::clamp(app.splat_export.sh_degree, 0, 3);
    sync_splat_export_path(app, app.splat_export.path[0] == '\0');
    app.splat_export.show = true;
}

bool copy_existing_file(
    const std::filesystem::path& source, const std::filesystem::path& destination) {
    std::error_code error;
    if (source.empty() || !std::filesystem::exists(source, error)) return false;
    std::filesystem::create_directories(destination.parent_path(), error);
    std::filesystem::copy_file(
        source, destination,
        std::filesystem::copy_options::overwrite_existing, error);
    return !error;
}

void write_sparse_ply(
    const SparseScene& scene, const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create PLY: " + path.string());
    const bool has_colour =
        scene.colours.size() == scene.points.size() && !scene.colours.empty();
    output << "ply\nformat ascii 1.0\nelement vertex " << scene.points.size()
           << "\nproperty float x\nproperty float y\nproperty float z\n";
    if (has_colour)
        output << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    output << "end_header\n";
    for (std::size_t i = 0; i < scene.points.size(); ++i) {
        const Vec3& point = scene.points[i];
        output << point.x << ' ' << point.y << ' ' << point.z;
        if (has_colour) {
            const std::uint32_t colour = scene.colours[i];
            output << ' ' << ((colour >> IM_COL32_R_SHIFT) & 0xFF) << ' '
                   << ((colour >> IM_COL32_G_SHIFT) & 0xFF) << ' '
                   << ((colour >> IM_COL32_B_SHIFT) & 0xFF);
        }
        output << '\n';
    }
}

photara::mvs::Mesh preview_mesh_to_mvs(const PreviewMesh& mesh) {
    photara::mvs::Mesh out;
    out.vertices.reserve(mesh.vertices.size());
    for (const Vec3& vertex : mesh.vertices)
        out.vertices.emplace_back(vertex.x, vertex.y, vertex.z);
    if (mesh.normals.size() == mesh.vertices.size()) {
        out.normals.reserve(mesh.normals.size());
        for (const Vec3& normal : mesh.normals)
            out.normals.emplace_back(normal.x, normal.y, normal.z);
    }
    if (mesh.colours.size() == mesh.vertices.size()) {
        out.colors.reserve(mesh.colours.size());
        for (const std::uint32_t colour : mesh.colours) {
            out.colors.emplace_back(
                static_cast<float>((colour >> IM_COL32_R_SHIFT) & 0xFF) / 255.F,
                static_cast<float>((colour >> IM_COL32_G_SHIFT) & 0xFF) / 255.F,
                static_cast<float>((colour >> IM_COL32_B_SHIFT) & 0xFF) / 255.F);
        }
    }
    out.faces.reserve(mesh.faces.size());
    for (const auto& face : mesh.faces)
        out.faces.emplace_back(
            static_cast<int>(face[0]), static_cast<int>(face[1]),
            static_cast<int>(face[2]));
    return out;
}

bool can_export_sparse(const App& app) {
    return app.has_sparse || app.scene.has_points();
}

bool can_export_model(const App& app) { return app.has_model; }

bool can_export_textured_mesh(const App& app) {
    if (!app.imported_mesh_source.empty()) return false;
    return app.has_texture || textured_mesh_on_disk(app.layout.working_texture);
}

bool can_export_mesh_file(const App& app) {
    return app.has_mesh || app.mesh.has() || can_export_textured_mesh(app);
}

void poll_splat_export(App& app) {
    auto& state = app.splat_export;
    if (!state.job.valid() || state.job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    try {
        state.job.get();
        state.error.clear();
        set_message(app, "Exported splat assets: " + path_to_utf8(state.output), theme::success);
    } catch (const std::exception& error) {
        state.error = error.what();
        set_message(app, state.error, theme::danger);
    }
    state.cancel.reset();
    state.progress.reset();
}

// Export source model. A multi-model session composes the same merged cloud
// the viewport shows, so exported files always match the preview.
photara::splat::GaussianModel load_export_model(
    const App& app, const tinytensor::Device device) {
    if (gaussian_models_managed(app)) {
        if (app.composed_model) return *app.composed_model;
        std::vector<photara::splat::GaussianModel> parts;
        for (const GaussianModelEntry& entry : app.gaussian_models) {
            if (!entry.visible) continue;
            parts.push_back(photara::splat::transform_gaussians(
                *gaussian_source_model(entry),
                gaussian_entry_transform(entry)));
        }
        return photara::splat::merge_gaussians(parts);
    }
    const auto source = existing_splat_model(app);
    if (!source.empty())
        return photara::splat::load_gaussians(source, device);
    const auto archive =
        photara::project::Archive::open(app.layout.project_file);
    return photara::splat::decode_gaussians(
        archive.chunk(photara::project::ChunkType::gaussians));
}

void export_trained_model(App& app) {
    if (app.job.running() || app.splat_export.job.valid() || !can_export_model(app)) return;
    const auto format = splat_export_format(app);
    const auto* extension = splat_export_extension_wide(app);
    app.splat_export.sh_degree = std::clamp(app.splat_export.sh_degree, 0, 3);
    std::filesystem::path out =
        path_from_utf8_field(app.splat_export.path.data());
    if (out.empty()) {
        if (!browse_splat_export_path(app)) return;
        out = path_from_utf8_field(app.splat_export.path.data());
    }
    if (out.empty()) return;
    if (app.splat_export.format != 4 && app.splat_export.format != 5 && out.extension().empty()) {
        out += ".";
        out += std::filesystem::path(extension);
        store_path_field(app.splat_export.path, out);
    }
    try {
        const auto source = existing_splat_model(app);
        photara::splat::GaussianModel model;
        const bool from_edit =
            app.splat_edit.dirty() && app.splat_edit.copy_model(model);
        // Embedded or cached sources can differ from the original file, even
        // for a single untransformed model. Always export the loaded cloud.
        const bool composed_export = gaussian_models_managed(app);
        if (!from_edit) {
            model = load_export_model(
                app, app.splat_export.format >= 4 ? tinytensor::Device::CPU
                                                  : tinytensor::Device::CUDA);
        }
        const unsigned export_degree =
            std::min(model.sh_degree,
                     static_cast<unsigned>(app.splat_export.sh_degree));
        if (app.splat_export.format == 4 || app.splat_export.format == 5) {
            photara::splat::restrict_sh_degree(model, export_degree);
            auto cancel = std::make_shared<std::atomic<bool>>(false);
            auto progress = std::make_shared<std::atomic<float>>(0.F);
            auto lod = app.splat_export.lod;
            auto collision = app.splat_export.collision;
            collision.fill = static_cast<photara::splat::CollisionOptions::Fill>(app.splat_export.fill_mode);
            lod.cancelled = collision.cancelled = [cancel] { return cancel->load(); };
            lod.progress = collision.progress = [progress](float p) { progress->store(p); };
            app.splat_export.cancel = cancel;
            app.splat_export.progress = progress;
            app.splat_export.output = std::filesystem::absolute(out);
            app.splat_export.error.clear();
            const bool mesh = app.splat_export.format == 5;
            app.splat_export.job = std::async(std::launch::async,
                [model = std::move(model), out = app.splat_export.output, lod, collision, mesh] {
                    if (mesh) photara::splat::export_collision(model, out, collision);
                    else photara::splat::export_streamed_sog(model, out, lod);
                });
            app.splat_export.show = true;
            set_message(app, "Processing splat assets on Vulkan...", theme::success);
            return;
        }
        const auto dest_ext = lower_path_extension(out);
        const auto source_ext = lower_path_extension(source);
        if (!from_edit && !composed_export &&
            export_degree >= model.sh_degree && !source.empty() &&
            source_ext == dest_ext && copy_existing_file(source, out)) {
            app.settings.splat_format =
                splat_settings_format_from_export(app.splat_export.format);
            set_message(app, "Exported splat", theme::success);
            return;
        }
        photara::splat::restrict_sh_degree(model, export_degree);
        photara::splat::save_gaussians(model, out, format);
        app.settings.splat_format =
            splat_settings_format_from_export(app.splat_export.format);
        std::string message = "Exported splat (SH ";
        message += std::to_string(export_degree);
        message += ")";
        set_message(app, message, theme::success);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

enum class MeshExportFormat { ply, obj, glb };

MeshExportFormat mesh_export_format(const App& app) {
    switch (app.mesh_export.format) {
        case 1: return MeshExportFormat::obj;
        case 2: return MeshExportFormat::glb;
        default: return MeshExportFormat::ply;
    }
}

bool mesh_export_wants_texture(const App& app) {
    return app.mesh_export.include_texture &&
           mesh_export_format(app) != MeshExportFormat::ply &&
           can_export_textured_mesh(app);
}

const wchar_t* mesh_export_extension_wide(const MeshExportFormat format) {
    switch (format) {
        case MeshExportFormat::obj: return L"obj";
        case MeshExportFormat::glb: return L"glb";
        case MeshExportFormat::ply:
        default: return L"ply";
    }
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Failed to read " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("Failed to size " + path.string());
    const auto size = static_cast<std::size_t>(end);
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    if (size > 0U)
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error("Failed to read " + path.string());
    return bytes;
}

std::filesystem::path ensure_textured_stem(App& app) {
    if (!app.imported_mesh_source.empty()) return {};
    if (textured_mesh_on_disk(app.layout.working_texture))
        return app.layout.working_texture;
#if defined(PHOTARA_HAS_TEXTURE)
    if (!app.layout.project_file.empty()) {
        const auto archive =
            photara::project::Archive::open(app.layout.project_file);
        if (archive.has(photara::project::ChunkType::texture)) {
            photara::texture::decode_textured_obj(
                archive.chunk(photara::project::ChunkType::texture),
                app.layout.working_texture);
            refresh_artifacts(app);
            if (textured_mesh_on_disk(app.layout.working_texture))
                return app.layout.working_texture;
        }
    }
#endif
    return {};
}

photara::mvs::Mesh load_geometry_mesh(App& app) {
    const auto ply = existing_mesh_path(app);
    if (!app.imported_mesh_source.empty()) {
        const auto loaded = load_preview_mesh(ply, {});
        if (!loaded.ok) throw std::runtime_error(loaded.error);
        return preview_mesh_to_mvs(loaded.mesh);
    }
    if (!ply.empty()) return photara::mvs::load_mesh_ply(ply);
    if (!app.layout.project_file.empty()) {
        const auto archive =
            photara::project::Archive::open(app.layout.project_file);
        if (archive.has(photara::project::ChunkType::mesh))
            return photara::mvs::decode_mesh(
                archive.chunk(photara::project::ChunkType::mesh));
    }
    if (app.mesh.has()) return preview_mesh_to_mvs(app.mesh);
    const auto textured = ensure_textured_stem(app);
    if (!textured.empty()) {
        const MeshLoad loaded = load_preview_textured_mesh(textured);
        if (loaded.ok) return preview_mesh_to_mvs(loaded.mesh);
        if (!loaded.error.empty()) throw std::runtime_error(loaded.error);
    }
    throw std::runtime_error("No mesh to export yet");
}

void export_textured_obj_files(
    App& app, const std::filesystem::path& destination_stem) {
    const auto source = ensure_textured_stem(app);
#if defined(PHOTARA_HAS_TEXTURE)
    if (!source.empty()) {
        photara::texture::copy_textured_obj(source, destination_stem);
        return;
    }
    if (!app.layout.project_file.empty()) {
        const auto archive =
            photara::project::Archive::open(app.layout.project_file);
        if (archive.has(photara::project::ChunkType::texture)) {
            photara::texture::decode_textured_obj(
                archive.chunk(photara::project::ChunkType::texture),
                destination_stem);
            return;
        }
    }
#endif
    if (source.empty())
        throw std::runtime_error("No textured mesh to export yet");
    if (!copy_existing_file(
            textured_obj_path(source), textured_obj_path(destination_stem)) ||
        !copy_existing_file(
            textured_albedo_path(source),
            textured_albedo_path(destination_stem)))
        throw std::runtime_error("Failed to copy textured mesh files");
    copy_existing_file(
        textured_mtl_path(source), textured_mtl_path(destination_stem));
}

void export_textured_glb(
    App& app, const std::filesystem::path& destination) {
    const auto source = ensure_textured_stem(app);
    if (source.empty())
        throw std::runtime_error("No textured mesh to export yet");
    const MeshLoad loaded = load_preview_textured_mesh(source);
    if (!loaded.ok)
        throw std::runtime_error(
            loaded.error.empty() ? "Textured mesh has no faces" : loaded.error);
    photara::mvs::Mesh mesh = preview_mesh_to_mvs(loaded.mesh);
    std::vector<photara::mvs::Vec2f> uvs;
    if (loaded.mesh.uvs.size() == loaded.mesh.vertices.size()) {
        uvs.reserve(loaded.mesh.uvs.size());
        for (const Vec2& uv : loaded.mesh.uvs)
            uvs.emplace_back(uv.x, 1.F - uv.y);
    }
    const auto png_path = loaded.mesh.albedo_path.empty()
        ? textured_albedo_path(source)
        : loaded.mesh.albedo_path;
    photara::io::save_mesh_glb(
        mesh, destination, uvs, read_file_bytes(png_path));
}

void open_mesh_export_panel(App& app) {
    if (app.job.running() || !can_export_mesh_file(app)) return;
    if (!app.mesh_export.initialized) {
        if (can_export_textured_mesh(app)) {
            app.mesh_export.format = 2;
            app.mesh_export.include_texture = true;
        }
        app.mesh_export.initialized = true;
    }
    app.mesh_export.show = true;
}

void export_mesh_file(App& app) {
    if (app.job.running() || !can_export_mesh_file(app)) return;
    const auto format = mesh_export_format(app);
    const bool with_texture = mesh_export_wants_texture(app);
    const wchar_t* extension = mesh_export_extension_wide(format);
    std::array<char, 1024> destination{};
    const std::wstring name =
        export_stem_wide(app) + L"_mesh." + std::wstring(extension);
    if (!pick_export_path(
            L"Export Mesh", destination, FilePickKind::mesh, name.c_str(),
            extension))
        return;
    std::filesystem::path out = path_from_utf8_field(destination.data());
    if (out.extension().empty()) {
        out += ".";
        out += std::filesystem::path(extension);
    }
    try {
        if (with_texture && format == MeshExportFormat::obj) {
            std::filesystem::path stem = out;
            stem.replace_extension();
            export_textured_obj_files(app, stem);
            set_message(
                app, "Exported mesh (OBJ + MTL + albedo PNG)", theme::success);
            return;
        }
        if (with_texture && format == MeshExportFormat::glb) {
            export_textured_glb(app, out);
            set_message(app, "Exported mesh (GLB with albedo)", theme::success);
            return;
        }
        if (format == MeshExportFormat::ply) {
            const auto source = existing_mesh_path(app);
            if (copy_existing_file(source, out)) {
                set_message(app, "Exported mesh", theme::success);
                return;
            }
            photara::mvs::save_mesh_ply(load_geometry_mesh(app), out);
            set_message(app, "Exported mesh", theme::success);
            return;
        }
        if (format == MeshExportFormat::obj) {
            photara::mvs::save_mesh_obj(load_geometry_mesh(app), out);
            set_message(app, "Exported mesh", theme::success);
            return;
        }
        photara::io::save_mesh_glb(load_geometry_mesh(app), out);
        set_message(app, "Exported mesh", theme::success);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}
std::array<float, 16> mesh_world_to_clip(
    const SplatPreviewCamera& camera, const float near_z, const float far_z) {
    const float* m = camera.world_to_camera.data();
    const float r00 = m[0], r10 = m[1], r20 = m[2];
    const float r01 = m[4], r11 = m[5], r21 = m[6];
    const float r02 = m[8], r12 = m[9], r22 = m[10];
    const float tx = m[12], ty = m[13], tz = m[14];
    const float width = static_cast<float>(std::max(1U, camera.width));
    const float height = static_cast<float>(std::max(1U, camera.height));
    const float x_offset = 2.F * (camera.cx + 0.5F) / width - 1.F;
    const float y_offset = 2.F * (camera.cy + 0.5F) / height - 1.F;
    const float depth_scale = (far_z + near_z) / (far_z - near_z);
    const float depth_offset = -2.F * far_z * near_z / (far_z - near_z);
    const float fx_term = 2.F * camera.fx / width;
    const float fy_term = 2.F * camera.fy / height;
    std::array<float, 16> clip{};
    clip[0 * 4 + 0] = fx_term * r00 + x_offset * r20;
    clip[0 * 4 + 1] = fx_term * r01 + x_offset * r21;
    clip[0 * 4 + 2] = fx_term * r02 + x_offset * r22;
    clip[0 * 4 + 3] = fx_term * tx + x_offset * tz;
    clip[1 * 4 + 0] = fy_term * r10 + y_offset * r20;
    clip[1 * 4 + 1] = fy_term * r11 + y_offset * r21;
    clip[1 * 4 + 2] = fy_term * r12 + y_offset * r22;
    clip[1 * 4 + 3] = fy_term * ty + y_offset * tz;
    clip[2 * 4 + 0] = depth_scale * r20;
    clip[2 * 4 + 1] = depth_scale * r21;
    clip[2 * 4 + 2] = depth_scale * r22;
    clip[2 * 4 + 3] = depth_scale * tz + depth_offset;
    clip[3 * 4 + 0] = r20;
    clip[3 * 4 + 1] = r21;
    clip[3 * 4 + 2] = r22;
    clip[3 * 4 + 3] = tz;
    return clip;
}

bool update_gpu_mesh_preview(App& app, const ImVec2 min, const ImVec2 max) {
    if (!app.mesh.has() || app.mesh_gpu_failed || !app.mesh_gpu_data ||
        app.mesh_gpu_data->indices.empty())
        return false;
    std::uint32_t width = static_cast<std::uint32_t>(
        std::max(1.F, std::floor(max.x - min.x)));
    std::uint32_t height = static_cast<std::uint32_t>(
        std::max(1.F, std::floor(max.y - min.y)));
    SplatPreviewCamera camera =
        make_preview_camera(app.camera, width, height);
    const Vec3 world_centroid = app.mesh.centroid;
    const float world_radius = app.mesh.radius;
    const float* w2c = camera.world_to_camera.data();
    const float cam_z =
        w2c[2] * world_centroid.x + w2c[6] * world_centroid.y +
        w2c[10] * world_centroid.z + w2c[14];
    float near_z = std::max(1e-4F, std::abs(cam_z) - world_radius * 1.25F);
    float far_z = std::max(near_z + 1e-3F, std::abs(cam_z) + world_radius * 1.25F);
    gpu::MeshPreviewUniforms uniforms;
    uniforms.world_to_clip = mesh_world_to_clip(camera, near_z, far_z);
    uniforms.world_to_camera = camera.world_to_camera;
    uniforms.vertex_colour = app.view_options.mesh_vertex_colour;
    uniforms.textured = app.view_options.mesh_texture && app.mesh.has_texture();
    uniforms.wireframe = app.view_options.mesh_wireframe;
    try {
        if (!app.mesh_gpu_uploaded) {
            app.mesh_renderer.set_mesh(app.mesh_gpu_data);
            app.mesh_gpu_uploaded = true;
        }
        return app.mesh_renderer.draw(width, height, uniforms);
    } catch (const std::exception& failure) {
        app.mesh_gpu_failed = true;
        set_message(
            app, std::string("Mesh GPU rasterizer: ") + failure.what(),
            theme::warning);
        return false;
    }
}

void show_mesh_view(App& app, const bool frame_when_ready) {
    app.view_mode = VisualizationMode::mesh;
    if (app.mesh.has()) {
        // Manual mode switching keeps the orbit camera so mesh and 3DGS can
        // be compared from the exact same viewpoint; only callers that just
        // produced a new mesh ask for reframing.
        if (frame_when_ready) frame_reconstruction(app);
        return;
    }
    app.mesh_load_failed = false;
    request_mesh_load(app, frame_when_ready);
}

photara::splat::VisualizeOptions editor_visualize_options(const App& app) {
    photara::splat::VisualizeOptions options;
    const bool vulkan_training = live_preview_active(app) &&
                                 app.settings.training_backend == 1;
    options.mode = vulkan_training && app.view_mode == VisualizationMode::points
        ? photara::splat::VisualizationMode::splat
        : app.view_mode == VisualizationMode::points
        ? photara::splat::VisualizationMode::points
        : app.view_mode == VisualizationMode::rings
            ? photara::splat::VisualizationMode::rings
            : photara::splat::VisualizationMode::splat;
    options.point_size_px = app.view_options.point_size;
    options.ring_scale = app.view_options.ring_scale;
    if (app.workspace == ViewportWorkspace::scene_3d && app.view_mode == VisualizationMode::splat) {
        options.channel = app.display_channel == DisplayChannel::depth
            ? photara::splat::VisualizationChannel::depth
            : app.display_channel == DisplayChannel::normal
                ? photara::splat::VisualizationChannel::normal
                : photara::splat::VisualizationChannel::color;
    }
    options.automatic_depth = app.display_options.automatic_depth;
    options.grayscale_depth = app.display_options.grayscale_depth;
    options.world_normals = app.display_options.world_normals;
    options.depth_near = app.display_options.depth_near;
    options.depth_far = app.display_options.depth_far;
    return options;
}

void write_preview_vis(App& app) {
    if (app.layout.preview_vis_file.empty()) return;
    photara::splat::write_visualization_sidecar(
        app.layout.preview_vis_file, editor_visualize_options(app),
        app.preview_camera_revision);
}

void publish_preview_vis(App& app) {
    ++app.preview_camera_revision;
    write_preview_vis(app);
}

void publish_preview_ack(App& app) {
    if (app.layout.preview_ack_file.empty()) return;
    // Only the external-memory transport has a shared image to hand over; the
    // offline preview paths have no handshake for the trainer to relax.
    if (!app.preview.timeline) return;
    const std::uint64_t copied = gpu::copied_preview_frames();
    if (copied == app.preview_ack_frames) return;
    std::ofstream output(app.layout.preview_ack_file, std::ios::trunc);
    if (!output) return;
    output << copied << '\n';
    if (!output) return;
    app.preview_ack_frames = copied;
}

void set_visualization_mode(App& app, const VisualizationMode mode) {
    app.view_mode = mode;
    // Changing the rail mode is an explicit request to show scene data again.
    if (mode == VisualizationMode::points)
        app.suppress_scene_auto_load = false;
    if (mode == VisualizationMode::mesh) {
        show_mesh_view(app, false);
        write_preview_vis(app);
        return;
    }
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    if (training) {
        write_preview_vis(app);
        sync_live_preview_camera(
            app, true, app.preview_raster_width, app.preview_raster_height);
        return;
    }
    // After training the splat image is drawn in-process. CUDA stays on the
    // training job.
    if (mode == VisualizationMode::points && !app.scene.has_points())
        ensure_sparse_loaded(app);
    write_preview_vis(app);
}
void apply_opened_project(App& app) {
    clear_loaded_result(app);
    app.settings.runtime_dir.fill('\0');
    app.saved_results_discarded = false;
    app.project_results_dirty = false;
    const auto snapshot_cache = resolve_layout(app.settings).working_sfm.parent_path();
    store_utf8_path_field(app.settings.runtime_dir, snapshot_cache /
        ("opened-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
    refresh_artifacts(app);
    try {
        auto archive =
            photara::project::Archive::open(app.layout.project_file);
        apply_project_settings(app, photara::project::read_settings(archive));
        auto sources = photara::project::read_gaussian_sources(archive);
        if (!sources.empty() && sources.size() != app.gaussian_models.size())
            throw std::runtime_error("Project Gaussian source count does not match its model list");
        for (std::size_t i = 0; i < sources.size(); ++i)
            app.gaussian_models[i].embedded =
                std::make_shared<const std::vector<std::uint8_t>>(std::move(sources[i]));
        if (gaussian_models_managed(app)) request_gaussian_scene_load(app);

        refresh_artifacts(app);
        mark_project_saved(app);
        if (app.training_checkpoint_info) {
            const auto& bounds = app.training_checkpoint_info->focus_bounds;
            if (bounds.valid && bounds.axes.isApprox(photara::mvs::Mat3f::Identity(), 1e-6F)) {
                const auto minimum = (bounds.center - bounds.half_extent).eval();
                const auto maximum = (bounds.center + bounds.half_extent).eval();
                app.reconstruction_box.min = {minimum.x(), minimum.y(), minimum.z()};
                app.reconstruction_box.max = {maximum.x(), maximum.y(), maximum.z()};
                app.reconstruction_box.valid = true;
                app.reconstruction_box.user_set = true;
            }
            set_message(app, "Saved training progress found. Use Resume Training to resume.", theme::accent);
        }
#if defined(PHOTARA_HAS_TEXTURE)
        if (archive.has(photara::project::ChunkType::texture) &&
            !textured_mesh_on_disk(app.layout.working_texture)) {
            try {
                photara::texture::decode_textured_obj(
                    archive.chunk(photara::project::ChunkType::texture),
                    app.layout.working_texture);
                refresh_artifacts(app);
            } catch (...) {
            }
        }
#endif
    } catch (const std::exception& error) {
        set_message(app, error.what(), theme::danger);
        return;
    }
    if (has_external_dataset(app))
        // The imported alignment drives training for this project; internal
        // SfM artifacts, if any, describe a different capture setup.
        request_dataset_scene_load(app);
    else if (app.has_sparse)
        request_ascan_scene_load(app);
    else if (app.has_asfm)
        request_asfm_scene_load(
            app, app.layout.sparse_asfm, app.layout.sparse_asfm.filename().string());
    else
        set_message(app, "Project opened; no SfM stage yet", theme::text_muted);
}

void select_project_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (defer_project_transition(app, [&app] { select_project_folder(app); })) return;
    auto field = app.settings.project_dir;
    if (!pick_project_file(
            L"Open Photara Project", field, false))
        return;
    const auto path = path_from_utf8_field(field.data());
    try { (void)photara::project::read_settings(photara::project::Archive::open(path)); }
    catch (const std::exception& error) { set_message(app, error.what(), theme::danger); return; }
    store_utf8_path_field(app.settings.project_dir, path);
    apply_opened_project(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot open a project while a job is running", theme::warning);
        return;
    }
    if (defer_project_transition(app, [&app, path] { open_project_from_path(app, path); })) return;
    try { (void)photara::project::read_settings(photara::project::Archive::open(path)); }
    catch (const std::exception& error) { set_message(app, error.what(), theme::danger); return; }
    store_utf8_path_field(app.settings.project_dir, path);
    apply_opened_project(app);
}

std::filesystem::path inferred_images_dir_near(
    const std::filesystem::path& asfm) {
    const auto parent = asfm.parent_path();
    if (directory_has_images(parent)) return parent;
    const auto images = parent / "images";
    if (directory_has_images(images)) return images;
    return {};
}

void infer_images_dir_from_scene(App& app) {
    if (app.settings.images_dir[0] != '\0') return;
    std::optional<std::filesystem::path> parent;
    for (const ViewPose& view : app.scene.views) {
        if (view.image_path.empty() || view.image_path.is_relative()) continue;
        const auto directory = view.image_path.parent_path();
        if (directory.empty()) continue;
        if (!parent)
            parent = directory;
        else if (!same_directory(*parent, directory))
            return;
    }
    if (parent && directory_has_images(*parent))
        store_utf8_path_field(app.settings.images_dir, *parent);
}

void open_asfm_from_path(App& app, const std::filesystem::path& asfm) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot open SfM while a job is running", theme::warning);
        return;
    }
    if (defer_project_transition(app, [&app, asfm] { open_asfm_from_path(app, asfm); })) return;
    clear_loaded_result(app);
    app.settings.project_dir.fill('\0');
    app.settings.runtime_dir.fill('\0');
    app.saved_project_settings.clear();
    app.project_results_dirty = true;

    const auto sibling = asfm.parent_path() / (asfm.stem().string() + ".ascan");
    std::error_code error;
    if (std::filesystem::exists(sibling, error)) {
        store_utf8_path_field(app.settings.project_dir, sibling);
        try {
            refresh_artifacts(app);
            const auto archive =
                photara::project::Archive::open(app.layout.project_file);
            apply_project_settings(
                app, photara::project::read_settings(archive));
        } catch (...) {
        }
    }

    if (app.settings.images_dir[0] == '\0') {
        const auto inferred = inferred_images_dir_near(asfm);
        if (!inferred.empty())
            store_utf8_path_field(app.settings.images_dir, inferred);
    }
    ensure_task_cache(app);
    refresh_artifacts(app);
    app.layout.sparse_asfm = asfm;
    if (!app.layout.working_sfm.empty()) {
        const auto scene = load_working_sfm(asfm, reconstruction_images_path(app));
        photara::sfm::AsfmOptions options;
        options.path_base = reconstruction_images_path(app);
        std::filesystem::create_directories(app.layout.working_sfm.parent_path());
        photara::sfm::save_asfm(scene, app.layout.working_sfm, options);
    }
    app.has_sparse = true;
    app.has_asfm = true;
    request_asfm_scene_load(app, asfm, asfm.filename().string());
}

bool choose_project_save_path(App& app, const bool save_as = false) {
    ensure_task_cache(app);
    refresh_artifacts(app);
    if (!save_as && !app.layout.project_file.empty()) {
        app.pending_project_save_path = app.layout.project_file;
        return true;
    }
    auto field = app.settings.project_dir;
    if (!pick_project_file(L"Save Photara Project", field, true)) return false;
    auto path = path_from_utf8_field(field.data());
    if (lower_path_extension(path) != ".ascan") path.replace_extension(".ascan");
    app.pending_project_save_path = path;
    return true;
}

void save_project_as(App& app) {
    if (app.job.running() || !choose_project_save_path(app, true)) return;
    if (photara::splat::training_checkpoint_in_use(app.pending_project_save_path)) {
        set_message(app, "This project is being trained in another process. Stop that task before saving.", theme::warning);
        return;
    }
    if (save_project_to_path(app, app.pending_project_save_path)) {
        app.pending_project_save_path.clear();
        set_message(app, "Project saved", theme::success);
        if (app.pending_project_action) complete_project_transition(app);
    }
}

void save_project(App& app) {
    request_project_save(app, false);
}

void cancel_project_save(App& app) {
    if (!app.saving_training) return;
    std::error_code ignored;
    // Truncation is safe even when a Windows reader briefly holds the file.
    { std::ofstream cancel(app.layout.training_control, std::ios::trunc); }
    std::filesystem::remove(app.layout.training_control, ignored);
    if (app.job.running() && app.paused_before_save) app.job.pause();
    app.saving_training = false;
    app.exit_after_save = false;
}

void request_project_save(App& app, const bool exit_after_save) {
    if (app.saving_training) return;
    app.checkpoint_save_error.clear();
    if (!choose_project_save_path(app)) return;
    if (photara::splat::training_checkpoint_in_use(app.pending_project_save_path)) {
        app.checkpoint_save_error = "This project is being trained in another process. Stop that task before saving.";
        set_message(app, app.checkpoint_save_error, theme::warning);
        return;
    }
    if (app.job.running() && app.active_job == JobKind::train) {
        try {
            std::filesystem::create_directories(app.layout.training_control.parent_path());
            auto reply = app.layout.training_control; reply += ".reply";
            std::error_code ignored; std::filesystem::remove(reply, ignored);
            app.checkpoint_request = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            // Atomic request publication: the child must never see a partial ID.
            auto staged = app.layout.training_control; staged += ".tmp";
            {
                std::ofstream output(staged, std::ios::trunc);
                output << app.checkpoint_request << '\n'; output.close();
                if (!output) throw std::runtime_error("Cannot request a training checkpoint");
            }
            std::filesystem::remove(app.layout.training_control, ignored);
            std::filesystem::rename(staged, app.layout.training_control);
            app.paused_before_save = app.job.paused();
            app.job.resume(); // An OS-suspended trainer cannot acknowledge a save.
            app.saving_training = true;
            app.exit_after_save = exit_after_save;
            app.checkpoint_save_started = std::chrono::steady_clock::now();
            set_message(app, "Saving project and training progress...", theme::accent);
        } catch (const std::exception& error) {
            app.checkpoint_save_error = error.what();
            set_message(app, error.what(), theme::danger);
        }
        return;
    }
    ensure_task_cache(app);
    refresh_artifacts(app);
    if (!app.job.running() && photara::splat::training_checkpoint_in_use(app.layout.project_file)) {
        app.checkpoint_save_error = "This project is being trained in another process. Stop that task before saving.";
        set_message(app, app.checkpoint_save_error, theme::warning); return;
    }
    const bool was_paused = app.job.paused();
    if (app.job.running()) app.job.pause();
    if (save_project_to_path(app, app.pending_project_save_path)) {
        app.pending_project_save_path.clear();
        app.close_approved = exit_after_save;
        set_message(app, "Project saved", theme::success);
        if (app.pending_project_action) complete_project_transition(app);
    } else app.checkpoint_save_error = app.message;
    if (app.job.running() && !was_paused) app.job.resume();
}

void poll_project_save(App& app) {
    if (!app.saving_training) return;
    auto reply = app.layout.training_control; reply += ".reply";
    std::ifstream input(reply);
    std::string request, response;
    std::getline(input, request); std::getline(input, response);
    const bool acknowledged = request == app.checkpoint_request && !response.empty();
    if (!acknowledged && app.job.running()) return;
    if (acknowledged && response.starts_with("error ")) {
        app.checkpoint_save_error = response.substr(6);
        cancel_project_save(app);
        set_message(app, app.checkpoint_save_error, theme::danger);
        return;
    }
    if (acknowledged && !response.starts_with("saved ")) return;
    if (!acknowledged && app.job.exit_code() != 0) {
        app.checkpoint_save_error = "Training stopped before confirming the save. The last automatic checkpoint is retained.";
        cancel_project_save(app);
        set_message(app, app.checkpoint_save_error, theme::danger);
        return;
    }
    const bool exit_after_save = app.exit_after_save;
    if (save_project_to_path(app, app.pending_project_save_path)) {
        app.pending_project_save_path.clear();
        if (exit_after_save) {
            app.close_approved = true; // Keep the child at its safe boundary until shutdown.
            app.saving_training = false;
        } else cancel_project_save(app);
        set_message(app, "Project and training progress saved", theme::success);
    } else {
        app.checkpoint_save_error = app.message;
        cancel_project_save(app);
    }
}

bool has_reconstruction_result(const App& app) {
    if (app.scene.has_points() || app.has_sparse || app.has_asfm ||
        app.has_mvs || app.has_model || app.has_mesh || app.has_texture)
        return true;
    std::error_code dense_error;
    if (!app.layout.dense_ply.empty() &&
        std::filesystem::exists(app.layout.dense_ply, dense_error))
        return true;
    if (!app.layout.working_dense.empty() &&
        std::filesystem::exists(app.layout.working_dense, dense_error))
        return true;
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.cache, error);
}

void delete_reconstruction_results(App& app) {
    if (app.job.running() || app.loading_scene || app.layout.root.empty()) return;

    clear_loaded_result(app);
    app.suppress_scene_auto_load = true;
    const std::array<std::filesystem::path, 25> generated_files = {
        app.layout.sparse_ply, app.layout.sparse_asfm, app.layout.sparse_mvs,
        app.layout.sparse_poses, app.layout.splat_ply, app.layout.splat_sog,
        app.layout.splat_spz, app.layout.splat_glb, app.layout.splat_usdc, app.layout.mesh_ply,
        app.layout.mvs_mesh_ply, app.layout.mvs_raw_mesh_ply,
        app.layout.dense_ply, app.layout.working_splat, app.layout.working_mesh,
        app.layout.working_dense, textured_obj_path(app.layout.working_texture),
        textured_mtl_path(app.layout.working_texture),
        textured_albedo_path(app.layout.working_texture),
        app.layout.align_log, app.layout.train_log, app.layout.dense_log,
        app.layout.texture_log, app.layout.export_log};

    std::uintmax_t removed = 0;
    std::string failure;
    try { photara::splat::remove_training_checkpoint(app.layout.training_checkpoint); }
    catch (const std::exception& error) { failure = error.what(); }
    for (const std::filesystem::path& path : generated_files) {
        if (path.empty()) continue;
        std::error_code error;
        if (std::filesystem::remove(path, error)) ++removed;
        if (error && failure.empty()) failure = error.message();
    }

    app.saved_results_discarded = true;
    app.project_results_dirty = true;

    // resolve_layout() keeps cache as a child of the project directory.
    // relationship before a recursive removal so a malformed path can never
    // broaden the deletion target.
    const std::filesystem::path root = app.layout.root.lexically_normal();
    const std::filesystem::path cache = app.layout.cache.lexically_normal();
    const bool safe_cache = !root.empty() && !cache.empty() && cache != root &&
                            cache.parent_path() == root &&
                            (cache.filename() == "cache" ||
                             cache.extension() == ".cache");
    if (safe_cache) {
        std::error_code error;
        removed += std::filesystem::remove_all(cache, error);
        if (error && failure.empty()) failure = error.message();
    }

    // The live sfm.bin and preview sidecars live in a per-project namespace:
    // <cache root>/<project-hash> when a cache folder is configured, otherwise
    // %TEMP%/Photara/<project-hash>. Delete that exact namespaced directory
    // as well; leaving it behind makes refresh_artifacts() resurrect the
    // supposedly deleted scene. The project-adjacent default is covered by the
    // .cache check above.
    const std::filesystem::path runtime =
        app.layout.working_sfm.parent_path().lexically_normal();
    std::error_code temp_error;
    const std::filesystem::path temp_root =
        (std::filesystem::temp_directory_path(temp_error) / "Photara")
            .lexically_normal();
    const std::filesystem::path configured_root =
        path_from_utf8_field(app.settings.cache_dir.data()).lexically_normal();
    const bool under_temp = !temp_error && runtime.parent_path() == temp_root;
    const bool under_configured = !configured_root.empty() &&
                                  runtime.parent_path() == configured_root;
    const bool safe_runtime = !runtime.empty() && runtime != temp_root &&
                              (under_temp || under_configured);
    if (safe_runtime) {
        std::error_code error;
        removed += std::filesystem::remove_all(runtime, error);
        if (error && failure.empty()) failure = error.message();
    }

    refresh_artifacts(app);
    if (failure.empty()) {
        set_message(
            app,
            "Reconstruction results cleared (" + std::to_string(removed) +
                " generated entries removed)",
            theme::success);
    } else {
        set_message(
            app, "Some reconstruction results could not be removed: " + failure,
            theme::danger);
    }
}

void poll_camera_photos(App& app) {
    if (app.scene.views.empty()) {
        if (app.photos.size() != 0) app.photos.clear();
        return;
    }
    attach_view_image_paths(app.scene, reconstruction_images_path(app));
    app.photos.resize(app.scene.views.size());
    if (app.workspace == ViewportWorkspace::image_2d) {
        const int count = static_cast<int>(app.scene.views.size());
        const int selected = std::clamp(app.image_qa.selected, 0, count - 1);
        const int begin = std::max(0, selected - 18);
        const int end = std::min(count, selected + 19);
        const int strip_begin = std::max(0, app.image_qa.filmstrip_first);
        const int strip_end = std::min(count, app.image_qa.filmstrip_last + 1);
        for (int i = begin; i < end; ++i)
            app.photos.request(
                static_cast<std::size_t>(i), app.scene.views[static_cast<std::size_t>(i)].image_path);
        for (int i = strip_begin; i < strip_end; ++i)
            app.photos.request(
                static_cast<std::size_t>(i), app.scene.views[static_cast<std::size_t>(i)].image_path);
    } else if (app.view_options.show_views && app.view_options.show_camera_photos) {
        std::vector<std::size_t> markers;
        sampled_view_indices(app.scene, markers);
        for (const std::size_t index : markers)
            app.photos.request(index, app.scene.views[index].image_path);
    }
    app.photos.poll();
}

void poll_mesh_load(App& app) {
    if (!app.pending_mesh_load.valid()) return;
    if (app.pending_mesh_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    MeshLoad loaded = app.pending_mesh_load.get();
    app.loading_scene = false;
    if (app.pending_scene_load_generation != app.scene_load_generation) return;
    if (!loaded.ok) {
        app.mesh_load_failed = true;
        set_message(app, "Mesh: " + loaded.error, theme::danger);
        return;
    }
    app.mesh = std::move(loaded.mesh);
    app.mesh_gpu_data = std::move(loaded.gpu_data);
    app.mesh_load_failed = false;
    if (app.mesh.has_texture()) app.view_options.mesh_texture = true;
    // A mesh mode switch does not change the Gaussian cloud the existing
    // subject box was fitted to. Preserve that cached fit instead of walking
    // millions of Gaussian centres again on the UI thread.
    if (!app.scene.has_points() && !app.reconstruction_box.user_set) {
        app.reconstruction_box.min = loaded.reconstruction_bounds[0];
        app.reconstruction_box.max = loaded.reconstruction_bounds[1];
        app.reconstruction_box.valid = loaded.reconstruction_bounds_valid;
        app.reconstruction_box.auto_fitted = true;
    }
    try {
        app.mesh_renderer.set_mesh(app.mesh_gpu_data);
        app.mesh_renderer.set_albedo(loaded.albedo);
        app.mesh_gpu_uploaded = true;
        app.mesh_gpu_failed = false;
    } catch (const std::exception& failure) {
        app.mesh_gpu_failed = true;
        set_message(app, std::string("Mesh GPU upload: ") + failure.what(), theme::danger);
        return;
    }
    // Frame only when this load was requested with framing intent (fresh
    // mesh from extraction or first session entry). A load triggered by
    // switching to the mesh tab keeps the current camera for 3DGS/mesh
    // comparison.
    if (app.frame_mesh_on_load) frame_reconstruction(app);
    app.frame_mesh_on_load = false;
    app.view_mode = VisualizationMode::mesh;
    set_message(
        app,
        app.mesh.has_texture()
            ? ("Textured mesh: " + format_count(app.mesh.vertices.size()) +
               " vertices, " + format_count(app.mesh.faces.size()) + " faces")
            : ("Mesh: " + format_count(app.mesh.vertices.size()) +
               " vertices, " + format_count(app.mesh.faces.size()) + " faces"),
        theme::success);
}

void poll_scene_load(App& app) {
    if (!app.loading_scene || !app.pending_load.valid()) return;
    if (app.pending_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    SceneLoad loaded = app.pending_load.get();
    app.loading_scene = false;
    if (app.pending_scene_load_generation != app.scene_load_generation) {
        if (app.gaussian_composition_reload) {
            app.gaussian_composition_reload = false;
            request_gaussian_scene_load(app);
        }
        return;
    }
    if (!loaded.ok) {
        set_message(app, "Point cloud: " + loaded.error, theme::danger);
        return;
    }
    app.photos.clear();
    app.image_qa_session.clear();
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    if (loaded.composed_model) {
        app.composed_model = std::move(loaded.composed_model);
        app.composed_model_key = std::move(loaded.composed_key);
        for (std::size_t i = 0; i < loaded.source_models.size() && i < app.gaussian_models.size(); ++i) {
            if (!loaded.source_models[i]) continue;
            app.gaussian_models[i].cached_model = std::move(loaded.source_models[i]);
            app.gaussian_models[i].cached_key = std::move(loaded.source_keys[i]);
            app.gaussian_models[i].splat_count = app.gaussian_models[i].cached_model->size();
        }
    }
    app.scene = std::move(loaded.scene);
    invalidate_reconstruction_box(app.reconstruction_box);
    app.alignment_preview_seen = false;
    // When an internal SfM or Gaussian scene replaces the imported cameras,
    // allow the next ensure pass to restore the dataset view poses, which are
    // the ones training actually uses.
    if (app.scene_source != "External dataset" &&
        app.scene_source != "Gaussian centres" &&
        app.scene_source != "Gaussian models")
        app.dataset_scene_key.clear();
    if (app.image_qa.selected >= static_cast<int>(app.scene.views.size()))
        app.image_qa.selected = app.scene.views.empty() ? -1 : 0;
    infer_images_dir_from_scene(app);
    attach_view_image_paths(app.scene, reconstruction_images_path(app));
    ensure_reconstruction_box(app);
    if (!live_preview_active(app) && app.view_mode != VisualizationMode::mesh)
        frame_reconstruction(app);
    if (app.view_mode != VisualizationMode::splat &&
        app.view_mode != VisualizationMode::rings &&
        app.view_mode != VisualizationMode::mesh)
        app.view_mode = VisualizationMode::points;
    const auto position_audit = app.scene.position_audit_counts();
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered" +
            (position_audit.unverified > 0
                ? "; " + std::to_string(position_audit.unverified) + " positions need review" : ""),
        position_audit.unverified > 0 ? theme::warning : theme::success);
}

void poll_align_live(App& app) {
    const bool aligning = alignment_job_running(app);
    if (!aligning) return;

    photara::sfm::AlignLiveFrame frame;
    const bool have_live =
        photara::sfm::load_align_live_frame(app.layout.align_live, frame);
    if (have_live &&
        (frame.revision != app.align_live.revision ||
         frame.kind != app.align_live.kind ||
         frame.index_a != app.align_live.index_a ||
         frame.index_b != app.align_live.index_b)) {
        app.align_live = std::move(frame);
        refresh_image_qa_folder(
            app.image_qa, reconstruction_images_path(app));
        const int count = image_qa_count(app.image_qa, app.scene);
        const Stage stage = app.monitor.stage();
        const bool follow_pair =
            stage == Stage::matching &&
            photara::sfm::is_live_pair_kind(app.align_live.kind);
        const bool follow_features =
            app.align_live.kind == photara::sfm::AlignLiveKind::features &&
            (stage == Stage::features || stage == Stage::preparing ||
             stage == Stage::matching);
        if ((follow_pair || follow_features) && app.align_live.index_a >= 0) {
            const int index = image_qa_index_for_path(
                app.image_qa, app.scene, app.align_live.path_a,
                app.align_live.index_a);
            select_image_qa_view(app.image_qa, index, count);
        }
        if (follow_features)
            app.image_qa.mode = ImageQaMode::features;
    }

    if (app.alignment_workspace_user_override) return;
    const Stage stage = app.monitor.stage();
    const bool live_2d =
        app.align_live.kind == photara::sfm::AlignLiveKind::features ||
        photara::sfm::is_live_pair_kind(app.align_live.kind);
    refresh_image_qa_folder(app.image_qa, reconstruction_images_path(app));
    const int count = image_qa_count(app.image_qa, app.scene);
    if ((stage == Stage::features || stage == Stage::matching) &&
        (live_2d || count > 0))
        set_viewport_workspace(app, ViewportWorkspace::image_2d, false);
    else if (
        stage == Stage::tracks || stage == Stage::mapping ||
        stage == Stage::exporting)
        set_viewport_workspace(app, ViewportWorkspace::scene_3d, false);
}

void poll_alignment_preview(App& app) {
    const bool aligning = app.job.running() && app.active_job == JobKind::align;
    if (app.alignment_preview_load.valid()) {
        if (app.alignment_preview_load.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        auto loaded = app.alignment_preview_load.get();
        if (aligning && !app.loading_scene && loaded.ok &&
            app.alignment_preview_load_generation == app.alignment_preview_generation) {
            const bool first = !app.alignment_preview_seen;
            app.scene = std::move(loaded.scene);
            invalidate_reconstruction_box(app.reconstruction_box);
            app.scene_source = "Alignment preview";
            attach_view_image_paths(app.scene, reconstruction_images_path(app));
            app.alignment_preview_seen = true;
            if (first) {
                app.photos.clear();
                frame_reconstruction(app);
                app.view_mode = VisualizationMode::points;
            }
            ensure_reconstruction_box(app);
            app.image_qa.metrics_dirty = true;
            set_message(app, "Aligning: " + std::to_string(app.scene.registered_views) +
                " cameras (partial result)", theme::accent);
        }
    }
    if (!aligning || app.loading_scene || app.layout.working_sfm.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < app.alignment_preview_poll) return;
    app.alignment_preview_poll = now + std::chrono::milliseconds(500);
    const auto path = app.layout.align_preview;
    if (path.empty()) return;
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(path, error);
    if (error || stamp == app.alignment_preview_stamp) return;
    app.alignment_preview_stamp = stamp;
    app.alignment_preview_load_generation = app.alignment_preview_generation;
    const std::filesystem::path images = reconstruction_images_path(app);
    app.alignment_preview_load = std::async(std::launch::async, [path, images] {
        try { return sparse_scene_from_sfm(load_working_sfm(path, images), false); }
        catch (const std::exception& error) {
            SceneLoad result;
            result.error = error.what();
            return result;
        }
    });
}

void start_align(App& app) {
    if (app.job.running()) return;
    if (!ensure_sam_ready(app)) return;
    if (has_external_dataset(app)) {
        set_message(
            app,
            "External dataset selected; use Train 3DGS instead of Align Photos",
            theme::warning);
        return;
    }
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    if (!app.layout.working_sfm.empty()) {
        std::error_code cache_error;
        std::filesystem::create_directories(
            app.layout.working_sfm.parent_path(), cache_error);
    }
    {
        std::error_code stale;
        std::filesystem::remove(app.layout.sparse_mvs, stale);
    }
    if (is_video_source(app.settings)) {
        std::error_code video_error;
        const std::filesystem::path video =
            path_from_utf8_field(app.settings.images_dir.data());
        if (!std::filesystem::is_regular_file(video, video_error)) {
            set_message(app, "Video file not found", theme::danger);
            return;
        }
        const auto frames = reconstruction_images_path(app);
        if (!directory_has_images(frames) &&
            !photara::io::ffmpeg_available()) {
            set_message(
                app,
                "ffmpeg was not found. Install ffmpeg and add it to PATH.",
                theme::danger);
            return;
        }
    } else if (!directory_has_images(
                   path_from_utf8_field(app.settings.images_dir.data()))) {
        set_message(
            app, "No images found in the selected source folder", theme::danger);
        return;
    }
    try {
        const Stage opening = is_video_source(app.settings)
            ? Stage::preparing
            : (app.settings.sam_masks ? Stage::masking : Stage::features);
        app.monitor.begin(JobKind::align, opening);
        const auto preview_path = app.layout.align_preview;
        std::error_code preview_error;
        if (!preview_path.empty())
            std::filesystem::remove(preview_path, preview_error);
        std::error_code live_error;
        std::filesystem::remove(app.layout.align_live, live_error);
        app.align_live = {};
        app.align_match_session.clear();
        app.alignment_workspace_user_override = false;
        app.alignment_preview_stamp = {};
        if (preview_error) {
            app.alignment_preview_stamp = std::filesystem::last_write_time(preview_path, preview_error);
            if (preview_error) app.alignment_preview_stamp = {};
        }
        app.alignment_preview_seen = false;
        ++app.alignment_preview_generation;
        app.alignment_preview_poll = {};
        app.log.open(app.layout.align_log);
        app.job.start(
            build_align_command(
                cli_path().c_str(), app.settings, app.layout),
            app.layout.align_log);
        app.active_job = JobKind::align;
        // Drop the previous cloud so the next frame only shows this run.
        // The actual GPU/photo teardown waits until after present(); this
        // frame's draw list still references those descriptors.
        ++app.scene_load_generation;
        app.suppress_scene_auto_load = true;
        app.pending_align_viewport_clear = true;
        set_message(app, "Aligning cameras...", theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

bool alignment_cache_present(const App& app) {
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.working_sfm, error) ||
           std::filesystem::exists(app.layout.cache, error);
}

bool can_export_sfm(const App& app) {
    return !has_external_dataset(app) &&
           app.settings.images_dir[0] != '\0' &&
           (app.has_sparse || alignment_cache_present(app));
}

const char* running_job_caption(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return tr("ALIGNING");
        case JobKind::export_sfm: return tr("EXPORTING");
        case JobKind::train: return tr("TRAINING");
        case JobKind::dense: return tr("EXTRACT MESH");
        case JobKind::texture: return tr("BAKE TEXTURE");
        case JobKind::none: return tr("READY");
    }
    return tr("READY");
}

const char* job_state_caption(const App& app) {
    if (!app.job.running()) return tr("READY");
    if (app.job.paused()) return tr("PAUSED");
    return running_job_caption(app.active_job);
}

const char* pause_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return tr("Pause Alignment");
        case JobKind::export_sfm: return tr("Pause Export");
        case JobKind::dense: return tr("Pause Extract Mesh");
        case JobKind::texture: return tr("Pause Bake Texture");
        default: return tr("Pause Training");
    }
}

const char* resume_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return tr("Resume Alignment");
        case JobKind::export_sfm: return tr("Resume Export");
        case JobKind::dense: return tr("Resume Extract Mesh");
        case JobKind::texture: return tr("Resume Bake Texture");
        default: return tr("Resume Training");
    }
}

const char* stop_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return tr("Stop Alignment");
        case JobKind::export_sfm: return tr("Stop Export");
        case JobKind::dense: return tr("Stop Extract Mesh");
        case JobKind::texture: return tr("Stop Bake Texture");
        default: return tr("Stop Training");
    }
}


void start_export_sfm(App& app) {
    if (app.job.running()) return;
    if (!ensure_sam_ready(app)) return;
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    if (!can_export_sfm(app)) {
        set_message(
            app, "Align photos before exporting SfM alignment", theme::warning);
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    try {
        if (!app.layout.working_sfm.empty()) {
            std::error_code exists_error;
            if (std::filesystem::exists(app.layout.working_sfm, exists_error)) {
                const auto scene = load_working_sfm(
                    app.layout.working_sfm, reconstruction_images_path(app));
                photara::sfm::save_asfm(scene, app.layout.sparse_asfm);
                photara::sfm::export_openmvs_interface(
                    scene, app.layout.sparse_mvs);
                refresh_artifacts(app);
                set_message(
                    app, "Exported SfM to .asfm and OpenMVS .mvs",
                    theme::success);
                return;
            }
        }
        if (!app.layout.project_file.empty()) {
            std::error_code exists_error;
            if (std::filesystem::exists(app.layout.project_file, exists_error)) {
                const auto archive =
                    photara::project::Archive::open(app.layout.project_file);
                const auto scene = photara::project::read_sfm(archive);
                if (scene) {
                    photara::sfm::save_asfm(*scene, app.layout.sparse_asfm);
                    photara::sfm::export_openmvs_interface(
                        *scene, app.layout.sparse_mvs);
                    refresh_artifacts(app);
                    set_message(
                        app, "Exported SfM to .asfm and OpenMVS .mvs",
                        theme::success);
                    return;
                }
            }
        }
        app.monitor.begin(JobKind::export_sfm);
        app.log.open(app.layout.export_log);
        app.job.start(
            build_export_sfm_command(
                cli_path().c_str(), app.settings, app.layout),
            app.layout.export_log);
        app.active_job = JobKind::export_sfm;
        set_message(app, "Exporting SfM alignment...", theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

bool can_export_alignment(const App& app) {
    return can_export_sfm(app) || can_export_sparse(app);
}

bool alignment_mvs_supported(const App& app) {
    for (const auto& view : app.scene.views) {
        // Neither a fisheye nor an equirectangular source image can be resampled
        // into the pinhole working camera the dense pipeline requires.
        if (view.registered && (view.camera_model == "OpenCV Fisheye" ||
                                view.camera_model == "Equirectangular"))
            return false;
    }
    return true;
}

std::optional<photara::sfm::Scene> load_alignment_scene(const App& app) {
    std::error_code error;
    if (!app.layout.working_sfm.empty() &&
        std::filesystem::exists(app.layout.working_sfm, error)) {
        return load_working_sfm(
            app.layout.working_sfm, reconstruction_images_path(app));
    }
    if (!app.layout.project_file.empty() &&
        std::filesystem::exists(app.layout.project_file, error)) {
        const auto archive =
            photara::project::Archive::open(app.layout.project_file);
        return photara::project::read_sfm(archive);
    }
    return std::nullopt;
}


AlignmentExportFormat alignment_export_format(const App& app) {
    switch (app.alignment_export.format) {
        case 1: return AlignmentExportFormat::colmap;
        case 2: return AlignmentExportFormat::nerfstudio;
        case 3: return AlignmentExportFormat::openmvs;
        default: return AlignmentExportFormat::asfm;
    }
}

const wchar_t* alignment_export_extension_wide(const AlignmentExportFormat format) {
    switch (format) {
        case AlignmentExportFormat::colmap: return L"";
        case AlignmentExportFormat::nerfstudio: return L"json";
        case AlignmentExportFormat::openmvs: return L"mvs";
        case AlignmentExportFormat::asfm:
        default: return L"asfm";
    }
}

std::filesystem::path alignment_export_directory(const App& app) {
    return export_output_directory(app);
}

std::filesystem::path alignment_export_stem_name(const App& app) {
    return export_stem_path(app);
}

std::filesystem::path suggested_alignment_export_path(
    const App& app, const AlignmentExportFormat format,
    std::filesystem::path directory, std::filesystem::path stem) {
    if (directory.empty()) directory = alignment_export_directory(app);
    if (stem.empty()) stem = alignment_export_stem_name(app);
    auto path = directory / stem;
    if (format == AlignmentExportFormat::colmap)
        path += "_colmap";
    else if (format == AlignmentExportFormat::nerfstudio)
        path += ".json";
    else if (format == AlignmentExportFormat::openmvs)
        path += ".mvs";
    else
        path += ".asfm";
    return path;
}

std::filesystem::path alignment_export_stem_from_path(
    const std::filesystem::path& current, const bool current_is_colmap) {
    std::filesystem::path name = current_is_colmap || current.extension().empty()
        ? current.filename()
        : current.stem();
    const std::wstring text = name.wstring();
    constexpr wchar_t suffix[] = L"_colmap";
    const std::size_t suffix_n = 7;
    if (text.size() > suffix_n &&
        text.compare(text.size() - suffix_n, suffix_n, suffix) == 0)
        name = text.substr(0, text.size() - suffix_n);
    if (name.empty()) name = "project";
    return name;
}

void sync_alignment_export_path(App& app, const bool force) {
    const auto format = alignment_export_format(app);
    if (!force && app.alignment_export.path_format == app.alignment_export.format &&
        app.alignment_export.path[0] != '\0')
        return;
    std::filesystem::path directory = alignment_export_directory(app);
    std::filesystem::path stem = alignment_export_stem_name(app);
    if (app.alignment_export.path[0] != '\0') {
        const auto current =
            path_from_utf8_field(app.alignment_export.path.data());
        const bool was_colmap = app.alignment_export.path_format == 1;
        auto parent = current.parent_path();
        if (!parent.empty()) directory = parent;
        const auto current_stem =
            alignment_export_stem_from_path(current, was_colmap);
        if (!current_stem.empty()) stem = current_stem;
    }
    store_path_field(
        app.alignment_export.path,
        suggested_alignment_export_path(app, format, directory, stem));
    app.alignment_export.path_format = app.alignment_export.format;
}

bool browse_alignment_export_path(App& app) {
    const auto format = alignment_export_format(app);
    sync_alignment_export_path(app, false);
    if (format == AlignmentExportFormat::colmap)
        return pick_folder(L"Export COLMAP Model", app.alignment_export.path);
    const wchar_t* extension = alignment_export_extension_wide(format);
    const auto current = path_from_utf8_field(app.alignment_export.path.data());
    const std::wstring name = current.filename().empty()
        ? export_stem_wide(app) + L"." + extension
        : current.filename().wstring();
    return pick_export_path(
        L"Export SfM Alignment", app.alignment_export.path,
        FilePickKind::alignment, name.c_str(), extension);
}

void open_alignment_export_panel(App& app) {
    if (app.job.running() || !can_export_alignment(app)) return;
    if (!app.alignment_export.initialized) {
        app.alignment_export.format = 0;
        app.alignment_export.write_ply = true;
        app.alignment_export.initialized = true;
    }
    sync_alignment_export_path(app, app.alignment_export.path[0] == '\0');
    app.alignment_export.show = true;
}

void export_alignment(App& app) {
    if (app.job.running() || !can_export_alignment(app)) return;
    const auto format = alignment_export_format(app);
    const bool write_ply = app.alignment_export.write_ply;
    if (format == AlignmentExportFormat::openmvs &&
        !alignment_mvs_supported(app)) {
        set_message(
            app, "OpenMVS export needs rectified pinhole images",
            theme::warning);
        return;
    }

    std::filesystem::path out =
        path_from_utf8_field(app.alignment_export.path.data());
    if (out.empty()) {
        if (!browse_alignment_export_path(app)) return;
        out = path_from_utf8_field(app.alignment_export.path.data());
    }
    if (out.empty()) return;
    if (format != AlignmentExportFormat::colmap && out.extension().empty()) {
        const wchar_t* extension = alignment_export_extension_wide(format);
        out += ".";
        out += std::filesystem::path(extension);
        store_path_field(app.alignment_export.path, out);
    }

    try {
        auto scene = load_alignment_scene(app);
        if (!scene) {
            set_message(
                app, "Align photos before exporting SfM alignment",
                theme::warning);
            return;
        }
        const auto images = reconstruction_images_path(app);
        std::filesystem::path ply;
        if (write_ply) {
            if (format == AlignmentExportFormat::colmap)
                ply = out / "points3D.ply";
            else {
                ply = out;
                ply.replace_extension();
                ply += "_sparse.ply";
            }
        }

        if (format == AlignmentExportFormat::asfm)
            photara::sfm::save_asfm(*scene, out);
        else if (format == AlignmentExportFormat::colmap)
            photara::sfm::save_colmap_text(
                *scene, out, images, write_ply);
        else if (format == AlignmentExportFormat::nerfstudio)
            photara::sfm::save_nerfstudio_transforms(
                *scene, out, images, ply);
        else
            photara::sfm::export_openmvs_interface(*scene, out);

        if (write_ply) photara::sfm::save_sparse_ply(*scene, ply);
        refresh_artifacts(app);
        const char* label = format == AlignmentExportFormat::colmap
            ? "COLMAP"
            : (format == AlignmentExportFormat::nerfstudio
                   ? "Nerfstudio / Blender"
                   : (format == AlignmentExportFormat::openmvs ? "OpenMVS"
                                                               : "ASFM"));
        std::string message = "Exported SfM alignment (";
        message += label;
        if (write_ply) message += ", PLY";
        message += ")";
        set_message(app, message, theme::success);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void start_train(App& app, const bool smoke, const bool resume_training) {
    if (app.job.running() || app.pending_train_prepare.valid()) return;
    if (!smoke && !ensure_sam_ready(app)) return;
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    if (resume_training && !app.training_checkpoint_info) {
        set_message(app, "No saved training progress is available.", theme::warning);
        return;
    }
    if (!smoke && photara::splat::training_checkpoint_in_use(app.layout.training_checkpoint)) {
        set_message(app, "This project's training checkpoint is in use by another process.", theme::warning);
        return;
    }
    if (!smoke && !app.training_inputs_ready) {
        const auto layout = app.layout;
        const auto images = reconstruction_images_path(app);
        app.train_prepare_resume = resume_training;
        app.train_preparation = std::make_shared<TrainPreparation>();
        const auto preparation = app.train_preparation;
        app.pending_train_prepare = std::async(std::launch::async,
            [layout, images, resume_training, preparation] {
                try {
                    prepare_training_inputs(layout, images, resume_training,
                        [preparation] { return preparation->cancelled.load(); },
                        [preparation](TrainPrepareStage stage) { preparation->stage = stage; });
                    return std::string{};
                } catch (const std::exception& failure) { return std::string(failure.what()); }
            });
        set_message(app, "Checking training inputs...", theme::accent);
        return;
    }
    app.training_inputs_ready = false;
    write_working_subject_bounds(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (!app.layout.preview_view_file.empty()) {
        std::error_code preview_error;
        std::filesystem::create_directories(
            app.layout.preview_view_file.parent_path(), preview_error);
    }
    app.preview_view = 0;
    app.preview_follow_view = true;
    if (has_external_dataset(app)) {
        // Keep the already-reviewed imported cameras and cloud on screen.
        // The child CLI reloads the same dataset; wiping the viewport made
        // Train look like a reset.
        if (!app.scene.has_points() && app.scene.views.empty())
            request_dataset_scene_load(app);
    } else {
        load_view_poses(app.layout.sparse_poses, app.scene);
        attach_view_image_paths(app.scene, reconstruction_images_path(app));
    }
    if (const ViewPose* pose = first_registered_view(app.scene))
        snap_orbit_to_view(app.camera, *pose);
    write_preview_view_index(app.layout, app.preview_view);
    app.view_mode = VisualizationMode::splat;
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);
    ensure_sparse_loaded(app);

    // Keep unsaved edits on the CPU, but always release the preview model,
    // sorting workspaces and frame targets before starting the trainer.
    release_splat_preview(app);
    if (!app.splat_edit.dirty()) {
        app.splat_edit.clear();
    }

    // The trainer inherits the shared allocation, so the exported handles are
    // recreated per run: the timeline counter has to restart from zero.
    app.preview.create(k_preview_extent, k_preview_extent);
    PreviewHandles handles;
#if defined(_WIN32)
    handles.memory =
        reinterpret_cast<std::uintptr_t>(app.preview.memory_handle);
    handles.semaphore =
        reinterpret_cast<std::uintptr_t>(app.preview.semaphore_handle);
#else
    if (app.preview.memory_handle >= 0)
        handles.memory = static_cast<std::uint64_t>(app.preview.memory_handle);
    if (app.preview.semaphore_handle >= 0)
        handles.semaphore =
            static_cast<std::uint64_t>(app.preview.semaphore_handle);
#endif
    handles.allocation_size = app.preview.allocation_size;
    handles.width = app.preview.width;
    handles.height = app.preview.height;
    handles.device_luid = gpu::device_luid();
    handles.device_node_mask = gpu::device_node_mask();
    handles.device_uuid = gpu::device_uuid_hex();

    try {
        std::string command;
        if (smoke) {
            // Fixed COLMAP dataset used only by --interop-smoke. The dataset
            // root is the parent of the images directory, so the harness
            // follows PHOTARA_SMOKE_IMAGES without a second hard-coded path.
            const std::filesystem::path smoke_dataset =
                path_from_utf8_field(app.settings.images_dir.data())
                    .parent_path();
            std::ostringstream smoke_command;
            smoke_command
                << '"' << cli_path().c_str() << "\" --images \""
                << app.settings.images_dir.data() << "\" --output \""
                << app.layout.model_output.string()
                << "\" --splat-dataset \""
                << smoke_dataset.string() << "\""
                << " --splat-use-mask false"
                << " --splat --backend "
                << (app.settings.training_backend == 1 ? "vulkan" : "cuda")
                << " --splat-strategy "
                << (app.settings.training_backend == 1 ? "adc_igs" : "adc_plus")
                << " --splat-iterations "
                << app.settings.iterations << " --splat-preview-interval "
                << app.settings.preview_interval
                << " --splat-preview-view 0 --splat-preview-view-file \""
                << app.layout.preview_view_file.string() << '"'
                << " --splat-preview-camera-file \""
                << app.layout.preview_camera_file.string() << '"'
                << " --splat-preview-vis-file \""
                << app.layout.preview_vis_file.string() << '"'
                << " --splat-preview-ack-file \""
                << app.layout.preview_ack_file.string() << '"'
                << " --splat-preview-vk-memory-handle " << handles.memory
                << " --splat-preview-vk-semaphore-handle " << handles.semaphore
                << " --splat-preview-vk-allocation-size " << handles.allocation_size
                << " --splat-preview-vk-width " << handles.width
                << " --splat-preview-vk-height " << handles.height
                << " --splat-preview-vk-device-luid " << handles.device_luid
                << " --splat-preview-vk-device-node-mask "
                << handles.device_node_mask;
            if (!handles.device_uuid.empty())
                smoke_command << " --splat-preview-vk-device-uuid "
                              << handles.device_uuid;
            command = smoke_command.str();
        } else {
            command = build_train_command(
                cli_path().c_str(), app.settings, app.layout, handles);
        }

        if (!smoke) command += resume_training ? " --splat-resume=true" : " --splat-resume=false";
        app.monitor.begin(JobKind::train);
        app.log.open(app.layout.train_log);
        app.job.start(command, app.layout.train_log);
        app.active_job = JobKind::train;
        app.project_results_dirty = true;
        set_message(
            app,
            has_external_dataset(app)
                ? "Training from imported cameras..."
                : (mesh_from_gaussians(app.settings)
                       ? "Training with depth/normal geometry supervision..."
                       : "Training from aligned cameras..."),
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
    // The child has inherited the handles; the parent copies are no longer
    // needed and must not leak across runs.
    app.preview.close_export_handles();
}

void poll_train_start(App& app) {
    if (!app.pending_train_prepare.valid() ||
        app.pending_train_prepare.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    const auto error = app.pending_train_prepare.get();
    const bool cancelled = app.train_preparation && app.train_preparation->cancelled;
    app.train_preparation.reset();
    if (cancelled) { set_message(app, "Training preparation cancelled", theme::text_muted); return; }
    if (!error.empty()) { set_message(app, error, theme::danger); return; }
    refresh_artifacts(app);
    // A requested close takes precedence over starting a new GPU task.
    if (app.close_requested || app.show_close_confirmation || app.close_approved) return;
    app.training_inputs_ready = true;
    start_train(app, false, app.train_prepare_resume);
    app.training_inputs_ready = false;
}

void start_splat_mesh(App& app) {
    if (app.job.running()) return;
    if (!ensure_sam_ready(app)) return;
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    if (!alignment_ready(app) && !alignment_cache_present(app)) {
        set_message(
            app,
            "Align photos or load an external camera dataset before Extract Mesh",
            theme::warning);
        return;
    }
    refresh_artifacts(app);
    auto model = existing_splat_model(app);
    if (const auto edited = write_edited_splat_file(app); !edited.empty())
        model = edited;
    if (model.empty()) {
        start_train(app, false);
        return;
    }
    app.settings.build_mesh = true;
    write_working_subject_bounds(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    app.view_mode = VisualizationMode::splat;
    try {
        app.monitor.begin(JobKind::dense, Stage::meshing);
        app.log.open(app.layout.dense_log);
        app.job.start(
            build_splat_mesh_command(
                cli_path().c_str(), app.settings, app.layout, model),
            app.layout.dense_log);
        app.active_job = JobKind::dense;
        set_message(
            app, "Extracting mesh from the trained 3DGS model...",
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void start_dense(App& app) {
    if (app.job.running()) return;
    // From Gaussians extracts the surface from the saved 3DGS model.
    // From MVS fuses the mesh from the dense photogrammetry cloud.
    if (mesh_from_gaussians(app.settings)) {
        refresh_artifacts(app);
        if (app.has_model) {
            start_splat_mesh(app);
            return;
        }
        start_train(app, false);
        return;
    }
    if (!ensure_sam_ready(app)) return;
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    if (!alignment_ready(app) && !alignment_cache_present(app)) {
        set_message(
            app,
            "Align photos or load an external camera dataset before Extract Mesh",
            theme::warning);
        return;
    }
    app.settings.build_mesh = true;
    app.settings.mesh_source = 1;
    if (app.settings.mesh_method == 3) app.settings.mesh_method = 0;
    refresh_artifacts(app);
    write_working_subject_bounds(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    if (has_external_dataset(app)) {
        if (!app.scene.has_points() && app.scene.views.empty())
            request_dataset_scene_load(app);
    }
    app.view_mode = VisualizationMode::points;
    try {
        app.monitor.begin(JobKind::dense);
        app.log.open(app.layout.dense_log);
        app.job.start(
            build_dense_command(
                cli_path().c_str(), app.settings, app.layout),
            app.layout.dense_log);
        app.active_job = JobKind::dense;
        set_message(
            app,
            has_external_dataset(app)
                ? "Building MVS mesh from imported cameras..."
                : "Building MVS mesh from aligned cameras...",
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void start_texture(App& app) {
    if (app.job.running()) return;
    if (!ensure_sam_ready(app)) return;
    ensure_task_cache(app);
    if (!app.settings.images_dir[0] && !app.settings.project_dir[0] && !app.settings.dataset_source[0]) {
        set_message(app, "Choose an image source or open a project first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    if (!alignment_ready(app) && !alignment_cache_present(app)) {
        set_message(
            app,
            "Align photos or load an external camera dataset before Bake Texture",
            theme::warning);
        return;
    }
    if (!app.has_mesh) {
        set_message(
            app, "Extract a mesh before baking a texture", theme::warning);
        return;
    }
    const auto mesh = existing_mesh_path(app);
    if (mesh.empty() && !app.has_mesh) {
        set_message(app, "No mesh file found to texture", theme::warning);
        return;
    }
    if (!mesh.empty() && !app.layout.working_mesh.empty() &&
        mesh != app.layout.working_mesh)
        copy_existing_file(mesh, app.layout.working_mesh);
#if !defined(PHOTARA_HAS_TEXTURE)
    set_message(
        app,
        "This build was compiled without texture baking (Vulkan + photara_drender).",
        theme::danger);
    return;
#endif
    app.settings.atlas_resolution = std::max(64, app.settings.atlas_resolution);
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    if (!app.layout.working_texture.empty()) {
        std::error_code parent_error;
        std::filesystem::create_directories(
            app.layout.working_texture.parent_path(), parent_error);
    }
    app.view_mode = VisualizationMode::mesh;
    try {
        app.monitor.begin(JobKind::texture);
        app.log.open(app.layout.texture_log);
        app.job.start(
            build_texture_command(
                cli_path().c_str(), app.settings, app.layout),
            app.layout.texture_log);
        app.active_job = JobKind::texture;
        set_message(
            app,
            app.settings.texture_delight
                ? "Baking albedo texture (delight + projection)..."
                : "Baking texture from calibrated photos...",
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void on_job_finished(App& app) {
    const int code = app.job.exit_code();
    app.monitor.mark_finished(code);
    const JobKind kind = app.active_job;
    if (kind != JobKind::none && kind != JobKind::export_sfm) app.project_results_dirty = true;
    app.active_job = JobKind::none;
    if (kind == JobKind::train && code != 0) {
        // The child is already reaped. Drop inherited-preview resources and
        // cached model workspaces, then avoid immediately loading the previous
        // saved model back into VRAM from the idle splat viewport.
        app.preview.reset();
        release_splat_preview(app);
        if (!app.splat_edit.dirty()) app.splat_edit.clear();
        if (app.view_mode == VisualizationMode::splat ||
            app.view_mode == VisualizationMode::rings)
            app.view_mode = VisualizationMode::points;
    }
    refresh_artifacts(app);

    if (code == 2) {
        set_message(
            app,
            std::string(job_name(kind)) +
                (kind == JobKind::train && app.training_checkpoint_info
                    ? " stopped. Resume Training restores the last saved checkpoint."
                    : " stopped. You can change images, video, or parameters and run again."),
            theme::warning);
        return;
    }
    if (code != 0) {
        std::string reason = app.monitor.last_error();
        if (reason.empty())
            reason = "Exited with code " + std::to_string(code);
        set_message(
            app, std::string(job_name(kind)) + " failed: " + reason,
            theme::danger);
        return;
    }

    if (kind == JobKind::train && !app.smoke_mode) {
        try {
            photara::splat::remove_training_checkpoint(app.layout.training_checkpoint);
        } catch (const std::exception& error) {
            set_message(app, error.what(), theme::danger); return;
        }
    }
    if (kind == JobKind::align) {
        app.align_live = {};
        app.align_match_session.clear();
        app.alignment_workspace_user_override = false;
        app.suppress_scene_auto_load = false;
        if (app.has_sparse) {
            request_ascan_scene_load(app);
        } else {
            set_message(
                app, "Alignment finished but no SfM stage was written",
                theme::warning);
        }
        return;
    }
    if (kind == JobKind::dense) {
        std::error_code error;
        const bool has_dense =
            std::filesystem::exists(app.layout.working_dense, error) ||
            std::filesystem::exists(app.layout.dense_ply, error);
        if (mesh_from_gaussians(app.settings)) {
            set_message(
                app,
                app.has_mesh
                    ? "Mesh extracted from the trained 3DGS model"
                    : "Extract Mesh finished but no mesh was written",
                app.has_mesh ? theme::success : theme::warning);
        } else if (mesh_from_mvs(app.settings)) {
            set_message(
                app,
                app.has_mesh ? "MVS mesh finished"
                             : (has_dense
                                    ? "MVS finished but no mesh was written"
                                    : "MVS finished but no surface was written"),
                app.has_mesh ? theme::success : theme::warning);
        } else {
            set_message(
                app,
                has_dense ? "MVS finished"
                          : "MVS finished but no dense cloud was written",
                has_dense ? theme::success : theme::warning);
        }
        if (app.has_mesh) {
            app.mesh.clear();
            show_mesh_view(app, true);
        } else if (app.has_sparse || has_external_dataset(app)) {
            ensure_sparse_loaded(app);
        }
        return;
    }
    if (kind == JobKind::texture) {
        refresh_artifacts(app);
        set_message(
            app,
            app.has_texture
                ? (app.settings.texture_delight
                       ? "Textured mesh finished (albedo atlas)"
                       : "Textured mesh finished")
                : "Texture bake finished but no OBJ/atlas was written",
            app.has_texture ? theme::success : theme::warning);
        if (app.has_mesh || app.has_texture) {
            app.mesh.clear();
            show_mesh_view(app, true);
        }
        app.atlas_preview.reset();
        app.atlas_preview_path.clear();
        return;
    }
    if (kind == JobKind::export_sfm) {
        std::error_code error;
        const bool has_asfm =
            std::filesystem::exists(app.layout.sparse_asfm, error);
        set_message(
            app,
            has_asfm ? "Exported SfM alignment to .asfm"
                     : (app.has_mvs
                            ? "Exported SfM alignment to sparse.mvs"
                            : "Export finished but no SfM file was written"),
            has_asfm || app.has_mvs ? theme::success : theme::warning);
        return;
    }
    set_message(
        app,
        mesh_from_gaussians(app.settings)
            ? (app.has_mesh ? "Training and mesh extraction finished"
                            : "Training finished, mesh extraction produced no "
                              "surface")
            : "Training finished",
        mesh_from_gaussians(app.settings) && !app.has_mesh ? theme::warning
                                                           : theme::success);
    if (app.has_mesh) {
        app.mesh.clear();
        show_mesh_view(app, true);
    }
}

bool ensure_splat_renderer(App& app) {
    if (!app.has_model) return false;
    // The CUDA live preview owns the viewport while it streams.
    if (live_preview_active(app)) return false;
    if (!app.splat_renderer.attached()) {
        app.splat_renderer.attach(gpu::runtime());
    }
    std::string key;
    try {
        photara::splat::GaussianModel model;
        if (gaussian_models_managed(app)) {
            // Rendering never inspects live drag values or reads source files.
            // Only a completed background composition changes this cache key.
            if (!app.composed_model) return false;
            key = app.composed_model_key;
            if (key == app.splat_load_failed_key) return false;
            if (key == app.splat_renderer.source_key() && app.splat_edit.bound_to(key))
                return app.splat_renderer.supported();
            model = *app.composed_model;
        } else if (const auto path = existing_splat_model(app);
                   !path.empty()) {
            std::error_code error;
            const auto stamp = std::filesystem::last_write_time(path, error);
            key = path.string();
            if (!error) {
                key += ":";
                key += std::to_string(
                    static_cast<long long>(stamp.time_since_epoch().count()));
            }
            if (key == app.splat_load_failed_key) return false;
            if (key == app.splat_renderer.source_key() &&
                app.splat_edit.bound_to(key))
                return app.splat_renderer.supported();
            model = photara::splat::load_gaussians(path);
        } else if (!app.layout.project_file.empty()) {
            const auto archive =
                photara::project::Archive::open(app.layout.project_file);
            if (!archive.has(photara::project::ChunkType::gaussians))
                return false;
            key = app.layout.project_file.string();
            key += ":gaussians:";
            key += std::to_string(archive.writer_version());
            if (key == app.splat_load_failed_key) return false;
            if (key == app.splat_renderer.source_key() &&
                app.splat_edit.bound_to(key))
                return app.splat_renderer.supported();
            model = photara::splat::decode_gaussians(
                archive.chunk(photara::project::ChunkType::gaussians));
        } else {
            return false;
        }

        const auto count = static_cast<std::uint32_t>(model.size());
        const auto means = model.means.to_vector();
        const auto scales = model.log_scales.to_vector();
        const auto quaternions = model.quaternions.to_vector();
        const auto opacity = model.opacity_logits.to_vector();
        const auto sh =
            model.sh.is_valid() ? model.sh.to_vector() : std::vector<float>{};
        const auto filter = model.filter_3d.is_valid() &&
                                    model.filter_3d.numel() == count
                                ? model.filter_3d.to_vector()
                                : std::vector<float>{};
        std::uint32_t bases = 1;
        if (model.sh.is_valid() && model.sh.shape().rank() >= 2)
            bases = static_cast<std::uint32_t>(model.sh.shape()[1]);
        if (means.size() < static_cast<std::size_t>(count) * 3U ||
            scales.size() < static_cast<std::size_t>(count) * 3U ||
            quaternions.size() < static_cast<std::size_t>(count) * 4U ||
            opacity.size() < count) {
            throw std::runtime_error(
                "Trained Gaussian model has an unexpected layout");
        }
        splat_render::GaussianCloud cloud;
        cloud.count = count;
        cloud.sh_degree = model.sh_degree;
        cloud.sh_bases = bases;
        cloud.means = means.empty() ? nullptr : means.data();
        cloud.log_scales = scales.empty() ? nullptr : scales.data();
        cloud.quaternions = quaternions.empty() ? nullptr : quaternions.data();
        cloud.opacity_logits = opacity.empty() ? nullptr : opacity.data();
        cloud.sh = sh.empty() ? nullptr : sh.data();
        cloud.filter_3d = filter.empty() ? nullptr : filter.data();
        if (!app.splat_renderer.upload(cloud, key)) {
            app.splat_load_failed_key = key;
            if (!app.splat_renderer.failure().empty())
                set_message(app, app.splat_renderer.failure(), theme::danger);
            return false;
        }
        app.splat_load_failed_key.clear();
        SplatEdit::Host host;
        host.means = means.empty() ? nullptr : means.data();
        host.log_scales = scales.empty() ? nullptr : scales.data();
        host.quaternions = quaternions.empty() ? nullptr : quaternions.data();
        host.opacity_logits = opacity.empty() ? nullptr : opacity.data();
        host.sh = sh.empty() ? nullptr : sh.data();
        host.count = count;
        host.sh_degree = model.sh_degree;
        host.sh_bases = bases;
        const auto normals = model.normal_features.is_valid() &&
                                     model.normal_features.numel() ==
                                         static_cast<std::size_t>(count) * 4U
                                 ? model.normal_features.to_vector()
                                 : std::vector<float>{};
        host.normals = normals.empty() ? nullptr : normals.data();
        host.filter_3d = filter.empty() ? nullptr : filter.data();
        app.splat_edit.sync(key, host);
        return true;
    } catch (const std::exception& failure) {
        app.splat_load_failed_key = key;
        set_message(app, failure.what(), theme::danger);
        return false;
    }
}

}  // namespace editor
