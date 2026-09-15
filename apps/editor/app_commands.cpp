#include "app.hpp"
#include "file_dialogs.hpp"

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
#include "splat/trainer.hpp"
#include "splat/visualize.hpp"
#if defined(AETHERSCAN_HAS_TEXTURE)
#include "texture/export.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
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

#ifndef AETHERSCAN_CLI_PATH
#define AETHERSCAN_CLI_PATH "aetherscan"
#endif

namespace editor {
using i18n::tr;

void stop_splat_view(App& app) {
    if (app.viewer.running()) app.viewer.stop();
}


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
    if (app.settings.scene_mode || !app.reconstruction_box.valid) return;
    write_reconstruction_box(
        app.reconstruction_box, app.layout.working_subject_bounds);
}

void frame_reconstruction(App& app) {
    app.camera.frame(
        reconstruction_local_centroid(app), reconstruction_local_radius(app));
}

std::filesystem::path existing_splat_model(const App& app) {
    const std::filesystem::path imported(app.settings.splat_model_source.data());
    std::error_code error;
    if (!imported.empty() && std::filesystem::exists(imported, error))
        return imported;
    const std::array<std::filesystem::path, 6> candidates = {
        app.layout.working_splat, app.layout.splat_model, app.layout.splat_ply,
        app.layout.splat_sog, app.layout.splat_spz, app.layout.splat_glb};
    for (const auto& candidate : candidates)
        if (std::filesystem::exists(candidate, error)) return candidate;
    return {};
}

bool live_preview_active(const App& app) {
    return (app.job.running() && app.active_job == JobKind::train) ||
           app.viewer.running();
}


std::filesystem::path resolve_editor_ini() {
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA"))
        return std::filesystem::path(appdata) / "AetherScan" / "editor.ini";
#endif
    return std::filesystem::current_path() / "aetherscan_editor.ini";
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
    // A normal editor session starts without a project. Do not resolve an
    // empty path relative to the process working directory, otherwise a
    // stray sparse.ply next to the executable could be treated as the active
    // reconstruction.
    if (app.settings.project_dir[0] == '\0') {
        app.layout = {};
        app.has_sparse = false;
        app.has_asfm = false;
        app.has_mvs = false;
        app.has_model = false;
        app.has_mesh = false;
        app.has_texture = false;
        app.project_writer_version = 0;
        app.project_min_reader_version = 0;
        return;
    }
    app.layout = resolve_layout(app.settings);
    std::error_code error;
    app.has_asfm = std::filesystem::exists(app.layout.sparse_asfm, error);
    app.has_mvs = std::filesystem::exists(app.layout.sparse_mvs, error);
    app.has_sparse =
        std::filesystem::exists(app.layout.working_sfm, error) ||
        std::filesystem::exists(app.layout.sparse_ply, error);
    app.has_model = !existing_splat_model(app).empty();
    app.has_mesh =
        std::filesystem::exists(app.layout.working_mesh, error) ||
        std::filesystem::exists(app.layout.mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_raw_mesh_ply, error);
    app.has_texture = textured_mesh_on_disk(app.layout.working_texture);
    app.project_writer_version = 0;
    app.project_min_reader_version = 0;
    if (app.layout.project_file.empty()) return;
    try {
        if (!std::filesystem::exists(app.layout.project_file, error)) return;
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        app.project_writer_version = archive.writer_version();
        app.project_min_reader_version = archive.min_reader_version();
        app.has_sparse =
            archive.has(aetherscan::project::ChunkType::sfm) || app.has_sparse;
        app.has_model =
            archive.has(aetherscan::project::ChunkType::gaussians) ||
            app.has_model;
        app.has_mesh =
            archive.has(aetherscan::project::ChunkType::mesh) || app.has_mesh;
        app.has_texture =
            archive.has(aetherscan::project::ChunkType::texture) ||
            app.has_texture;
    } catch (...) {
    }
}

std::filesystem::path reconstruction_images_path(const App& app) {
    return reconstruction_images_dir(app.settings);
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);

void assign_default_project_folder(App& app) {
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] != '\0')
        return;

    const std::filesystem::path images =
        path_from_utf8_field(app.settings.images_dir.data());
    const std::filesystem::path parent = images.parent_path();
    std::filesystem::path project_name = is_video_source(app.settings)
        ? images.stem()
        : images.filename();
    project_name += ".ascan";
    const std::filesystem::path project =
        parent.empty() ? project_name : parent / project_name;
    store_utf8_path_field(app.settings.project_dir, project);
    app.project_folder_automatic = true;
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

aetherscan::sfm::Scene load_working_sfm(
    const std::filesystem::path& path,
    const std::filesystem::path& images_dir) {
    aetherscan::sfm::AsfmOptions options;
    options.path_base = images_dir;
    return aetherscan::sfm::load_asfm(path, options);
}

aetherscan::project::Settings collect_project_settings(const App& app) {
    aetherscan::project::Settings settings;
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
            : app.settings.splat_format == 4 ? "glb" : "auto";
    settings.camera_model = app.settings.camera_model;
    settings.sfm_mode = app.settings.sfm_mode;
    settings.reuse_cache = app.settings.reuse_cache;
    settings.max_features = static_cast<unsigned>(
        std::max(0, app.settings.max_features));
    settings.scene_mode = app.settings.scene_mode;
    settings.iterations = app.settings.iterations;
    settings.densification_cap = app.settings.densification_cap;
    settings.sh_degree = app.settings.sh_degree;
    settings.preview_interval = app.settings.preview_interval;
    settings.strategy = app.settings.strategy;
    settings.max_resolution = app.settings.max_resolution;
    settings.progressive_resolution = app.settings.progressive_resolution;
    settings.use_mask = app.settings.use_mask;
    settings.build_mesh = app.settings.build_mesh;
    settings.mesh_source = app.settings.mesh_source;
    settings.mesh_method = app.settings.mesh_method;
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
    return settings;
}

void apply_project_settings(
    App& app, const aetherscan::project::Settings& settings) {
    if (!settings.image_directory.empty())
        store_path_field(app.settings.images_dir, settings.image_directory);
    store_path_field(app.settings.dataset_source, settings.dataset_source);
    store_path_field(
        app.settings.dataset_initial_cloud, settings.dataset_initial_cloud);
    store_path_field(app.settings.splat_model_source, settings.splat_model_source);
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
    else
        app.settings.splat_format = 0;
    app.settings.camera_model = settings.camera_model;
    app.settings.sfm_mode = settings.sfm_mode;
    app.settings.reuse_cache = settings.reuse_cache;
    app.settings.max_features = static_cast<int>(settings.max_features);
    app.settings.scene_mode = settings.scene_mode;
    app.settings.iterations = settings.iterations;
    app.settings.densification_cap = settings.densification_cap;
    app.settings.sh_degree = settings.sh_degree;
    app.settings.preview_interval = settings.preview_interval;
    app.settings.strategy = settings.strategy;
    app.settings.max_resolution = settings.max_resolution;
    app.settings.progressive_resolution = settings.progressive_resolution;
    app.settings.use_mask = settings.use_mask;
    app.settings.build_mesh = settings.build_mesh;
    app.settings.mesh_source = settings.mesh_source == 1 ? 1 : 0;
    app.settings.mesh_method = settings.mesh_method;
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
    assign_default_project_folder(app);
    request_dataset_scene_load(app);
    set_message(
        app,
        "External cameras loaded. Align Photos is skipped; Train 3DGS or "
        "Extract Mesh can run next.",
        theme::accent);
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
         app.scene_source == "Gaussian centres");
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
                        aetherscan::project::Archive::open(ascan);
                    const auto scene = aetherscan::project::read_sfm(archive);
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

bool save_project_to_path(App& app, const std::filesystem::path& path) {
    try {
        std::error_code error;
        aetherscan::project::Archive archive =
            aetherscan::project::Archive::create();
        const auto current = app.layout.project_file;
        if (!current.empty() && std::filesystem::exists(current, error))
            archive = aetherscan::project::Archive::open(current);
        else if (std::filesystem::exists(path, error))
            archive = aetherscan::project::Archive::open(path);
        aetherscan::project::write_settings(
            archive, collect_project_settings(app), path);
        std::error_code working_error;
        if (!app.layout.working_sfm.empty() &&
            std::filesystem::exists(app.layout.working_sfm, working_error)) {
            const auto scene = load_working_sfm(
                app.layout.working_sfm, reconstruction_images_path(app));
            aetherscan::project::write_sfm(archive, scene, path);
        }
        const auto splat = existing_splat_model(app);
        if (!splat.empty()) {
            try {
                archive.set_chunk(
                    aetherscan::project::ChunkType::gaussians,
                    aetherscan::splat::encode_gaussians(
                        aetherscan::splat::load_gaussians(splat)));
            } catch (...) {
            }
        }
        const std::array<std::filesystem::path, 4> mesh_candidates = {
            app.layout.working_mesh, app.layout.mesh_ply,
            app.layout.mvs_mesh_ply, app.layout.mvs_raw_mesh_ply};
        for (const auto& mesh_path : mesh_candidates) {
            if (mesh_path.empty() ||
                !std::filesystem::exists(mesh_path, working_error))
                continue;
            try {
                archive.set_chunk(
                    aetherscan::project::ChunkType::mesh,
                    aetherscan::mvs::encode_mesh(
                        aetherscan::mvs::load_mesh_ply(mesh_path)));
            } catch (...) {
            }
            break;
        }
#if defined(AETHERSCAN_HAS_TEXTURE)
        if (textured_mesh_on_disk(app.layout.working_texture)) {
            try {
                archive.set_chunk(
                    aetherscan::project::ChunkType::texture,
                    aetherscan::texture::encode_textured_obj(
                        app.layout.working_texture));
            } catch (...) {
            }
        }
#endif
        archive.save(path);
        store_path_field(app.settings.project_dir, path);
        app.project_folder_automatic = false;
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
    app.mesh_load_failed = false;
    app.mesh_gpu_positions.clear();
    app.mesh_gpu_normals.clear();
    app.mesh_gpu_colours.clear();
    app.mesh_gpu_uvs.clear();
    app.mesh_gpu_indices.clear();
    app.mesh_gpu_uploaded = false;
    app.mesh_gpu_failed = false;
    app.mesh_renderer.set_mesh({}, {}, {}, {});
    app.mesh_renderer.set_albedo({});
    app.atlas_preview.reset();
    app.atlas_preview_path.clear();
    app.scene_source.clear();
    app.camera = {};
    app.reconstruction_box.clear();
    app.gizmo.box = {};
    app.view_mode = VisualizationMode::points;
    ++app.scene_load_generation;
    app.loading_scene = false;
}

void clear_loaded_result(App& app) {
    clear_viewport_scene(app);
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
    stop_splat_view(app);
    clear_loaded_result(app);
    app.settings = {};
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
    app.project_folder_automatic = false;
    set_message(app, "New project", theme::text_muted);
}

void apply_image_directory_selection(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    // An image-folder selection starts a different reconstruction. Retaining
    // an explicitly opened project here makes refresh_artifacts() immediately
    // reload that project's old SfM result into the new dataset.
    app.settings.project_dir.fill('\0');
    app.project_folder_automatic = false;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    set_message(
        app, "Image dataset selected; previous viewport result cleared",
        theme::text_muted);
}

void select_image_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_folder(
            L"Select the capture image folder", app.settings.images_dir))
        return;
    app.settings.video_frames_dir.fill('\0');
    apply_image_directory_selection(app);
}

void apply_video_selection(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    app.settings.video_frames_dir.fill('\0');
    app.settings.project_dir.fill('\0');
    app.project_folder_automatic = false;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    set_message(
        app, "Video selected; Align Photos will extract sharp frames, then run SfM",
        theme::text_muted);
}

void select_video_file(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_video_file(
            L"Select a capture video", app.settings.images_dir))
        return;
    apply_video_selection(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path);
void open_asfm_from_path(App& app, const std::filesystem::path& asfm);

void apply_dropped_image_source(
    App& app, const std::vector<std::string>& dropped) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Stop the running job first to change images or video",
            theme::warning);
        return;
    }
    std::vector<std::filesystem::path> videos;
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        std::error_code status;
        if (std::filesystem::is_regular_file(path, status) &&
            aetherscan::io::is_video_path(path))
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
            "Drop photos, a video, an .asfm scene, or an .ascan project on the viewport",
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

void request_gaussian_scene_load(App& app) {
    if (app.loading_scene) return;
    const auto model_path = existing_splat_model(app);
    const auto ascan = app.layout.project_file;
    const auto poses = app.layout.sparse_poses;
    if (model_path.empty() && ascan.empty()) return;
    const bool dataset = has_external_dataset(app);
    const std::filesystem::path dataset_source(app.settings.dataset_source.data());
    const std::filesystem::path dataset_initial_cloud(
        app.settings.dataset_initial_cloud.data());
    const std::filesystem::path dataset_images = reconstruction_images_path(app);
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "Gaussian centres";
    app.pending_load = std::async(
        std::launch::async,
        [model_path, ascan, poses, dataset, dataset_source,
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
                if (!model_path.empty())
                {
                    SceneLoad loaded = load_gaussian_scene(model_path, poses);
                    merge_dataset_views(loaded);
                    return loaded;
                }
                std::error_code error;
                if (std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        aetherscan::project::Archive::open(ascan);
                    if (archive.has(aetherscan::project::ChunkType::gaussians))
                    {
                        SceneLoad loaded = gaussian_scene_from_model(
                            aetherscan::splat::decode_gaussians(
                                archive.chunk(
                                    aetherscan::project::ChunkType::gaussians)),
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
    if (app.scene.has_gaussians() || app.loading_scene || !app.has_model) return;
    request_gaussian_scene_load(app);
}

std::filesystem::path existing_mesh_path(const App& app) {
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
    const bool have_textured = textured_mesh_on_disk(textured);
    if (ply.empty() && ascan.empty() && !have_textured) {
        app.mesh_load_failed = true;
        set_message(app, "No mesh found to preview", theme::warning);
        return;
    }
    app.loading_scene = true;
    app.mesh_load_failed = false;
    app.frame_mesh_on_load = frame_when_ready;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.pending_mesh_load = std::async(
        std::launch::async,
        [ply, ascan, textured, have_textured] {
            if (have_textured) return load_preview_textured_mesh(textured);
            return load_preview_mesh(ply, ascan);
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

aetherscan::splat::GaussianFormat splat_export_format(const App& app) {
    switch (app.splat_export.format) {
        case 1: return aetherscan::splat::GaussianFormat::sog;
        case 2: return aetherscan::splat::GaussianFormat::spz;
        case 3: return aetherscan::splat::GaussianFormat::glb;
        default: return aetherscan::splat::GaussianFormat::ply;
    }
}

const wchar_t* splat_export_extension_wide(const App& app) {
    switch (app.splat_export.format) {
        case 1: return L"sog";
        case 2: return L"spz";
        case 3: return L"glb";
        default: return L"ply";
    }
}

int splat_export_format_from_settings(const int settings_format) {
    switch (settings_format) {
        case 2: return 1;
        case 3: return 2;
        case 4: return 3;
        default: return 0;
    }
}

int splat_settings_format_from_export(const int export_format) {
    switch (export_format) {
        case 1: return 2;
        case 2: return 3;
        case 3: return 4;
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
    const wchar_t* extension = splat_export_extension_wide(app);
    const auto current = path_from_utf8_field(app.splat_export.path.data());
    const std::wstring name = current.filename().empty()
        ? export_stem_wide(app) + L"_splat." + extension
        : current.filename().wstring();
    return pick_export_path(
        L"Export Splat", app.splat_export.path, FilePickKind::splat_model,
        name.c_str(), extension);
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

aetherscan::mvs::Mesh preview_mesh_to_mvs(const PreviewMesh& mesh) {
    aetherscan::mvs::Mesh out;
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
    return app.has_texture || textured_mesh_on_disk(app.layout.working_texture);
}

bool can_export_mesh_file(const App& app) {
    return app.has_mesh || app.mesh.has() || can_export_textured_mesh(app);
}

void export_trained_model(App& app) {
    if (app.job.running() || !can_export_model(app)) return;
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
    if (out.extension().empty()) {
        out += ".";
        out += std::filesystem::path(extension);
        store_path_field(app.splat_export.path, out);
    }
    try {
        const auto source = existing_splat_model(app);
        aetherscan::splat::GaussianModel model;
        if (!source.empty()) {
            model = aetherscan::splat::load_gaussians(source);
        } else {
            const auto archive =
                aetherscan::project::Archive::open(app.layout.project_file);
            model = aetherscan::splat::decode_gaussians(
                archive.chunk(aetherscan::project::ChunkType::gaussians));
        }
        const unsigned export_degree =
            std::min(model.sh_degree,
                     static_cast<unsigned>(app.splat_export.sh_degree));
        const auto dest_ext = lower_path_extension(out);
        const auto source_ext = lower_path_extension(source);
        if (export_degree >= model.sh_degree && !source.empty() &&
            source_ext == dest_ext && copy_existing_file(source, out)) {
            app.settings.splat_format =
                splat_settings_format_from_export(app.splat_export.format);
            set_message(app, "Exported splat", theme::success);
            return;
        }
        aetherscan::splat::restrict_sh_degree(model, export_degree);
        aetherscan::splat::save_gaussians(model, out, format);
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
    if (textured_mesh_on_disk(app.layout.working_texture))
        return app.layout.working_texture;
#if defined(AETHERSCAN_HAS_TEXTURE)
    if (!app.layout.project_file.empty()) {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        if (archive.has(aetherscan::project::ChunkType::texture)) {
            aetherscan::texture::decode_textured_obj(
                archive.chunk(aetherscan::project::ChunkType::texture),
                app.layout.working_texture);
            refresh_artifacts(app);
            if (textured_mesh_on_disk(app.layout.working_texture))
                return app.layout.working_texture;
        }
    }
#endif
    return {};
}

aetherscan::mvs::Mesh load_geometry_mesh(App& app) {
    const auto ply = existing_mesh_path(app);
    if (!ply.empty()) return aetherscan::mvs::load_mesh_ply(ply);
    if (!app.layout.project_file.empty()) {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        if (archive.has(aetherscan::project::ChunkType::mesh))
            return aetherscan::mvs::decode_mesh(
                archive.chunk(aetherscan::project::ChunkType::mesh));
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
#if defined(AETHERSCAN_HAS_TEXTURE)
    if (!source.empty()) {
        aetherscan::texture::copy_textured_obj(source, destination_stem);
        return;
    }
    if (!app.layout.project_file.empty()) {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        if (archive.has(aetherscan::project::ChunkType::texture)) {
            aetherscan::texture::decode_textured_obj(
                archive.chunk(aetherscan::project::ChunkType::texture),
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
    aetherscan::mvs::Mesh mesh = preview_mesh_to_mvs(loaded.mesh);
    std::vector<aetherscan::mvs::Vec2f> uvs;
    if (loaded.mesh.uvs.size() == loaded.mesh.vertices.size()) {
        uvs.reserve(loaded.mesh.uvs.size());
        for (const Vec2& uv : loaded.mesh.uvs)
            uvs.emplace_back(uv.x, 1.F - uv.y);
    }
    const auto png_path = loaded.mesh.albedo_path.empty()
        ? textured_albedo_path(source)
        : loaded.mesh.albedo_path;
    aetherscan::io::save_mesh_glb(
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
            aetherscan::mvs::save_mesh_ply(load_geometry_mesh(app), out);
            set_message(app, "Exported mesh", theme::success);
            return;
        }
        if (format == MeshExportFormat::obj) {
            aetherscan::mvs::save_mesh_obj(load_geometry_mesh(app), out);
            set_message(app, "Exported mesh", theme::success);
            return;
        }
        aetherscan::io::save_mesh_glb(load_geometry_mesh(app), out);
        set_message(app, "Exported mesh", theme::success);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void pack_mesh_gpu_buffers(App& app) {
    app.mesh_gpu_positions.clear();
    app.mesh_gpu_normals.clear();
    app.mesh_gpu_colours.clear();
    app.mesh_gpu_uvs.clear();
    app.mesh_gpu_indices.clear();
    app.mesh_gpu_uploaded = false;
    if (!app.mesh.has()) {
        app.mesh_renderer.set_mesh({}, {}, {}, {});
        app.mesh_renderer.set_albedo({});
        return;
    }
    const std::size_t count = app.mesh.vertices.size();
    app.mesh_gpu_positions.resize(count * 3U);
    for (std::size_t i = 0; i < count; ++i) {
        app.mesh_gpu_positions[3U * i] = app.mesh.vertices[i].x;
        app.mesh_gpu_positions[3U * i + 1U] = app.mesh.vertices[i].y;
        app.mesh_gpu_positions[3U * i + 2U] = app.mesh.vertices[i].z;
    }
    if (app.mesh.normals.size() == count) {
        app.mesh_gpu_normals.resize(count * 3U);
        for (std::size_t i = 0; i < count; ++i) {
            app.mesh_gpu_normals[3U * i] = app.mesh.normals[i].x;
            app.mesh_gpu_normals[3U * i + 1U] = app.mesh.normals[i].y;
            app.mesh_gpu_normals[3U * i + 2U] = app.mesh.normals[i].z;
        }
    }
    if (app.mesh.colours.size() == count) {
        app.mesh_gpu_colours.resize(count * 3U);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t packed = app.mesh.colours[i];
            app.mesh_gpu_colours[3U * i] =
                static_cast<float>((packed >> IM_COL32_R_SHIFT) & 255U) / 255.F;
            app.mesh_gpu_colours[3U * i + 1U] =
                static_cast<float>((packed >> IM_COL32_G_SHIFT) & 255U) / 255.F;
            app.mesh_gpu_colours[3U * i + 2U] =
                static_cast<float>((packed >> IM_COL32_B_SHIFT) & 255U) / 255.F;
        }
    }
    if (app.mesh.uvs.size() == count) {
        app.mesh_gpu_uvs.resize(count * 2U);
        for (std::size_t i = 0; i < count; ++i) {
            app.mesh_gpu_uvs[2U * i] = app.mesh.uvs[i].x;
            app.mesh_gpu_uvs[2U * i + 1U] = app.mesh.uvs[i].y;
        }
    }
    app.mesh_gpu_indices.reserve(app.mesh.faces.size() * 3U);
    for (const auto& face : app.mesh.faces) {
        app.mesh_gpu_indices.push_back(face[0]);
        app.mesh_gpu_indices.push_back(face[1]);
        app.mesh_gpu_indices.push_back(face[2]);
    }
    app.mesh_renderer.set_mesh(
        app.mesh_gpu_positions, app.mesh_gpu_normals, app.mesh_gpu_colours,
        app.mesh_gpu_indices, app.mesh_gpu_uvs);
    if (!app.mesh.albedo_path.empty()) {
        try {
            app.mesh_renderer.set_albedo(
                aetherscan::io::load_rgb(app.mesh.albedo_path));
        } catch (...) {
            app.mesh_renderer.set_albedo({});
        }
    } else {
        app.mesh_renderer.set_albedo({});
    }
    app.mesh_gpu_uploaded = true;
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
    if (!app.mesh.has() || app.mesh_gpu_failed || app.mesh_gpu_indices.empty())
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
            app.mesh_renderer.set_mesh(
                app.mesh_gpu_positions, app.mesh_gpu_normals,
                app.mesh_gpu_colours, app.mesh_gpu_indices, app.mesh_gpu_uvs);
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
    stop_splat_view(app);
    if (app.mesh.has()) {
        frame_reconstruction(app);
        return;
    }
    app.mesh_load_failed = false;
    request_mesh_load(app, frame_when_ready);
}

aetherscan::splat::VisualizeOptions editor_visualize_options(const App& app) {
    aetherscan::splat::VisualizeOptions options;
    options.mode = app.view_mode == VisualizationMode::points
        ? aetherscan::splat::VisualizationMode::points
        : app.view_mode == VisualizationMode::rings
            ? aetherscan::splat::VisualizationMode::rings
            : aetherscan::splat::VisualizationMode::splat;
    options.point_size_px = app.view_options.point_size;
    options.ring_scale = app.view_options.ring_scale;
    return options;
}

void write_preview_vis(App& app) {
    if (app.layout.preview_vis_file.empty()) return;
    aetherscan::splat::write_visualization_sidecar(
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
        show_mesh_view(app, !app.mesh.has());
        write_preview_vis(app);
        return;
    }
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    if (app.has_model && !training)
        start_splat_view(app);
    else if (mode == VisualizationMode::points && !live_preview_active(app) &&
             !app.scene.has_points())
        ensure_sparse_loaded(app);
    write_preview_vis(app);
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);
}
void apply_opened_project(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    app.project_folder_automatic = false;
    refresh_artifacts(app);
    try {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        apply_project_settings(app, aetherscan::project::read_settings(archive));
        refresh_artifacts(app);
#if defined(AETHERSCAN_HAS_TEXTURE)
        if (archive.has(aetherscan::project::ChunkType::texture) &&
            !textured_mesh_on_disk(app.layout.working_texture)) {
            try {
                aetherscan::texture::decode_textured_obj(
                    archive.chunk(aetherscan::project::ChunkType::texture),
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
    if (!pick_project_file(
            L"Open AetherScan Project", app.settings.project_dir, false))
        return;
    apply_opened_project(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot open a project while a job is running", theme::warning);
        return;
    }
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
    stop_splat_view(app);
    clear_loaded_result(app);

    const auto sibling = asfm.parent_path() / (asfm.stem().string() + ".ascan");
    std::error_code error;
    if (std::filesystem::exists(sibling, error)) {
        store_utf8_path_field(app.settings.project_dir, sibling);
        app.project_folder_automatic = false;
        try {
            refresh_artifacts(app);
            const auto archive =
                aetherscan::project::Archive::open(app.layout.project_file);
            apply_project_settings(
                app, aetherscan::project::read_settings(archive));
        } catch (...) {
        }
    } else {
        store_utf8_path_field(
            app.settings.project_dir,
            asfm.parent_path() / (asfm.stem().string() + ".ascan"));
        app.project_folder_automatic = true;
    }

    if (app.settings.images_dir[0] == '\0') {
        const auto inferred = inferred_images_dir_near(asfm);
        if (!inferred.empty())
            store_utf8_path_field(app.settings.images_dir, inferred);
    }
    if (app.settings.project_dir[0] != '\0') refresh_artifacts(app);
    app.has_sparse = true;
    app.has_asfm = true;
    request_asfm_scene_load(app, asfm, asfm.filename().string());
}

void save_project_as(App& app) {
    if (app.job.running()) return;
    if (!pick_project_file(
            L"Save AetherScan Project", app.settings.project_dir, true))
        return;
    {
        std::filesystem::path path =
            path_from_utf8_field(app.settings.project_dir.data());
        std::string extension = path.extension().string();
        std::transform(
            extension.begin(), extension.end(), extension.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
        if (extension != ".ascan") path.replace_extension(".ascan");
        store_path_field(app.settings.project_dir, path);
    }
    app.project_folder_automatic = false;
    refresh_artifacts(app);
    if (save_project_to_path(app, app.layout.project_file))
        set_message(app, "Project saved", theme::success);
}

void save_project(App& app) {
    if (app.job.running()) return;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    if (app.layout.project_file.empty()) {
        save_project_as(app);
        return;
    }
    if (save_project_to_path(app, app.layout.project_file))
        set_message(app, "Project saved", theme::success);
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

    stop_splat_view(app);
    clear_loaded_result(app);
    app.suppress_scene_auto_load = true;
    const std::array<std::filesystem::path, 24> generated_files = {
        app.layout.sparse_ply, app.layout.sparse_asfm, app.layout.sparse_mvs,
        app.layout.sparse_poses, app.layout.splat_ply, app.layout.splat_sog,
        app.layout.splat_spz, app.layout.splat_glb, app.layout.mesh_ply,
        app.layout.mvs_mesh_ply, app.layout.mvs_raw_mesh_ply,
        app.layout.dense_ply, app.layout.working_splat, app.layout.working_mesh,
        app.layout.working_dense, textured_obj_path(app.layout.working_texture),
        textured_mtl_path(app.layout.working_texture),
        textured_albedo_path(app.layout.working_texture),
        app.layout.align_log, app.layout.train_log, app.layout.dense_log,
        app.layout.texture_log, app.layout.export_log, app.layout.view_log};

    std::uintmax_t removed = 0;
    std::string failure;
    for (const std::filesystem::path& path : generated_files) {
        if (path.empty()) continue;
        std::error_code error;
        if (std::filesystem::remove(path, error)) ++removed;
        if (error && failure.empty()) failure = error.message();
    }

    if (!app.layout.project_file.empty()) {
        try {
            if (std::filesystem::exists(app.layout.project_file)) {
                auto archive =
                    aetherscan::project::Archive::open(app.layout.project_file);
                archive.erase_chunk(aetherscan::project::ChunkType::sfm);
                archive.erase_chunk(aetherscan::project::ChunkType::gaussians);
                archive.erase_chunk(aetherscan::project::ChunkType::mesh);
                archive.erase_chunk(aetherscan::project::ChunkType::texture);
                aetherscan::project::write_settings(
                    archive, collect_project_settings(app),
                    app.layout.project_file);
                archive.save(app.layout.project_file);
            }
        } catch (const std::exception& error) {
            if (failure.empty()) failure = error.what();
        }
    }

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

    // With reuse_cache disabled, resolve_layout() places the live sfm.bin and
    // preview sidecars under %TEMP%/AetherScan/<project-hash>, not in
    // layout.cache. Delete that exact namespaced directory as well; leaving it
    // behind makes refresh_artifacts() resurrect the supposedly deleted scene.
    const std::filesystem::path runtime =
        app.layout.working_sfm.parent_path().lexically_normal();
    std::error_code temp_error;
    const std::filesystem::path temp_root =
        (std::filesystem::temp_directory_path(temp_error) / "AetherScan")
            .lexically_normal();
    const bool safe_runtime =
        !temp_error && !runtime.empty() && runtime != temp_root &&
        runtime.parent_path() == temp_root;
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
    invalidate_reconstruction_box(app.reconstruction_box);
    app.mesh_load_failed = false;
    if (app.mesh.has_texture()) app.view_options.mesh_texture = true;
    pack_mesh_gpu_buffers(app);
    ensure_reconstruction_box(app);
    frame_reconstruction(app);
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
    if (app.pending_scene_load_generation != app.scene_load_generation) return;
    if (!loaded.ok) {
        set_message(app, "Point cloud: " + loaded.error, theme::danger);
        return;
    }
    app.photos.clear();
    app.image_qa_session.clear();
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    app.scene = std::move(loaded.scene);
    invalidate_reconstruction_box(app.reconstruction_box);
    app.alignment_preview_seen = false;
    // When an internal SfM or Gaussian scene replaces the imported cameras,
    // allow the next ensure pass to restore the dataset view poses, which are
    // the ones training actually uses.
    if (app.scene_source != "External dataset" &&
        app.scene_source != "Gaussian centres")
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
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered",
        theme::success);
}

void poll_align_live(App& app) {
    const bool aligning = alignment_job_running(app);
    if (!aligning) return;

    aetherscan::sfm::AlignLiveFrame frame;
    const bool have_live =
        aetherscan::sfm::load_align_live_frame(app.layout.align_live, frame);
    if (have_live &&
        (frame.revision != app.align_live.revision ||
         frame.kind != app.align_live.kind ||
         frame.index_a != app.align_live.index_a ||
         frame.index_b != app.align_live.index_b)) {
        app.align_live = std::move(frame);
        refresh_image_qa_folder(
            app.image_qa, reconstruction_images_path(app));
        const int count = image_qa_count(app.image_qa, app.scene);
        if (app.align_live.index_a >= 0)
            select_image_qa_view(app.image_qa, app.align_live.index_a, count);
        if (app.align_live.kind == aetherscan::sfm::AlignLiveKind::features)
            app.image_qa.mode = ImageQaMode::features;
    }

    if (app.alignment_workspace_user_override) return;
    const Stage stage = app.monitor.stage();
    const bool live_2d =
        app.align_live.kind == aetherscan::sfm::AlignLiveKind::features ||
        app.align_live.kind == aetherscan::sfm::AlignLiveKind::matching;
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
    auto path = app.layout.working_sfm;
    path += ".preview.asfm";
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
    stop_splat_view(app);
    if (has_external_dataset(app)) {
        set_message(
            app,
            "External dataset selected; use Train 3DGS instead of Align Photos",
            theme::warning);
        return;
    }
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
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
            !aetherscan::io::ffmpeg_available()) {
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
        app.monitor.begin(JobKind::align);
        auto preview_path = app.layout.working_sfm;
        preview_path += ".preview.asfm";
        std::error_code preview_error;
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
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
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
           app.settings.project_dir[0] != '\0' &&
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
    if (!app.job.running()) return app.viewer.running() ? tr("VIEWING") : tr("READY");
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
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
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
                aetherscan::sfm::save_asfm(scene, app.layout.sparse_asfm);
                aetherscan::sfm::export_openmvs_interface(
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
                    aetherscan::project::Archive::open(app.layout.project_file);
                const auto scene = aetherscan::project::read_sfm(archive);
                if (scene) {
                    aetherscan::sfm::save_asfm(*scene, app.layout.sparse_asfm);
                    aetherscan::sfm::export_openmvs_interface(
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
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
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

std::optional<aetherscan::sfm::Scene> load_alignment_scene(const App& app) {
    std::error_code error;
    if (!app.layout.working_sfm.empty() &&
        std::filesystem::exists(app.layout.working_sfm, error)) {
        return load_working_sfm(
            app.layout.working_sfm, reconstruction_images_path(app));
    }
    if (!app.layout.project_file.empty() &&
        std::filesystem::exists(app.layout.project_file, error)) {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        return aetherscan::project::read_sfm(archive);
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
            aetherscan::sfm::save_asfm(*scene, out);
        else if (format == AlignmentExportFormat::colmap)
            aetherscan::sfm::save_colmap_text(
                *scene, out, images, write_ply);
        else if (format == AlignmentExportFormat::nerfstudio)
            aetherscan::sfm::save_nerfstudio_transforms(
                *scene, out, images, ply);
        else
            aetherscan::sfm::export_openmvs_interface(*scene, out);

        if (write_ply) aetherscan::sfm::save_sparse_ply(*scene, ply);
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

void start_splat_view(App& app) {
    if (app.job.running() || app.viewer.running() || !app.has_model) return;
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] == '\0')
        return;
    refresh_artifacts(app);
    if (!app.has_model) return;
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (!app.layout.preview_camera_file.empty()) {
        std::error_code preview_error;
        std::filesystem::create_directories(
            app.layout.preview_camera_file.parent_path(), preview_error);
    }
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);

    app.preview.create(k_preview_extent, k_preview_extent);
    PreviewHandles handles;
    handles.memory =
        reinterpret_cast<std::uintptr_t>(app.preview.memory_handle);
    handles.semaphore =
        reinterpret_cast<std::uintptr_t>(app.preview.semaphore_handle);
    handles.allocation_size = app.preview.allocation_size;
    handles.width = app.preview.width;
    handles.height = app.preview.height;
    handles.device_luid = gpu::device_luid();
    handles.device_node_mask = gpu::device_node_mask();
    try {
        app.viewer.start(
            build_view_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout, handles),
            app.layout.view_log);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
    app.preview.close_export_handles();
}

void start_train(App& app, const bool smoke) {
    if (app.job.running()) return;
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
        return;
    }
    refresh_artifacts(app);
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

    // The trainer inherits the shared allocation, so the exported handles are
    // recreated per run: the timeline counter has to restart from zero.
    app.preview.create(k_preview_extent, k_preview_extent);
    PreviewHandles handles;
    handles.memory =
        reinterpret_cast<std::uintptr_t>(app.preview.memory_handle);
    handles.semaphore =
        reinterpret_cast<std::uintptr_t>(app.preview.semaphore_handle);
    handles.allocation_size = app.preview.allocation_size;
    handles.width = app.preview.width;
    handles.height = app.preview.height;
    handles.device_luid = gpu::device_luid();
    handles.device_node_mask = gpu::device_node_mask();

    std::string command;
    if (smoke) {
        // Fixed COLMAP dataset used only by --interop-smoke.
        std::ostringstream smoke_command;
        smoke_command
            << '"' << AETHERSCAN_CLI_PATH << "\" --images \""
            << app.settings.images_dir.data() << "\" --output \""
            << app.layout.model_output.string()
            << "\" --splat-dataset \"D:\\ScanVideo\\ori_img\""
            << " --splat-use-mask false"
            << " --splat --splat-strategy adc_plus --splat-iterations "
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
        command = smoke_command.str();
    } else {
        command = build_train_command(
            AETHERSCAN_CLI_PATH, app.settings, app.layout, handles);
    }

    try {
        app.monitor.begin(JobKind::train);
        app.log.open(app.layout.train_log);
        app.job.start(command, app.layout.train_log);
        app.active_job = JobKind::train;
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

void start_dense(App& app) {
    if (app.job.running()) return;
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
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
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
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
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
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
#if !defined(AETHERSCAN_HAS_TEXTURE)
    set_message(
        app,
        "This build was compiled without texture baking (Vulkan + aether_drender)",
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
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
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
    app.active_job = JobKind::none;
    refresh_artifacts(app);

    if (code == 2) {
        set_message(
            app,
            std::string(job_name(kind)) +
                " stopped. You can change images, video, or parameters and run again.",
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
        if (mesh_from_mvs(app.settings)) {
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
    } else if (!app.smoke_mode && app.has_model) {
        start_splat_view(app);
    }
}

}  // namespace editor
