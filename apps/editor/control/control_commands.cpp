#include "control/control_commands.hpp"

#include "control/control_bus.hpp"
#include "control/control_state.hpp"

#include "app.hpp"
#include "core/version.hpp"
#include "io/image.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace editor::control {
namespace {

std::int64_t current_pid() {
#if defined(_WIN32)
    return static_cast<std::int64_t>(_getpid());
#else
    return static_cast<std::int64_t>(getpid());
#endif
}

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension;
}

bool path_within(
    const std::filesystem::path& candidate,
    const std::filesystem::path& root) {
    if (root.empty()) return false;
    std::error_code error;
    const auto absolute_candidate =
        std::filesystem::weakly_canonical(candidate, error);
    if (error) return false;
    const auto absolute_root =
        std::filesystem::weakly_canonical(root, error);
    if (error) return false;
    const auto relative = absolute_candidate.lexically_relative(absolute_root);
    if (relative.empty()) return false;
    for (const auto& part : relative) {
        if (part == "..") return false;
    }
    return true;
}

bool allowed_capture_path(
    const App& app, const std::filesystem::path& path) {
    if (path.is_relative()) return false;
    if (path_within(path, app.layout.root) ||
        path_within(path, app.layout.cache))
        return true;
    std::error_code error;
    const auto temporary =
        std::filesystem::temp_directory_path(error);
    if (!error && path_within(path, temporary)) return true;
    if (const char* roots = std::getenv("PHOTARA_CONTROL_ALLOWED_ROOTS");
        roots != nullptr && roots[0] != '\0') {
        std::string text(roots);
        std::size_t begin = 0;
        while (begin <= text.size()) {
            const std::size_t end = text.find(';', begin);
            const std::string item = text.substr(
                begin, end == std::string::npos ? std::string::npos
                                                : end - begin);
            if (!item.empty() && path_within(path, item)) return true;
            if (end == std::string::npos) break;
            begin = end + 1;
        }
    }
    return false;
}

json require_string(
    const json& params, const char* name, std::string& output) {
    if (!params.contains(name) || !params[name].is_string())
        return failure(
            "INVALID_REQUEST", std::string(name) + " must be a string");
    output = params[name].get<std::string>();
    if (output.empty())
        return failure(
            "INVALID_REQUEST", std::string(name) + " must not be empty");
    return success();
}

json job_failure(const App& app, const std::string& stage) {
    const std::string message = app.message.empty()
        ? "The job did not start" : app.message;
    return failure(
        "PRECONDITION_FAILED", message,
        {{"stage", stage}, {"suggested_action", "studio.get_state"}});
}

json output_path(const App& app, const json& params,
                 const std::string& extension, std::filesystem::path& path) {
    std::string text;
    const json required = require_string(params, "path", text);
    if (!ok(required)) return required;
    path = path_from_utf8_field(text.c_str());
    if (path.extension().empty()) path.replace_extension(extension);
    if (lower_extension(path) != extension)
        return failure("INVALID_REQUEST", "Output extension must be " + extension);
    if (!allowed_capture_path(app, path))
        return failure("PERMISSION_DENIED", "Output must be an absolute path under the project, cache, temp or PHOTARA_CONTROL_ALLOWED_ROOTS");
    std::error_code error;
    if (std::filesystem::exists(path, error) && !params.value("overwrite", false))
        return failure("ALREADY_EXISTS", "Output exists; specify overwrite=true to replace it");
    if (error) return failure("INTERNAL", error.message());
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return failure("INTERNAL", error.message());
    return success();
}

}  // namespace

json execute_command(
    App& app, ControlBus& bus, const std::string& method,
    const json& params, const std::string& client) {
    if (method == "studio.ping") {
        return success({
            {"version", PHOTARA_VERSION},
            {"schema_version", k_schema_version},
            {"session_id", bus.session_id()},
            {"pid", current_pid()},
            {"revision", bus.revision()},
        });
    }
    if (method == "studio.get_state")
        return snapshot_state(app, bus);
    if (method == "studio.params.describe")
        return describe_schema();
    if (method == "studio.params.patch")
        return apply_parameter_patch(app, params.value("patch", json::object()));
    if (method == "studio.job.status")
        return bus.operation_status(app, params.value("op_id", std::string{}));
    if (method == "studio.inspect.logs") {
        const auto limit = static_cast<std::size_t>(
            std::clamp(params.value("limit", 200), 1, 2000));
        const auto& lines = app.log.lines();
        json output = json::array();
        const std::size_t begin =
            lines.size() > limit ? lines.size() - limit : 0;
        for (std::size_t i = begin; i < lines.size(); ++i) {
            const ConsoleLine& line = lines[i];
            const char* severity = "other";
            switch (line.severity) {
                case ConsoleSeverity::debug: severity = "debug"; break;
                case ConsoleSeverity::info: severity = "info"; break;
                case ConsoleSeverity::warning: severity = "warning"; break;
                case ConsoleSeverity::error: severity = "error"; break;
                case ConsoleSeverity::other: break;
            }
            output.push_back({
                {"severity", severity},
                {"time", line.time},
                {"message", line.message},
            });
        }
        return success({
            {"lines", std::move(output)},
            {"generation", app.log.generation()},
        });
    }
    if (method == "studio.project.save") {
        if (app.job.running() || app.pending_train_prepare.valid() || app.loading_scene ||
            app.saving_training || app.splat_export.job.valid())
            return failure("BUSY", "Finish the active task before saving through the control API");
        json options = params;
        if (!options.contains("path")) {
            if (app.layout.project_file.empty())
                return failure("INVALID_REQUEST", "A path is required for the first save");
            options["path"] = path_to_utf8(app.layout.project_file);
            options["overwrite"] = true;
        }
        std::filesystem::path path;
        const json checked = output_path(app, options, ".ascan", path);
        if (!ok(checked)) return checked;
        if (photara::splat::training_checkpoint_in_use(path))
            return failure("BUSY", "The destination is being trained in another process");
        ensure_task_cache(app);
        refresh_artifacts(app);
        if (!save_project_to_path(app, path)) return failure("INTERNAL", app.message);
        set_message(app, "Project saved", theme::success);
        return success({{"path", path_to_utf8(path)}, {"completed", true}});
    }
    if (method == "studio.edit.apply") {
        if (app.job.running() || app.pending_train_prepare.valid() || app.loading_scene ||
            app.pending_splat_model.valid() || app.splat_export.job.valid() || app.splat_edit.gesture_active())
            return failure("BUSY", "Finish the active task or gesture before editing");
        if (!app.has_model || !app.splat_renderer)
            return failure("PRECONDITION_FAILED", "Load the splat model in the viewport before editing");
        const std::string action = params.value("action", std::string{});
        auto& edit = app.splat_edit;
        if (action == "select_all") edit.select_all();
        else if (action == "clear_selection") edit.clear_selection();
        else if (action == "invert_selection") edit.invert_selection();
        else if (action == "delete_selected") {
            if (!edit.selected_count()) return failure("PRECONDITION_FAILED", "No Gaussians selected");
            edit.delete_selected(app);
        } else if (action == "undo") {
            if (!edit.can_undo()) return failure("PRECONDITION_FAILED", "Undo history is empty");
            edit.undo_edit(app);
        } else if (action == "redo") {
            if (!edit.can_redo()) return failure("PRECONDITION_FAILED", "Redo history is empty");
            edit.redo_edit(app);
        } else return failure("INVALID_REQUEST", "Unknown edit action");
        return success({{"gaussians", edit.gaussian_count()}, {"selected", edit.selected_count()},
            {"can_undo", edit.can_undo()}, {"can_redo", edit.can_redo()}});
    }
    if (method == "studio.export.mesh") {
        if (app.job.running() || app.pending_train_prepare.valid() || app.loading_scene)
            return failure("BUSY", "An active task prevents export");
        if (!can_export_mesh_file(app)) return failure("PRECONDITION_FAILED", "No mesh to export");
        const std::string format = params.value("format", std::string("ply"));
        const std::vector<std::string> formats{"ply", "obj", "glb"};
        const auto selected = std::find(formats.begin(), formats.end(), format);
        if (selected == formats.end()) return failure("INVALID_REQUEST", "Mesh formats: ply, obj, glb");
        const bool textured = params.value("include_texture", false);
        if (textured && (format == "ply" || !can_export_textured_mesh(app)))
            return failure("PRECONDITION_FAILED", "Textured export requires a textured mesh and OBJ/GLB");
        std::filesystem::path path;
        const json checked = output_path(app, params, "." + format, path);
        if (!ok(checked)) return checked;
        // OBJ creates sidecars; only expose single-file textured GLB initially.
        if (textured && format == "obj")
            return failure("UNSUPPORTED", "Use GLB for textured control export");
        app.mesh_export.format = static_cast<int>(selected - formats.begin());
        app.mesh_export.include_texture = textured;
        if (!export_mesh_to_path(app, path)) return failure("INTERNAL", app.message);
        return success({{"path", path_to_utf8(path)}, {"completed", true}});
    }
    if (method == "studio.inspect.quality") {
        const double min_ratio = params.value("min_registered_ratio", 0.8);
        const double max_error = params.value("max_mean_reprojection", 2.0);
        const double max_loss = params.value("max_training_loss", 0.1);
        if (!std::isfinite(min_ratio) || min_ratio < 0 || min_ratio > 1 ||
            !std::isfinite(max_error) || max_error <= 0 || !std::isfinite(max_loss) || max_loss <= 0)
            return failure("INVALID_REQUEST", "Invalid quality thresholds");
        const auto& training = app.monitor.training();
        const bool alignment_known = app.scene.total_views > 0 && !app.loading_scene;
        const double ratio = alignment_known ? static_cast<double>(app.scene.registered_views) / app.scene.total_views : 0;
        const bool training_known = training.valid && std::isfinite(training.loss);
        json checks = json::array();
        checks.push_back({{"name", "registered_ratio"}, {"status", !alignment_known ? "unknown" : ratio >= min_ratio ? "passed" : "failed"},
            {"value", alignment_known ? json(ratio) : json(nullptr)}, {"minimum", min_ratio}});
        checks.push_back({{"name", "mean_reprojection"}, {"status", !alignment_known || !std::isfinite(app.scene.mean_reprojection) ? "unknown" : app.scene.mean_reprojection <= max_error ? "passed" : "failed"},
            {"value", alignment_known ? json(app.scene.mean_reprojection) : json(nullptr)}, {"maximum", max_error}});
        checks.push_back({{"name", "training_loss"}, {"status", !training_known ? "unknown" : training.loss <= max_loss ? "passed" : "failed"},
            {"value", training_known ? json(training.loss) : json(nullptr)}, {"maximum", max_loss}});
        return success({{"checks", checks}, {"scope", "loaded_scene_and_latest_monitor"},
            {"note", "Unknown metrics are not a pass; training loss is not a perceptual quality score"}});
    }
    if (method == "studio.export.model") {
        if (app.job.running() || app.pending_train_prepare.valid() || app.loading_scene ||
            app.splat_export.job.valid()) return failure("BUSY", "An active task prevents export");
        if (!can_export_model(app)) return failure("PRECONDITION_FAILED", "No splat model to export");
        const std::string format = params.value("format", std::string("ply"));
        const std::vector<std::string> formats{"ply", "sog", "spz", "glb"};
        const auto selected = std::find(formats.begin(), formats.end(), format);
        if (selected == formats.end())
            return failure("UNSUPPORTED", "Supported formats: ply, sog, spz, glb");
        const json degree = params.value("sh_degree", json(3));
        if (!degree.is_number_integer() || degree.get<int>() < 0 || degree.get<int>() > 3)
            return failure("INVALID_REQUEST", "sh_degree must be an integer in [0, 3]");
        std::filesystem::path path;
        const json checked = output_path(app, params, "." + format, path);
        if (!ok(checked)) return checked;
        app.splat_export.format = static_cast<int>(selected - formats.begin());
        app.splat_export.sh_degree = degree.get<int>();
        if (!export_trained_model_to_path(app, path)) return failure("INTERNAL", app.message);
        return success({{"path", path_to_utf8(path)}, {"format", format}, {"completed", true}});
    }
    if (method == "studio.view.patch") {
        if (app.loading_scene) return failure("BUSY", "Scene is loading");
        const json patch = params.value("patch", json::object());
        if (!patch.is_object()) return failure("INVALID_REQUEST", "patch must be an object");
        auto camera = app.camera;
        auto mode = app.view_mode;
        auto channel = app.display_channel;
        auto workspace = app.workspace;
        const auto choice = [](const json& value, const std::vector<std::string>& names) {
            if (!value.is_string()) return -1;
            const auto found = std::find(names.begin(), names.end(), value.get<std::string>());
            return found == names.end() ? -1 : static_cast<int>(found - names.begin());
        };
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            const auto& key = it.key();
            const auto& value = it.value();
            if (key == "mode" || key == "channel" || key == "workspace" || key == "projection") {
                int index = -1;
                if (key == "mode") {
                    index = choice(value, {"points", "splat", "rings", "mesh"});
                    if (index >= 0) mode = static_cast<VisualizationMode>(index);
                } else if (key == "channel") {
                    index = choice(value, {"color", "depth", "normal"});
                    if (index >= 0) channel = static_cast<DisplayChannel>(index);
                } else if (key == "workspace") {
                    index = choice(value, {"scene_3d", "image_2d"});
                    if (index >= 0) workspace = static_cast<ViewportWorkspace>(index);
                } else {
                    index = choice(value, {"perspective", "orthographic", "fisheye", "panorama"});
                    if (index >= 0) camera.projection = static_cast<EditorProjection>(index);
                }
                if (index < 0) return failure("INVALID_REQUEST", "Invalid view value: " + key);
            } else if (key == "target") {
                if (!value.is_array() || value.size() != 3)
                    return failure("INVALID_REQUEST", "target must contain three finite coordinates");
                float coordinates[3];
                for (int i = 0; i < 3; ++i) {
                    if (!value[i].is_number() || !std::isfinite(value[i].get<double>()) ||
                        std::abs(value[i].get<double>()) > 1e12)
                        return failure("INVALID_REQUEST", "Invalid target coordinate");
                    coordinates[i] = value[i].get<float>();
                }
                camera.target = {coordinates[0], coordinates[1], coordinates[2]};
            } else {
                if (key != "yaw" && key != "pitch" && key != "distance" &&
                    key != "fov_degrees" && key != "ortho_height")
                    return failure("INVALID_REQUEST", "Unknown view field: " + key);
                if (!value.is_number() || !std::isfinite(value.get<double>()))
                    return failure("INVALID_REQUEST", "View value must be finite: " + key);
                const double number = value.get<double>();
                const double min = key == "pitch" ? -1.5707 : key == "yaw" ? -1e6 :
                    key == "fov_degrees" ? 1.0 : 1e-5;
                const double max = key == "pitch" ? 1.5707 : key == "fov_degrees" ? 179.0 : 1e6;
                if (number < min || number > max)
                    return failure("INVALID_REQUEST", "View value out of range: " + key);
                const float n = static_cast<float>(number);
                if (key == "yaw") camera.yaw = n;
                else if (key == "pitch") camera.pitch = n;
                else if (key == "distance") camera.distance = n;
                else if (key == "fov_degrees") camera.fov_degrees = n;
                else camera.ortho_height = n;
            }
        }
        set_viewport_workspace(app, workspace);
        app.camera = camera;
        app.view_mode = mode;
        app.display_channel = channel;
        publish_preview_vis(app);
        return success({{"view", snapshot_state(app, bus)["view"]}});
    }
    if (method == "studio.job.start") {
        if (app.job.running() || app.pending_train_prepare.valid())
            return failure("BUSY", "A Studio job is already running");
        std::string stage;
        const json required = require_string(params, "stage", stage);
        if (!ok(required)) return required;
        if (stage != "align" && stage != "train" && stage != "resume_training" &&
            stage != "dense" && stage != "texture" && stage != "export_sfm")
            return failure(
                "INVALID_REQUEST", "Unsupported stage",
                {{"stage", stage},
                 {"supported", json::array(
                      {"align", "train", "resume_training", "dense",
                       "texture", "export_sfm"})}});

        const bool dry_run = params.value("dry_run", false);
        if (dry_run) {
            const json actions = snapshot_state(app, bus)["actions"];
            const std::string action =
                stage == "resume_training" ? "resume_training" : stage;
            if (!actions.contains(action))
                return failure("UNSUPPORTED", "Stage has no action state");
            const json& state = actions[action];
            if (!state.value("available", false))
                return failure(
                    "PRECONDITION_FAILED", "Stage is not available",
                    {{"stage", stage}, {"reason", state.value("reason", "unknown")}});
            return success({{"dry_run", true}, {"stage", stage}});
        }

        if (stage == "export_sfm") {
            if (!can_export_sfm(app))
                return failure(
                    "PRECONDITION_FAILED",
                    "Align photos before exporting SfM alignment");
            editor::start_export_sfm(app);
            if (!app.job.running() && !app.pending_train_prepare.valid())
                return success({
                    {"completed", true},
                    {"stage", stage},
                    {"message", app.message},
                });
        } else if (stage == "align") {
            editor::start_align(app);
        } else if (stage == "train") {
            editor::start_train(app, false, false);
        } else if (stage == "resume_training") {
            editor::start_train(app, false, true);
        } else if (stage == "dense") {
            editor::start_dense(app);
        } else if (stage == "texture") {
            editor::start_texture(app);
        }

        if (!app.job.running() && !app.pending_train_prepare.valid())
            return job_failure(app, stage);
        const std::string op_id = bus.begin_operation(stage, client);
        return success({
            {"op_id", op_id},
            {"stage", stage},
            {"revision", bus.revision()},
        });
    }
    if (method == "studio.job.control") {
        std::string action;
        const json required = require_string(params, "action", action);
        if (!ok(required)) return required;
        if (action == "pause") {
            if (!app.job.running())
                return failure("PRECONDITION_FAILED", "No job is running");
            app.job.pause();
            if (!app.job.paused())
                return failure("INTERNAL", "Could not pause the running job");
            app.monitor.pause_clock();
            editor::set_message(app, "Paused by agent", theme::warning);
            return success({{"action", action}, {"paused", true}});
        }
        if (action == "resume") {
            if (!app.job.running())
                return failure("PRECONDITION_FAILED", "No job is running");
            app.job.resume();
            if (app.job.paused())
                return failure("INTERNAL", "Could not resume the paused job");
            app.monitor.resume_clock();
            editor::set_message(app, "Resumed by agent", theme::accent);
            return success({{"action", action}, {"paused", false}});
        }
        if (action == "stop") {
            if (!app.job.running())
                return failure("PRECONDITION_FAILED", "No job is running");
            // pump_post_present() calls this after the draw list is submitted.
            bus.mark_operation_cancelled();
            app.job.stop();
            if (app.job.consume_completion())
                editor::on_job_finished(app);
            else
                editor::set_message(
                    app,
                    "Stopped by agent. You can change inputs or parameters.",
                    theme::warning);
            return success({{"action", action}, {"running", false}});
        }
        return failure(
            "INVALID_REQUEST", "action must be pause, resume, or stop",
            {{"action", action}});
    }
    if (method == "studio.view.capture") {
        std::string text;
        const json required = require_string(params, "path", text);
        if (!ok(required)) return required;
        std::filesystem::path path = path_from_utf8_field(text.c_str());
        if (lower_extension(path).empty()) path.replace_extension(".png");
        if (lower_extension(path) != ".png")
            return failure("INVALID_REQUEST", "Only PNG capture is supported");
        if (!allowed_capture_path(app, path))
            return failure(
                "PERMISSION_DENIED",
                "Capture path must be under the project, cache, temp directory, "
                "or PHOTARA_CONTROL_ALLOWED_ROOTS");
        const std::uint32_t max_long_edge = static_cast<std::uint32_t>(
            std::clamp(params.value("max_long_edge", 0), 0, 8192));
        photara::io::RgbImage image;
        bool captured = false;
        if (app.splat_renderer) {
            std::vector<std::uint8_t> pixels;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            if (app.splat_renderer->download_rgb(
                    pixels, width, height, max_long_edge)) {
                image.width = width;
                image.height = height;
                image.pixels = std::move(pixels);
                captured = true;
            }
        }
        if (!captured &&
            app.preview.display.download_rgb(image, max_long_edge))
            captured = true;
        if (!captured)
            return failure(
                "UNSUPPORTED",
                "No rendered preview is available yet; start a preview or train first");
        std::error_code error;
        if (!path.parent_path().empty())
            std::filesystem::create_directories(path.parent_path(), error);
        if (error)
            return failure("INTERNAL", "Cannot create capture directory");
        try {
            photara::io::save_rgb_png(image, path);
        } catch (const std::exception& exception) {
            return failure("INTERNAL", exception.what());
        }
        return success({
            {"path", path_to_utf8(path)},
            {"width", image.width},
            {"height", image.height},
        });
    }
    if (method == "studio.project.open") {
        if (app.job.running() || app.loading_scene)
            return failure("BUSY", "Cannot open a project while a job is running");
        std::string text;
        const json required = require_string(params, "path", text);
        if (!ok(required)) return required;
        const std::filesystem::path path =
            path_from_utf8_field(text.c_str());
        std::error_code error;
        if (!std::filesystem::exists(path, error))
            return failure("NOT_FOUND", "Project path does not exist");
        if (has_unsaved_project_changes(app))
            return failure(
                "CONFIRMATION_REQUIRED",
                "The current project has unsaved changes; save or discard them in Studio");
        const bool previous = app.project_transition_approved;
        app.project_transition_approved = true;
        try {
            editor::open_project_from_path(app, path);
        } catch (...) {
            app.project_transition_approved = previous;
            throw;
        }
        app.project_transition_approved = previous;
        const std::filesystem::path opened =
            path_from_utf8_field(app.settings.project_dir.data());
        std::error_code compare_error;
        if (opened.empty() ||
            !std::filesystem::equivalent(opened, path, compare_error)) {
            return failure(
                "PRECONDITION_FAILED",
                app.message.empty() ? "Project was not opened" : app.message);
        }
        return success({
            {"project", path_to_utf8(opened)},
            {"message", app.message},
        });
    }
    if (method == "studio.window.open") {
        std::string name;
        const json required = require_string(params, "window", name);
        if (!ok(required)) return required;
        if (name == "video") {
            open_video_window(app);
        } else if (name == "mask") {
            open_mask_editor(app);
        } else {
            return failure(
                "INVALID_ARGUMENT",
                "window must be \"video\" or \"mask\"");
        }
        return success({{"window", name}, {"visible", true}});
    }
    if (method == "studio.project.set_source") {
        if (app.job.running() || app.loading_scene)
            return failure("BUSY", "Cannot change the source while a job is running");
        std::string text;
        const json required = require_string(params, "path", text);
        if (!ok(required)) return required;
        const std::filesystem::path path =
            path_from_utf8_field(text.c_str());
        std::error_code error;
        if (!std::filesystem::exists(path, error))
            return failure("NOT_FOUND", "Image source does not exist");
        if (has_unsaved_project_changes(app))
            return failure(
                "CONFIRMATION_REQUIRED",
                "The current project has unsaved changes; save or discard them in Studio");
        const bool previous = app.project_transition_approved;
        app.project_transition_approved = true;
        try {
            editor::change_image_source(app, path);
        } catch (...) {
            app.project_transition_approved = previous;
            throw;
        }
        app.project_transition_approved = previous;
        const std::filesystem::path selected =
            path_from_utf8_field(app.settings.images_dir.data());
        std::error_code compare_error;
        if (selected.empty() ||
            !std::filesystem::equivalent(selected, path, compare_error)) {
            return failure(
                "PRECONDITION_FAILED",
                app.message.empty() ? "Image source was not selected" : app.message);
        }
        return success({
            {"source", path_to_utf8(selected)},
            {"message", app.message},
        });
    }
    return failure(
        "UNSUPPORTED", "Unknown control method", {{"method", method}});
}

}  // namespace editor::control
