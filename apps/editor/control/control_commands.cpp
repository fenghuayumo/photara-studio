#include "control/control_commands.hpp"

#include "control/control_bus.hpp"
#include "control/control_state.hpp"

#include "app.hpp"
#include "core/version.hpp"
#include "io/image.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
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
        std::filesystem::absolute(candidate, error).lexically_normal();
    if (error) return false;
    const auto absolute_root =
        std::filesystem::absolute(root, error).lexically_normal();
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
        return success({{"job", snapshot_job(app, bus)}});
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
