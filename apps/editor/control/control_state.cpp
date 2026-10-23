#include "control/control_state.hpp"

#include "control/control_bus.hpp"

#include "app.hpp"
#include "core/version.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace editor::control {
namespace {

std::string path_text(const std::filesystem::path& path) {
    return path.empty() ? std::string{} : path_to_utf8(path);
}

template <std::size_t N>
std::string field_text(const std::array<char, N>& field) {
    return field[0] == '\0' ? std::string{} : std::string(field.data());
}

const char* workspace_name(const ViewportWorkspace workspace) {
    return workspace == ViewportWorkspace::image_2d ? "image_2d" : "scene_3d";
}

const char* mode_name(const VisualizationMode mode) {
    switch (mode) {
        case VisualizationMode::splat: return "splat";
        case VisualizationMode::rings: return "rings";
        case VisualizationMode::mesh: return "mesh";
        case VisualizationMode::points:
        default: return "points";
    }
}

const char* channel_name(const DisplayChannel channel) {
    switch (channel) {
        case DisplayChannel::depth: return "depth";
        case DisplayChannel::normal: return "normal";
        case DisplayChannel::color:
        default: return "color";
    }
}

const char* severity_name(const ConsoleSeverity severity) {
    switch (severity) {
        case ConsoleSeverity::debug: return "debug";
        case ConsoleSeverity::info: return "info";
        case ConsoleSeverity::warning: return "warning";
        case ConsoleSeverity::error: return "error";
        default: return "other";
    }
}

const char* job_kind_token(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "align";
        case JobKind::train: return "train";
        case JobKind::dense: return "dense";
        case JobKind::texture: return "texture";
        case JobKind::export_sfm: return "export_sfm";
        case JobKind::none:
        default: return "none";
    }
}

const char* stage_token(const Stage stage) {
    switch (stage) {
        case Stage::idle: return "idle";
        case Stage::features: return "features";
        case Stage::matching: return "matching";
        case Stage::tracks: return "tracks";
        case Stage::mapping: return "mapping";
        case Stage::exporting: return "exporting";
        case Stage::preparing: return "preparing";
        case Stage::masking: return "masking";
        case Stage::dense: return "dense";
        case Stage::training: return "training";
        case Stage::meshing: return "meshing";
        case Stage::texturing: return "texturing";
        case Stage::complete: return "complete";
        case Stage::failed: return "failed";
        default: return "idle";
    }
}

const char* prepare_stage_name(const TrainPrepareStage stage) {
    switch (stage) {
        case TrainPrepareStage::checking: return "checking";
        case TrainPrepareStage::reading_sfm: return "reading_sfm";
        case TrainPrepareStage::encoding_sfm: return "encoding_sfm";
        case TrainPrepareStage::restoring_progress: return "restoring_progress";
        case TrainPrepareStage::ready: return "ready";
        default: return "checking";
    }
}

json setting_spec(
    const std::string& name, const std::string& type,
    const std::string& description, json minimum = nullptr,
    json maximum = nullptr, json default_value = nullptr,
    json values = nullptr) {
    json spec{
        {"name", name},
        {"type", type},
        {"description", description},
    };
    if (!minimum.is_null()) spec["min"] = std::move(minimum);
    if (!maximum.is_null()) spec["max"] = std::move(maximum);
    if (!default_value.is_null()) spec["default"] = std::move(default_value);
    if (!values.is_null()) spec["values"] = std::move(values);
    return spec;
}

json settings_json(const ProjectSettings& settings) {
    return json{
        {"camera_model", settings.camera_model},
        {"sfm_mode", settings.sfm_mode},
        {"max_features", settings.max_features},
        {"reuse_cache", settings.reuse_cache},
        {"iterations", settings.iterations},
        {"strategy", settings.strategy},
        {"training_mode", settings.training_mode},
        {"sh_degree", settings.sh_degree},
        {"densification_cap", settings.densification_cap},
        {"max_resolution", settings.max_resolution},
        {"progressive_resolution", settings.progressive_resolution},
        {"preview_interval", settings.preview_interval},
        {"geometry_from_iter", settings.geometry_from_iter},
        {"normal_field", settings.normal_field},
        {"ai_priors", settings.ai_priors},
        {"ai_prior_source", settings.ai_prior_source},
        {"ai_depth_weight", settings.ai_depth_weight},
        {"ai_normal_weight", settings.ai_normal_weight},
        {"ai_prior_tokens", settings.ai_prior_tokens},
        {"ppisp_layout", settings.ppisp_layout},
        {"bilateral_grid", settings.bilateral_grid},
        {"build_mesh", settings.build_mesh},
        {"mesh_source", settings.mesh_source},
        {"mesh_method", settings.mesh_method},
        {"mesh_quality", settings.mesh_quality},
        {"mesh_postprocess_quality", settings.mesh_postprocess_quality},
        {"mesh_simplify", settings.mesh_simplify},
        {"mesh_target_faces", settings.mesh_target_faces},
        {"mesh_remesh", settings.mesh_remesh},
        {"mesh_remesh_target_faces", settings.mesh_remesh_target_faces},
        {"mesh_remesh_smooth_iterations", settings.mesh_remesh_smooth_iterations},
        {"texture_quality", settings.texture_quality},
        {"atlas_resolution", settings.atlas_resolution},
        {"texture_delight", settings.texture_delight},
        {"texture_optimize", settings.texture_optimize},
        {"texture_photometric_steps", settings.texture_photometric_steps},
        {"texture_seam_steps", settings.texture_seam_steps},
        {"texture_blend_linear", settings.texture_blend_linear},
        {"video_fps", settings.video_fps},
        {"video_max_frames", settings.video_max_frames},
        {"video_sharp_window", settings.video_sharp_window},
        {"video_quality", settings.video_quality},
        {"video_scale", settings.video_scale},
        {"video_rotate", settings.video_rotate},
        {"sam_masks", settings.sam_masks},
        {"sam_use_mask_for_sfm", settings.sam_use_mask_for_sfm},
        {"sam_text", field_text(settings.sam_text)},
        {"sam_negative_text", field_text(settings.sam_negative_text)},
        {"sam_keep_prompted", settings.sam_keep_prompted},
        {"sam_video", settings.sam_video},
        {"sam_max_size", settings.sam_max_size},
        {"splat_format", settings.splat_format},
    };
}

bool read_bool(
    const json& value, const char* name, bool& output, std::string& message) {
    if (!value.is_boolean()) {
        message = std::string(name) + " must be a boolean";
        return false;
    }
    output = value.get<bool>();
    return true;
}

bool read_integer(
    const json& value, const char* name, const long long minimum,
    const long long maximum, long long& output, std::string& message) {
    if (!value.is_number()) {
        message = std::string(name) + " must be an integer";
        return false;
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::floor(number) != number) {
        message = std::string(name) + " must be an integer";
        return false;
    }
    if (number < static_cast<double>(minimum) ||
        number > static_cast<double>(maximum)) {
        message = std::string(name) + " must be in [" +
                  std::to_string(minimum) + ", " + std::to_string(maximum) + "]";
        return false;
    }
    output = static_cast<long long>(number);
    return true;
}

bool read_number(
    const json& value, const char* name, const double minimum,
    const double maximum, float& output, std::string& message) {
    if (!value.is_number()) {
        message = std::string(name) + " must be a number";
        return false;
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < minimum || number > maximum) {
        message = std::string(name) + " is out of range";
        return false;
    }
    output = static_cast<float>(number);
    return true;
}

bool read_string(
    const json& value, const char* name, const std::size_t maximum,
    std::string& output, std::string& message) {
    if (!value.is_string()) {
        message = std::string(name) + " must be a string";
        return false;
    }
    output = value.get<std::string>();
    if (output.size() > maximum) {
        message = std::string(name) + " is too long";
        return false;
    }
    return true;
}

bool read_choice(
    const json& value, const char* name, const std::vector<const char*>& tokens,
    int& output, std::string& message) {
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (text == tokens[i]) {
                output = static_cast<int>(i);
                return true;
            }
        }
    } else if (value.is_number()) {
        const double number = value.get<double>();
        if (std::isfinite(number) && std::floor(number) == number &&
            number >= 0.0 && number < static_cast<double>(tokens.size())) {
            output = static_cast<int>(number);
            return true;
        }
    }
    message = std::string(name) + " has an unsupported value";
    return false;
}

bool read_sfm_mode(
    const json& value, int& output, std::string& message) {
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        if (text == "global") {
            output = 0;
            return true;
        }
        if (text == "hierarchical") {
            output = 2;
            return true;
        }
    } else if (value.is_number()) {
        const double number = value.get<double>();
        if (number == 0.0 || number == 2.0) {
            output = static_cast<int>(number);
            return true;
        }
    }
    message = "sfm_mode must be global, hierarchical, 0, or 2";
    return false;
}

json action_state(const bool available, const char* reason) {
    return json{
        {"available", available},
        {"reason", available ? json(nullptr) : json(reason)},
    };
}

}  // namespace

void normalize_settings(ProjectSettings& settings) {
    settings.sanitize();
    settings.iterations = std::max(settings.iterations, 1);
    settings.densification_cap =
        std::clamp(settings.densification_cap, 10'000, 50'000'000);
    settings.sh_degree = std::clamp(settings.sh_degree, 0, 3);
    settings.mask_mode = settings.mask_mode == 1 ? 1 : 0;
    settings.preview_interval = std::max(settings.preview_interval, 1);
    settings.max_features = std::max(settings.max_features, 512);
    settings.geometry_from_iter = std::max(settings.geometry_from_iter, 0);
    settings.video_fps = std::clamp(settings.video_fps, 0.05F, 60.F);
    settings.video_sharp_window = std::max(1, settings.video_sharp_window);
    settings.video_max_frames = std::max(0, settings.video_max_frames);
    settings.video_quality = std::clamp(settings.video_quality, -1, 100);
    settings.video_scale = std::clamp(settings.video_scale, 0.05F, 4.F);
    settings.video_rotate =
        std::clamp(settings.video_rotate / 90, 0, 3) * 90;
    settings.texture_quality = std::clamp(settings.texture_quality, 0, 2);
    settings.atlas_resolution =
        std::clamp(settings.atlas_resolution, 64, 8192);
    settings.max_resolution =
        std::clamp(settings.max_resolution, 256, 8192);
    settings.mesh_quality = std::clamp(settings.mesh_quality, 0, 2);
    settings.mesh_postprocess_quality =
        std::clamp(settings.mesh_postprocess_quality, 0, 2);
    settings.mesh_target_faces =
        std::max<std::uint64_t>(settings.mesh_target_faces, 1);
    settings.mesh_remesh_target_faces =
        std::max<std::uint64_t>(settings.mesh_remesh_target_faces, 1);
}

std::string state_revision_key(const App& app) {
    const json key{
        {"project", settings_json(app.settings)},
        {"has_sparse", app.has_sparse},
        {"has_asfm", app.has_asfm},
        {"has_mvs", app.has_mvs},
        {"has_model", app.has_model},
        {"has_mesh", app.has_mesh},
        {"has_texture", app.has_texture},
        {"dirty", app.project_results_dirty},
        {"job", {
            {"kind", static_cast<int>(app.active_job)},
            {"running", app.job.running()},
            {"paused", app.job.paused()},
            {"preparing", app.pending_train_prepare.valid()},
            {"stage", static_cast<int>(app.monitor.stage())},
            {"fraction", app.monitor.fraction()},
            {"iteration", app.monitor.training().iteration},
            {"loss", app.monitor.training().loss},
        }},
        {"view", {
            {"workspace", static_cast<int>(app.workspace)},
            {"mode", static_cast<int>(app.view_mode)},
            {"channel", static_cast<int>(app.display_channel)},
            {"yaw", app.camera.yaw},
            {"pitch", app.camera.pitch},
            {"distance", app.camera.distance},
            {"target", {app.camera.target.x, app.camera.target.y, app.camera.target.z}},
            {"fov_degrees", app.camera.fov_degrees},
            {"ortho_height", app.camera.ortho_height},
            {"projection", static_cast<int>(app.camera.projection)},
        }},
        {"message", app.message},
    };
    return key.dump();
}

json snapshot_job(const App& app, const ControlBus& bus) {
    const TrainingStats& training = app.monitor.training();
    const char* kind = job_kind_token(app.active_job);
    const char* stage = stage_token(app.monitor.stage());
    const float fraction = app.monitor.fraction();
    const double eta = app.monitor.eta_seconds();
    const double elapsed = app.monitor.elapsed_seconds();
    const std::string& last_error = app.monitor.last_error();
    json job = json::object();
    job["kind"] = kind;
    job["last_kind"] = job_kind_token(app.monitor.kind());
    job["running"] = app.job.running();
    job["paused"] = app.job.paused();
    job["preparing"] = app.pending_train_prepare.valid();
    job["finishing"] = !app.job.running() && !app.pending_train_prepare.valid() &&
        app.active_job != JobKind::none;
    if (app.train_preparation)
        job["prepare_stage"] =
            prepare_stage_name(app.train_preparation->stage.load());
    else
        job["prepare_stage"] = json(nullptr);
    job["stage"] = stage;
    job["fraction"] = fraction;
    job["eta_seconds"] = eta;
    job["elapsed_seconds"] = elapsed;
    job["op_id"] = bus.active_operation().empty()
        ? json(nullptr) : json(bus.active_operation());
    job["client"] = bus.active_client().empty()
        ? json(nullptr) : json(bus.active_client());
    job["last_error"] = last_error;
    json training_json = json::object();
    training_json["iteration"] = training.iteration;
    training_json["total_iterations"] = training.total_iterations;
    training_json["gaussians"] = training.gaussians;
    training_json["loss"] = training.loss;
    training_json["rgb_loss"] = training.rgb_loss;
    training_json["alpha_loss"] = training.alpha_loss;
    training_json["depth_loss"] = training.depth_loss;
    training_json["normal_loss"] = training.normal_loss;
    training_json["multi_view_geometry_loss"] = training.multi_view_geometry_loss;
    training_json["multi_view_ncc_loss"] = training.multi_view_ncc_loss;
    training_json["resolution_scale"] = training.resolution_scale;
    training_json["sh_degree"] = training.sh_degree;
    training_json["step_milliseconds"] = training.step_milliseconds;
    training_json["vram_used"] = training.vram_used;
    training_json["vram_total"] = training.vram_total;
    training_json["valid"] = training.valid;
    job["training"] = std::move(training_json);
    if (!app.job.running() && app.job.exit_code() >= 0)
        job["exit_code"] = app.job.exit_code();
    else
        job["exit_code"] = nullptr;
    return job;
}

json describe_schema() {
    json groups = json::array();

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "camera_model", "enum",
            "Camera model used by alignment: pinhole, opencv_fisheye, automatic, equirectangular.",
            0, 3, 2,
            json::array({"pinhole", "opencv_fisheye", "automatic", "equirectangular"})));
        fields.push_back(setting_spec(
            "sfm_mode", "enum", "SfM mode: global or hierarchical.",
            nullptr, nullptr, 0, json::array({"global", "hierarchical"})));
        fields.push_back(setting_spec(
            "max_features", "int", "Maximum features per image.",
            512, 1000000, 27000));
        fields.push_back(setting_spec(
            "reuse_cache", "bool", "Reuse an existing alignment cache when compatible.",
            nullptr, nullptr, false));
        groups.push_back({
            {"id", "alignment"},
            {"label", "Alignment"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "iterations", "int", "Total 3DGS training iterations.",
            1, 1000000, 30000));
        fields.push_back(setting_spec(
            "strategy", "enum", "Gaussian densification strategy.",
            nullptr, nullptr, 0,
            json::array({"adc_igs", "adc_plus", "dense_adaptive"})));
        fields.push_back(setting_spec(
            "training_mode", "enum", "Training quality preset.",
            nullptr, nullptr, 1, json::array({"fast", "standard", "high"})));
        fields.push_back(setting_spec(
            "sh_degree", "int", "Spherical-harmonics degree.", 0, 3, 3));
        fields.push_back(setting_spec(
            "densification_cap", "int", "Maximum Gaussian count.",
            10000, 50000000, 3000000));
        fields.push_back(setting_spec(
            "max_resolution", "int", "Maximum training image long edge.",
            256, 8192, 1920));
        fields.push_back(setting_spec(
            "progressive_resolution", "bool",
            "Enable progressive-resolution training.", nullptr, nullptr, true));
        fields.push_back(setting_spec(
            "preview_interval", "int", "Preview publication interval in iterations.",
            1, 10000, 50));
        fields.push_back(setting_spec(
            "geometry_from_iter", "int",
            "Iteration after which geometry supervision starts.",
            0, 1000000, 7000));
        fields.push_back(setting_spec(
            "normal_field", "bool", "Enable GaussianWrapping normal-field training.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "ai_priors", "bool", "Enable MoGe depth/normal priors.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "ai_depth_weight", "float", "AI depth prior weight.",
            0, 10, 0.01));
        fields.push_back(setting_spec(
            "ai_normal_weight", "float", "AI normal prior weight.",
            0, 10, 0.01));
        fields.push_back(setting_spec(
            "ppisp_layout", "int", "PPISP exposure/white-balance layout; 0 disables.",
            0, 2, 0));
        fields.push_back(setting_spec(
            "bilateral_grid", "bool", "Enable bilateral-grid colour correction.",
            nullptr, nullptr, false));
        groups.push_back({
            {"id", "training"},
            {"label", "Training"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "build_mesh", "bool", "Build a mesh after reconstruction.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "mesh_source", "enum", "Mesh source: gaussians or mvs.",
            nullptr, nullptr, 0, json::array({"gaussians", "mvs"})));
        fields.push_back(setting_spec(
            "mesh_method", "enum", "Mesh extraction method.",
            nullptr, nullptr, 0, json::array({"auto", "tsdf", "delaunay", "pam"})));
        fields.push_back(setting_spec(
            "mesh_quality", "enum", "Mesh extraction quality.",
            nullptr, nullptr, 1, json::array({"fast", "standard", "high"})));
        fields.push_back(setting_spec(
            "mesh_simplify", "bool", "Simplify the extracted mesh.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "mesh_target_faces", "int", "Simplified mesh target face count.",
            1, 100000000, 1000000));
        fields.push_back(setting_spec(
            "mesh_remesh", "bool", "Run Instant Meshes remeshing.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "mesh_remesh_target_faces", "int", "Remesh target face count.",
            1, 100000000, 1000000));
        groups.push_back({
            {"id", "mesh"},
            {"label", "Mesh"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "texture_quality", "enum", "Texture baking quality.",
            nullptr, nullptr, 1, json::array({"fast", "standard", "high"})));
        fields.push_back(setting_spec(
            "atlas_resolution", "int", "Texture atlas resolution.",
            64, 8192, 2048));
        fields.push_back(setting_spec(
            "texture_delight", "bool", "Remove baked-in illumination.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "texture_optimize", "bool", "Optimize texture projection.",
            nullptr, nullptr, true));
        fields.push_back(setting_spec(
            "texture_blend_linear", "bool", "Blend in scene-linear space.",
            nullptr, nullptr, false));
        groups.push_back({
            {"id", "texture"},
            {"label", "Texture"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "video_fps", "float", "Video frame extraction rate.",
            0.05, 60, 2.0));
        fields.push_back(setting_spec(
            "video_max_frames", "int", "Maximum extracted video frames; 0 means unlimited.",
            0, 1000000, 0));
        fields.push_back(setting_spec(
            "video_sharp_window", "int", "Sharpness selection window.",
            1, 101, 3));
        fields.push_back(setting_spec(
            "video_quality", "int", "JPEG quality for extracted frames.",
            -1, 100, 95));
        fields.push_back(setting_spec(
            "video_scale", "float", "Scale applied to extracted frames.",
            0.05, 4, 1.0));
        fields.push_back(setting_spec(
            "video_rotate", "int", "Clockwise rotation in degrees.",
            0, 270, 0));
        groups.push_back({
            {"id", "video"},
            {"label", "Video"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "sam_masks", "bool", "Generate SAM masks before reconstruction.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "sam_use_mask_for_sfm", "bool", "Mask SfM observations.",
            nullptr, nullptr, false));
        fields.push_back(setting_spec(
            "sam_text", "string", "Positive SAM prompt.", nullptr, nullptr, ""));
        fields.push_back(setting_spec(
            "sam_negative_text", "string", "Negative SAM prompt.",
            nullptr, nullptr, ""));
        fields.push_back(setting_spec(
            "sam_keep_prompted", "bool", "Keep prompted regions.",
            nullptr, nullptr, true));
        groups.push_back({
            {"id", "sam"},
            {"label", "SAM"},
            {"fields", std::move(fields)},
        });
    }

    {
        json fields = json::array();
        fields.push_back(setting_spec(
            "splat_format", "enum", "Gaussian export format.",
            nullptr, nullptr, 0,
            json::array({"auto", "ply", "sog", "spz", "glb", "usdc"})));
        groups.push_back({
            {"id", "export"},
            {"label", "Export"},
            {"fields", std::move(fields)},
        });
    }

    return success({
        {"schema_version", k_schema_version},
        {"groups", std::move(groups)},
    });
}

json apply_parameter_patch(App& app, const json& patch) {
    if (app.job.running() || app.pending_train_prepare.valid())
        return failure(
            "BUSY", "Cannot change reconstruction parameters while a job is running");
    if (!patch.is_object())
        return failure("INVALID_REQUEST", "patch must be a JSON object");

    const json current = settings_json(app.settings);
    json unknown = json::array();
    for (auto it = patch.begin(); it != patch.end(); ++it) {
        if (!current.contains(it.key())) unknown.push_back(it.key());
    }
    if (!unknown.empty())
        return failure(
            "INVALID_REQUEST", "Unknown parameter name",
            {{"unknown", std::move(unknown)}});

    ProjectSettings settings = app.settings;
    std::string message;
    json changed = json::array();
    const auto note = [&changed](const char* name) { changed.push_back(name); };
    const auto invalid = [](const std::string& text) {
        return failure("INVALID_REQUEST", text);
    };

    if (patch.contains("camera_model")) {
        int value{};
        if (!read_choice(
                patch["camera_model"], "camera_model",
                {"pinhole", "opencv_fisheye", "automatic", "equirectangular"},
                value, message))
            return invalid(message);
        settings.camera_model = value;
        note("camera_model");
    }
    if (patch.contains("sfm_mode")) {
        int value{};
        if (!read_sfm_mode(patch["sfm_mode"], value, message))
            return invalid(message);
        settings.sfm_mode = value;
        note("sfm_mode");
    }
    if (patch.contains("max_features")) {
        long long value{};
        if (!read_integer(
                patch["max_features"], "max_features", 512, 1000000,
                value, message))
            return invalid(message);
        settings.max_features = static_cast<int>(value);
        note("max_features");
    }
    if (patch.contains("reuse_cache")) {
        bool value{};
        if (!read_bool(patch["reuse_cache"], "reuse_cache", value, message))
            return invalid(message);
        settings.reuse_cache = value;
        note("reuse_cache");
    }
    if (patch.contains("iterations")) {
        long long value{};
        if (!read_integer(
                patch["iterations"], "iterations", 1, 1000000, value, message))
            return invalid(message);
        settings.iterations = static_cast<int>(value);
        note("iterations");
    }
    if (patch.contains("strategy")) {
        int value{};
        if (!read_choice(
                patch["strategy"], "strategy",
                {"adc_igs", "adc_plus", "dense_adaptive"}, value, message))
            return invalid(message);
        settings.strategy = value;
        note("strategy");
    }
    if (patch.contains("training_mode")) {
        int value{};
        if (!read_choice(
                patch["training_mode"], "training_mode",
                {"fast", "standard", "high"}, value, message))
            return invalid(message);
        settings.training_mode = value;
        note("training_mode");
    }
    if (patch.contains("sh_degree")) {
        long long value{};
        if (!read_integer(
                patch["sh_degree"], "sh_degree", 0, 3, value, message))
            return invalid(message);
        settings.sh_degree = static_cast<int>(value);
        note("sh_degree");
    }
    if (patch.contains("densification_cap")) {
        long long value{};
        if (!read_integer(
                patch["densification_cap"], "densification_cap",
                10000, 50000000, value, message))
            return invalid(message);
        settings.densification_cap = static_cast<int>(value);
        note("densification_cap");
    }
    if (patch.contains("max_resolution")) {
        long long value{};
        if (!read_integer(
                patch["max_resolution"], "max_resolution",
                256, 8192, value, message))
            return invalid(message);
        settings.max_resolution = static_cast<int>(value);
        note("max_resolution");
    }
    if (patch.contains("progressive_resolution")) {
        bool value{};
        if (!read_bool(
                patch["progressive_resolution"], "progressive_resolution",
                value, message))
            return invalid(message);
        settings.progressive_resolution = value;
        note("progressive_resolution");
    }
    if (patch.contains("preview_interval")) {
        long long value{};
        if (!read_integer(
                patch["preview_interval"], "preview_interval",
                1, 10000, value, message))
            return invalid(message);
        settings.preview_interval = static_cast<int>(value);
        note("preview_interval");
    }
    if (patch.contains("geometry_from_iter")) {
        long long value{};
        if (!read_integer(
                patch["geometry_from_iter"], "geometry_from_iter",
                0, 1000000, value, message))
            return invalid(message);
        settings.geometry_from_iter = static_cast<int>(value);
        note("geometry_from_iter");
    }
    if (patch.contains("normal_field")) {
        bool value{};
        if (!read_bool(patch["normal_field"], "normal_field", value, message))
            return invalid(message);
        settings.normal_field = value;
        note("normal_field");
    }
    if (patch.contains("ai_priors")) {
        bool value{};
        if (!read_bool(patch["ai_priors"], "ai_priors", value, message))
            return invalid(message);
        settings.ai_priors = value;
        note("ai_priors");
    }
    if (patch.contains("ai_prior_source")) {
        long long value{};
        if (!read_integer(
                patch["ai_prior_source"], "ai_prior_source",
                0, 1, value, message))
            return invalid(message);
        settings.ai_prior_source = static_cast<int>(value);
        note("ai_prior_source");
    }
    if (patch.contains("ai_depth_weight")) {
        float value{};
        if (!read_number(
                patch["ai_depth_weight"], "ai_depth_weight",
                0, 10, value, message))
            return invalid(message);
        settings.ai_depth_weight = value;
        note("ai_depth_weight");
    }
    if (patch.contains("ai_normal_weight")) {
        float value{};
        if (!read_number(
                patch["ai_normal_weight"], "ai_normal_weight",
                0, 10, value, message))
            return invalid(message);
        settings.ai_normal_weight = value;
        note("ai_normal_weight");
    }
    if (patch.contains("ai_prior_tokens")) {
        long long value{};
        if (!read_integer(
                patch["ai_prior_tokens"], "ai_prior_tokens",
                0, 10000, value, message))
            return invalid(message);
        settings.ai_prior_tokens = static_cast<int>(value);
        note("ai_prior_tokens");
    }
    if (patch.contains("ppisp_layout")) {
        long long value{};
        if (!read_integer(
                patch["ppisp_layout"], "ppisp_layout", 0, 2, value, message))
            return invalid(message);
        settings.ppisp_layout = static_cast<int>(value);
        note("ppisp_layout");
    }
    if (patch.contains("bilateral_grid")) {
        bool value{};
        if (!read_bool(patch["bilateral_grid"], "bilateral_grid", value, message))
            return invalid(message);
        settings.bilateral_grid = value;
        note("bilateral_grid");
    }
    if (patch.contains("build_mesh")) {
        bool value{};
        if (!read_bool(patch["build_mesh"], "build_mesh", value, message))
            return invalid(message);
        settings.build_mesh = value;
        note("build_mesh");
    }
    if (patch.contains("mesh_source")) {
        int value{};
        if (!read_choice(
                patch["mesh_source"], "mesh_source",
                {"gaussians", "mvs"}, value, message))
            return invalid(message);
        settings.mesh_source = value;
        note("mesh_source");
    }
    if (patch.contains("mesh_method")) {
        int value{};
        if (!read_choice(
                patch["mesh_method"], "mesh_method",
                {"auto", "tsdf", "delaunay", "pam"}, value, message))
            return invalid(message);
        settings.mesh_method = value;
        note("mesh_method");
    }
    if (patch.contains("mesh_quality")) {
        int value{};
        if (!read_choice(
                patch["mesh_quality"], "mesh_quality",
                {"fast", "standard", "high"}, value, message))
            return invalid(message);
        settings.mesh_quality = value;
        note("mesh_quality");
    }
    if (patch.contains("mesh_postprocess_quality")) {
        int value{};
        if (!read_choice(
                patch["mesh_postprocess_quality"], "mesh_postprocess_quality",
                {"fast", "standard", "high"}, value, message))
            return invalid(message);
        settings.mesh_postprocess_quality = value;
        note("mesh_postprocess_quality");
    }
    if (patch.contains("mesh_simplify")) {
        bool value{};
        if (!read_bool(patch["mesh_simplify"], "mesh_simplify", value, message))
            return invalid(message);
        settings.mesh_simplify = value;
        note("mesh_simplify");
    }
    if (patch.contains("mesh_target_faces")) {
        long long value{};
        if (!read_integer(
                patch["mesh_target_faces"], "mesh_target_faces",
                1, 100000000, value, message))
            return invalid(message);
        settings.mesh_target_faces = static_cast<std::uint64_t>(value);
        note("mesh_target_faces");
    }
    if (patch.contains("mesh_remesh")) {
        bool value{};
        if (!read_bool(patch["mesh_remesh"], "mesh_remesh", value, message))
            return invalid(message);
        settings.mesh_remesh = value;
        note("mesh_remesh");
    }
    if (patch.contains("mesh_remesh_target_faces")) {
        long long value{};
        if (!read_integer(
                patch["mesh_remesh_target_faces"], "mesh_remesh_target_faces",
                1, 100000000, value, message))
            return invalid(message);
        settings.mesh_remesh_target_faces =
            static_cast<std::uint64_t>(value);
        note("mesh_remesh_target_faces");
    }
    if (patch.contains("mesh_remesh_smooth_iterations")) {
        long long value{};
        if (!read_integer(
                patch["mesh_remesh_smooth_iterations"],
                "mesh_remesh_smooth_iterations", 0, 100, value, message))
            return invalid(message);
        settings.mesh_remesh_smooth_iterations = static_cast<int>(value);
        note("mesh_remesh_smooth_iterations");
    }
    if (patch.contains("texture_quality")) {
        long long value{};
        if (!read_integer(
                patch["texture_quality"], "texture_quality",
                0, 2, value, message))
            return invalid(message);
        settings.texture_quality = static_cast<int>(value);
        note("texture_quality");
    }
    if (patch.contains("atlas_resolution")) {
        long long value{};
        if (!read_integer(
                patch["atlas_resolution"], "atlas_resolution",
                64, 8192, value, message))
            return invalid(message);
        settings.atlas_resolution = static_cast<int>(value);
        note("atlas_resolution");
    }
    if (patch.contains("texture_delight")) {
        bool value{};
        if (!read_bool(
                patch["texture_delight"], "texture_delight", value, message))
            return invalid(message);
        settings.texture_delight = value;
        note("texture_delight");
    }
    if (patch.contains("texture_optimize")) {
        bool value{};
        if (!read_bool(
                patch["texture_optimize"], "texture_optimize", value, message))
            return invalid(message);
        settings.texture_optimize = value;
        note("texture_optimize");
    }
    if (patch.contains("texture_photometric_steps")) {
        long long value{};
        if (!read_integer(
                patch["texture_photometric_steps"], "texture_photometric_steps",
                0, 100000, value, message))
            return invalid(message);
        settings.texture_photometric_steps = static_cast<int>(value);
        note("texture_photometric_steps");
    }
    if (patch.contains("texture_seam_steps")) {
        long long value{};
        if (!read_integer(
                patch["texture_seam_steps"], "texture_seam_steps",
                0, 10000, value, message))
            return invalid(message);
        settings.texture_seam_steps = static_cast<int>(value);
        note("texture_seam_steps");
    }
    if (patch.contains("texture_blend_linear")) {
        bool value{};
        if (!read_bool(
                patch["texture_blend_linear"], "texture_blend_linear",
                value, message))
            return invalid(message);
        settings.texture_blend_linear = value;
        note("texture_blend_linear");
    }
    if (patch.contains("video_fps")) {
        float value{};
        if (!read_number(
                patch["video_fps"], "video_fps", 0.05, 60, value, message))
            return invalid(message);
        settings.video_fps = value;
        note("video_fps");
    }
    if (patch.contains("video_max_frames")) {
        long long value{};
        if (!read_integer(
                patch["video_max_frames"], "video_max_frames",
                0, 1000000, value, message))
            return invalid(message);
        settings.video_max_frames = static_cast<int>(value);
        note("video_max_frames");
    }
    if (patch.contains("video_sharp_window")) {
        long long value{};
        if (!read_integer(
                patch["video_sharp_window"], "video_sharp_window",
                1, 101, value, message))
            return invalid(message);
        settings.video_sharp_window = static_cast<int>(value);
        note("video_sharp_window");
    }
    if (patch.contains("video_quality")) {
        long long value{};
        if (!read_integer(
                patch["video_quality"], "video_quality",
                -1, 100, value, message))
            return invalid(message);
        settings.video_quality = static_cast<int>(value);
        note("video_quality");
    }
    if (patch.contains("video_scale")) {
        float value{};
        if (!read_number(
                patch["video_scale"], "video_scale", 0.05, 4, value, message))
            return invalid(message);
        settings.video_scale = value;
        note("video_scale");
    }
    if (patch.contains("video_rotate")) {
        long long value{};
        if (!read_integer(
                patch["video_rotate"], "video_rotate", 0, 270, value, message))
            return invalid(message);
        settings.video_rotate = static_cast<int>(value / 90 * 90);
        note("video_rotate");
    }
    if (patch.contains("sam_masks")) {
        bool value{};
        if (!read_bool(patch["sam_masks"], "sam_masks", value, message))
            return invalid(message);
        settings.sam_masks = value;
        note("sam_masks");
    }
    if (patch.contains("sam_use_mask_for_sfm")) {
        bool value{};
        if (!read_bool(
                patch["sam_use_mask_for_sfm"], "sam_use_mask_for_sfm",
                value, message))
            return invalid(message);
        settings.sam_use_mask_for_sfm = value;
        note("sam_use_mask_for_sfm");
    }
    if (patch.contains("sam_text")) {
        std::string value;
        if (!read_string(
                patch["sam_text"], "sam_text", 511, value, message))
            return invalid(message);
        std::snprintf(
            settings.sam_text.data(), settings.sam_text.size(), "%s",
            value.c_str());
        note("sam_text");
    }
    if (patch.contains("sam_negative_text")) {
        std::string value;
        if (!read_string(
                patch["sam_negative_text"], "sam_negative_text",
                511, value, message))
            return invalid(message);
        std::snprintf(
            settings.sam_negative_text.data(),
            settings.sam_negative_text.size(), "%s", value.c_str());
        note("sam_negative_text");
    }
    if (patch.contains("sam_keep_prompted")) {
        bool value{};
        if (!read_bool(
                patch["sam_keep_prompted"], "sam_keep_prompted",
                value, message))
            return invalid(message);
        settings.sam_keep_prompted = value;
        note("sam_keep_prompted");
    }
    if (patch.contains("sam_video")) {
        bool value{};
        if (!read_bool(patch["sam_video"], "sam_video", value, message))
            return invalid(message);
        settings.sam_video = value;
        note("sam_video");
    }
    if (patch.contains("sam_max_size")) {
        long long value{};
        if (!read_integer(
                patch["sam_max_size"], "sam_max_size",
                0, 8192, value, message))
            return invalid(message);
        settings.sam_max_size = static_cast<int>(value);
        note("sam_max_size");
    }
    if (patch.contains("splat_format")) {
        int value{};
        if (!read_choice(
                patch["splat_format"], "splat_format",
                {"auto", "ply", "sog", "spz", "glb", "usdc"}, value, message))
            return invalid(message);
        settings.splat_format = value;
        note("splat_format");
    }

    normalize_settings(settings);
    app.settings = settings;
    const json normalized_all = settings_json(settings);
    json normalized = json::object();
    for (const auto& name : changed) {
        const std::string key = name.get<std::string>();
        normalized[key] = normalized_all[key];
    }
    return success({
        {"changed", std::move(changed)},
        {"normalized", std::move(normalized)},
        {"settings", normalized_all},
    });
}

json snapshot_state(const App& app, const ControlBus& bus) {
    const std::filesystem::path project =
        app.layout.project_file.empty()
            ? path_from_utf8_field(app.settings.project_dir.data())
            : app.layout.project_file;
    const std::string project_text = path_text(project);
    std::string project_name = path_text(project.stem());
    if (project_name.empty())
        project_name = path_text(
            path_from_utf8_field(app.settings.images_dir.data()).filename());

    const bool external = has_external_dataset(app);
    const bool alignment =
        alignment_ready(app) || app.has_sparse || app.has_asfm;
    const bool can_export =
        can_export_model(app) || can_export_mesh_file(app) ||
        can_export_alignment(app) || can_export_sfm(app) ||
        can_export_sparse(app);
    const bool dirty = has_unsaved_project_changes(app);
    const json job = snapshot_job(app, bus);
    const json settings = settings_json(app.settings);
    const std::uint32_t gaussian_count = app.splat_edit.gaussian_count();
    const std::uint32_t selected_count = app.splat_edit.selected_count();

    json result = success({
        {"schema_version", k_schema_version},
        {"session_id", bus.session_id()},
        {"revision", bus.revision()},
        {"version", PHOTARA_VERSION},
        {"project", {
            {"path", project_text.empty() ? json(nullptr) : json(project_text)},
            {"name", project_name},
            {"dirty", dirty},
            {"images", field_text(app.settings.images_dir).empty()
                 ? json(nullptr)
                 : json(field_text(app.settings.images_dir))},
            {"dataset_source", field_text(app.settings.dataset_source).empty()
                 ? json(nullptr)
                 : json(field_text(app.settings.dataset_source))},
            {"root", app.layout.root.empty()
                 ? json(nullptr) : json(path_text(app.layout.root))},
            {"cache", app.layout.cache.empty()
                 ? json(nullptr) : json(path_text(app.layout.cache))},
        }},
        {"capabilities", {
            {"vulkan_training", studio_vulkan_training()},
            {"ba_cuda", studio_ba_cuda()},
            {"sam", studio_has_sam()},
            {"sam_cuda", studio_sam_cuda()},
            {"sam_vulkan", studio_sam_vulkan()},
#if defined(PHOTARA_HAS_TEXTURE)
            {"texture", true},
#else
            {"texture", false},
#endif
#if defined(PHOTARA_HAS_ONNXRUNTIME)
            {"onnx", true},
#else
            {"onnx", false},
#endif
        }},
        {"artifacts", {
            {"sparse", app.has_sparse},
            {"asfm", app.has_asfm},
            {"mvs", app.has_mvs},
            {"splat", app.has_model},
            {"mesh", app.has_mesh},
            {"texture", app.has_texture},
        }},
        {"job", job},
        {"actions", {
            {"align", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    !external &&
                    (app.settings.images_dir[0] != '\0' ||
                     app.settings.project_dir[0] != '\0'),
                app.job.running() ? "job_running" :
                app.pending_train_prepare.valid() ? "preparing" :
                external ? "external_dataset" : "no_source")},
            {"train", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    (app.settings.images_dir[0] != '\0' ||
                     app.settings.project_dir[0] != '\0' ||
                     app.settings.dataset_source[0] != '\0'),
                app.job.running() ? "job_running" :
                app.pending_train_prepare.valid() ? "preparing" :
                "no_source")},
            {"resume_training", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    app.training_checkpoint_info.has_value(),
                app.job.running() ? "job_running" :
                app.pending_train_prepare.valid() ? "preparing" :
                "no_checkpoint")},
            {"dense", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    alignment,
                app.job.running() ? "job_running" :
                app.pending_train_prepare.valid() ? "preparing" :
                alignment ? "available" : "no_alignment")},
            {"texture", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    app.has_mesh && alignment,
                app.job.running() ? "job_running" :
                app.pending_train_prepare.valid() ? "preparing" :
                !app.has_mesh ? "no_mesh" : "no_alignment")},
            {"export_sfm", action_state(
                !app.job.running() && !app.pending_train_prepare.valid() &&
                    can_export_sfm(app), "no_alignment_or_busy")},
            {"export", action_state(can_export, "no_exportable_result")},
            {"pause", action_state(app.job.running() && !app.job.paused(),
                                   "job_not_running")},
            {"resume", action_state(app.job.running() && app.job.paused(),
                                    "job_not_paused")},
            {"stop", action_state(app.job.running(), "job_not_running")},
        }},
        {"view", {
            {"workspace", workspace_name(app.workspace)},
            {"mode", mode_name(app.view_mode)},
            {"channel", channel_name(app.display_channel)},
            {"camera", {
                {"target", {app.camera.target.x, app.camera.target.y,
                            app.camera.target.z}},
                {"yaw", app.camera.yaw},
                {"pitch", app.camera.pitch},
                {"distance", app.camera.distance},
                {"fov_degrees", app.camera.fov_degrees},
                {"ortho_height", app.camera.ortho_height},
                {"projection", static_cast<int>(app.camera.projection)},
            }},
        }},
        {"selection", {
            {"gaussians", gaussian_count},
            {"selected", selected_count},
        }},
        {"settings", settings},
        {"message", {
            {"text", app.message},
        }},
    });
    return result;
}

}  // namespace editor::control
