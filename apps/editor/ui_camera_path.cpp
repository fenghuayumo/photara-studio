#include "ui.hpp"
#include "camera_timeline.hpp"
#include "camera_capture_path.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "shortcuts.hpp"
#include "io/image.hpp"
#include "io/video_frames.hpp"
#include "file_dialogs.hpp"
#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <thread>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace editor {
namespace {
using Key = photara::project::CameraKeyframe;
using i18n::tr;

Key capture(const OrbitCamera& c, float time) {
    Key k; k.time = time;
    k.target[0] = c.target.x; k.target[1] = c.target.y; k.target[2] = c.target.z;
    k.yaw = c.yaw; k.pitch = c.pitch; k.roll = c.roll; k.distance = c.distance;
    k.fov_degrees = c.fov_degrees; k.ortho_height = c.ortho_height;
    k.fisheye_k1 = c.fisheye_k1; k.projection = static_cast<int>(c.projection);
    return k;
}
bool same_pose(const OrbitCamera& c, const Key& k) {
    return c.target.x == k.target[0] && c.target.y == k.target[1] && c.target.z == k.target[2] &&
        c.roll == k.roll && c.yaw == k.yaw && c.pitch == k.pitch && c.distance == k.distance &&
        c.fov_degrees == k.fov_degrees && c.ortho_height == k.ortho_height &&
        c.fisheye_k1 == k.fisheye_k1 && static_cast<int>(c.projection) == k.projection;
}
void apply(App& app, const Key& k) {
    auto& c = app.camera;
    c.target = {k.target[0], k.target[1], k.target[2]};
    c.yaw = k.yaw; c.pitch = k.pitch; c.roll = k.roll; c.distance = k.distance;
    c.fov_degrees = k.fov_degrees; c.ortho_height = k.ortho_height;
    c.fisheye_k1 = k.fisheye_k1; c.projection = static_cast<EditorProjection>(k.projection);
    c.interacting = false; app.preview_follow_view = false;
}
Key evaluate(const std::vector<Key>& keys, float t) {
    return camera_timeline::evaluate(keys, t);
}
void seek(App& app) {
    if (!app.camera_keyframes.empty()) apply(app, evaluate(app.camera_keyframes, app.camera_path_time));
}
void select_key(App& app, int i) {
    app.camera_key_selected = i;
    app.camera_path_time = app.camera_keyframes[static_cast<std::size_t>(i)].time;
    app.camera_timeline_view.reveal(camera_timeline::span(app.camera_keyframes.back().time),app.camera_path_time);
    app.camera_path_playing = false;
    apply(app, app.camera_keyframes[static_cast<std::size_t>(i)]);
}
void sync_thumbnails(App& app) {
    auto& cache = app.camera_thumbnails;
    while (cache.size() > app.camera_keyframes.size()) {
        cache.back().texture.reset(); cache.pop_back();
    }
    cache.resize(app.camera_keyframes.size());
    for (std::size_t i = 0; i < cache.size(); ++i) {
        const auto& k = app.camera_keyframes[i]; auto& item = cache[i];
        if (item.pixels == k.thumbnail_rgb.data() && item.size == k.thumbnail_rgb.size()) continue;
        item.texture.reset(); item.pixels = k.thumbnail_rgb.data(); item.size = k.thumbnail_rgb.size();
        if (!k.thumbnail_rgb.empty()) {
            try { item.texture.upload({k.thumbnail_width, k.thumbnail_height, k.thumbnail_rgb}); }
            catch (...) { item.texture.reset(); }
        }
    }
}

constexpr float k_record_max_seconds = 600.F;
constexpr std::size_t k_max_path_keys = 10000;

float recording_elapsed(const App::CameraPathRecording& r) {
    const auto now = r.paused ? r.paused_at : std::chrono::steady_clock::now();
    return std::clamp(std::chrono::duration<float>(now-r.started).count()-r.paused_seconds, 0.F, k_record_max_seconds);
}
void start_path_recording(App& app) {
    auto& r = app.camera_path_recording;
    r.active = true; r.paused = false; r.review = false; r.preview_playing = false;
    r.started = std::chrono::steady_clock::now(); r.paused_seconds = 0.F;
    r.result.clear();
    r.from_capture=false;
    r.take.begin(capture(app.camera, 0.F), camera_recording::quality_for(r.quality, app.camera.distance), r.preserve_pauses);
    app.camera_path_playing = false; app.camera_timeline_drag = -1;
    app.preview_follow_view = false;
}
void finish_path_recording(App& app) {
    auto& r = app.camera_path_recording;
    if (!r.active) return;
    r.result = r.take.finish(r.paused ? r.take.anchor : capture(app.camera, 0.F), recording_elapsed(r));
    r.active = false; r.paused = false;
    r.preview_camera = app.camera;
    r.review = !r.result.empty();
    if (!r.review) set_message(app, tr("No camera movement detected. No path was created."), theme::text_muted);
}
void discard_path_recording(App& app) {
    auto& r = app.camera_path_recording;
    if (r.review) apply(app, capture(r.preview_camera, 0.F));
    r.active = r.review = r.paused = r.preview_playing = false;
    r.result.clear(); r.take.samples.clear();
}
void apply_path_recording(App& app) {
    auto& r = app.camera_path_recording;
    const float start = r.replace_path || app.camera_keyframes.empty() ? 0.F : app.camera_keyframes.back().time+2.F;
    if (r.result.empty() || start+r.result.back().time > 36000.F ||
        (r.replace_path ? 0 : app.camera_keyframes.size())+r.result.size() > k_max_path_keys) {
        set_message(app, tr("The recording exceeds the camera path limit."), theme::warning); return;
    }
    if (r.replace_path) app.camera_keyframes.clear();
    const int first = static_cast<int>(app.camera_keyframes.size());
    for (auto key : r.result) { key.time += start; app.camera_keyframes.push_back(std::move(key)); }
    r.review = r.preview_playing = false; r.result.clear(); r.take.samples.clear();
    sync_thumbnails(app); select_key(app, first);
    app.camera_thumbnail_pending = first;
    set_message(app, tr("Camera path added."), theme::accent);
}
void update_path_recording(App& app) {
    auto& r = app.camera_path_recording;
    if (r.active && !r.paused) {
        r.take.observe(capture(app.camera, 0.F), recording_elapsed(r));
        if (recording_elapsed(r) >= k_record_max_seconds || r.take.samples.size() >= k_max_path_keys-3)
            finish_path_recording(app);
    }
    if (r.review && r.preview_playing) {
        const float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now()-r.preview_started).count();
        apply(app, evaluate(r.result, elapsed));
        if (elapsed >= r.result.back().time) r.preview_playing = false;
    }
}

// Recording belongs to the Camera Path dock, never the rendered viewport.
void draw_recording_controls(App& app) {
    auto& r = app.camera_path_recording;
    const float width = ImGui::GetContentRegionAvail().x;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float action_w = std::max(106.F, ImGui::CalcTextSize(tr(r.review ? "Preview" : "Resume")).x+40.F);
    const float cancel_w = std::max(80.F, ImGui::CalcTextSize(tr(r.review ? "Discard" : "Cancel")).x+24.F);
    const float finish_w = std::max(116.F, ImGui::CalcTextSize(tr(r.review ? "Use path" : "Finish")).x+28.F);
    const float actions_width = action_w+cancel_w+finish_w+16.F;
    const bool single_row = width >= actions_width+300.F;
    const float seconds = r.review ? r.result.back().time : recording_elapsed(r);
    char clock[32];
    std::snprintf(clock,sizeof(clock),"%02d:%05.2f",static_cast<int>(seconds)/60,std::fmod(seconds,60.F));
    ImGui::SetCursorScreenPos({origin.x,origin.y+7.F});
    ImGui::GetWindowDrawList()->AddCircleFilled({origin.x+5.F,origin.y+15.F},4.F,
        theme::u32(r.review ? theme::accent : r.paused ? theme::warning : theme::danger));
    ImGui::Dummy({14.F,16.F}); ImGui::SameLine(0.F,6.F);
    ImGui::TextUnformatted(tr(r.review ? (r.from_capture ? "Capture path ready" : "Your flight is ready") : r.paused ? "Recording paused" : "Flight recording"));
    ImGui::SameLine(0.F,16.F); ImGui::TextColored(theme::text_bright,"%s",clock);
    ImGui::SetCursorScreenPos({single_row ? origin.x+width-actions_width : origin.x,
                              single_row ? origin.y : origin.y+35.F});
    if (r.review) {
        if (icons::labeled_button("##preview_take",r.preview_playing ? icons::Icon::pause : icons::Icon::play,
                tr(r.preview_playing ? "Pause" : "Preview"),{action_w,30.F})) {
            r.preview_playing=!r.preview_playing; r.preview_started=std::chrono::steady_clock::now();
        }
        ImGui::SameLine(0.F,8.F);
        if (ImGui::Button(tr("Discard"),{cancel_w,30.F})) discard_path_recording(app);
        ImGui::SameLine(0.F,8.F);
        if (theme::primary_button(tr("Use path"),{finish_w,30.F})) apply_path_recording(app);
    } else {
        if (icons::labeled_button("##pause_take",r.paused ? icons::Icon::play : icons::Icon::pause,
                tr(r.paused ? "Resume" : "Pause"),{action_w,30.F})) {
            if (!r.paused) { r.paused_at=std::chrono::steady_clock::now(); r.paused=true; }
            else { r.paused_seconds+=std::chrono::duration<float>(std::chrono::steady_clock::now()-r.paused_at).count(); r.paused=false; }
        }
        ImGui::SameLine(0.F,8.F);
        if (ImGui::Button(tr("Cancel"),{cancel_w,30.F})) discard_path_recording(app);
        ImGui::SameLine(0.F,8.F);
        if (theme::primary_button(tr("Finish"),{finish_w,30.F})) finish_path_recording(app);
    }
    ImGui::SetCursorScreenPos({origin.x,origin.y+(single_row ? 40.F : 75.F)});
    ImGui::Separator(); ImGui::Spacing();
    if (r.review) {
        if (r.from_capture) ImGui::Text(tr("%zu keyframes from %zu capture cameras"),r.result.size(),r.source_count);
        else ImGui::Text(tr("%zu keyframes selected from %zu motion samples"),r.result.size(),r.take.samples.size());
        ImGui::Checkbox(tr("Replace current path"),&r.replace_path);
        if (r.replace_path && !app.camera_keyframes.empty()) {
            ImGui::SameLine(0.F,12.F);
            ImGui::TextColored(theme::warning,"%s",tr("Replaces the current path when applied."));
        }
    } else if (r.active) {
        ImGui::TextWrapped("%s",tr(r.preserve_pauses ? "Stillness keeps its timing. Preparation is trimmed." : "Idle time is trimmed. Move to build the path."));
        ImGui::TextDisabled(tr("%zu motion samples"),r.take.samples.size());
    }
}

void draw_recording_setup(App& app) {
    auto& r = app.camera_path_recording;
    if (r.show_setup) { ImGui::OpenPopup("##flight_setup"); r.show_setup=false; }
    const auto* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({vp->WorkPos.x+vp->WorkSize.x*.5F,vp->WorkPos.y+vp->WorkSize.y*.5F},ImGuiCond_Appearing,{.5F,.5F});
    ImGui::SetNextWindowSize({440.F,0.F});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{22.F,20.F});
    const bool visible = ImGui::BeginPopupModal("##flight_setup",nullptr,ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    if (!visible) return;
    icons::inline_icon(icons::Icon::camera,theme::u32(theme::accent),20.F); ImGui::SameLine(0.F,10.F);
    ImGui::TextUnformatted(tr("Fly a new path"));
    ImGui::Spacing(); ImGui::TextWrapped("%s",tr("Explore the scene. We choose keyframes that preserve your camera motion."));
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (ImGui::BeginTable("##flight_options",2,ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Labels",ImGuiTableColumnFlags_WidthFixed,98.F);
        ImGui::TableSetupColumn("Values",ImGuiTableColumnFlags_WidthStretch);
        const auto row=[](const char* label) { ImGui::TableNextColumn(); ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s",tr(label)); ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1.F); };
        row("Path detail"); const char* detail[]{tr("Precise"),tr("Balanced"),tr("Compact")};
        ImGui::Combo("##detail",&r.quality,detail,3);
        row("Idle time"); int pause=r.preserve_pauses ? 1 : 0;
        const char* pauses[]{tr("Trim pauses"),tr("Keep pauses")};
        if (ImGui::Combo("##idle",&pause,pauses,2)) r.preserve_pauses=pause==1;
        row("Destination"); int destination=r.replace_path ? 1 : 0;
        const char* destinations[]{tr("Append to path"),tr("Replace current path")};
        if (ImGui::Combo("##destination",&destination,destinations,2)) r.replace_path=destination==1;
        ImGui::EndTable();
    }
    ImGui::Spacing(); ImGui::TextWrapped("%s",tr(camera_navigation_hint(app.camera)));
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextDisabled("%s",tr("Nothing changes until you preview and use the recording."));
    ImGui::PopTextWrapPos();
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (ImGui::Button(tr("Cancel"),{110.F,32.F})) ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (theme::primary_button(tr("Start flight"),{272.F,32.F})) { start_path_recording(app); ImGui::CloseCurrentPopup(); }
    ImGui::EndPopup();
}

void draw_capture_path_setup(App& app) {
    auto& options=app.camera_capture_path;
    if (options.show_setup) { ImGui::OpenPopup("##capture_path_setup"); options.show_setup=false; options.first=1; options.last=0; }
    const auto* vp=ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({vp->WorkPos.x+vp->WorkSize.x*.5F,vp->WorkPos.y+vp->WorkSize.y*.5F},ImGuiCond_Appearing,{.5F,.5F});
    ImGui::SetNextWindowSize({470.F,0.F});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,{22.F,20.F});
    const bool open=ImGui::BeginPopupModal("##capture_path_setup",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    if (!open) return;
    auto indices=camera_capture_path::ordered(app.scene.views,options.order==0,options.reverse);
    const int count=static_cast<int>(indices.size());
    if (options.last==0) {
        options.last=count;
        options.duration=std::clamp(std::max(options.duration,float(std::max(1,count)-1)*.02F),.1F,600.F);
    }
    options.first=std::clamp(options.first,1,std::max(1,count));
    options.last=std::clamp(options.last,options.first,std::max(1,count));
    icons::inline_icon(icons::Icon::camera,theme::u32(theme::accent),20.F); ImGui::SameLine(0.F,10.F);
    ImGui::TextUnformatted(tr("From capture cameras"));
    ImGui::Spacing(); ImGui::TextWrapped("%s",tr("Build a path from the cameras used for training or reconstruction."));
    ImGui::TextDisabled(tr("%zu usable capture cameras"),indices.size());
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (ImGui::BeginTable("##capture_options",2,ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Labels",ImGuiTableColumnFlags_WidthFixed,112.F);
        ImGui::TableSetupColumn("Values",ImGuiTableColumnFlags_WidthStretch);
        const auto row=[](const char* label){ImGui::TableNextColumn();ImGui::AlignTextToFramePadding();ImGui::TextDisabled("%s",tr(label));ImGui::TableNextColumn();ImGui::SetNextItemWidth(-1.F);};
        row("Camera order"); const char* orders[]{tr("File name (natural)"),tr("Dataset order")};
        ImGui::Combo("##capture_order",&options.order,orders,2);
        row("Direction"); ImGui::Checkbox(tr("Reverse order"),&options.reverse);
        row("Camera range"); ImGui::PushMultiItemsWidths(2,ImGui::GetContentRegionAvail().x);
        ImGui::DragInt("##first_camera",&options.first,1.F,1,std::max(1,count),"%d",ImGuiSliderFlags_AlwaysClamp); ImGui::PopItemWidth(); ImGui::SameLine();
        ImGui::DragInt("##last_camera",&options.last,1.F,options.first,std::max(1,count),"%d",ImGuiSliderFlags_AlwaysClamp); ImGui::PopItemWidth();
        row("Duration (s)"); ImGui::DragFloat("##capture_duration",&options.duration,.1F,.1F,600.F,"%.1f",ImGuiSliderFlags_AlwaysClamp);
        row("Keyframe selection"); ImGui::Checkbox(tr("Automatic reduction"),&options.automatic);
        if (options.automatic) { row("Path detail"); const char* qualities[]{tr("Precise"),tr("Balanced"),tr("Compact")}; ImGui::Combo("##capture_quality",&options.quality,qualities,3); }
        else { row("Camera interval"); ImGui::DragInt("##capture_step",&options.step,1.F,1,std::max(1,count),"%d",ImGuiSliderFlags_AlwaysClamp); }
        row("Destination"); int destination=app.camera_path_recording.replace_path ? 1 : 0;
        const char* destinations[]{tr("Append to path"),tr("Replace current path")};
        if (ImGui::Combo("##capture_destination",&destination,destinations,2)) app.camera_path_recording.replace_path=destination==1;
        ImGui::EndTable();
    }
    // Re-evaluate after changing order or range in this frame.
    indices=camera_capture_path::ordered(app.scene.views,options.order==0,options.reverse);
    options.first=std::clamp(options.first,1,std::max(1,count));
    options.last=std::clamp(options.last,options.first,std::max(1,count));
    const int selected=options.last-options.first+1;
    if (count>0) {
        ImGui::Spacing(); ImGui::TextDisabled("%s",app.scene.views[indices[options.first-1]].name.c_str());
        ImGui::TextDisabled("%s",app.scene.views[indices[options.last-1]].name.c_str());
    }
    ImGui::Spacing(); ImGui::TextWrapped("%s",tr("Capture timestamps are unavailable. Cameras are spaced evenly over the chosen duration."));
    const int needed=options.automatic ? selected : (selected-1)/std::max(1,options.step)+1+((selected-1)%std::max(1,options.step)!=0);
    const bool valid=count>=2 && selected>=2 && needed<=10000 && options.duration/float(selected-1)>=.01F;
    if (!valid) ImGui::TextWrapped("%s",tr("Choose at least two cameras, at most 10000 samples, and at least 0.01 seconds per source camera."));
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    if (ImGui::Button(tr("Cancel"),{110.F,32.F})) ImGui::CloseCurrentPopup();
    ImGui::SameLine(); ImGui::BeginDisabled(!valid);
    if (theme::primary_button(tr("Generate preview"),{302.F,32.F})) {
        const float distance=std::max(.01F,app.scene.radius);
        auto samples=camera_capture_path::generate<Key>(app.scene.views,indices,
            static_cast<std::size_t>(options.first-1),static_cast<std::size_t>(options.last-1),options.duration,
            options.automatic ? 1 : static_cast<std::size_t>(std::max(1,options.step)),distance,
            [](const auto& view){return view.model==photara::CameraModel::equirectangular ? 3 : view.model==photara::CameraModel::opencv_fisheye ? 2 : 0;});
        auto& result=app.camera_path_recording;
        result.result=options.automatic ? camera_recording::reduce(samples,camera_recording::quality_for(options.quality,distance)) : std::move(samples);
        result.active=false; result.review=true; result.from_capture=true; result.source_count=selected;
        result.preview_playing=false; result.preview_camera=app.camera; result.take.samples.clear();
        app.camera_path_playing=false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled(); ImGui::EndPopup();
}

void format_time(float seconds, char* out, std::size_t size) {
    const int centiseconds = static_cast<int>(std::lround(std::max(0.F, seconds) * 100.F));
    const int minutes = centiseconds / 6000;
    std::snprintf(out, size, "%02d:%02d.%02d", minutes, (centiseconds / 100) % 60, centiseconds % 100);
}

void add_camera_view(App& app) {
    auto& keys = app.camera_keyframes;
    const float t = keys.empty() ? 0.F : std::min(36000.F, std::max(app.camera_path_time, keys.back().time + 2.F));
    if (!keys.empty() && t <= keys.back().time) return;
    keys.push_back(capture(app.camera, t));
    app.camera_key_selected = static_cast<int>(keys.size()) - 1;
    app.camera_path_time = t;
    app.camera_timeline_view.reveal(camera_timeline::span(keys.back().time),t);
    app.camera_path_playing = false;
    app.camera_thumbnail_pending = app.camera_key_selected;
    sync_thumbnails(app);
}

std::string process_quote(const std::filesystem::path& path) {
    const auto input = path_to_utf8(path);
#ifdef _WIN32
    // CreateProcess parses Windows argv using backslash-before-quote rules.
    std::string out = "\"";
    std::size_t slashes = 0;
    for (const char c : input) {
        if (c == '\\') { ++slashes; continue; }
        if (c == '"') { out.append(slashes * 2 + 1, '\\'); out += c; }
        else { out.append(slashes, '\\'); out += c; }
        slashes = 0;
    }
    out.append(slashes * 2, '\\');
    return out + '"';
#else
    std::string out = "'";
    for (const char c : input) out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + '\'';
#endif
}

struct TemporaryFrames {
    std::filesystem::path path;
    ~TemporaryFrames() { std::error_code ignored; if (!path.empty()) std::filesystem::remove_all(path, ignored); }
};

void install_rendered_video(const std::filesystem::path& encoded,
                            const std::filesystem::path& destination,
                            const std::string& unique_name) {
    const auto parent = destination.parent_path().empty() ? std::filesystem::current_path() : destination.parent_path();
    const auto staged = parent / ("." + unique_name + ".mp4.tmp");
    try {
        std::filesystem::copy_file(encoded, staged);
#ifdef _WIN32
        if (!MoveFileExW(staged.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot install rendered video (Windows error " + std::to_string(GetLastError()) + ")");
#else
        std::filesystem::rename(staged, destination);
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(staged, ignored);
        throw;
    }
}

std::string render_video_worker(
    std::shared_ptr<const photara::splat::GaussianModel> model,
    std::vector<splat_render::ModelTransform> transforms, std::vector<Key> keys,
    OrbitCamera orbit, float near_clip, int width, int height, int fps, int codec,
    VisualizationMode view_mode, std::shared_ptr<const SparseScene> point_scene,
    ViewOptions view_options,
    std::filesystem::path destination, std::filesystem::path ffmpeg,
    photara::vk::Device runtime, const std::shared_ptr<App::VideoRenderProgress>& progress) {
    try {
        TemporaryFrames temp;
        const auto base = std::filesystem::temp_directory_path();
        for (unsigned attempt = 0; attempt < 100; ++attempt) {
            const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
            auto candidate = base / ("photara_video_" + std::to_string(id) + "_" + std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) { temp.path = std::move(candidate); break; }
        }
        if (temp.path.empty()) throw std::runtime_error("Cannot create temporary video render directory");
        if (progress->cancel) return "Cancelled";
        splat_render::Renderer renderer;
        if (view_mode != VisualizationMode::points) {
            renderer.attach(runtime);
            splat_render::GaussianCloud cloud;
            cloud.count = static_cast<std::uint32_t>(model->size());
            cloud.sh_degree = model->sh_degree;
            cloud.sh_bases = model->sh.is_valid() ? static_cast<std::uint32_t>(model->sh.shape()[1]) : 1U;
            cloud.render_method = static_cast<splat_render::RenderMethod>(model->render_method);
            cloud.filter_mode = static_cast<splat_render::FilterMode>(model->filter_mode);
            cloud.means = model->means.ptr<float>(); cloud.log_scales = model->log_scales.ptr<float>();
            cloud.quaternions = model->quaternions.ptr<float>(); cloud.opacity_logits = model->opacity_logits.ptr<float>();
            cloud.sh = model->sh.is_valid() ? model->sh.ptr<float>() : nullptr;
            cloud.filter_3d = model->filter_3d.is_valid() && model->filter_3d.numel() == model->size()
                ? model->filter_3d.ptr<float>() : nullptr;
            if (!renderer.upload(cloud, "video-render") || !renderer.set_model_transforms(transforms))
                throw std::runtime_error(renderer.failure());
        }
        splat_render::DisplayOptions display;
        display.near_clip = near_clip;
        const float duration = keys.back().time - keys.front().time;
        const int count = std::max(2, static_cast<int>(std::ceil(duration * fps)) + 1);
        progress->total = count;
        for (int i = 0; i < count; ++i) {
            if (progress->cancel) return "Cancelled";
            const auto key = evaluate(keys, std::min(keys.back().time, keys.front().time + float(i) / fps));
            orbit.target = {key.target[0], key.target[1], key.target[2]};
            orbit.yaw = key.yaw; orbit.pitch = key.pitch; orbit.roll = key.roll; orbit.distance = key.distance;
            orbit.fov_degrees = key.fov_degrees; orbit.ortho_height = key.ortho_height;
            orbit.fisheye_k1 = key.fisheye_k1; orbit.projection = static_cast<EditorProjection>(key.projection);
            const auto source = make_preview_camera(orbit, width, height);
            splat_render::Camera camera;
            camera.world_to_camera = source.world_to_camera; camera.position = source.position;
            camera.fx = source.fx; camera.fy = source.fy; camera.cx = source.cx; camera.cy = source.cy;
            camera.k1 = source.k1; camera.k2 = source.k2; camera.k3 = source.k3; camera.k4 = source.k4;
            camera.width = width; camera.height = height;
            switch (orbit.projection) {
            case EditorProjection::orthographic:
                camera.model = splat_render::k_camera_orthographic;
                camera.fx = camera.fy = float(height) / std::max(1e-4F, orbit.ortho_height);
                camera.cx = width * .5F; camera.cy = height * .5F;
                camera.k1 = camera.k2 = camera.k3 = camera.k4 = 0.F;
                break;
            case EditorProjection::fisheye: camera.model = splat_render::k_camera_fisheye; break;
            case EditorProjection::panorama: camera.model = splat_render::k_camera_equirectangular; break;
            default: camera.model = splat_render::k_camera_pinhole; break;
            }
            photara::io::RgbImage image;
            if (view_mode == VisualizationMode::points) {
                image = render_point_cloud_image(*point_scene, orbit, view_options, width, height);
            } else {
                splat_render::FrameTarget frame;
                const auto shading = view_mode == VisualizationMode::rings
                    ? splat_render::Shading::rings : splat_render::Shading::gaussian;
                camera.ring_sigma = view_options.ring_scale;
                if (!renderer.draw(camera, frame, shading, display))
                    throw std::runtime_error(renderer.failure());
                if (!renderer.download_rgb(image.pixels, image.width, image.height))
                    throw std::runtime_error("Cannot download a rendered video frame");
            }
            char filename[32]; std::snprintf(filename, sizeof(filename), "frame_%08d.png", i);
            photara::io::save_rgb_png(image, temp.path / filename);
            progress->completed = i + 1;
        }
        if (progress->cancel) return "Cancelled";
        progress->encoding = true;
        const auto encoded = temp.path / "video.mp4";
        const std::string command = process_quote(ffmpeg) + " -nostdin -hide_banner -loglevel error -y -framerate " +
            std::to_string(fps) + " -i " + process_quote(temp.path / "frame_%08d.png") +
            (codec == 1 ? " -c:v libx265 -tag:v hvc1" : " -c:v libx264") +
            " -pix_fmt yuv420p -movflags +faststart " + process_quote(encoded);
        ProcessJob encoder;
        encoder.start(command, temp.path / "ffmpeg.log");
        while (encoder.running()) {
            if (progress->cancel) { encoder.stop(); return "Cancelled"; }
            encoder.poll();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (encoder.exit_code() != 0) {
            std::ifstream log(temp.path / "ffmpeg.log");
            std::string details((std::istreambuf_iterator<char>(log)), {});
            throw std::runtime_error("Video encoding failed: " + details.substr(0, 500));
        }
        if (progress->cancel) return "Cancelled";
        install_rendered_video(encoded, destination, temp.path.filename().string());
        return {};
    } catch (const std::exception& error) { return error.what(); }
}

void timeline(App& app) {
    auto& keys=app.camera_keyframes;
    auto& view=app.camera_timeline_view;
    const float total=camera_timeline::span(keys.back().time);
    if (app.camera_timeline_drag<0) view.constrain(total);
    if (app.camera_path_playing) view.reveal(total,app.camera_path_time);
    const ImVec2 origin=ImGui::GetCursorScreenPos();
    const float width=std::max(1.F,ImGui::GetContentRegionAvail().x);
    const float left=origin.x+80.F,right=std::max(left+1.F,origin.x+width-8.F),plot_width=right-left;
    const float line_y=origin.y+46.F;
    ImGui::PushID("timeline_zoom");
    const float centre=std::clamp(view.fraction(app.camera_path_time),0.F,1.F);
    ImGui::SetCursorScreenPos(origin);
    if (ImGui::Button("-",{20.F,20.F})) view.zoom(total,1.F/1.5F,centre);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tr("Zoom out"));
    ImGui::SameLine(0.F,3.F);
    if (ImGui::Button("+",{20.F,20.F})) view.zoom(total,1.5F,centre);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s",tr("Zoom in"));
    ImGui::SameLine(0.F,3.F);
    if (icons::ghost_button("##fit",icons::Icon::frame,{20.F,20.F},false,true,tr("Fit full path"))) {
        view={};view.constrain(total);
    }
    ImGui::PopID();
    ImGui::SetCursorScreenPos({left,origin.y});
    ImGui::InvisibleButton("##timeline",{plot_width,60.F},ImGuiButtonFlags_MouseButtonLeft|ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered=ImGui::IsItemHovered(),active=ImGui::IsItemActive();
    auto& io=ImGui::GetIO();
    if (hovered) {
        ImGui::SetKeyOwner(ImGuiKey_MouseWheelY,ImGui::GetItemID());
        ImGui::SetKeyOwner(ImGuiKey_MouseWheelX,ImGui::GetItemID());
    }
    if (hovered && !active) {
        if (io.MouseWheel!=0.F) {
            if (io.KeyShift) view.pan(total,-io.MouseWheel*view.duration*.15F);
            else view.zoom(total,std::pow(1.25F,io.MouseWheel),(io.MousePos.x-left)/plot_width);
        }
        if (io.MouseWheelH!=0.F) view.pan(total,-io.MouseWheelH*view.duration*.15F);
        if (io.MouseWheel!=0.F || io.MouseWheelH!=0.F) ImGui::SetKeyOwner(ImGuiKey_MouseWheelY,ImGui::GetItemID());
    }
    const auto x_for=[&](float time){return left+plot_width*view.fraction(time);};
    if (ImGui::IsItemActivated()) {
        app.camera_timeline_drag=-1;
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && std::abs(io.MousePos.y-line_y)<12.F) {
            float closest=9.F;
            for (int i=0;i<static_cast<int>(keys.size());++i) {
                if (keys[i].time<view.start || keys[i].time>view.start+view.duration) continue;
                const float distance=std::abs(io.MousePos.x-x_for(keys[i].time));
                if (distance<closest) {closest=distance;app.camera_timeline_drag=i;}
            }
            if (app.camera_timeline_drag>=0) select_key(app,app.camera_timeline_drag);
        }
    }
    if (active) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) view.pan(total,-io.MouseDelta.x/plot_width*view.duration);
        else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const float time=view.time_at((io.MousePos.x-left)/plot_width);
            const int index=app.camera_timeline_drag;
            if (index>=0 && index<static_cast<int>(keys.size()) && ImGui::IsMouseDragging(ImGuiMouseButton_Left,2.F)) {
                keys[index].time=camera_timeline::bounded_time(keys,static_cast<std::size_t>(index),time);
                app.camera_path_time=keys[index].time;app.camera_path_playing=false;
            } else if (index<0) {
                app.camera_path_time=std::clamp(time,keys.front().time,keys.back().time);
                app.camera_path_playing=false;seek(app);
            }
        }
    }
    if (ImGui::IsItemDeactivated()) app.camera_timeline_drag=-1;
    if (hovered && !active) ImGui::SetTooltip("%s",tr("Wheel: zoom · Shift+wheel or middle-drag: pan"));
    auto* draw=ImGui::GetWindowDrawList();
    draw->AddRectFilled({origin.x,origin.y+31.F},{origin.x+width,origin.y+59.F},theme::u32(theme::surface_1),4.F);
    draw->AddText({origin.x+10.F,line_y-ImGui::GetTextLineHeight()*.5F},theme::u32(theme::text_muted),tr("Camera"));
    draw->PushClipRect({left-6.F,origin.y},{right+6.F,origin.y+61.F},true);
    const float major=camera_timeline::tick_step(view.duration*85.F/plot_width),minor=major/5.F;
    const auto first_tick=static_cast<long long>(std::ceil(view.start/minor));
    for (int i=0;i<1000;++i) {
        const auto tick=first_tick+i;
        const float time=static_cast<float>(double(tick)*minor);
        if (time>view.start+view.duration+minor*.001F) break;
        const bool is_major=tick%5==0;
        const float x=x_for(time);
        draw->AddLine({x,line_y-(is_major ? 10.F : 5.F)},{x,line_y-3.F},theme::u32(is_major ? theme::text_faint : theme::border));
        if (is_major) {
            char label[32];
            std::snprintf(label,sizeof(label),major<.1F ? "%.2f s" : major<1.F ? "%.1f s" : "%.0f s",time);
            const float label_width=ImGui::CalcTextSize(label).x;
            draw->AddText({std::clamp(x-label_width*.5F,left,std::max(left,right-label_width)),origin.y},theme::u32(theme::text_muted),label);
        }
    }
    draw->AddLine({left,line_y},{right,line_y},theme::u32(theme::surface_3),2.F);
    const float playhead=x_for(app.camera_path_time);
    draw->AddLine({left,line_y},{std::clamp(playhead,left,right),line_y},theme::u32(theme::accent_deep),2.F);
    for (int i=0;i<static_cast<int>(keys.size());++i) {
        const float x=x_for(keys[i].time);
        if (x<left-6.F || x>right+6.F) continue;
        const bool selected=i==app.camera_key_selected;
        const float radius=selected ? 6.F : 4.F;
        draw->AddQuadFilled({x,line_y-radius},{x+radius,line_y},{x,line_y+radius},{x-radius,line_y},theme::u32(selected ? theme::accent : theme::text_muted));
        if (selected) draw->AddQuadFilled({x,line_y-3.F},{x+3.F,line_y},{x,line_y+3.F},{x-3.F,line_y},theme::u32(theme::surface_0));
    }
    if (playhead>=left && playhead<=right) {
        draw->AddLine({playhead,origin.y+16.F},{playhead,line_y+11.F},theme::u32(theme::accent),1.5F);
        draw->AddTriangleFilled({playhead-4.F,origin.y+16.F},{playhead+4.F,origin.y+16.F},{playhead,origin.y+21.F},theme::u32(theme::accent));
    }
    draw->PopClipRect();
    // Overview doubles as a horizontal scrollbar for the visible time range.
    const float thumb_width=std::min(plot_width,std::max(18.F,plot_width*view.duration/total));
    const float travel=plot_width-thumb_width;
    float thumb_x=left+(total>view.duration ? view.start/(total-view.duration)*travel : 0.F);
    ImGui::SetCursorScreenPos({left,origin.y+64.F});
    ImGui::InvisibleButton("##timeline_scroll",{plot_width,8.F});
    if (ImGui::IsItemActivated()) app.camera_timeline_scroll_grab=io.MousePos.x>=thumb_x && io.MousePos.x<=thumb_x+thumb_width ? io.MousePos.x-thumb_x : thumb_width*.5F;
    if (ImGui::IsItemActive() && travel>0.F) {
        view.start=std::clamp((io.MousePos.x-left-app.camera_timeline_scroll_grab)/travel,0.F,1.F)*(total-view.duration);
        thumb_x=left+view.start/(total-view.duration)*travel;
    }
    draw->AddRectFilled({left,origin.y+65.F},{right,origin.y+71.F},theme::u32(theme::surface_1),3.F);
    draw->AddRectFilled({thumb_x,origin.y+65.F},{thumb_x+thumb_width,origin.y+71.F},theme::u32(ImGui::IsItemHovered() || ImGui::IsItemActive() ? theme::accent : theme::surface_3),3.F);
    ImGui::SetCursorScreenPos({origin.x,origin.y+76.F});
}

} // namespace

void select_camera_path_keyframe(App& app, int index) {
    if (app.camera_path_recording.active) return;
    if (app.camera_path_recording.review) {
        auto& recording=app.camera_path_recording;
        if (index>=0 && index<static_cast<int>(recording.result.size())) {
            recording.preview_playing=false; apply(app,recording.result[index]);
        }
        return;
    }
    if (index>=0 && index<static_cast<int>(app.camera_keyframes.size())) select_key(app,index);
}

bool finish_camera_path_recording(App& app) { finish_path_recording(app); return !app.camera_path_recording.review; }

bool start_video_render(App& app, const std::filesystem::path& output, const std::filesystem::path& ffmpeg) {
    auto& render = app.video_render;
    try {
        if (render.job.valid()) throw std::runtime_error("A video render is already running");
        if (app.camera_path_recording.active || app.camera_path_recording.review) throw std::runtime_error("Stop camera path recording before rendering");
        const auto view_mode = app.view_mode;
        if (view_mode != VisualizationMode::points && view_mode != VisualizationMode::splat &&
            view_mode != VisualizationMode::rings)
            throw std::runtime_error("Select Point Cloud, 3DGS, or Rings before rendering video");
        if (ffmpeg.empty()) throw std::runtime_error("ffmpeg was not found");
        if (render.width < 2 || render.width > 8192 || render.width % 2 ||
            render.height < 2 || render.height > 8192 || render.height % 2 ||
            render.fps < 1 || render.fps > 120 || render.codec < 0 || render.codec > 1 || app.camera_keyframes.size() < 2 ||
            !(app.camera_keyframes.back().time > app.camera_keyframes.front().time) ||
            (app.camera_keyframes.back().time - app.camera_keyframes.front().time) * render.fps > 100000.F)
            throw std::runtime_error("Invalid video dimensions, frame rate, or camera path");
        float previous = -std::numeric_limits<float>::infinity();
        for (const auto& key : app.camera_keyframes) {
            if (!std::isfinite(key.roll) || !std::isfinite(key.time) || key.time <= previous ||
                key.projection < static_cast<int>(EditorProjection::perspective) ||
                key.projection > static_cast<int>(EditorProjection::panorama))
                throw std::runtime_error("Camera path has invalid or unordered keyframes");
            previous = key.time;
        }
        auto extension = output.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".mp4") throw std::runtime_error("Video output must use the .mp4 extension");
        std::shared_ptr<const photara::splat::GaussianModel> model;
        std::shared_ptr<const SparseScene> point_scene;
        std::vector<splat_render::ModelTransform> transforms;
        if (view_mode == VisualizationMode::points) {
            if (app.loading_scene || !app.scene.has_points())
                throw std::runtime_error("Wait for a point cloud to load before rendering video");
            auto snapshot = std::make_shared<SparseScene>();
            snapshot->points = app.scene.points;
            snapshot->colours = app.scene.colours;
            snapshot->placements = app.scene.placements;
            point_scene = std::move(snapshot);
        } else {
            if (!app.gaussian_models.empty() && !app.gaussian_preview_baked &&
                !app.splat_edit.dirty() && app.composed_model) {
                model = app.composed_model;
                transforms = app.gaussian_placements;
            } else if (app.final_splat_ready || app.gaussian_preview_baked || app.splat_edit.dirty()) {
                auto editable = std::make_shared<photara::splat::GaussianModel>();
                if (app.splat_edit.copy_model(*editable)) model = std::move(editable);
                else if (app.splat_edit.dirty())
                    throw std::runtime_error("Cannot snapshot the edited 3DGS model");
            }
            // Export can start before the viewport has uploaded/synced its model.
            if (!model)
                model = std::make_shared<photara::splat::GaussianModel>(
                    load_export_model(app, tinytensor::Device::CPU));
            if (model->size() == 0) throw std::runtime_error("No 3DGS model is available to render");
        }
        auto keys = app.camera_keyframes;
        for (auto& key : keys) key.thumbnail_rgb.clear();
        const auto progress = std::make_shared<App::VideoRenderProgress>();
        render.progress = progress; render.output = output;
        const auto runtime = view_mode == VisualizationMode::points
            ? photara::vk::Device{} : gpu::runtime();
        const auto orbit = app.camera;
        const auto near_clip = app.navigation_near_clip;
        const auto width = render.width, height = render.height, fps = render.fps, codec = render.codec;
        const auto view_options = app.view_options;
        render.job = std::async(std::launch::async,
            [model, transforms = std::move(transforms), keys = std::move(keys), orbit,
             near_clip, width, height, fps, codec, view_mode, point_scene, view_options,
             output, ffmpeg, runtime, progress]() mutable {
                return render_video_worker(model, std::move(transforms), std::move(keys), orbit,
                    near_clip, width, height, fps, codec, view_mode, point_scene, view_options,
                    output, ffmpeg, runtime, progress);
            });
        app.camera_path_playing = false;
        return true;
    } catch (const std::exception& error) {
        render.progress.reset(); set_message(app, error.what(), theme::danger);
        return false;
    }
}

void poll_video_render(App& app) {
    auto& render = app.video_render;
    if (!render.job.valid() || render.job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    try {
        const auto error = render.job.get();
        if (error.empty()) set_message(app, std::string(tr("Rendered video")) + ": " + path_to_utf8(render.output), theme::success);
        else if (error == "Cancelled") set_message(app, tr("Video rendering cancelled"), theme::text_muted);
        else set_message(app, error, theme::danger);
    } catch (const std::exception& error) { set_message(app, error.what(), theme::danger); }
    render.progress.reset();
}

void shutdown_video_render(App& app) {
    if (app.video_render.progress) app.video_render.progress->cancel = true;
    if (app.video_render.job.valid()) app.video_render.job.wait();
}

void draw_video_render_settings(App& app) {
    auto& render = app.video_render;
    if (render.show_settings) { ImGui::OpenPopup("##render_video_settings"); render.show_settings = false; }
    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * .5F,
                            viewport->WorkPos.y + viewport->WorkSize.y * .5F}, ImGuiCond_Appearing, {.5F, .5F});
    ImGui::SetNextWindowSize({420.F, 0.F}, ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints({420.F, 0.F}, {420.F, viewport->WorkSize.y - 24.F});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {22.F, 20.F});
    const bool settings_open = ImGui::BeginPopupModal("##render_video_settings", nullptr,
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
    ImGui::PopStyleVar();
    if (settings_open) {
        icons::inline_icon(icons::Icon::render_video, theme::u32(theme::accent), 19.F);
        ImGui::SameLine(0.F, 10.F);
        ImGui::TextUnformatted(tr("Render video"));
        ImGui::Spacing();
        ImGui::TextDisabled("%s", tr("Export your camera path as an MP4 video."));
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        constexpr int widths[]{3840, 2560, 1920, 1280};
        constexpr int heights[]{2160, 1440, 1080, 720};
        const char* presets[]{"4K   ·   3840 × 2160", "2K   ·   2560 × 1440",
                              "1080p   ·   1920 × 1080", "720p   ·   1280 × 720"};
        render.resolution_preset = std::clamp(render.resolution_preset, 0, 3);
        ImGui::TextUnformatted(tr("Resolution"));
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::Combo("##video_resolution", &render.resolution_preset, presets, 4)) {
            render.width = widths[render.resolution_preset];
            render.height = heights[render.resolution_preset];
        }
        ImGui::Spacing();
        ImGui::TextUnformatted(tr("Frame rate"));
        ImGui::SetNextItemWidth(-1.F);
        char current_fps[24]; std::snprintf(current_fps, sizeof(current_fps), "%d fps", render.fps);
        if (ImGui::BeginCombo("##video_fps", current_fps)) {
            for (const int fps : {24, 25, 30, 50, 60}) {
                char label[24]; std::snprintf(label, sizeof(label), "%d fps", fps);
                if (ImGui::Selectable(label, render.fps == fps)) render.fps = fps;
                if (render.fps == fps) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::Spacing();
        ImGui::TextUnformatted(tr("Encoding"));
        ImGui::SetNextItemWidth(-1.F);
        const char* codecs[]{"H.264 / AVC", "H.265 / HEVC"};
        ImGui::Combo("##video_codec", &render.codec, codecs, 2);
        ImGui::Spacing();
        ImGui::TextDisabled("%s", tr("Container: MP4"));
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        const bool valid = render.width >= 2 && render.width <= 8192 && render.width % 2 == 0 &&
            render.height >= 2 && render.height <= 8192 && render.height % 2 == 0 &&
            render.fps >= 1 && render.fps <= 120 && app.camera_keyframes.size() >= 2 &&
            app.camera_keyframes.back().time > app.camera_keyframes.front().time &&
            (app.camera_keyframes.back().time - app.camera_keyframes.front().time) * render.fps <= 100000.F;
        if (!valid) ImGui::TextWrapped("%s", tr("Use even dimensions from 2 to 8192, 1–120 fps, and a valid camera path."));
        const float cancel_w = std::max(82.F, ImGui::CalcTextSize(tr("Cancel")).x + 28.F);
        const float action_w = std::max(152.F, ImGui::CalcTextSize(tr("Choose output..." )).x + 36.F);
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - cancel_w - action_w - 8.F));
        if (ImGui::Button(tr("Cancel"), {cancel_w, 32.F})) ImGui::CloseCurrentPopup();
        ImGui::SameLine(0.F, 8.F);
        ImGui::BeginDisabled(!valid || render.job.valid());
        if (theme::primary_button(tr("Choose output..."), {action_w, 32.F})) {
            const auto ffmpeg = photara::io::locate_ffmpeg();
            if (ffmpeg.empty()) {
                set_message(app, tr("ffmpeg was not found. Install ffmpeg and add it to PATH."), theme::warning);
            } else {
                std::array<char, 1024> path{};
                if (pick_export_path(L"Render video", path, FilePickKind::video, L"render.mp4", L"mp4")) {
                    auto output = path_from_utf8_field(path.data());
                    if (output.extension().empty()) output.replace_extension(".mp4");
                    if (start_video_render(app, output, ffmpeg)) ImGui::CloseCurrentPopup();
                }
            }
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }
    if (render.job.valid() && render.progress) {
        ImGui::OpenPopup("##render_video_progress");
        const ImVec2 progress_size{
            std::min(420.F, std::max(240.F, viewport->WorkSize.x - 24.F)),
            std::min(210.F, std::max(160.F, viewport->WorkSize.y - 24.F))};
        ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * .5F,
                                viewport->WorkPos.y + viewport->WorkSize.y * .5F},
                               ImGuiCond_Always, {.5F, .5F});
        ImGui::SetNextWindowSize(progress_size, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {22.F, 20.F});
        const bool progress_open = ImGui::BeginPopupModal("##render_video_progress", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleVar();
        if (progress_open) {
            icons::inline_icon(icons::Icon::render_video, theme::u32(theme::accent), 19.F);
            ImGui::SameLine(0.F, 10.F);
            ImGui::TextUnformatted(tr("Rendering video"));
            ImGui::Spacing(); ImGui::Separator();
            const int total = render.progress->total.load();
            const int completed = std::clamp(render.progress->completed.load(), 0, std::max(0, total));
            const bool encoding = render.progress->encoding.load();
            const bool cancelling = render.progress->cancel.load();
            ImGui::SetCursorPosY(progress_size.y < 195.F ? 58.F : 76.F);
            if (cancelling) ImGui::TextUnformatted(tr("Cancelling..."));
            else if (encoding) ImGui::TextUnformatted(tr("Encoding video..."));
            else if (total > 0) ImGui::Text(tr("Rendering frame %d / %d"), completed, total);
            else ImGui::TextUnformatted(tr("Preparing model..."));
            if (!encoding && !cancelling && total > 0) {
                char percentage[16]; std::snprintf(percentage, sizeof(percentage), "%d%%", completed * 100 / total);
                ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 8.F,
                    ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize(percentage).x));
                ImGui::TextColored(theme::accent, "%s", percentage);
            }
            ImGui::SetCursorPosY(progress_size.y < 195.F ? 82.F : 103.F);
            theme::progress_track({-1.F, 6.F},
                encoding || cancelling || total == 0 ? -1.F : float(completed) / total, theme::accent);
            ImGui::SetCursorPosY(progress_size.y - 53.F);
            const float cancel_width = std::max(94.F, ImGui::CalcTextSize(tr("Cancel")).x + 30.F);
            ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - cancel_width));
            ImGui::BeginDisabled(cancelling);
            if (theme::toolbar_button(tr("Cancel"), {cancel_width, 32.F})) render.progress->cancel = true;
            ImGui::EndDisabled();
            ImGui::EndPopup();
        }
    } else if (ImGui::BeginPopupModal("##render_video_progress")) {
        ImGui::CloseCurrentPopup(); ImGui::EndPopup();
    }
}

void clear_camera_path_thumbnails(App& app) {
    app.camera_path_recording = {};
    for (auto& item : app.camera_thumbnails) item.texture.reset();
    app.camera_thumbnails.clear(); app.camera_thumbnail_pending = -1; app.camera_key_delete_pending = -1;
    app.camera_timeline_drag = -1;
    app.camera_timeline_view = {};
}
void capture_camera_path_thumbnail(App& app) {
    const int i = app.camera_thumbnail_pending;
    if (i < 0 || i >= static_cast<int>(app.camera_keyframes.size())) return;
    app.camera_thumbnail_pending = -1;
    auto& key = app.camera_keyframes[static_cast<std::size_t>(i)];
    if (!same_pose(app.camera, key) || !app.splat_renderer) return;
    try {
        std::vector<std::uint8_t> rgb; std::uint32_t width = 0, height = 0;
        if (app.splat_renderer->download_rgb(rgb, width, height, 256)) {
            key.thumbnail_width = width; key.thumbnail_height = height; key.thumbnail_rgb = std::move(rgb);
            if (i < static_cast<int>(app.camera_thumbnails.size())) app.camera_thumbnails[i].pixels = nullptr;
        }
    } catch (...) { key.thumbnail_width = key.thumbnail_height = 0; key.thumbnail_rgb.clear(); }
}

void draw_camera_path_panel(App& app) {
    update_path_recording(app);
    auto& keys = app.camera_keyframes;
    if (app.camera_key_delete_pending >= 0 && app.camera_key_delete_pending < static_cast<int>(keys.size())) {
        const auto i = static_cast<std::size_t>(app.camera_key_delete_pending);
        if (i < app.camera_thumbnails.size()) { app.camera_thumbnails[i].texture.reset(); app.camera_thumbnails.erase(app.camera_thumbnails.begin() + i); }
        keys.erase(keys.begin() + i); app.camera_key_selected = -1; app.camera_path_playing = false;
        app.camera_path_time = keys.empty() ? 0.F : std::min(app.camera_path_time, keys.back().time);
        app.camera_thumbnail_pending = -1; app.camera_key_delete_pending = -1; app.camera_timeline_drag = -1;
    }
    if (app.camera_path_playing && keys.size() >= 2) {
        app.camera_path_time += ImGui::GetIO().DeltaTime;
        if (app.camera_path_time >= keys.back().time) { app.camera_path_time = keys.back().time; app.camera_path_playing = false; }
        seek(app);
    }
    if (!app.show_camera_path) return;
    // The first implementation saved a narrow floating window in existing ini
    // layouts. Migrate that window into Console's dock, preserving wider custom
    // floating layouts and all existing sidebar splits.
    if (app.camera_path_dock_pending) {
        const auto* saved = ImGui::FindWindowSettingsByID(ImHashStr("###CameraPath"));
        if (saved && (saved->DockId || saved->Size.x >= 480)) app.camera_path_dock_pending = false;
        else {
            ImGuiID dock = 0;
            if (const auto* console = ImGui::FindWindowByName("###Console")) dock = console->DockId;
            if (!dock) if (const auto* settings = ImGui::FindWindowSettingsByID(ImHashStr("###Console"))) dock = settings->DockId;
            if (dock && ImGui::DockBuilderGetNode(dock)) {
                ImGui::SetNextWindowDockID(dock, ImGuiCond_Always);
                ImGui::SetNextWindowFocus();
                app.camera_path_dock_pending = false;
            }
        }
    }
    ImGui::SetNextWindowSize({1100.F, 200.F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints({430.F, 170.F}, {FLT_MAX, FLT_MAX});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16.F, 12.F});
    if (app.camera_path_focus_requested) { ImGui::SetNextWindowFocus(); app.camera_path_focus_requested=false; }
    const bool open = ImGui::Begin(i18n::id("Camera Path", "###CameraPath"), &app.show_camera_path);
    ImGui::PopStyleVar();
    if (!open) { ImGui::End(); return; }
    if (app.camera_path_recording.active || app.camera_path_recording.review) {
        draw_recording_controls(app);
        ImGui::End(); return;
    }
    const auto& io = ImGui::GetIO();
    if (!app.camera_path_recording.active && !app.camera_path_recording.review && !shortcuts::suspended &&
        app.camera_key_selected >= 0 && app.camera_key_selected < static_cast<int>(keys.size()) &&
        ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) &&
        !io.WantTextInput && !ImGui::IsAnyItemActive() && shortcuts::modifiers() == 0 &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) &&
        ImGui::IsKeyPressed(ImGuiKey_Delete, false))
        app.camera_key_delete_pending = app.camera_key_selected;
    ImGui::BeginDisabled(app.camera_path_recording.active || app.camera_path_recording.review);
    sync_thumbnails(app);
    const float width = ImGui::GetContentRegionAvail().x;
    const ImVec2 header = ImGui::GetCursorScreenPos();
    const bool compact = width < 680.F;
    ImGui::SetCursorScreenPos({header.x, header.y + 7.F});
    ImGui::TextUnformatted(tr("Camera Path"));
    if (width >= 520.F) {
        ImGui::SameLine(0.F, 16.F);
        ImGui::TextDisabled("%zu %s  ·  %.1f s", keys.size(), tr("views"), keys.empty() ? 0.F : keys.back().time);
    }
    const float transport_x = header.x + (width - 170.F) * 0.5F;
    const float transport_y = header.y + (compact ? 38.F : 0.F);
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled({transport_x - 4.F, transport_y - 2.F}, {transport_x + 174.F, transport_y + 32.F},
        theme::u32(theme::surface_1), 7.F);
    draw->AddRect({transport_x - 4.F, transport_y - 2.F}, {transport_x + 174.F, transport_y + 32.F},
        theme::u32(theme::border), 7.F);
    ImGui::SetCursorScreenPos({transport_x, transport_y});
    const bool can_play = keys.size() >= 2;
    if (icons::ghost_button("##previous", icons::Icon::previous_keyframe, {30.F, 30.F}, false, !keys.empty(), tr("Previous view"))) {
        select_key(app, static_cast<int>(camera_timeline::previous(keys, app.camera_path_time)));
    }
    ImGui::SameLine(0.F, 5.F);
    if (icons::ghost_button("##rewind",icons::Icon::rewind,{30.F,30.F},false,!keys.empty(),tr("Rewind 5 seconds"))) {
        app.camera_path_time=camera_timeline::jump_time(app.camera_path_time,-5.F,keys.front().time,keys.back().time);
        seek(app);app.camera_timeline_view.reveal(camera_timeline::span(keys.back().time),app.camera_path_time);
    }
    ImGui::SameLine(0.F, 5.F);
    if (icons::button("##play", app.camera_path_playing ? icons::Icon::pause : icons::Icon::play,
            {30.F, 30.F}, app.camera_path_playing, can_play, tr(app.camera_path_playing ? "Pause" : "Play"))) {
        app.camera_path_playing = !app.camera_path_playing;
        if (app.camera_path_playing && app.camera_path_time >= keys.back().time) app.camera_path_time = keys.front().time;
    }
    ImGui::SameLine(0.F, 5.F);
    if (icons::ghost_button("##fast_forward",icons::Icon::fast_forward,{30.F,30.F},false,!keys.empty(),tr("Forward 5 seconds"))) {
        app.camera_path_time=camera_timeline::jump_time(app.camera_path_time,5.F,keys.front().time,keys.back().time);
        seek(app);app.camera_timeline_view.reveal(camera_timeline::span(keys.back().time),app.camera_path_time);
    }
    ImGui::SameLine(0.F, 5.F);
    if (icons::ghost_button("##next", icons::Icon::next_keyframe, {30.F, 30.F}, false, !keys.empty(), tr("Next view"))) {
        select_key(app, static_cast<int>(camera_timeline::next(keys, app.camera_path_time)));
    }
    if (!compact && width >= 940.F && !keys.empty()) {
        char current[32], total[32], text[72];
        format_time(app.camera_path_time, current, sizeof(current));
        format_time(keys.back().time, total, sizeof(total));
        std::snprintf(text, sizeof(text), "%s  /  %s", current, total);
        const float available = header.x + width - 315.F - (transport_x + 191.F);
        if (ImGui::CalcTextSize(text).x <= available)
            draw->AddText({transport_x + 191.F, header.y + 8.F}, theme::u32(theme::text_muted), text);
    }
    const float add_width = std::max(122.F, ImGui::CalcTextSize(tr("Add view")).x + 44.F);
    if (!compact && width >= 940.F) {
        const float record_width = std::max(92.F, ImGui::CalcTextSize(tr("Fly a new path")).x + 44.F);
        ImGui::SetCursorScreenPos({header.x + width - add_width - 44.F - record_width, header.y});
        if (icons::labeled_button("##record", icons::Icon::camera, tr("Fly a new path"),
                {record_width, 30.F}, icons::ButtonStyle::normal, true,
                false, tr("Explore the scene and generate a camera path."))) app.camera_path_recording.show_setup = true;
    }
    ImGui::SetCursorScreenPos({header.x + width - add_width - 36.F, header.y});
    if (icons::labeled_button("##add", icons::Icon::add, tr("Add view"), {add_width, 30.F}, icons::ButtonStyle::primary)) {
        add_camera_view(app);
    }
    ImGui::SameLine(0.F, 6.F);
    if (icons::ghost_button("##path_actions", icons::Icon::more, {30.F, 30.F}, false, true, tr("Path tools")))
        ImGui::OpenPopup("##path_tools");
    if (ImGui::BeginPopup("##path_tools")) {
        const auto capture_count=std::count_if(app.scene.views.begin(),app.scene.views.end(),[](const auto& view){return camera_capture_path::usable(view);});
        if (ImGui::MenuItem(tr("From capture cameras"),nullptr,false,capture_count>=2)) {
            app.camera_capture_path.show_setup=true;
            ImGui::CloseCurrentPopup();
        }
        if (capture_count<2 && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s",tr("Load training or reconstruction cameras to use this option."));
        if (ImGui::MenuItem(tr("Fly a new path"))) {
            app.camera_path_recording.show_setup = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextUnformatted(tr("Orbit around target"));
        ImGui::TextDisabled("%s", tr("Uses the current view and orbit target."));
        static float duration = 12.F;
        static bool clockwise = false;
        ImGui::SetNextItemWidth(120.F);
        ImGui::DragFloat(tr("Duration (s)"), &duration, .1F, .1F, 600.F, "%.1f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::Checkbox(tr("Clockwise"), &clockwise);
        ImGui::TextDisabled("%s", tr("Adds 9 keyframes at the end of the path."));
        const float start = keys.empty() ? 0.F : keys.back().time + 2.F;
        ImGui::BeginDisabled(start + duration > 36000.F);
        if (ImGui::Button(tr("Append orbit"))) {
            const auto first = camera_timeline::append_orbit(keys, capture(app.camera, 0.F), duration, clockwise);
            sync_thumbnails(app);
            select_key(app, static_cast<int>(first));
            app.camera_thumbnail_pending = static_cast<int>(first);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }
    draw_recording_setup(app);
    draw_capture_path_setup(app);
    ImGui::SetCursorScreenPos({header.x, header.y + (compact ? 78.F : 43.F)});
    if (keys.empty()) {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float h = std::max(0.F, ImGui::GetContentRegionAvail().y);
        auto* d = ImGui::GetWindowDrawList();
        const bool short_panel = h < 145.F;
        const float icon_size = short_panel ? 32.F : 48.F;
        const float group_h = short_panel ? 100.F : 144.F;
        const float x = origin.x + width * .5F;
        const float y = origin.y + std::max(0.F, (h - group_h) * .5F);
        const ImVec2 box_min{x - icon_size * .5F, y}, box_max{x + icon_size * .5F, y + icon_size};
        d->AddRectFilled(box_min, box_max, theme::u32(theme::surface_1), 7.F);
        d->AddRect(box_min, box_max, theme::u32(theme::border), 7.F);
        const float inset = short_panel ? 7.F : 10.F;
        icons::draw(d, icons::Icon::camera, {box_min.x + inset, box_min.y + inset},
            {box_max.x - inset, box_max.y - inset}, theme::u32(theme::text_muted));
        const char* title = tr("No camera views yet");
        const char* hint = tr("Frame the scene, then add your first view.");
        const float title_y = box_max.y + 8.F;
        d->AddText({x - ImGui::CalcTextSize(title).x * .5F, title_y}, theme::u32(theme::text_bright), title);
        const float hint_y = title_y + ImGui::GetTextLineHeight() + 4.F;
        if (!short_panel)
            d->AddText({x - ImGui::CalcTextSize(hint).x * .5F, hint_y}, theme::u32(theme::text_muted), hint);
        const float button_y = short_panel ? hint_y : hint_y + ImGui::GetTextLineHeight() + 10.F;
        const float button_w = std::max(130.F, ImGui::CalcTextSize(tr("Add first view")).x + 44.F);
        if (button_y + 30.F <= origin.y + h) {
            ImGui::SetCursorScreenPos({x - button_w * .5F, button_y});
            if (icons::labeled_button("##add_first", icons::Icon::add, tr("Add first view"),
                    {button_w, 30.F}, icons::ButtonStyle::primary)) add_camera_view(app);
        }
        ImGui::EndDisabled(); ImGui::End(); return;
    }
    timeline(app);
    ImGui::Separator();
    ImGui::Spacing();
    int selected = app.camera_key_selected;
    if (selected >= 0 && selected < static_cast<int>(keys.size())) {
        const float inspector_width = ImGui::GetContentRegionAvail().x;
        const bool wide = inspector_width >= 1040.F;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14.F, 10.F});
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.F);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.F, 5.F});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8.F, 3.F});
        ImGui::BeginChild("##key_properties_compact", {0.F, 0.F}, ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::AlignTextToFramePadding();
        icons::inline_icon(icons::Icon::camera, theme::u32(theme::accent), 16.F);
        ImGui::SameLine(0.F, 8.F);
        char key_label[64];
        std::snprintf(key_label, sizeof(key_label), "%s %02d / %zu", tr("Keyframe"), selected + 1, keys.size());
        ImGui::SetNextItemWidth(160.F);
        if (ImGui::BeginCombo("##selected_key", key_label)) {
            for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
                char label[80];
                std::snprintf(label, sizeof(label), "%02d   ·   %.2f s", i + 1, keys[i].time);
                if (ImGui::Selectable(label, selected == i)) { select_key(app, i); selected = i; }
            }
            ImGui::EndCombo();
        }
        Key& k = keys[selected];
        ImGui::SameLine(0.F, 6.F);
        if (icons::ghost_button("##view_key", icons::Icon::camera, {24.F, 24.F}, false, true, tr("View keyframe")))
            select_key(app, selected);
        ImGui::SameLine(0.F, 12.F);
        if (icons::ghost_button("##delete", icons::Icon::trash, {24.F, 24.F}, false, true, tr("Delete keyframe (Delete)")))
            app.camera_key_delete_pending = selected;
        ImGui::SameLine(0.F, 6.F);
        if (icons::ghost_button("##lens", icons::Icon::sliders, {24.F, 24.F}, false, true, tr("Lens settings")))
            ImGui::OpenPopup("##lens_settings");
        bool lens_changed = false;
        ImGui::SetNextWindowSize({290.F, 0.F}, ImGuiCond_Appearing);
        if (ImGui::BeginPopup("##lens_settings")) {
            ImGui::TextUnformatted(tr("Lens settings"));
            ImGui::Separator();
            ImGui::TextDisabled("%s", tr("Projection"));
            const char* projections[]{tr("Perspective"), tr("Orthographic"), tr("Fisheye"), tr("Panorama")};
            ImGui::SetNextItemWidth(-1.F);
            lens_changed |= ImGui::Combo("##key_projection", &k.projection, projections, 4);
            if (k.projection == static_cast<int>(EditorProjection::orthographic)) {
                ImGui::TextDisabled("%s", tr("Orthographic height"));
                ImGui::SetNextItemWidth(-1.F);
                lens_changed |= ImGui::DragFloat("##key_ortho", &k.ortho_height, .01F, .001F, 1e6F, "%.3f", ImGuiSliderFlags_AlwaysClamp);
            } else if (k.projection != static_cast<int>(EditorProjection::panorama)) {
                ImGui::TextDisabled("%s", tr("Field of view"));
                ImGui::SetNextItemWidth(-1.F);
                lens_changed |= ImGui::DragFloat("##key_lens_fov", &k.fov_degrees, .1F, 1.F, 179.F, "%.1f°", ImGuiSliderFlags_AlwaysClamp);
                if (k.projection == static_cast<int>(EditorProjection::fisheye)) {
                    ImGui::TextDisabled("%s", tr("Fisheye distortion (k1)"));
                    ImGui::SetNextItemWidth(-1.F);
                    lens_changed |= ImGui::DragFloat("##key_k1", &k.fisheye_k1, .001F, -1.F, 1.F, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                }
            } else ImGui::TextDisabled("360° × 180°");
            ImGui::TextDisabled("%s", tr("Camera roll"));
            ImGui::SetNextItemWidth(-1.F);
            float roll_degrees = k.roll * 57.2957795F;
            if (ImGui::DragFloat("##key_roll", &roll_degrees, .1F, -180.F, 180.F, "%.1f°", ImGuiSliderFlags_AlwaysClamp)) {
                k.roll = roll_degrees / 57.2957795F; lens_changed = true;
            }
            ImGui::EndPopup();
        }
        const float update_width = std::max(126.F, ImGui::CalcTextSize(tr("Update pose")).x + 36.F);
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 8.F, ImGui::GetWindowContentRegionMax().x - update_width));
        if (icons::labeled_button("##update", icons::Icon::camera, tr("Update pose"), {update_width, 26.F})) {
            const int interpolation = k.interpolation;
            k = capture(app.camera, k.time);
            k.interpolation = interpolation;
            app.camera_path_playing = false;
            app.camera_thumbnail_pending = selected;
        }
        ImGui::Spacing();
        bool changed = lens_changed;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {10.F, 2.F});
        if (ImGui::BeginTable("##inspector", wide ? 2 : 1, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Properties", ImGuiTableColumnFlags_WidthFixed, wide ? 680.F : 410.F);
            if (wide) ImGui::TableSetupColumn("TransitionPreview", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableNextColumn();
            // One compact label column keeps numeric controls aligned rather than stretched.
            if (ImGui::BeginTable("##pose_fields", wide ? 4 : 2, ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 90.F);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, wide ? 155.F : 290.F);
                if (wide) {
                    ImGui::TableSetupColumn("PoseLabel", ImGuiTableColumnFlags_WidthFixed, 70.F);
                    ImGui::TableSetupColumn("PoseValue", ImGuiTableColumnFlags_WidthFixed, 275.F);
                }
                const auto row = [](const char* label) {
                    ImGui::TableNextColumn();
                    ImGui::AlignTextToFramePadding(); ImGui::TextDisabled("%s", tr(label));
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(110.F);
                };
                row("Time (s)");
                ImGui::SetNextItemWidth(110.F);
                float t = k.time;
                if (ImGui::DragFloat("##keytime", &t, .02F, 0.F, 36000.F, "%.2f s", ImGuiSliderFlags_AlwaysClamp)) {
                    k.time = camera_timeline::bounded_time(keys, static_cast<std::size_t>(selected), t);
                    app.camera_path_time = k.time; app.camera_path_playing = false;
                }
                row("Target");
                ImGui::SetNextItemWidth(268.F);
                changed |= ImGui::DragFloat3("##target", k.target, .01F, 0.F, 0.F, "%.3f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("X / Y / Z");
                row(k.projection == static_cast<int>(EditorProjection::orthographic) ? "Orthographic height" : "Field of view");
                if (k.projection == static_cast<int>(EditorProjection::orthographic))
                    changed |= ImGui::DragFloat("##ortho_height", &k.ortho_height, .01F, .001F, 1e6F, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                else {
                    ImGui::BeginDisabled(k.projection == static_cast<int>(EditorProjection::panorama));
                    changed |= ImGui::DragFloat("##fov", &k.fov_degrees, .1F, 1.F, 179.F, "%.1f°", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::EndDisabled();
                }
                row("Orbit");
                float orbit[]{k.yaw * 57.2957795F, k.pitch * 57.2957795F, k.distance};
                ImGui::PushMultiItemsWidths(3, 268.F);
                if (ImGui::DragFloat("##yaw", &orbit[0], .1F, 0.F, 0.F, "%.1f°")) {
                    k.yaw = orbit[0] / 57.2957795F; changed = true;
                }
                ImGui::PopItemWidth(); ImGui::SameLine();
                if (ImGui::DragFloat("##pitch", &orbit[1], .1F, -89.9F, 89.9F, "%.1f°", ImGuiSliderFlags_AlwaysClamp)) {
                    k.pitch = orbit[1] / 57.2957795F; changed = true;
                }
                ImGui::PopItemWidth(); ImGui::SameLine();
                changed |= ImGui::DragFloat("##distance", &k.distance, .01F, .001F, 1e6F, "%.3f", ImGuiSliderFlags_AlwaysClamp);
                ImGui::PopItemWidth();
                ImGui::EndTable();
            }
            if (changed) {
                apply(app, k); app.camera_path_time = k.time; app.camera_path_playing = false;
                k.thumbnail_rgb.clear(); k.thumbnail_width = k.thumbnail_height = 0;
                app.camera_thumbnail_pending = selected;
            }
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding(); ImGui::TextDisabled("%s", tr("Interpolation"));
            ImGui::SameLine(0.F, 14.F);
            ImGui::SetNextItemWidth(150.F);
            const char* methods[]{tr("Smooth"), tr("Linear"), tr("Hold")};
            if (ImGui::Combo("##interpolation", &k.interpolation, methods, 3)) {
                app.camera_path_playing = false;
                seek(app);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Transition to the next keyframe"));
            if (wide) {
                const ImVec2 pos = ImGui::GetCursorScreenPos();
                const ImVec2 size{144.F, 81.F};
                auto* d = ImGui::GetWindowDrawList();
                d->AddRectFilled(pos, {pos.x + size.x, pos.y + size.y}, theme::u32(theme::viewport_bg), 5.F);
                const auto& thumb = app.camera_thumbnails[selected].texture;
                if (!changed && thumb.descriptor && thumb.width && thumb.height) {
                    const float scale = std::min(size.x / thumb.width, size.y / thumb.height);
                    const ImVec2 offset{(size.x - thumb.width * scale) * .5F, (size.y - thumb.height * scale) * .5F};
                    d->AddImageRounded(reinterpret_cast<ImTextureID>(thumb.descriptor),
                        {pos.x + offset.x, pos.y + offset.y},
                        {pos.x + size.x - offset.x, pos.y + size.y - offset.y},
                        {0.F, 0.F}, {1.F, 1.F}, IM_COL32_WHITE, 5.F);
                } else {
                    const char* label = tr("No preview");
                    d->AddText({pos.x + (size.x - ImGui::CalcTextSize(label).x) * .5F, pos.y + 32.F},
                        theme::u32(theme::text_faint), label);
                }
                d->AddRect(pos, {pos.x + size.x, pos.y + size.y}, theme::u32(theme::border), 5.F);
                ImGui::Dummy(size);
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        ImGui::EndChild();
        ImGui::PopStyleVar(4);
        app.camera_path_edit = false;
    } else ImGui::TextDisabled("%s", tr("Select a view to edit"));
    ImGui::EndDisabled();
    ImGui::End();
}
} // namespace editor
