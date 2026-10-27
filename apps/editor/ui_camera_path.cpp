#include "ui.hpp"
#include "camera_timeline.hpp"
#include "i18n.hpp"
#include "icons.hpp"
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
    k.yaw = c.yaw; k.pitch = c.pitch; k.distance = c.distance;
    k.fov_degrees = c.fov_degrees; k.ortho_height = c.ortho_height;
    k.fisheye_k1 = c.fisheye_k1; k.projection = static_cast<int>(c.projection);
    return k;
}
bool same_pose(const OrbitCamera& c, const Key& k) {
    return c.target.x == k.target[0] && c.target.y == k.target[1] && c.target.z == k.target[2] &&
        c.yaw == k.yaw && c.pitch == k.pitch && c.distance == k.distance &&
        c.fov_degrees == k.fov_degrees && c.ortho_height == k.ortho_height &&
        c.fisheye_k1 == k.fisheye_k1 && static_cast<int>(c.projection) == k.projection;
}
void apply(App& app, const Key& k) {
    auto& c = app.camera;
    c.target = {k.target[0], k.target[1], k.target[2]};
    c.yaw = k.yaw; c.pitch = k.pitch; c.distance = k.distance;
    c.fov_degrees = k.fov_degrees; c.ortho_height = k.ortho_height;
    c.fisheye_k1 = k.fisheye_k1; c.projection = static_cast<EditorProjection>(k.projection);
    c.interacting = false; app.preview_follow_view = false;
}
Key evaluate(const std::vector<Key>& keys, float t) {
    if (t <= keys.front().time) return keys.front();
    if (t >= keys.back().time) return keys.back();
    const auto right = std::upper_bound(keys.begin(), keys.end(), t,
        [](float time, const Key& key) { return time < key.time; });
    const Key& b = *right; const Key& a = *(right - 1);
    float u = (t - a.time) / (b.time - a.time);
    u = u * u * (3.F - 2.F * u);
    const auto mix = [u](float x, float y) { return x + (y - x) * u; };
    Key result; result.projection = a.projection;
    for (int i = 0; i < 3; ++i) result.target[i] = mix(a.target[i], b.target[i]);
    result.yaw = a.yaw + std::remainder(b.yaw - a.yaw, 6.28318530718F) * u;
    result.pitch = mix(a.pitch, b.pitch); result.distance = mix(a.distance, b.distance);
    result.fov_degrees = mix(a.fov_degrees, b.fov_degrees);
    result.ortho_height = mix(a.ortho_height, b.ortho_height);
    result.fisheye_k1 = mix(a.fisheye_k1, b.fisheye_k1);
    return result;
}
void seek(App& app) {
    if (!app.camera_keyframes.empty()) apply(app, evaluate(app.camera_keyframes, app.camera_path_time));
}
void select_key(App& app, int i) {
    app.camera_key_selected = i;
    app.camera_path_time = app.camera_keyframes[static_cast<std::size_t>(i)].time;
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
void draw_card(App& app, int index, ImVec2 size) {
    const Key& key = app.camera_keyframes[static_cast<std::size_t>(index)];
    const bool selected = app.camera_key_selected == index;
    ImGui::PushID(index);
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton("##key", size);
    if (ImGui::IsItemClicked()) select_key(app, index);
    const bool hovered = ImGui::IsItemHovered();
    auto* draw = ImGui::GetWindowDrawList();
    const ImVec2 max{min.x + size.x, min.y + size.y};
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg), 5.F);
    const auto& thumb = app.camera_thumbnails[static_cast<std::size_t>(index)].texture;
    if (thumb.descriptor && thumb.width && thumb.height) {
        const float scale = std::min(size.x / static_cast<float>(thumb.width), size.y / static_cast<float>(thumb.height));
        const ImVec2 image_size{thumb.width * scale, thumb.height * scale};
        const ImVec2 image_min{min.x + (size.x - image_size.x) * 0.5F, min.y + (size.y - image_size.y) * 0.5F};
        const ImVec2 image_max{image_min.x + image_size.x, image_min.y + image_size.y};
        draw->AddImageRounded(reinterpret_cast<ImTextureID>(thumb.descriptor), image_min, image_max,
            {0.F, 0.F}, {1.F, 1.F}, IM_COL32_WHITE, 5.F);
        draw->AddRectFilled({min.x, max.y - 25.F}, max, IM_COL32(0, 0, 0, 150), 5.F, ImDrawFlags_RoundCornersBottom);
    } else {
        const ImVec2 icon_min{min.x + size.x * 0.5F - 11.F, min.y + size.y * 0.5F - 13.F};
        icons::draw(draw, icons::Icon::camera, icon_min, {icon_min.x + 22.F, icon_min.y + 22.F}, theme::u32(theme::text_faint));
        const char* label = tr("No preview");
        const float w = ImGui::CalcTextSize(label).x;
        draw->AddText({min.x + (size.x - w) * 0.5F, min.y + size.y * 0.5F + 14.F}, theme::u32(theme::text_faint), label);
    }
    draw->AddRect(min, max, theme::u32(selected ? theme::accent : hovered ? theme::text_faint : theme::border), 5.F, 0, selected ? 2.F : 1.F);
    char text[32]; std::snprintf(text, sizeof(text), "%02d", index + 1);
    draw->AddRectFilled({min.x + 7.F, min.y + 6.F}, {min.x + 31.F, min.y + 25.F}, IM_COL32(12, 16, 22, 190), 4.F);
    draw->AddText({min.x + 11.F, min.y + 7.F}, theme::u32(selected ? theme::accent : theme::text_bright), text);
    std::snprintf(text, sizeof(text), "%.2f s", key.time);
    draw->AddText({min.x + 9.F, max.y - 20.F}, theme::u32(theme::text_bright), text);
    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    if (selected || hovered) {
        ImGui::SetCursorScreenPos({max.x - 61.F, min.y + 6.F});
        if (icons::button("##remove", icons::Icon::trash, {24.F, 24.F}, false, true, tr("Delete keyframe")))
            app.camera_key_delete_pending = index;
        ImGui::SameLine(0.F, 4.F);
        if (icons::button("##more", icons::Icon::more, {24.F, 24.F}, false, true, tr("More options"))) ImGui::OpenPopup("##view_actions");
    }
    if (ImGui::BeginPopup("##view_actions")) {
        if (ImGui::MenuItem(tr("Edit"))) { app.camera_key_selected = index; app.camera_path_edit = true; }
        if (ImGui::MenuItem(tr("Update pose"))) {
            auto& k = app.camera_keyframes[static_cast<std::size_t>(index)];
            k = capture(app.camera, k.time); app.camera_path_playing = false; app.camera_thumbnail_pending = index;
        }
        if (ImGui::MenuItem(tr("Delete keyframe"))) app.camera_key_delete_pending = index;
        ImGui::EndPopup();
    }
    ImGui::SetCursorScreenPos(cursor);
    ImGui::PopID();
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
        splat_render::DisplayOptions display;
        display.near_clip = near_clip;
        const float duration = keys.back().time - keys.front().time;
        const int count = std::max(2, static_cast<int>(std::ceil(duration * fps)) + 1);
        progress->total = count;
        for (int i = 0; i < count; ++i) {
            if (progress->cancel) return "Cancelled";
            const auto key = evaluate(keys, std::min(keys.back().time, keys.front().time + float(i) / fps));
            orbit.target = {key.target[0], key.target[1], key.target[2]};
            orbit.yaw = key.yaw; orbit.pitch = key.pitch; orbit.distance = key.distance;
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
            splat_render::FrameTarget frame;
            if (!renderer.draw(camera, frame, splat_render::Shading::gaussian, display))
                throw std::runtime_error(renderer.failure());
            photara::io::RgbImage image;
            if (!renderer.download_rgb(image.pixels, image.width, image.height))
                throw std::runtime_error("Cannot download a rendered video frame");
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
    auto& keys = app.camera_keyframes;
    const float natural_span = camera_timeline::span(keys.back().time);
    const float span = app.camera_timeline_drag >= 0 ? app.camera_timeline_drag_span : natural_span;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(1.F, ImGui::GetContentRegionAvail().x);
    const float left = origin.x + 8.F, right = std::max(left + 1.F, origin.x + width - 8.F);
    const float line_y = origin.y + 31.F;
    ImGui::InvisibleButton("##timeline", {width, 46.F});
    const auto x_for = [&](float t) { return left + (right - left) * std::clamp(t / span, 0.F, 1.F); };
    if (ImGui::IsItemActivated()) {
        app.camera_timeline_drag = -1;
        app.camera_timeline_drag_span = natural_span;
        const float mouse_x = ImGui::GetIO().MousePos.x;
        float closest = 9.F;
        for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
            const float distance = std::abs(mouse_x - x_for(keys[i].time));
            if (distance < closest) { closest = distance; app.camera_timeline_drag = i; }
        }
        if (app.camera_timeline_drag >= 0) select_key(app, app.camera_timeline_drag);
    }
    if (ImGui::IsItemActive() && right > left) {
        const float t = std::clamp((ImGui::GetIO().MousePos.x - left) / (right - left), 0.F, 1.F) * span;
        const int index = app.camera_timeline_drag;
        if (index >= 0 && index < static_cast<int>(keys.size()) && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.F)) {
            keys[index].time = camera_timeline::bounded_time(keys, static_cast<std::size_t>(index), t);
            app.camera_path_time = keys[index].time; app.camera_path_playing = false;
        } else if (index < 0) {
            app.camera_path_time = std::min(t, keys.back().time);
            app.camera_path_playing = false; seek(app);
        }
    }
    if (ImGui::IsItemDeactivated()) app.camera_timeline_drag = -1;
    auto* d = ImGui::GetWindowDrawList();
    const float major = camera_timeline::tick_step(span * 85.F / (right - left));
    const float minor = major / 5.F;
    for (int i = 0; i <= 500; ++i) {
        const float t = i * minor;
        if (t > span + minor * 0.001F) break;
        const bool is_major = i % 5 == 0;
        const float x = x_for(t);
        d->AddLine({x, line_y - (is_major ? 10.F : 5.F)}, {x, line_y - 3.F},
            theme::u32(is_major ? theme::text_faint : theme::border));
        if (is_major) {
            char label[32];
            if (major < 1.F) std::snprintf(label, sizeof(label), "%.1f s", t);
            else std::snprintf(label, sizeof(label), "%.0f s", t);
            const float label_width = ImGui::CalcTextSize(label).x;
            d->AddText({std::clamp(x - label_width * 0.5F, left, std::max(left, right - label_width)), origin.y},
                theme::u32(theme::text_muted), label);
        }
    }
    d->AddLine({left, line_y}, {right, line_y}, theme::u32(theme::surface_3), 2.F);
    const float playhead = x_for(app.camera_path_time);
    d->AddLine({left, line_y}, {playhead, line_y}, theme::u32(theme::accent_deep), 2.F);
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
        const float x = x_for(keys[i].time); const bool selected = i == app.camera_key_selected;
        const float r = selected ? 6.F : 5.F;
        const ImU32 edge = theme::u32(selected ? theme::accent : theme::text_muted);
        d->AddQuadFilled({x, line_y - r}, {x + r, line_y}, {x, line_y + r}, {x - r, line_y}, edge);
        if (selected) d->AddQuadFilled({x, line_y - 3.F}, {x + 3.F, line_y}, {x, line_y + 3.F}, {x - 3.F, line_y}, theme::u32(theme::surface_0));
    }
    d->AddLine({playhead, origin.y + 16.F}, {playhead, line_y + 11.F}, theme::u32(theme::accent), 1.5F);
    d->AddTriangleFilled({playhead - 4.F, origin.y + 16.F}, {playhead + 4.F, origin.y + 16.F},
        {playhead, origin.y + 21.F}, theme::u32(theme::accent));
}

} // namespace

bool start_video_render(App& app, const std::filesystem::path& output, const std::filesystem::path& ffmpeg) {
    auto& render = app.video_render;
    try {
        if (render.job.valid()) throw std::runtime_error("A video render is already running");
        if (ffmpeg.empty()) throw std::runtime_error("ffmpeg was not found");
        if (render.width < 2 || render.width > 8192 || render.width % 2 ||
            render.height < 2 || render.height > 8192 || render.height % 2 ||
            render.fps < 1 || render.fps > 120 || render.codec < 0 || render.codec > 1 || app.camera_keyframes.size() < 2 ||
            !(app.camera_keyframes.back().time > app.camera_keyframes.front().time) ||
            (app.camera_keyframes.back().time - app.camera_keyframes.front().time) * render.fps > 100000.F)
            throw std::runtime_error("Invalid video dimensions, frame rate, or camera path");
        float previous = -std::numeric_limits<float>::infinity();
        for (const auto& key : app.camera_keyframes) {
            if (!std::isfinite(key.time) || key.time <= previous ||
                key.projection < static_cast<int>(EditorProjection::perspective) ||
                key.projection > static_cast<int>(EditorProjection::panorama))
                throw std::runtime_error("Camera path has invalid or unordered keyframes");
            previous = key.time;
        }
        auto extension = output.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".mp4") throw std::runtime_error("Video output must use the .mp4 extension");
        std::shared_ptr<const photara::splat::GaussianModel> model;
        std::vector<splat_render::ModelTransform> transforms;
        if (!app.gaussian_models.empty() && !app.gaussian_preview_baked && !app.splat_edit.dirty()) {
            model = app.composed_model;
            transforms = app.gaussian_placements;
        } else {
            auto editable = std::make_shared<photara::splat::GaussianModel>();
            if (app.splat_edit.copy_model(*editable)) model = std::move(editable);
        }
        if (!model || model->size() == 0) throw std::runtime_error("No 3DGS model is available to render");
        auto keys = app.camera_keyframes;
        for (auto& key : keys) key.thumbnail_rgb.clear();
        const auto progress = std::make_shared<App::VideoRenderProgress>();
        render.progress = progress; render.output = output;
        const auto runtime = gpu::runtime();
        const auto orbit = app.camera;
        const auto near_clip = app.navigation_near_clip;
        const auto width = render.width, height = render.height, fps = render.fps, codec = render.codec;
        render.job = std::async(std::launch::async,
            [model, transforms = std::move(transforms), keys = std::move(keys), orbit,
             near_clip, width, height, fps, codec, output, ffmpeg, runtime, progress]() mutable {
                return render_video_worker(model, std::move(transforms), std::move(keys), orbit,
                    near_clip, width, height, fps, codec, output, ffmpeg, runtime, progress);
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
    for (auto& item : app.camera_thumbnails) item.texture.reset();
    app.camera_thumbnails.clear(); app.camera_thumbnail_pending = -1; app.camera_key_delete_pending = -1;
    app.camera_timeline_drag = -1;
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
        ImGuiID bottom_id = 0;
        if (const auto* console = ImGui::FindWindowByName("###Console")) bottom_id = console->DockId;
        if (!bottom_id) if (const auto* settings = ImGui::FindWindowSettingsByID(ImHashStr("###Console"))) bottom_id = settings->DockId;
        if (auto* bottom = ImGui::DockBuilderGetNode(bottom_id)) {
            if (bottom->ParentNode && bottom->ParentNode->SplitAxis == ImGuiAxis_Y && bottom->Size.y < 300.F) {
                const float height = std::min(300.F, bottom->ParentNode->Size.y * 0.45F);
                bottom->SizeRef.y = height;
                auto* parent = bottom->ParentNode;
                auto* sibling = parent->ChildNodes[parent->ChildNodes[0] == bottom ? 1 : 0];
                if (sibling) sibling->SizeRef.y = std::max(1.F, parent->Size.y - height - ImGui::GetStyle().DockingSeparatorSize);
                ImGui::MarkIniSettingsDirty();
            }
        }
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
    ImGui::SetNextWindowSize({850.F, 290.F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints({430.F, 170.F}, {FLT_MAX, FLT_MAX});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16.F, 12.F});
    const bool open = ImGui::Begin(i18n::id("Camera Path", "###CameraPath"), &app.show_camera_path);
    ImGui::PopStyleVar();
    if (!open) { ImGui::End(); return; }
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
    const float transport_x = header.x + (width - 104.F) * 0.5F;
    const float transport_y = header.y + (compact ? 38.F : 0.F);
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled({transport_x - 4.F, transport_y - 2.F}, {transport_x + 108.F, transport_y + 32.F},
        theme::u32(theme::surface_1), 7.F);
    draw->AddRect({transport_x - 4.F, transport_y - 2.F}, {transport_x + 108.F, transport_y + 32.F},
        theme::u32(theme::border), 7.F);
    ImGui::SetCursorScreenPos({transport_x, transport_y});
    const bool can_play = keys.size() >= 2;
    if (icons::ghost_button("##previous", icons::Icon::previous_keyframe, {30.F, 30.F}, false, !keys.empty(), tr("Previous view"))) {
        select_key(app, static_cast<int>(camera_timeline::previous(keys, app.camera_path_time)));
    }
    ImGui::SameLine(0.F, 5.F);
    if (icons::button("##play", app.camera_path_playing ? icons::Icon::pause : icons::Icon::play,
            {30.F, 30.F}, app.camera_path_playing, can_play, tr(app.camera_path_playing ? "Pause" : "Play"))) {
        app.camera_path_playing = !app.camera_path_playing;
        if (app.camera_path_playing && app.camera_path_time >= keys.back().time) app.camera_path_time = keys.front().time;
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
        const float available = header.x + width - 142.F - (transport_x + 123.F);
        if (ImGui::CalcTextSize(text).x <= available)
            draw->AddText({transport_x + 123.F, header.y + 8.F}, theme::u32(theme::text_muted), text);
    }
    const float add_width = std::max(122.F, ImGui::CalcTextSize(tr("Add view")).x + 44.F);
    ImGui::SetCursorScreenPos({header.x + width - add_width, header.y});
    if (icons::labeled_button("##add", icons::Icon::add, tr("Add view"), {add_width, 30.F}, icons::ButtonStyle::primary)) {
        add_camera_view(app);
    }
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
        ImGui::End(); return;
    }
    const float available_h = ImGui::GetContentRegionAvail().y;
    // Treat the card as one 16:9 size. Never clamp its height independently:
    // a shallow dock must not turn the storyboard into a cropped film strip.
    const float preferred_w = std::clamp((width - 30.F) / 4.F, 160.F, 224.F);
    const float card_h = std::clamp(std::min(preferred_w * 9.F / 16.F, available_h - 103.F), 90.F, 126.F);
    const float card_w = card_h * 16.F / 9.F;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::BeginChild("##storyboard", {0.F, card_h + 17.F}, false, ImGuiWindowFlags_HorizontalScrollbar);
    for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
        if (i) ImGui::SameLine(0.F, 10.F);
        draw_card(app, i, {card_w, card_h});
    }
    ImGui::EndChild(); ImGui::PopStyleColor();
    timeline(app);
    ImGui::Separator();
    const int selected = app.camera_key_selected;
    if (selected >= 0 && selected < static_cast<int>(keys.size())) {
        Key& k = keys[selected];
        ImGui::AlignTextToFramePadding();
        icons::inline_icon(icons::Icon::camera, theme::u32(theme::text_muted), 15.F);
        ImGui::SameLine(0.F, 6.F); ImGui::Text("%s %02d", tr("View"), selected + 1);
        ImGui::SameLine(0.F, 18.F); icons::inline_icon(icons::Icon::clock, theme::u32(theme::text_faint), 14.F);
        ImGui::SameLine(0.F, 5.F); ImGui::Text("%.2f s", k.time);
        if (width >= 480.F) {
            ImGui::SameLine(0.F, 18.F); ImGui::TextDisabled("%.0f° FOV", k.fov_degrees);
        }
        if (width >= 600.F) {
            ImGui::SameLine(0.F, 18.F); ImGui::TextDisabled("%s", tr("Smooth"));
        }
        const float edit_width = std::max(74.F, ImGui::CalcTextSize(tr("Edit")).x + 40.F);
        const float edit_x = std::max(ImGui::GetCursorPosX() + 8.F, ImGui::GetWindowContentRegionMax().x - edit_width);
        ImGui::SameLine(edit_x);
        if (icons::labeled_button("##edit", icons::Icon::edit, tr("Edit"), {edit_width, 28.F}))
            app.camera_path_edit = true;
        if (app.camera_path_edit) { ImGui::OpenPopup("##edit_view"); app.camera_path_edit = false; }
        ImGui::SetNextWindowSize({320.F, 0.F}, ImGuiCond_Appearing);
        if (ImGui::BeginPopup("##edit_view")) {
            ImGui::Text("%s %02d", tr("View"), selected + 1);
            ImGui::Separator(); ImGui::Spacing();
            ImGui::TextUnformatted(tr("Time (s)"));
            ImGui::SetNextItemWidth(-1.F);
            float t = k.time;
            if (ImGui::DragFloat("##keytime", &t, 0.02F, 0.F, 36000.F, "%.2f")) {
                k.time = camera_timeline::bounded_time(keys, static_cast<std::size_t>(selected), t);
                app.camera_path_time = k.time; app.camera_path_playing = false;
            }
            ImGui::TextUnformatted("FOV"); ImGui::SetNextItemWidth(-1.F);
            if (ImGui::DragFloat("##fov", &k.fov_degrees, 0.1F, 1.F, 179.F, "%.1f°")) {
                apply(app, k); app.camera_path_playing = false;
                k.thumbnail_rgb.clear(); k.thumbnail_width = k.thumbnail_height = 0;
                app.camera_thumbnail_pending = selected;
            }
            ImGui::Spacing();
            const float update_width = std::max(145.F, ImGui::CalcTextSize(tr("Update pose")).x + 40.F);
            if (icons::labeled_button("##update", icons::Icon::camera, tr("Update pose"), {update_width, 30.F})) {
                k = capture(app.camera, k.time); app.camera_path_playing = false;
                app.camera_thumbnail_pending = selected;
            }
            ImGui::SameLine();
            if (icons::button("##delete", icons::Icon::trash, {30.F, 30.F}, false, true, tr("Delete keyframe"))) {
                app.camera_key_delete_pending = selected; ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    } else ImGui::TextDisabled("%s", tr("Select a view to edit"));
    ImGui::End();
}
} // namespace editor
