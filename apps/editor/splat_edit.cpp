#include "splat_edit.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <vector>

namespace editor {
using i18n::tr;
namespace {

constexpr float k_pi = 3.14159265F;
constexpr std::size_t k_undo_limit = 12;
constexpr float k_min_opacity = 1.F / 255.F;
constexpr float k_click_px = 4.F;
constexpr float k_close_px = 14.F;

float activate_opacity(const float logit) {
    if (logit >= 0.F) {
        const float z = std::exp(-logit);
        return 1.F / (1.F + z);
    }
    const float z = std::exp(std::max(logit, -80.F));
    return z / (1.F + z);
}

struct RayHit {
    float u{};
    float v{};
    float depth{};
    bool valid{};
};

RayHit project_center(
    const splat_render::Camera& camera, const float x, const float y, const float z) {
    const float* m = camera.world_to_camera.data();
    const float cx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const float cy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const float cz = m[2] * x + m[6] * y + m[10] * z + m[14];
    RayHit hit;
    hit.depth = cz;
    if (camera.model == splat_render::k_camera_equirectangular) {
        const float len = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (!(len > 1e-8F)) return hit;
        const float azimuth = std::atan2(cx, cz);
        const float elevation = std::asin(std::clamp(cy / len, -1.F, 1.F));
        hit.u = (azimuth / (2.F * k_pi) + 0.5F) * static_cast<float>(camera.width);
        hit.v = (elevation / k_pi + 0.5F) * static_cast<float>(camera.height);
        hit.depth = len;
        hit.valid = true;
        return hit;
    }
    if (camera.model == splat_render::k_camera_orthographic) {
        if (!(cz > 1e-4F)) return hit;
        hit.u = camera.fx * cx + camera.cx;
        hit.v = camera.fy * cy + camera.cy;
        hit.valid = true;
        return hit;
    }
    if (camera.model == splat_render::k_camera_fisheye) {
        if (!(cz > 1e-8F)) return hit;
        const float radius = std::sqrt(cx * cx + cy * cy);
        const float theta = std::atan2(radius, cz);
        const float t2 = theta * theta;
        const float poly = 1.F + t2 * (camera.k1 + t2 * (camera.k2 + t2 * (camera.k3 + t2 * camera.k4)));
        const float theta_d = theta * poly;
        if (theta_d < 0.F) return hit;
        if (radius < 1e-12F) {
            hit.u = camera.cx;
            hit.v = camera.cy;
        } else {
            const float scale = theta_d / radius;
            hit.u = camera.fx * scale * cx + camera.cx;
            hit.v = camera.fy * scale * cy + camera.cy;
        }
        hit.valid = true;
        return hit;
    }
    if (!(cz > 1e-4F)) return hit;
    hit.u = camera.fx * cx / cz + camera.cx;
    hit.v = camera.fy * cy / cz + camera.cy;
    hit.valid = true;
    return hit;
}

ImVec2 raster_to_view(
    const splat_render::Camera& camera, ImVec2 view_min, ImVec2 view_max, float u, float v) {
    const float width = std::max(1.F, view_max.x - view_min.x);
    const float height = std::max(1.F, view_max.y - view_min.y);
    return {
        view_min.x + u / std::max(1U, camera.width) * width,
        view_min.y + v / std::max(1U, camera.height) * height};
}

ImVec2 view_to_raster(
    const splat_render::Camera& camera, ImVec2 view_min, ImVec2 view_max, ImVec2 screen) {
    const float width = std::max(1.F, view_max.x - view_min.x);
    const float height = std::max(1.F, view_max.y - view_min.y);
    return {
        (screen.x - view_min.x) / width * static_cast<float>(camera.width),
        (screen.y - view_min.y) / height * static_cast<float>(camera.height)};
}

bool inside_polygon(const std::vector<ImVec2>& polygon, ImVec2 point) {
    bool inside = false;
    const std::size_t count = polygon.size();
    for (std::size_t i = 0, j = count - 1; i < count; j = i++) {
        const ImVec2 a = polygon[i];
        const ImVec2 b = polygon[j];
        const bool crosses = (a.y > point.y) != (b.y > point.y);
        if (!crosses) continue;
        const float x = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
        if (point.x < x) inside = !inside;
    }
    return inside;
}

enum class SelectMode { replace, add, subtract };

SelectMode select_mode() {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.KeyAlt) return SelectMode::subtract;
    if (io.KeyShift) return SelectMode::add;
    return SelectMode::replace;
}

ImU32 gesture_stroke() {
    const SelectMode mode = select_mode();
    if (mode == SelectMode::add) return theme::u32(theme::success);
    if (mode == SelectMode::subtract) return theme::u32(theme::danger);
    return IM_COL32(236, 238, 242, 230);
}

ImU32 gesture_fill() {
    const SelectMode mode = select_mode();
    if (mode == SelectMode::add) return theme::u32(theme::success, 0.18F);
    if (mode == SelectMode::subtract) return theme::u32(theme::danger, 0.18F);
    return IM_COL32(236, 238, 242, 40);
}

void show_tip(const char* title, const char* shortcut, const char* detail) {
    if (shortcut != nullptr && shortcut[0] != '\0')
        ImGui::SetTooltip("%s    %s\n%s", tr(title), shortcut, tr(detail));
    else
        ImGui::SetTooltip("%s\n%s", tr(title), tr(detail));
}

struct ScreenHit {
    std::uint32_t index{};
    float u{};
    float v{};
    float depth{};
};

std::uint64_t depth_cell(const float u, const float v) {
    constexpr float k_cell = 10.F;
    const auto x = static_cast<std::uint32_t>(static_cast<int>(std::floor(u / k_cell)));
    const auto y = static_cast<std::uint32_t>(static_cast<int>(std::floor(v / k_cell)));
    return (static_cast<std::uint64_t>(x) << 32) | y;
}

void keep_front_surface(std::vector<ScreenHit>& hits) {
    if (hits.size() < 2) return;
    std::unordered_map<std::uint64_t, float> nearest;
    nearest.reserve(hits.size());
    for (const ScreenHit& hit : hits) {
        const std::uint64_t key = depth_cell(hit.u, hit.v);
        const auto found = nearest.find(key);
        if (found == nearest.end() || hit.depth < found->second)
            nearest[key] = hit.depth;
    }
    hits.erase(
        std::remove_if(
            hits.begin(), hits.end(),
            [&](const ScreenHit& hit) {
                const float front = nearest[depth_cell(hit.u, hit.v)];
                return hit.depth > front * 1.08F + 1e-4F;
            }),
        hits.end());
}

}  // namespace

void SplatEdit::clear() {
    key_.clear();
    count_ = 0;
    selected_count_ = 0;
    centers_.clear();
    selected_.clear();
    polygon_.clear();
    undo_.clear();
    redo_.clear();
    tool_ = Tool::none;
    stroking_ = false;
    stroke_changed_ = false;
    stroke_moved_ = false;
}

void SplatEdit::sync(
    const std::string& key, const float* means, const float* opacity_logits,
    const std::uint32_t count) {
    if (bound_to(key)) return;
    clear();
    if (count == 0 || means == nullptr || opacity_logits == nullptr) return;
    key_ = key;
    count_ = count;
    centers_.resize(static_cast<std::size_t>(count) * 4U);
    selected_.assign(count, 0);
    for (std::uint32_t index = 0; index < count; ++index) {
        centers_[static_cast<std::size_t>(index) * 4U] = means[index * 3U];
        centers_[static_cast<std::size_t>(index) * 4U + 1U] = means[index * 3U + 1U];
        centers_[static_cast<std::size_t>(index) * 4U + 2U] = means[index * 3U + 2U];
        centers_[static_cast<std::size_t>(index) * 4U + 3U] =
            activate_opacity(opacity_logits[index]);
    }
}

void SplatEdit::push_undo(Snapshot snapshot) {
    undo_.push_back(std::move(snapshot));
    if (undo_.size() > k_undo_limit) undo_.erase(undo_.begin());
    redo_.clear();
}

void SplatEdit::restore(App& app, const Snapshot& snapshot) {
    if (snapshot.selected.size() == selected_.size()) {
        selected_ = snapshot.selected;
        selected_count_ = 0;
        for (const std::uint8_t bit : selected_)
            selected_count_ += bit ? 1U : 0U;
    }
    if (snapshot.xyz.size() == static_cast<std::size_t>(count_) * 3U) {
        for (std::uint32_t index = 0; index < count_; ++index) {
            centers_[static_cast<std::size_t>(index) * 4U] =
                snapshot.xyz[static_cast<std::size_t>(index) * 3U];
            centers_[static_cast<std::size_t>(index) * 4U + 1U] =
                snapshot.xyz[static_cast<std::size_t>(index) * 3U + 1U];
            centers_[static_cast<std::size_t>(index) * 4U + 2U] =
                snapshot.xyz[static_cast<std::size_t>(index) * 3U + 2U];
        }
        upload(app);
    }
}

void SplatEdit::upload(App& app) {
    if (count_ == 0) return;
    app.splat_renderer.update_centers(centers_.data(), count_);
}

void SplatEdit::begin_stroke(const bool capture_positions) {
    if (stroking_) return;
    stroke_before_.selected = selected_;
    stroke_before_.xyz.clear();
    if (capture_positions) {
        stroke_before_.xyz.resize(static_cast<std::size_t>(count_) * 3U);
        for (std::uint32_t index = 0; index < count_; ++index) {
            stroke_before_.xyz[static_cast<std::size_t>(index) * 3U] =
                centers_[static_cast<std::size_t>(index) * 4U];
            stroke_before_.xyz[static_cast<std::size_t>(index) * 3U + 1U] =
                centers_[static_cast<std::size_t>(index) * 4U + 1U];
            stroke_before_.xyz[static_cast<std::size_t>(index) * 3U + 2U] =
                centers_[static_cast<std::size_t>(index) * 4U + 2U];
        }
    }
    stroking_ = true;
    stroke_changed_ = false;
    stroke_moved_ = false;
    stroke_origin_ = ImGui::GetIO().MousePos;
}

void SplatEdit::end_stroke(App& app) {
    if (!stroking_) return;
    stroking_ = false;
    const bool selection_changed = stroke_before_.selected.size() == selected_.size() &&
                                   stroke_before_.selected != selected_;
    if (!selection_changed && !stroke_moved_) {
        stroke_changed_ = false;
        return;
    }
    if (stroke_moved_) upload(app);
    push_undo(std::move(stroke_before_));
}

void SplatEdit::select_at(
    const splat_render::Camera& camera, const ImVec2 view_min, const ImVec2 view_max,
    const ImVec2 mouse, const bool replace_first) {
    const SelectMode mode = select_mode();
    if (replace_first && mode == SelectMode::replace) {
        std::fill(selected_.begin(), selected_.end(), 0);
        selected_count_ = 0;
        stroke_changed_ = true;
    }
    const float view_span = std::max(1.F, view_max.x - view_min.x);
    const float pick_px = 12.F * static_cast<float>(camera.width) / view_span;
    const float gate = pick_px * pick_px;
    float best_screen = gate;
    float best_depth = 1e30F;
    int best_index = -1;
    const ImVec2 raster = view_to_raster(camera, view_min, view_max, mouse);
    for (std::uint32_t index = 0; index < count_; ++index) {
        const float* center = centers_.data() + static_cast<std::size_t>(index) * 4U;
        if (center[3] < k_min_opacity) continue;
        const RayHit hit = project_center(camera, center[0], center[1], center[2]);
        if (!hit.valid) continue;
        const float du = hit.u - raster.x;
        const float dv = hit.v - raster.y;
        const float d2 = du * du + dv * dv;
        if (d2 > gate) continue;
        const bool closer_front = front_only_ && hit.depth < best_depth;
        const bool closer_cursor = !front_only_ && d2 < best_screen;
        if (best_index < 0 || closer_front || closer_cursor) {
            best_index = static_cast<int>(index);
            best_screen = d2;
            best_depth = hit.depth;
        }
    }
    if (best_index < 0) return;
    std::uint8_t& bit = selected_[static_cast<std::size_t>(best_index)];
    if (mode == SelectMode::subtract) {
        if (bit) {
            bit = 0;
            --selected_count_;
            stroke_changed_ = true;
        }
    } else if (!bit) {
        bit = 1;
        ++selected_count_;
        stroke_changed_ = true;
    }
}

void SplatEdit::apply_region(
    const splat_render::Camera& camera, const ImVec2 view_min, const ImVec2 view_max,
    const int mode, const ImVec2 a, const ImVec2 b) {
    const SelectMode op = select_mode();
    const ImVec2 ra = view_to_raster(camera, view_min, view_max, a);
    const ImVec2 rb = view_to_raster(camera, view_min, view_max, b);
    const float view_w = std::max(1.F, view_max.x - view_min.x);
    const float raster_scale =
        static_cast<float>(camera.width) / view_w;
    std::vector<ImVec2> raster_poly;
    if (mode == 3) {
        raster_poly.reserve(polygon_.size());
        for (const ImVec2& point : polygon_)
            raster_poly.push_back(view_to_raster(camera, view_min, view_max, point));
    }
    std::vector<ScreenHit> hits;
    hits.reserve(256);
    for (std::uint32_t index = 0; index < count_; ++index) {
        const float* center = centers_.data() + static_cast<std::size_t>(index) * 4U;
        if (center[3] < k_min_opacity) continue;
        const RayHit hit = project_center(camera, center[0], center[1], center[2]);
        if (!hit.valid) continue;
        bool inside = false;
        if (mode == 0) {
            const float min_x = std::min(ra.x, rb.x);
            const float max_x = std::max(ra.x, rb.x);
            const float min_y = std::min(ra.y, rb.y);
            const float max_y = std::max(ra.y, rb.y);
            inside = hit.u >= min_x && hit.u <= max_x && hit.v >= min_y && hit.v <= max_y;
        } else if (mode == 1) {
            const float radius = std::sqrt(
                (rb.x - ra.x) * (rb.x - ra.x) + (rb.y - ra.y) * (rb.y - ra.y));
            const float du = hit.u - ra.x;
            const float dv = hit.v - ra.y;
            inside = du * du + dv * dv <= radius * radius;
        } else if (mode == 2) {
            const float du = hit.u - rb.x;
            const float dv = hit.v - rb.y;
            const float radius = brush_radius_ * raster_scale;
            inside = du * du + dv * dv <= radius * radius;
        } else if (raster_poly.size() >= 3) {
            inside = inside_polygon(raster_poly, {hit.u, hit.v});
        }
        if (!inside) continue;
        hits.push_back(ScreenHit{index, hit.u, hit.v, hit.depth});
    }
    if (front_only_) keep_front_surface(hits);
    for (const ScreenHit& hit : hits) {
        std::uint8_t& bit = selected_[hit.index];
        if (op == SelectMode::subtract) {
            if (bit) {
                bit = 0;
                --selected_count_;
                stroke_changed_ = true;
            }
        } else if (!bit) {
            bit = 1;
            ++selected_count_;
            stroke_changed_ = true;
        }
    }
}

void SplatEdit::focus_selection(App& app) const {
    if (count_ == 0) return;
    double sx = 0, sy = 0, sz = 0;
    std::uint32_t used = 0;
    float max_r = 0.F;
    for (std::uint32_t index = 0; index < count_; ++index) {
        if (selected_count_ > 0 && !selected_[index]) continue;
        const float* c = centers_.data() + static_cast<std::size_t>(index) * 4U;
        sx += c[0];
        sy += c[1];
        sz += c[2];
        ++used;
    }
    if (used == 0) return;
    const Vec3 centre{
        static_cast<float>(sx / used), static_cast<float>(sy / used),
        static_cast<float>(sz / used)};
    for (std::uint32_t index = 0; index < count_; ++index) {
        if (selected_count_ > 0 && !selected_[index]) continue;
        const float* c = centers_.data() + static_cast<std::size_t>(index) * 4U;
        const float dx = c[0] - centre.x;
        const float dy = c[1] - centre.y;
        const float dz = c[2] - centre.z;
        max_r = std::max(max_r, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    app.camera.frame(centre, std::max(max_r, 1e-3F));
}

void SplatEdit::recount() {
    selected_count_ = 0;
    for (const std::uint8_t bit : selected_)
        selected_count_ += bit ? 1U : 0U;
}

std::vector<float> SplatEdit::capture_xyz() const {
    std::vector<float> xyz(static_cast<std::size_t>(count_) * 3U);
    for (std::uint32_t index = 0; index < count_; ++index) {
        xyz[static_cast<std::size_t>(index) * 3U] =
            centers_[static_cast<std::size_t>(index) * 4U];
        xyz[static_cast<std::size_t>(index) * 3U + 1U] =
            centers_[static_cast<std::size_t>(index) * 4U + 1U];
        xyz[static_cast<std::size_t>(index) * 3U + 2U] =
            centers_[static_cast<std::size_t>(index) * 4U + 2U];
    }
    return xyz;
}

void SplatEdit::undo(App& app) {
    if (undo_.empty() || stroking_) return;
    Snapshot current{
        selected_, undo_.back().xyz.empty() ? std::vector<float>{} : capture_xyz()};
    restore(app, undo_.back());
    redo_.push_back(std::move(current));
    undo_.pop_back();
}

void SplatEdit::redo(App& app) {
    if (redo_.empty() || stroking_) return;
    Snapshot current{
        selected_, redo_.back().xyz.empty() ? std::vector<float>{} : capture_xyz()};
    restore(app, redo_.back());
    undo_.push_back(std::move(current));
    redo_.pop_back();
}

void SplatEdit::abort_stroke(App& app) {
    if (!stroking_) return;
    if (stroke_before_.selected.size() == selected_.size()) {
        selected_ = stroke_before_.selected;
        recount();
    }
    if (stroke_moved_ &&
        stroke_before_.xyz.size() == static_cast<std::size_t>(count_) * 3U) {
        for (std::uint32_t index = 0; index < count_; ++index) {
            centers_[static_cast<std::size_t>(index) * 4U] =
                stroke_before_.xyz[static_cast<std::size_t>(index) * 3U];
            centers_[static_cast<std::size_t>(index) * 4U + 1U] =
                stroke_before_.xyz[static_cast<std::size_t>(index) * 3U + 1U];
            centers_[static_cast<std::size_t>(index) * 4U + 2U] =
                stroke_before_.xyz[static_cast<std::size_t>(index) * 3U + 2U];
        }
        upload(app);
    }
    stroking_ = false;
    stroke_changed_ = false;
    stroke_moved_ = false;
}

void SplatEdit::set_tool(App& app, const Tool tool) {
    if (stroking_) {
        if (tool_ == Tool::brush || tool_ == Tool::move) end_stroke(app);
        else abort_stroke(app);
    }
    if (tool != Tool::polygon) polygon_.clear();
    tool_ = tool;
}

void SplatEdit::toggle_tool(App& app, const Tool tool) {
    set_tool(app, tool_ == tool ? Tool::none : tool);
}

void SplatEdit::clear_selection() {
    if (selected_count_ == 0 || stroking_) return;
    push_undo(Snapshot{selected_, {}});
    std::fill(selected_.begin(), selected_.end(), 0);
    selected_count_ = 0;
}

void SplatEdit::select_all() {
    if (count_ == 0 || stroking_ || selected_count_ == count_) return;
    push_undo(Snapshot{selected_, {}});
    std::fill(selected_.begin(), selected_.end(), 1);
    selected_count_ = count_;
}

void SplatEdit::finish_gesture(
    App& app, const splat_render::Camera& camera, const ImVec2 view_min,
    const ImVec2 view_max, const ImVec2 mouse) {
    if (!stroking_) return;
    const float dx = mouse.x - stroke_origin_.x;
    const float dy = mouse.y - stroke_origin_.y;
    const bool drag = dx * dx + dy * dy > k_click_px * k_click_px;
    if (tool_ == Tool::pick) {
        if (drag) {
            if (select_mode() == SelectMode::replace) {
                std::fill(selected_.begin(), selected_.end(), 0);
                selected_count_ = 0;
            }
            apply_region(camera, view_min, view_max, 0, stroke_origin_, mouse);
        } else {
            select_at(camera, view_min, view_max, mouse, true);
        }
    } else if (tool_ == Tool::circle && drag) {
        if (select_mode() == SelectMode::replace) {
            std::fill(selected_.begin(), selected_.end(), 0);
            selected_count_ = 0;
        }
        apply_region(camera, view_min, view_max, 1, stroke_origin_, mouse);
    }
    end_stroke(app);
}

void SplatEdit::handle_keys(App& app) {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    const bool mouse_busy = ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                            ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                            ImGui::IsMouseDown(ImGuiMouseButton_Middle);

    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false) && !app.job.running()) {
        if (stroking_) {
            abort_stroke(app);
            return;
        }
        if (!polygon_.empty()) {
            polygon_.clear();
            return;
        }
        if (tool_ != Tool::none) {
            tool_ = Tool::none;
            return;
        }
        clear_selection();
        return;
    }

    if (io.KeyCtrl && !io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Z, true)) {
        if (io.KeyShift) redo(app);
        else undo(app);
        return;
    }
    if (io.KeyCtrl && !io.KeyShift && !io.KeyAlt &&
        ImGui::IsKeyPressed(ImGuiKey_Y, true)) {
        redo(app);
        return;
    }
    if (io.KeyCtrl && !io.KeyAlt && !mouse_busy &&
        ImGui::IsKeyPressed(ImGuiKey_A, false)) {
        if (io.KeyShift) clear_selection();
        else select_all();
        return;
    }

    if (tool_ == Tool::brush && !io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true))
            brush_radius_ = std::max(4.F, brush_radius_ - 4.F);
        if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, true))
            brush_radius_ = std::min(180.F, brush_radius_ + 4.F);
    }

    if (mouse_busy || stroking_ || io.KeyCtrl || io.KeyAlt || io.KeyShift) return;
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) toggle_tool(app, Tool::pick);
    else if (ImGui::IsKeyPressed(ImGuiKey_C, false)) toggle_tool(app, Tool::circle);
    else if (ImGui::IsKeyPressed(ImGuiKey_P, false)) toggle_tool(app, Tool::polygon);
    else if (ImGui::IsKeyPressed(ImGuiKey_B, false)) toggle_tool(app, Tool::brush);
    else if (ImGui::IsKeyPressed(ImGuiKey_N, false)) front_only_ = !front_only_;
}

bool SplatEdit::consume_alt_wheel(const bool pointer_in_view) {
    if (tool_ != Tool::brush || !pointer_in_view) return false;
    ImGuiIO& io = ImGui::GetIO();
    if (!io.KeyAlt || io.KeyCtrl || io.MouseWheel == 0.F) return false;
    brush_radius_ = std::clamp(brush_radius_ + io.MouseWheel * 8.F, 4.F, 180.F);
    return true;
}

void SplatEdit::write_status(char* buffer, const std::size_t size) const {
    if (buffer == nullptr || size == 0) return;
    buffer[0] = '\0';
    if (count_ == 0) return;
    if (tool_ == Tool::none) {
        const char* orbit = tr(
            "LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  F frame  |  double-click focus");
        if (selected_count_ > 0)
            std::snprintf(
                buffer, size, "%s  |  %s", orbit, tr("Esc clears the selection"));
        else
            std::snprintf(buffer, size, "%s", orbit);
        return;
    }
    const char* gesture = tr("Click or drag a box");
    if (tool_ == Tool::circle) gesture = tr("Drag a circle");
    else if (tool_ == Tool::polygon) gesture = tr("Click points, Enter closes");
    else if (tool_ == Tool::brush) gesture = tr("Paint across Gaussians");
    std::snprintf(
        buffer, size, "%s  |  %s  |  %s",
        tr(front_only_ ? "Front surface" : "All depths"), gesture,
        tr("Shift add  |  Alt remove  |  RMB orbit  |  Esc steps back"));
}

bool SplatEdit::draw_toolbar(App& app, const ImVec2 view_min, const ImVec2 view_max) {
    toolbar_visible_ = false;
    if (count_ == 0) return false;
    struct Item {
        icons::Icon icon;
        const char* id;
        const char* tip;
        const char* detail;
        const char* shortcut;
        int action;
        bool gap;
    };
    // 0 undo, 1 redo, 2 navigate, 3+ selection tool, 20 front depth, 21 all depths.
    const Item items[] = {
        {icons::Icon::undo, "##splat_undo", "Undo", "Undo the last selection.", "Ctrl+Z", 0, false},
        {icons::Icon::redo, "##splat_redo", "Redo", "Redo the last selection.", "Ctrl+Shift+Z", 1, false},
        {icons::Icon::select, "##splat_orbit", "Navigate",
         "Left button orbits. A click does not select until a selection tool is on.",
         "Esc", 2, true},
        {icons::Icon::marquee, "##splat_pick", "Pick",
         "Click one Gaussian, or drag a rectangle. Shift adds and Alt removes.",
         "R", 3, true},
        {icons::Icon::circle_select, "##splat_circle", "Circle select",
         "Drag out a circle. A click without dragging leaves the selection unchanged.",
         "C", 4, false},
        {icons::Icon::polygon_select, "##splat_poly", "Polygon select",
         "Click to add corners. Close on the first point or press Enter. Backspace removes the last corner.",
         "P", 5, false},
        {icons::Icon::brush, "##splat_brush", "Brush select",
         "Paint over Gaussians. [ ] or Alt+wheel changes the brush size.",
         "B", 6, false},
        {icons::Icon::depth_front, "##splat_front", "Front surface",
         "The next gesture keeps only the nearest Gaussian in each small screen cell.",
         "N", 20, true},
        {icons::Icon::depth_through, "##splat_through", "All depths",
         "The next gesture selects every layer it covers, including Gaussians hidden behind the front one.",
         "N", 21, false},
    };
    constexpr float k_button = 34.F;
    constexpr float k_gap = 4.F;
    constexpr float k_pad = 6.F;
    constexpr float k_group = 16.F;
    float content_w = 0.F;
    for (const Item& item : items)
        content_w += k_button + k_gap + (item.gap ? k_group : 0.F);
    content_w -= k_gap;
    const float bar_w = k_pad * 2.F + content_w;
    const float bar_h = k_pad * 2.F + k_button;
    const float left = view_min.x + (view_max.x - view_min.x - bar_w) * 0.5F;
    const float top = view_min.y + 14.F;
    toolbar_min_ = {left, top};
    toolbar_max_ = {left + bar_w, top + bar_h};
    toolbar_visible_ = true;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        {toolbar_min_.x + 1.F, toolbar_min_.y + 3.F},
        {toolbar_max_.x + 1.F, toolbar_max_.y + 3.F},
        IM_COL32(0, 0, 0, 70), 9.F);
    draw->AddRectFilled(
        toolbar_min_, toolbar_max_, IM_COL32(22, 24, 28, 236), 9.F);
    draw->AddRect(
        toolbar_min_, toolbar_max_, IM_COL32(255, 255, 255, 18), 9.F);

    const Tool tools[] = {
        Tool::pick, Tool::circle, Tool::polygon, Tool::brush};
    float cursor_x = left + k_pad;
    for (int index = 0; index < static_cast<int>(std::size(items)); ++index) {
        const Item& item = items[index];
        if (item.gap) {
            const float separator_x = cursor_x + k_group * 0.5F;
            draw->AddLine(
                {separator_x, top + k_pad + 7.F},
                {separator_x, top + k_pad + k_button - 7.F},
                IM_COL32(255, 255, 255, 36), 1.F);
            cursor_x += k_group;
        }
        const ImVec2 min{cursor_x, top + k_pad};
        const ImVec2 max{min.x + k_button, min.y + k_button};
        cursor_x += k_button + k_gap;
        const bool enabled = item.action == 0 ? !undo_.empty()
            : item.action == 1 ? !redo_.empty()
                               : true;
        const bool active = item.action == 2 ? tool_ == Tool::none
            : item.action >= 3 && item.action <= 6
                ? tool_ == tools[item.action - 3]
            : item.action == 20 ? front_only_
            : item.action == 21 ? !front_only_
                                : false;
        ImGui::SetCursorScreenPos(min);
        ImGui::PushID(item.id);
        const bool pressed = ImGui::InvisibleButton(item.id, {k_button, k_button});
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        ImU32 fill = theme::u32(theme::surface_3);
        if (!enabled) fill = theme::u32(theme::fade(theme::surface_3, 0.45F));
        else if (active) fill = theme::u32(theme::fade(theme::accent, 0.24F));
        else if (hovered) fill = IM_COL32(58, 62, 72, 255);
        draw->AddRectFilled(min, max, fill, 6.F);
        if (enabled && active)
            draw->AddRect(min, max, theme::u32(theme::accent, 0.75F), 6.F, 0, 1.F);
        const ImU32 icon_colour = theme::u32(
            !enabled ? theme::text_faint
                     : active ? theme::accent
                     : hovered ? theme::text_bright
                               : theme::text_muted);
        icons::draw(
            draw, item.icon,
            {min.x + 7.F, min.y + 7.F}, {max.x - 7.F, max.y - 7.F}, icon_colour, 1.5F);
        if (hovered) {
            if (enabled) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            show_tip(item.tip, item.shortcut, item.detail);
        }
        if (!pressed || !enabled) continue;
        if (item.action == 0) undo(app);
        else if (item.action == 1) redo(app);
        else if (item.action == 2) set_tool(app, Tool::none);
        else if (item.action >= 3 && item.action <= 6)
            toggle_tool(app, tools[item.action - 3]);
        else if (item.action == 20) front_only_ = true;
        else if (item.action == 21) front_only_ = false;
    }
    return ImGui::IsMouseHoveringRect(toolbar_min_, toolbar_max_, false);
}

void SplatEdit::draw_overlay(
    ImDrawList* draw, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera) const {
    if (count_ == 0 || draw == nullptr) return;
    draw->PushClipRect(view_min, view_max, true);
    const std::uint32_t stride = std::max(1U, selected_count_ / 8000U);
    std::uint32_t seen = 0;
    const ImU32 mark = theme::u32(theme::accent);
    for (std::uint32_t index = 0; index < count_; ++index) {
        if (!selected_[index]) continue;
        if ((seen++ % stride) != 0) continue;
        const float* center = centers_.data() + static_cast<std::size_t>(index) * 4U;
        const RayHit hit = project_center(camera, center[0], center[1], center[2]);
        if (!hit.valid) continue;
        const ImVec2 screen = raster_to_view(camera, view_min, view_max, hit.u, hit.v);
        if (screen.x < view_min.x || screen.x > view_max.x || screen.y < view_min.y ||
            screen.y > view_max.y)
            continue;
        draw->AddCircleFilled(screen, 3.1F, IM_COL32(6, 10, 16, 170));
        draw->AddCircleFilled(screen, 1.8F, mark);
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool in_view = mouse.x >= view_min.x && mouse.x < view_max.x &&
                         mouse.y >= view_min.y && mouse.y < view_max.y;
    const bool over_bar = toolbar_visible_ &&
                          mouse.x >= toolbar_min_.x && mouse.x <= toolbar_max_.x &&
                          mouse.y >= toolbar_min_.y && mouse.y <= toolbar_max_.y;
    // Left mode rail: pad 10, top 52, 44px wide, through the scene toggle.
    const bool over_rail = mouse.x <= view_min.x + 62.F &&
                           mouse.y >= view_min.y + 48.F &&
                           mouse.y <= view_min.y + 340.F;
    const ImU32 stroke = gesture_stroke();
    const ImU32 fill = gesture_fill();
    const bool show_cursor = tool_ != Tool::none && in_view && !over_bar && !over_rail;
    if (show_cursor) ImGui::SetMouseCursor(ImGuiMouseCursor_None);

    if (tool_ == Tool::brush && (show_cursor || stroking_)) {
        draw->AddCircleFilled(mouse, brush_radius_, fill, 48);
        draw->AddCircle(mouse, brush_radius_, IM_COL32(0, 0, 0, 90), 48, 2.4F);
        draw->AddCircle(mouse, brush_radius_, stroke, 48, 1.25F);
        char label[16];
        std::snprintf(label, sizeof(label), "%.0f", brush_radius_);
        if (ImFont* font = theme::small_font())
            draw->AddText(
                font, font->FontSize, {mouse.x + 12.F, mouse.y + 8.F},
                theme::u32(theme::text_bright), label);
    } else if (show_cursor &&
               (tool_ == Tool::pick || tool_ == Tool::circle || tool_ == Tool::polygon)) {
        constexpr float k_gap = 4.F;
        constexpr float k_arm = 9.F;
        draw->AddLine({mouse.x - k_arm, mouse.y}, {mouse.x - k_gap, mouse.y}, stroke, 1.25F);
        draw->AddLine({mouse.x + k_gap, mouse.y}, {mouse.x + k_arm, mouse.y}, stroke, 1.25F);
        draw->AddLine({mouse.x, mouse.y - k_arm}, {mouse.x, mouse.y - k_gap}, stroke, 1.25F);
        draw->AddLine({mouse.x, mouse.y + k_gap}, {mouse.x, mouse.y + k_arm}, stroke, 1.25F);
    }

    if (stroking_ && tool_ == Tool::pick) {
        const ImVec2 box_min{std::min(stroke_origin_.x, mouse.x), std::min(stroke_origin_.y, mouse.y)};
        const ImVec2 box_max{std::max(stroke_origin_.x, mouse.x), std::max(stroke_origin_.y, mouse.y)};
        draw->AddRectFilled(box_min, box_max, fill);
        draw->AddRect(box_min, box_max, IM_COL32(0, 0, 0, 110), 0.F, 0, 2.4F);
        draw->AddRect(box_min, box_max, stroke, 0.F, 0, 1.25F);
    }
    if (stroking_ && tool_ == Tool::circle) {
        const float radius = std::sqrt(
            (mouse.x - stroke_origin_.x) * (mouse.x - stroke_origin_.x) +
            (mouse.y - stroke_origin_.y) * (mouse.y - stroke_origin_.y));
        draw->AddCircleFilled(stroke_origin_, radius, fill, 48);
        draw->AddCircle(stroke_origin_, radius, IM_COL32(0, 0, 0, 110), 48, 2.4F);
        draw->AddCircle(stroke_origin_, radius, stroke, 48, 1.25F);
    }
    if (tool_ == Tool::polygon && !polygon_.empty()) {
        const float dx = mouse.x - polygon_.front().x;
        const float dy = mouse.y - polygon_.front().y;
        const bool closable =
            polygon_.size() >= 3 && dx * dx + dy * dy < k_close_px * k_close_px;
        const ImVec2 tip = closable ? polygon_.front() : mouse;
        for (std::size_t i = 1; i < polygon_.size(); ++i)
            draw->AddLine(polygon_[i - 1], polygon_[i], stroke, 1.35F);
        draw->AddLine(polygon_.back(), tip, stroke, 1.15F);
        for (std::size_t i = 0; i < polygon_.size(); ++i) {
            const bool first = i == 0;
            const float radius = first && closable ? 6.F : 3.2F;
            draw->AddCircleFilled(polygon_[i], radius + 1.4F, IM_COL32(6, 10, 16, 160));
            draw->AddCircleFilled(polygon_[i], radius, first && closable ? mark : stroke);
        }
    }
    draw->PopClipRect();
}

void SplatEdit::handle_pointer(
    App& app, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera, const bool viewport_hot,
    const bool over_toolbar) {
    if (count_ == 0) return;
    const ImGuiIO& io = ImGui::GetIO();
    const bool keys_hot = viewport_hot && !io.WantTextInput;
    const bool hot = keys_hot && !over_toolbar;
    if (keys_hot) handle_keys(app);

    if (tool_ == Tool::none) {
        if (stroking_) abort_stroke(app);
        return;
    }

    const bool click = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    const bool down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    const bool release = ImGui::IsMouseReleased(ImGuiMouseButton_Left);
    const ImVec2 mouse = io.MousePos;

    const auto close_polygon = [&]() {
        if (polygon_.size() < 3 || stroking_) return;
        begin_stroke(false);
        if (select_mode() == SelectMode::replace) {
            std::fill(selected_.begin(), selected_.end(), 0);
            selected_count_ = 0;
        }
        apply_region(camera, view_min, view_max, 3, {}, {});
        end_stroke(app);
        polygon_.clear();
    };

    if (tool_ == Tool::polygon && keys_hot) {
        if (ImGui::IsKeyPressed(ImGuiKey_Backspace, false) && !polygon_.empty())
            polygon_.pop_back();
        if (polygon_.size() >= 3 &&
            (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
             ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)))
            close_polygon();
    }

    if (tool_ == Tool::polygon) {
        if (!(hot && click)) return;
        if (polygon_.size() >= 3) {
            const float dx = mouse.x - polygon_.front().x;
            const float dy = mouse.y - polygon_.front().y;
            if (dx * dx + dy * dy < k_close_px * k_close_px) {
                close_polygon();
                return;
            }
        }
        polygon_.push_back(mouse);
        return;
    }

    if (hot && click && !stroking_) {
        begin_stroke(tool_ == Tool::move);
        if (tool_ == Tool::brush && select_mode() == SelectMode::replace) {
            std::fill(selected_.begin(), selected_.end(), 0);
            selected_count_ = 0;
            stroke_changed_ = true;
        }
    }
    if (!stroking_) return;

    if (tool_ == Tool::brush && (down || release)) {
        const ImVec2 raster = view_to_raster(camera, view_min, view_max, mouse);
        apply_region(
            camera, view_min, view_max, 2, {},
            raster_to_view(camera, view_min, view_max, raster.x, raster.y));
    }
    if (tool_ == Tool::move && (down || release) && selected_count_ > 0) {
        const float* m = camera.world_to_camera.data();
        const float right[3] = {m[0], m[4], m[8]};
        const float down_axis[3] = {m[1], m[5], m[9]};
        double sx = 0, sy = 0, sz = 0;
        for (std::uint32_t index = 0; index < count_; ++index) {
            if (!selected_[index]) continue;
            sx += centers_[index * 4U];
            sy += centers_[index * 4U + 1U];
            sz += centers_[index * 4U + 2U];
        }
        const float inv = 1.F / static_cast<float>(selected_count_);
        const float wx = static_cast<float>(sx * inv);
        const float wy = static_cast<float>(sy * inv);
        const float wz = static_cast<float>(sz * inv);
        const float cz = m[2] * wx + m[6] * wy + m[10] * wz + m[14];
        const float view_w = std::max(1.F, view_max.x - view_min.x);
        const float view_h = std::max(1.F, view_max.y - view_min.y);
        const float du = io.MouseDelta.x / view_w * static_cast<float>(camera.width);
        const float dv = io.MouseDelta.y / view_h * static_cast<float>(camera.height);
        float scale_x = 1.F / std::max(camera.fx, 1e-4F);
        float scale_y = 1.F / std::max(camera.fy, 1e-4F);
        if (camera.model != splat_render::k_camera_orthographic) {
            scale_x *= std::max(cz, 1e-3F);
            scale_y *= std::max(cz, 1e-3F);
        }
        const float world_dx = du * scale_x;
        const float world_dy = dv * scale_y;
        if (world_dx != 0.F || world_dy != 0.F) {
            for (std::uint32_t index = 0; index < count_; ++index) {
                if (!selected_[index]) continue;
                centers_[index * 4U] += right[0] * world_dx + down_axis[0] * world_dy;
                centers_[index * 4U + 1U] += right[1] * world_dx + down_axis[1] * world_dy;
                centers_[index * 4U + 2U] += right[2] * world_dx + down_axis[2] * world_dy;
            }
            stroke_moved_ = true;
            upload(app);
        }
    }
    if (release) finish_gesture(app, camera, view_min, view_max, mouse);
}

}  // namespace editor