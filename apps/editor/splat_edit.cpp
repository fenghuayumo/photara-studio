#include "splat_edit.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "theme.hpp"

#include "core/camera_projection.hpp"
#include "splat/types.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
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

void take_strided(
    std::vector<float>& values, const std::uint32_t stride,
    const std::vector<std::uint8_t>& drop, const std::uint32_t count,
    std::vector<float>& removed) {
    removed.clear();
    if (values.empty() || stride == 0 || count == 0) return;
    std::vector<float> kept;
    kept.reserve(values.size());
    removed.reserve(static_cast<std::size_t>(count) * stride / 4U);
    for (std::uint32_t index = 0; index < count; ++index) {
        const float* src =
            values.data() + static_cast<std::size_t>(index) * stride;
        if (index < drop.size() && drop[index])
            removed.insert(removed.end(), src, src + stride);
        else
            kept.insert(kept.end(), src, src + stride);
    }
    values.swap(kept);
}

void keep_strided(
    std::vector<float>& values, const std::uint32_t stride,
    const std::vector<std::uint8_t>& drop, const std::uint32_t count) {
    if (values.empty() || stride == 0 || count == 0) return;
    std::vector<float> kept;
    kept.reserve(values.size());
    for (std::uint32_t index = 0; index < count; ++index) {
        if (index < drop.size() && drop[index]) continue;
        const float* src =
            values.data() + static_cast<std::size_t>(index) * stride;
        kept.insert(kept.end(), src, src + stride);
    }
    values.swap(kept);
}

std::vector<float> splice_strided(
    const std::vector<float>& kept, const std::uint32_t stride,
    const std::vector<std::uint32_t>& index, const std::vector<float>& removed) {
    if (stride == 0 || (kept.empty() && removed.empty())) return {};
    const std::uint32_t removed_count =
        static_cast<std::uint32_t>(index.size());
    const std::uint32_t kept_count =
        static_cast<std::uint32_t>(kept.size() / stride);
    const std::uint32_t total = kept_count + removed_count;
    std::vector<float> out(static_cast<std::size_t>(total) * stride);
    std::uint32_t kept_index = 0;
    std::uint32_t removed_index = 0;
    for (std::uint32_t dst = 0; dst < total; ++dst) {
        const float* src = nullptr;
        if (removed_index < removed_count && index[removed_index] == dst) {
            src = removed.data() + static_cast<std::size_t>(removed_index) * stride;
            ++removed_index;
        } else if (kept_index < kept_count) {
            src = kept.data() + static_cast<std::size_t>(kept_index) * stride;
            ++kept_index;
        }
        if (src == nullptr) continue;
        std::copy(
            src, src + stride,
            out.begin() + static_cast<std::ptrdiff_t>(dst) * stride);
    }
    return out;
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

float dot3(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

float length3(const float v[3]) {
    return std::sqrt(dot3(v, v));
}

bool normalize3(float v[3]) {
    const float len = length3(v);
    if (len < 1e-8F) return false;
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
    return true;
}

void quat_axes(const float q[4], float axis[3][3]) {
    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];
    axis[0][0] = 1.F - 2.F * (y * y + z * z);
    axis[0][1] = 2.F * (x * y + w * z);
    axis[0][2] = 2.F * (x * z - w * y);
    axis[1][0] = 2.F * (x * y - w * z);
    axis[1][1] = 1.F - 2.F * (x * x + z * z);
    axis[1][2] = 2.F * (y * z + w * x);
    axis[2][0] = 2.F * (x * z + w * y);
    axis[2][1] = 2.F * (y * z - w * x);
    axis[2][2] = 1.F - 2.F * (x * x + y * y);
}

void quat_mul(const float a[4], const float b[4], float out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

void quat_from_axis(const float axis[3], const float radians, float out[4]) {
    const float s = std::sin(radians * 0.5F);
    out[0] = std::cos(radians * 0.5F);
    out[1] = axis[0] * s;
    out[2] = axis[1] * s;
    out[3] = axis[2] * s;
}

void camera_basis(
    const splat_render::Camera& camera, float right[3], float up[3],
    float forward[3]) {
    const float* m = camera.world_to_camera.data();
    right[0] = m[0];
    right[1] = m[4];
    right[2] = m[8];
    up[0] = m[1];
    up[1] = m[5];
    up[2] = m[9];
    forward[0] = m[2];
    forward[1] = m[6];
    forward[2] = m[10];
}

struct WorldRay {
    float origin[3]{};
    float direction[3]{};
    bool valid{};
};

WorldRay camera_ray(
    const splat_render::Camera& camera, const ImVec2 view_min, const ImVec2 view_max,
    const ImVec2 mouse) {
    WorldRay ray;
    ray.origin[0] = camera.position[0];
    ray.origin[1] = camera.position[1];
    ray.origin[2] = camera.position[2];
    float right[3];
    float up[3];
    float forward[3];
    camera_basis(camera, right, up, forward);
    const ImVec2 raster = view_to_raster(camera, view_min, view_max, mouse);
    const auto aim = [&](const float x, const float y, const float z) {
        const float wx = right[0] * x + up[0] * y + forward[0] * z;
        const float wy = right[1] * x + up[1] * y + forward[1] * z;
        const float wz = right[2] * x + up[2] * y + forward[2] * z;
        const float len = std::sqrt(wx * wx + wy * wy + wz * wz);
        if (len < 1e-8F) return false;
        ray.direction[0] = wx / len;
        ray.direction[1] = wy / len;
        ray.direction[2] = wz / len;
        return true;
    };
    if (camera.model == splat_render::k_camera_orthographic) {
        const float cx = (raster.x - camera.cx) / std::max(camera.fx, 1e-6F);
        const float cy = (raster.y - camera.cy) / std::max(camera.fy, 1e-6F);
        ray.origin[0] += right[0] * cx + up[0] * cy;
        ray.origin[1] += right[1] * cx + up[1] * cy;
        ray.origin[2] += right[2] * cx + up[2] * cy;
        ray.valid = aim(0.F, 0.F, 1.F);
        return ray;
    }
    if (camera.model == splat_render::k_camera_fisheye) {
        const auto local = photara::unproject_fisheye_camera(
            raster.x, raster.y, camera.fx, camera.fy, camera.cx, camera.cy,
            camera.k1, camera.k2, camera.k3, camera.k4);
        if (!local.valid) return ray;
        ray.valid = aim(static_cast<float>(local.x), static_cast<float>(local.y),
                        static_cast<float>(local.z));
        return ray;
    }
    if (camera.model == splat_render::k_camera_equirectangular) {
        const auto local = photara::unproject_equirectangular_camera(
            raster.x, raster.y, static_cast<int>(camera.width),
            static_cast<int>(camera.height));
        if (!local.valid) return ray;
        ray.valid = aim(static_cast<float>(local.x), static_cast<float>(local.y),
                        static_cast<float>(local.z));
        return ray;
    }
    const float cx = (raster.x - camera.cx) / std::max(camera.fx, 1e-6F);
    const float cy = (raster.y - camera.cy) / std::max(camera.fy, 1e-6F);
    ray.valid = aim(cx, cy, 1.F);
    return ray;
}

bool project_world(
    const splat_render::Camera& camera, const ImVec2 view_min, const ImVec2 view_max,
    const float point[3], ImVec2& screen, float& depth) {
    const RayHit hit = project_center(camera, point[0], point[1], point[2]);
    if (!hit.valid) return false;
    screen = raster_to_view(camera, view_min, view_max, hit.u, hit.v);
    depth = hit.depth;
    return true;
}

float world_from_screen(
    const splat_render::Camera& camera, const ImVec2 view_min, const ImVec2 view_max,
    const float depth, const float screen_px) {
    const float view_w = std::max(1.F, view_max.x - view_min.x);
    const float raster_px =
        screen_px * static_cast<float>(std::max(1U, camera.width)) / view_w;
    if (camera.model == splat_render::k_camera_orthographic)
        return raster_px / std::max(camera.fx, 1e-4F);
    return raster_px * std::max(depth, 1e-3F) / std::max(camera.fx, 1.F);
}

float dist2_segment(const ImVec2 p, const ImVec2 a, const ImVec2 b) {
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float ab2 = abx * abx + aby * aby;
    const float t = ab2 > 1e-6F
        ? std::clamp(((p.x - a.x) * abx + (p.y - a.y) * aby) / ab2, 0.F, 1.F)
        : 0.F;
    const float dx = p.x - (a.x + abx * t);
    const float dy = p.y - (a.y + aby * t);
    return dx * dx + dy * dy;
}

bool ray_plane(
    const WorldRay& ray, const float point[3], const float normal[3], float hit[3]) {
    const float denom = dot3(ray.direction, normal);
    if (std::fabs(denom) < 1e-6F) return false;
    const float rel[3] = {
        point[0] - ray.origin[0], point[1] - ray.origin[1], point[2] - ray.origin[2]};
    const float t = dot3(rel, normal) / denom;
    if (t < 0.F) return false;
    hit[0] = ray.origin[0] + ray.direction[0] * t;
    hit[1] = ray.origin[1] + ray.direction[1] * t;
    hit[2] = ray.origin[2] + ray.direction[2] * t;
    return true;
}

}  // namespace

void SplatEdit::clear() {
    key_.clear();
    synced_ = false;
    dirty_ = false;
    rings_hit_ = false;
    hit_chosen_ = false;
    count_ = 0;
    selected_count_ = 0;
    sh_degree_ = 0;
    sh_bases_ = 1;
    centers_.clear();
    log_scales_.clear();
    quaternions_.clear();
    opacity_.clear();
    sh_.clear();
    normals_.clear();
    filter_.clear();
    selected_.clear();
    polygon_.clear();
    undo_.clear();
    redo_.clear();
    tool_ = Tool::none;
    stroking_ = false;
    stroke_changed_ = false;
    stroke_moved_ = false;
    volume_placed_ = false;
    volume_drag_ = -1;
    volume_gesture_ = VolumeGesture::move;
    volume_quat_[0] = 1.F;
    volume_quat_[1] = volume_quat_[2] = volume_quat_[3] = 0.F;
}

void SplatEdit::sync(const std::string& key, const Host& host) {
    if (bound_to(key)) return;
    clear();
    if (host.count == 0 || host.means == nullptr || host.log_scales == nullptr ||
        host.quaternions == nullptr || host.opacity_logits == nullptr)
        return;
    const std::uint32_t count = host.count;
    const std::uint32_t bases = std::max(1U, host.sh_bases);
    key_ = key;
    synced_ = true;
    count_ = count;
    sh_degree_ = host.sh == nullptr ? 0U : host.sh_degree;
    sh_bases_ = host.sh == nullptr ? 1U : bases;
    centers_.resize(static_cast<std::size_t>(count) * 4U);
    log_scales_.assign(
        host.log_scales, host.log_scales + static_cast<std::size_t>(count) * 3U);
    quaternions_.assign(
        host.quaternions, host.quaternions + static_cast<std::size_t>(count) * 4U);
    opacity_.assign(host.opacity_logits, host.opacity_logits + count);
    if (host.sh != nullptr)
        sh_.assign(
            host.sh, host.sh + static_cast<std::size_t>(count) * sh_bases_ * 3U);
    if (host.normals != nullptr)
        normals_.assign(
            host.normals, host.normals + static_cast<std::size_t>(count) * 4U);
    if (host.filter_3d != nullptr)
        filter_.assign(host.filter_3d, host.filter_3d + count);
    selected_.assign(count, 0);
    for (std::uint32_t index = 0; index < count; ++index) {
        centers_[static_cast<std::size_t>(index) * 4U] = host.means[index * 3U];
        centers_[static_cast<std::size_t>(index) * 4U + 1U] =
            host.means[index * 3U + 1U];
        centers_[static_cast<std::size_t>(index) * 4U + 2U] =
            host.means[index * 3U + 2U];
        centers_[static_cast<std::size_t>(index) * 4U + 3U] =
            activate_opacity(host.opacity_logits[index]);
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

void SplatEdit::note_view(const bool rings_view, const float ring_scale) {
    if (!hit_chosen_) rings_hit_ = rings_view;
    if (ring_scale > 0.F) ring_sigma_ = ring_scale;
}

bool SplatEdit::rings_ready() const noexcept {
    return count_ > 0 &&
           log_scales_.size() >= static_cast<std::size_t>(count_) * 3U &&
           quaternions_.size() >= static_cast<std::size_t>(count_) * 4U;
}

float SplatEdit::ellipse_reach(
    const splat_render::Camera& camera, const float depth,
    const std::uint32_t index) const {
    if (!rings_ready()) return 0.F;
    const float* scale = log_scales_.data() + static_cast<std::size_t>(index) * 3U;
    const float sigma = std::exp(std::min(8.F, std::max(scale[0], std::max(scale[1], scale[2]))));
    const float focal = std::max(camera.fx, camera.fy);
    float reach = 1024.F;
    if (camera.model == splat_render::k_camera_orthographic)
        reach = sigma * focal * ring_sigma_;
    else if (camera.model != splat_render::k_camera_equirectangular)
        reach = sigma * focal / std::max(depth, 1e-3F) * ring_sigma_;
    // One axis underestimates the screen-edge Jacobian. Dilation adds a couple of pixels.
    reach = std::min(1024.F, reach * 2.F + ring_sigma_ * 2.F);
    return std::max(6.F, reach);
}

bool SplatEdit::project_ellipse(
    const splat_render::Camera& camera, const std::uint32_t index,
    ScreenEllipse& ellipse) const {
    ellipse = {};
    if (!rings_ready() || index >= count_) return false;
    const float* center = centers_.data() + static_cast<std::size_t>(index) * 4U;
    const RayHit origin = project_center(camera, center[0], center[1], center[2]);
    if (!origin.valid) return false;
    const float* scale = log_scales_.data() + static_cast<std::size_t>(index) * 3U;
    const float* rotation = quaternions_.data() + static_cast<std::size_t>(index) * 4U;
    const float sx = std::exp(std::clamp(scale[0], -12.F, 8.F));
    const float sy = std::exp(std::clamp(scale[1], -12.F, 8.F));
    const float sz = std::exp(std::clamp(scale[2], -12.F, 8.F));
    float qw = rotation[0];
    float qx = rotation[1];
    float qy = rotation[2];
    float qz = rotation[3];
    const float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn < 1e-8F) return false;
    qw /= qn;
    qx /= qn;
    qy /= qn;
    qz /= qn;
    const float xx = qx * qx;
    const float yy = qy * qy;
    const float zz = qz * qz;
    const float xy = qx * qy;
    const float xz = qx * qz;
    const float yz = qy * qz;
    const float wx = qw * qx;
    const float wy = qw * qy;
    const float wz = qw * qz;
    const float axis_x[3] = {
        (1.F - 2.F * (yy + zz)) * sx, (2.F * (xy + wz)) * sx, (2.F * (xz - wy)) * sx};
    const float axis_y[3] = {
        (2.F * (xy - wz)) * sy, (1.F - 2.F * (xx + zz)) * sy, (2.F * (yz + wx)) * sy};
    const float axis_z[3] = {
        (2.F * (xz + wy)) * sz, (2.F * (yz - wx)) * sz, (1.F - 2.F * (xx + yy)) * sz};

    const float* m = camera.world_to_camera.data();
    const float cam_x = m[0] * center[0] + m[4] * center[1] + m[8] * center[2] + m[12];
    const float cam_y = m[1] * center[0] + m[5] * center[1] + m[9] * center[2] + m[13];
    const float cam_z = m[2] * center[0] + m[6] * center[1] + m[10] * center[2] + m[14];
    float px[3]{};
    float py[3]{};
    const float* axes[3] = {axis_x, axis_y, axis_z};
    const bool linear = camera.model == splat_render::k_camera_pinhole ||
                        camera.model == splat_render::k_camera_orthographic;
    if (linear && cam_z > 1e-4F) {
        const float inv_z = 1.F / cam_z;
        const float inv_z2 = inv_z * inv_z;
        for (int axis = 0; axis < 3; ++axis) {
            const float ax = m[0] * axes[axis][0] + m[4] * axes[axis][1] + m[8] * axes[axis][2];
            const float ay = m[1] * axes[axis][0] + m[5] * axes[axis][1] + m[9] * axes[axis][2];
            const float az = m[2] * axes[axis][0] + m[6] * axes[axis][1] + m[10] * axes[axis][2];
            if (camera.model == splat_render::k_camera_orthographic) {
                px[axis] = camera.fx * ax;
                py[axis] = camera.fy * ay;
            } else {
                px[axis] = camera.fx * inv_z * ax - camera.fx * cam_x * inv_z2 * az;
                py[axis] = camera.fy * inv_z * ay - camera.fy * cam_y * inv_z2 * az;
            }
        }
    } else {
        const float width = static_cast<float>(std::max(1U, camera.width));
        for (int axis = 0; axis < 3; ++axis) {
            const RayHit end = project_center(
                camera, center[0] + axes[axis][0], center[1] + axes[axis][1],
                center[2] + axes[axis][2]);
            if (!end.valid) continue;
            float du = end.u - origin.u;
            float dv = end.v - origin.v;
            if (camera.model == splat_render::k_camera_equirectangular) {
                if (du > width * 0.5F) du -= width;
                if (du < -width * 0.5F) du += width;
            }
            px[axis] = du;
            py[axis] = dv;
        }
    }

    // Same contour as ring_prepare.cs.hlsl: 0.3px dilation, minor-axis floor,
    // and a uniform cap so a long splat keeps its shape.
    const float a = px[0] * px[0] + px[1] * px[1] + px[2] * px[2] + 0.3F;
    const float b = px[0] * py[0] + px[1] * py[1] + px[2] * py[2];
    const float c = py[0] * py[0] + py[1] * py[1] + py[2] * py[2] + 0.3F;
    const float mid = 0.5F * (a + c);
    const float extent =
        0.5F * std::sqrt(std::max(0.F, (a - c) * (a - c) + 4.F * b * b));
    const float lambda1 = std::max(0.F, mid + extent);
    const float lambda2 = std::max(0.1F, mid - extent);
    float rx = ring_sigma_ * std::sqrt(lambda1);
    float ry = ring_sigma_ * std::sqrt(lambda2);
    const float cap = std::min(
        1024.F, std::min(
                    static_cast<float>(std::max(1U, camera.width)),
                    static_cast<float>(std::max(1U, camera.height))));
    const float major = std::max(rx, ry);
    if (major > cap && major > 1e-4F) {
        const float fit = cap / major;
        rx *= fit;
        ry *= fit;
    }
    ellipse.u = origin.u;
    ellipse.v = origin.v;
    ellipse.depth = origin.depth;
    ellipse.rx = rx;
    ellipse.ry = ry;
    ellipse.rotation = 0.5F * std::atan2(2.F * b, a - c);
    ellipse.valid = major >= 0.5F;
    return ellipse.valid;
}

bool SplatEdit::ellipse_contains(const ScreenEllipse& ellipse, const float x, const float y) {
    if (!ellipse.valid || ellipse.rx < 1e-3F || ellipse.ry < 1e-3F) return false;
    const float dx = x - ellipse.u;
    const float dy = y - ellipse.v;
    const float c = std::cos(ellipse.rotation);
    const float s = std::sin(ellipse.rotation);
    const float local_x = c * dx + s * dy;
    const float local_y = -s * dx + c * dy;
    return (local_x * local_x) / (ellipse.rx * ellipse.rx) +
               (local_y * local_y) / (ellipse.ry * ellipse.ry) <=
           1.F;
}

bool SplatEdit::ellipse_hits(
    const ScreenEllipse& ellipse, const int mode, const ImVec2 ra, const ImVec2 rb,
    const std::vector<ImVec2>* polygon, const float radius) const {
    if (!ellipse.valid) return false;
    const auto rim = [&](const auto& accept) {
        const float c = std::cos(ellipse.rotation);
        const float s = std::sin(ellipse.rotation);
        for (int step = 0; step < 8; ++step) {
            const float angle = static_cast<float>(step) * 0.78539816F;
            const float local_x = ellipse.rx * std::cos(angle);
            const float local_y = ellipse.ry * std::sin(angle);
            if (accept(ellipse.u + c * local_x - s * local_y,
                       ellipse.v + s * local_x + c * local_y))
                return true;
        }
        return false;
    };
    if (mode == 0) {
        const float min_x = std::min(ra.x, rb.x);
        const float max_x = std::max(ra.x, rb.x);
        const float min_y = std::min(ra.y, rb.y);
        const float max_y = std::max(ra.y, rb.y);
        if (ellipse.u >= min_x && ellipse.u <= max_x && ellipse.v >= min_y &&
            ellipse.v <= max_y)
            return true;
        const float closest_x = std::clamp(ellipse.u, min_x, max_x);
        const float closest_y = std::clamp(ellipse.v, min_y, max_y);
        if (ellipse_contains(ellipse, closest_x, closest_y)) return true;
        return rim([&](const float x, const float y) {
            return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
        });
    }
    if (mode == 1 || mode == 2) {
        const float du = ellipse.u - ra.x;
        const float dv = ellipse.v - ra.y;
        if (du * du + dv * dv <= radius * radius) return true;
        if (ellipse_contains(ellipse, ra.x, ra.y)) return true;
        return rim([&](const float x, const float y) {
            const float dx = x - ra.x;
            const float dy = y - ra.y;
            return dx * dx + dy * dy <= radius * radius;
        });
    }
    if (polygon == nullptr || polygon->size() < 3) return false;
    if (inside_polygon(*polygon, {ellipse.u, ellipse.v})) return true;
    for (const ImVec2& point : *polygon)
        if (ellipse_contains(ellipse, point.x, point.y)) return true;
    return rim([&](const float x, const float y) {
        return inside_polygon(*polygon, {x, y});
    });
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
        if (rings_hit_ && rings_ready()) {
            const float reach = ellipse_reach(camera, hit.depth, index);
            if (d2 > reach * reach) continue;
            ScreenEllipse ellipse;
            if (!project_ellipse(camera, index, ellipse) ||
                !ellipse_contains(ellipse, raster.x, raster.y))
                continue;
        } else if (d2 > gate) {
            continue;
        }
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
        const float rect_min_x = std::min(ra.x, rb.x);
        const float rect_max_x = std::max(ra.x, rb.x);
        const float rect_min_y = std::min(ra.y, rb.y);
        const float rect_max_y = std::max(ra.y, rb.y);
        const float circle_radius = mode == 1
            ? std::sqrt((rb.x - ra.x) * (rb.x - ra.x) + (rb.y - ra.y) * (rb.y - ra.y))
            : brush_radius_ * raster_scale;
        if (rings_hit_ && rings_ready()) {
            float clearance = 0.F;
            if (mode == 0) {
                const float dx = hit.u < rect_min_x ? rect_min_x - hit.u
                    : hit.u > rect_max_x ? hit.u - rect_max_x : 0.F;
                const float dy = hit.v < rect_min_y ? rect_min_y - hit.v
                    : hit.v > rect_max_y ? hit.v - rect_max_y : 0.F;
                clearance = std::sqrt(dx * dx + dy * dy);
            } else if (mode == 1 || mode == 2) {
                const ImVec2 origin = mode == 1 ? ra : rb;
                const float dx = hit.u - origin.x;
                const float dy = hit.v - origin.y;
                clearance = std::max(0.F, std::sqrt(dx * dx + dy * dy) - circle_radius);
            }
            if (clearance <= ellipse_reach(camera, hit.depth, index)) {
                ScreenEllipse ellipse;
                if (project_ellipse(camera, index, ellipse)) {
                    if (mode == 1)
                        inside = ellipse_hits(ellipse, 1, ra, rb, nullptr, circle_radius);
                    else if (mode == 2)
                        inside = ellipse_hits(ellipse, 1, rb, rb, nullptr, circle_radius);
                    else
                        inside = ellipse_hits(
                            ellipse, mode, ra, rb,
                            raster_poly.size() >= 3 ? &raster_poly : nullptr, 0.F);
                }
            }
        } else if (mode == 0) {
            inside = hit.u >= rect_min_x && hit.u <= rect_max_x &&
                     hit.v >= rect_min_y && hit.v <= rect_max_y;
        } else if (mode == 1) {
            const float du = hit.u - ra.x;
            const float dv = hit.v - ra.y;
            inside = du * du + dv * dv <= circle_radius * circle_radius;
        } else if (mode == 2) {
            const float du = hit.u - rb.x;
            const float dv = hit.v - rb.y;
            inside = du * du + dv * dv <= circle_radius * circle_radius;
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
    Snapshot step = std::move(undo_.back());
    undo_.pop_back();
    if (step.kind == Snapshot::Kind::deletion) {
        reinsert(step);
        redo_.push_back(std::move(step));
        upload_model(app);
        return;
    }
    if (step.kind == Snapshot::Kind::volume) {
        Snapshot current;
        current.kind = Snapshot::Kind::volume;
        std::copy(std::begin(volume_center_), std::end(volume_center_), current.volume_center);
        std::copy(std::begin(volume_quat_), std::end(volume_quat_), current.volume_quat);
        std::copy(std::begin(volume_size_), std::end(volume_size_), current.volume_size);
        std::copy(std::begin(step.volume_center), std::end(step.volume_center), volume_center_);
        std::copy(std::begin(step.volume_quat), std::end(step.volume_quat), volume_quat_);
        std::copy(std::begin(step.volume_size), std::end(step.volume_size), volume_size_);
        redo_.push_back(std::move(current));
        return;
    }
    Snapshot current;
    current.selected = selected_;
    if (!step.xyz.empty()) current.xyz = capture_xyz();
    restore(app, step);
    redo_.push_back(std::move(current));
}

void SplatEdit::redo(App& app) {
    if (redo_.empty() || stroking_) return;
    Snapshot step = std::move(redo_.back());
    redo_.pop_back();
    if (step.kind == Snapshot::Kind::deletion) {
        erase_recorded(step);
        undo_.push_back(std::move(step));
        upload_model(app);
        dirty_ = true;
        return;
    }
    if (step.kind == Snapshot::Kind::volume) {
        Snapshot current;
        current.kind = Snapshot::Kind::volume;
        std::copy(std::begin(volume_center_), std::end(volume_center_), current.volume_center);
        std::copy(std::begin(volume_quat_), std::end(volume_quat_), current.volume_quat);
        std::copy(std::begin(volume_size_), std::end(volume_size_), current.volume_size);
        std::copy(std::begin(step.volume_center), std::end(step.volume_center), volume_center_);
        std::copy(std::begin(step.volume_quat), std::end(step.volume_quat), volume_quat_);
        std::copy(std::begin(step.volume_size), std::end(step.volume_size), volume_size_);
        undo_.push_back(std::move(current));
        return;
    }
    Snapshot current;
    current.selected = selected_;
    if (!step.xyz.empty()) current.xyz = capture_xyz();
    restore(app, step);
    undo_.push_back(std::move(current));
}

void SplatEdit::upload_model(App& app) {
    std::vector<float> means(static_cast<std::size_t>(count_) * 3U);
    for (std::uint32_t index = 0; index < count_; ++index) {
        means[static_cast<std::size_t>(index) * 3U] =
            centers_[static_cast<std::size_t>(index) * 4U];
        means[static_cast<std::size_t>(index) * 3U + 1U] =
            centers_[static_cast<std::size_t>(index) * 4U + 1U];
        means[static_cast<std::size_t>(index) * 3U + 2U] =
            centers_[static_cast<std::size_t>(index) * 4U + 2U];
    }
    splat_render::GaussianCloud cloud;
    cloud.count = count_;
    cloud.sh_degree = sh_.empty() ? 0U : sh_degree_;
    cloud.sh_bases = sh_.empty() ? 1U : sh_bases_;
    cloud.means = count_ == 0 ? nullptr : means.data();
    cloud.log_scales = count_ == 0 ? nullptr : log_scales_.data();
    cloud.quaternions = count_ == 0 ? nullptr : quaternions_.data();
    cloud.opacity_logits = count_ == 0 ? nullptr : opacity_.data();
    cloud.sh = sh_.empty() ? nullptr : sh_.data();
    if (!app.splat_renderer.upload(cloud, key_)) {
        if (!app.splat_renderer.failure().empty())
            set_message(app, app.splat_renderer.failure(), theme::danger);
    }
}

void SplatEdit::reinsert(const Snapshot& snapshot) {
    const std::uint32_t kept = count_;
    centers_ = splice_strided(centers_, 4, snapshot.removed_index, snapshot.removed_centers);
    log_scales_ =
        splice_strided(log_scales_, 3, snapshot.removed_index, snapshot.removed_scales);
    quaternions_ =
        splice_strided(quaternions_, 4, snapshot.removed_index, snapshot.removed_quats);
    opacity_ =
        splice_strided(opacity_, 1, snapshot.removed_index, snapshot.removed_opacity);
    if (!sh_.empty() || !snapshot.removed_sh.empty())
        sh_ = splice_strided(
            sh_, sh_bases_ * 3U, snapshot.removed_index, snapshot.removed_sh);
    if (!normals_.empty() || !snapshot.removed_normals.empty())
        normals_ = splice_strided(
            normals_, 4, snapshot.removed_index, snapshot.removed_normals);
    if (!filter_.empty() || !snapshot.removed_filter.empty())
        filter_ = splice_strided(
            filter_, 1, snapshot.removed_index, snapshot.removed_filter);
    count_ = kept + static_cast<std::uint32_t>(snapshot.removed_index.size());
    if (snapshot.selected.size() == count_) {
        selected_ = snapshot.selected;
        recount();
    } else {
        selected_.assign(count_, 0);
        selected_count_ = 0;
    }
}

void SplatEdit::erase_recorded(const Snapshot& snapshot) {
    std::vector<std::uint8_t> drop(count_, 0);
    std::uint32_t removed = 0;
    for (const std::uint32_t index : snapshot.removed_index) {
        if (index >= count_ || drop[index]) continue;
        drop[index] = 1;
        ++removed;
    }
    keep_strided(centers_, 4, drop, count_);
    keep_strided(log_scales_, 3, drop, count_);
    keep_strided(quaternions_, 4, drop, count_);
    keep_strided(opacity_, 1, drop, count_);
    keep_strided(sh_, sh_bases_ * 3U, drop, count_);
    keep_strided(normals_, 4, drop, count_);
    keep_strided(filter_, 1, drop, count_);
    count_ = count_ - removed;
    selected_.assign(count_, 0);
    selected_count_ = 0;
}

void SplatEdit::delete_selected(App& app) {
    if (!synced_ || count_ == 0 || selected_count_ == 0 || stroking_) return;
    Snapshot step;
    step.kind = Snapshot::Kind::deletion;
    step.selected = selected_;
    const std::uint32_t removed = selected_count_;
    take_strided(centers_, 4, selected_, count_, step.removed_centers);
    take_strided(log_scales_, 3, selected_, count_, step.removed_scales);
    take_strided(quaternions_, 4, selected_, count_, step.removed_quats);
    take_strided(opacity_, 1, selected_, count_, step.removed_opacity);
    take_strided(sh_, sh_bases_ * 3U, selected_, count_, step.removed_sh);
    take_strided(normals_, 4, selected_, count_, step.removed_normals);
    take_strided(filter_, 1, selected_, count_, step.removed_filter);
    step.removed_index.reserve(removed);
    for (std::uint32_t index = 0; index < count_; ++index) {
        if (selected_[index]) step.removed_index.push_back(index);
    }
    count_ -= removed;
    selected_.assign(count_, 0);
    selected_count_ = 0;
    polygon_.clear();
    try {
        upload_model(app);
    } catch (const std::exception& failure) {
        reinsert(step);
        try {
            upload_model(app);
        } catch (...) {
        }
        set_message(app, failure.what(), theme::danger);
        return;
    }
    if (!app.splat_renderer.failure().empty()) {
        const std::string failure = app.splat_renderer.failure();
        reinsert(step);
        try {
            upload_model(app);
        } catch (...) {
        }
        set_message(app, failure, theme::danger);
        return;
    }
    dirty_ = true;
    if (app.scene.has_gaussians()) {
        app.scene.points.clear();
        app.scene.colours.clear();
        app.scene.gaussians.clear();
    }
    push_undo(std::move(step));
    set_message(
        app, std::to_string(removed) + " " + tr("gaussians removed"), theme::success);
}

bool SplatEdit::copy_model(photara::splat::GaussianModel& model) const {
    if (!synced_) return false;
    model = {};
    model.sh_degree = sh_degree_;
    const std::size_t count = count_;
    std::vector<float> means(count * 3U);
    for (std::uint32_t index = 0; index < count_; ++index) {
        means[static_cast<std::size_t>(index) * 3U] =
            centers_[static_cast<std::size_t>(index) * 4U];
        means[static_cast<std::size_t>(index) * 3U + 1U] =
            centers_[static_cast<std::size_t>(index) * 4U + 1U];
        means[static_cast<std::size_t>(index) * 3U + 2U] =
            centers_[static_cast<std::size_t>(index) * 4U + 2U];
    }
    const std::uint32_t bases = std::max(1U, sh_.empty() ? 1U : sh_bases_);
    std::vector<float> sh = sh_;
    if (sh.size() < count * bases * 3U) sh.resize(count * bases * 3U, 0.F);
    model.means = tinytensor::Tensor::from_vector(
        means, {count, 3}, tinytensor::Device::CPU);
    model.log_scales = tinytensor::Tensor::from_vector(
        log_scales_, {count, 3}, tinytensor::Device::CPU);
    model.quaternions = tinytensor::Tensor::from_vector(
        quaternions_, {count, 4}, tinytensor::Device::CPU);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        opacity_, {count, 1}, tinytensor::Device::CPU);
    model.sh = tinytensor::Tensor::from_vector(
        sh, {count, static_cast<std::size_t>(bases), 3}, tinytensor::Device::CPU);
    if (normals_.size() == count * 4U)
        model.normal_features = tinytensor::Tensor::from_vector(
            normals_, {count, 4}, tinytensor::Device::CPU);
    if (filter_.size() == count)
        model.filter_3d = tinytensor::Tensor::from_vector(
            filter_, {count, 1}, tinytensor::Device::CPU);
    return true;
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
    const Tool previous = tool_;
    tool_ = tool;
    volume_drag_ = -1;
    if (tool == Tool::box || tool == Tool::sphere) {
        if (!volume_placed_) place_volume(app);
        else if ((previous == Tool::box || previous == Tool::sphere) && previous != tool)
            adopt_volume_kind(tool);
        if (tool == Tool::sphere && volume_gesture_ == VolumeGesture::rotate)
            volume_gesture_ = VolumeGesture::move;
    }
}

void SplatEdit::toggle_tool(App& app, const Tool tool) {
    set_tool(app, tool_ == tool ? Tool::none : tool);
}

void SplatEdit::clear_selection() {
    if (selected_count_ == 0 || stroking_) return;
    Snapshot snap;
    snap.selected = selected_;
    push_undo(std::move(snap));
    std::fill(selected_.begin(), selected_.end(), 0);
    selected_count_ = 0;
}

void SplatEdit::select_all() {
    if (count_ == 0 || stroking_ || selected_count_ == count_) return;
    Snapshot snap;
    snap.selected = selected_;
    push_undo(std::move(snap));
    std::fill(selected_.begin(), selected_.end(), 1);
    selected_count_ = count_;
}

void SplatEdit::invert_selection() {
    if (count_ == 0 || stroking_) return;
    Snapshot snap;
    snap.selected = selected_;
    push_undo(std::move(snap));
    selected_count_ = 0;
    for (std::uint8_t& bit : selected_) {
        bit = bit ? 0 : 1;
        if (bit) ++selected_count_;
    }
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
    if (io.KeyCtrl && !io.KeyAlt && !io.KeyShift && !mouse_busy &&
        ImGui::IsKeyPressed(ImGuiKey_I, false)) {
        invert_selection();
        return;
    }
    if (!io.KeyCtrl && !io.KeyAlt && !io.KeyShift && !mouse_busy && !stroking_ &&
        ImGui::IsKeyPressed(ImGuiKey_Delete, false)) {
        polygon_.clear();
        delete_selected(app);
        return;
    }

    if (tool_ == Tool::brush && !io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true))
            brush_radius_ = std::max(4.F, brush_radius_ - 4.F);
        if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, true))
            brush_radius_ = std::min(180.F, brush_radius_ + 4.F);
    }
    if (volume_tool() && !mouse_busy) {
        const bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                           ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false);
        if (enter && io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
            apply_volume(VolumeOp::intersect);
            return;
        }
        if (enter && io.KeyShift && !io.KeyCtrl) {
            apply_volume(VolumeOp::add);
            return;
        }
        if (enter && io.KeyAlt && !io.KeyCtrl) {
            apply_volume(VolumeOp::remove);
            return;
        }
        if (!io.KeyCtrl && !io.KeyAlt && !io.KeyShift) {
            if (ImGui::IsKeyPressed(ImGuiKey_G, false))
                volume_gesture_ = VolumeGesture::move;
            else if (ImGui::IsKeyPressed(ImGuiKey_T, false) && tool_ == Tool::box)
                volume_gesture_ = VolumeGesture::rotate;
            else if (ImGui::IsKeyPressed(ImGuiKey_S, false))
                volume_gesture_ = VolumeGesture::scale;
            else if (enter) apply_volume(VolumeOp::replace);
            else if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true) ||
                     ImGui::IsKeyPressed(ImGuiKey_RightBracket, true)) {
                const float factor =
                    ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true) ? 0.9F : 1.1F;
                for (float& side : volume_size_)
                    side = std::max(0.02F, side * factor);
            }
        }
    }

    if (mouse_busy || stroking_ || io.KeyCtrl || io.KeyAlt || io.KeyShift) return;
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) toggle_tool(app, Tool::pick);
    else if (ImGui::IsKeyPressed(ImGuiKey_C, false)) toggle_tool(app, Tool::circle);
    else if (ImGui::IsKeyPressed(ImGuiKey_P, false)) toggle_tool(app, Tool::polygon);
    else if (ImGui::IsKeyPressed(ImGuiKey_B, false)) toggle_tool(app, Tool::brush);
    else if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) toggle_tool(app, Tool::box);
    else if (ImGui::IsKeyPressed(ImGuiKey_U, false)) toggle_tool(app, Tool::sphere);
    else if (ImGui::IsKeyPressed(ImGuiKey_N, false)) front_only_ = !front_only_;
}

bool SplatEdit::consume_alt_wheel(const bool pointer_in_view) {
    if (!pointer_in_view) return false;
    ImGuiIO& io = ImGui::GetIO();
    if (!io.KeyAlt || io.KeyCtrl || io.MouseWheel == 0.F) return false;
    if (tool_ == Tool::brush) {
        brush_radius_ = std::clamp(brush_radius_ + io.MouseWheel * 8.F, 4.F, 180.F);
        return true;
    }
    if (volume_tool()) {
        const float factor = io.MouseWheel > 0.F ? 1.08F : 1.F / 1.08F;
        for (float& side : volume_size_)
            side = std::max(0.02F, side * factor);
        return true;
    }
    return false;
}

void SplatEdit::write_status(char* buffer, const std::size_t size) const {
    if (buffer == nullptr || size == 0) return;
    buffer[0] = '\0';
    if (!synced_) return;
    if (count_ == 0) {
        std::snprintf(
            buffer, size, "%s",
            undo_.empty()
                ? tr("LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  F frame  |  double-click focus")
                : tr("Ctrl+Z restores the deleted Gaussians"));
        return;
    }
    if (tool_ == Tool::none) {
        const char* orbit = tr(
            "LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  F frame  |  double-click focus");
        if (selected_count_ > 0)
            std::snprintf(
                buffer, size, "%s  |  %s  |  %s", orbit,
                tr("Delete removes the selection"), tr("Esc clears the selection"));
        else
            std::snprintf(buffer, size, "%s", orbit);
        return;
    }
    if (volume_tool()) {
        const char* shape = tool_ == Tool::box ? tr("Box") : tr("Sphere");
        const char* gesture = volume_gesture_ == VolumeGesture::rotate
            ? tr("Rotate")
            : volume_gesture_ == VolumeGesture::scale ? tr("Scale")
                                                      : tr("Move");
        std::snprintf(
            buffer, size, "%s  |  %s  |  %s  |  %s", shape, gesture,
            tr("Drag the coloured squares or Alt+wheel to resize"),
            tr("Enter selects  |  Shift+Enter adds  |  Alt+Enter removes  |  Ctrl+Enter intersects"));
        return;
    }
    const char* gesture = tr("Click or drag a box");
    if (tool_ == Tool::circle) gesture = tr("Drag a circle");
    else if (tool_ == Tool::polygon) gesture = tr("Click points, Enter closes");
    else if (tool_ == Tool::brush) gesture = tr("Paint across Gaussians");
    if (rings_hit_) {
        std::snprintf(
            buffer, size, "%s  |  %s  |  %s  |  %s", tr("By ring"),
            tr(front_only_ ? "Front surface" : "All depths"), gesture,
            tr("Shift add  |  Alt remove  |  RMB orbit  |  Esc steps back"));
    } else {
        std::snprintf(
            buffer, size, "%s  |  %s  |  %s",
            tr(front_only_ ? "Front surface" : "All depths"), gesture,
            tr("Shift add  |  Alt remove  |  RMB orbit  |  Esc steps back"));
    }
    if (selected_count_ > 0) {
        const std::string gesture_line(buffer);
        std::snprintf(
            buffer, size, "%s  |  %s", gesture_line.c_str(),
            tr("Delete removes the selection"));
    }
}

bool SplatEdit::draw_toolbar(App& app, const ImVec2 view_min, const ImVec2 view_max) {
    toolbar_visible_ = false;
    if (!synced_) return false;
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
        {icons::Icon::box_select, "##splat_box", "Box select",
         "Place a box in the scene. Drag the coloured squares, or Alt+wheel, to resize it.",
         "Y", 7, false},
        {icons::Icon::sphere_select, "##splat_sphere", "Sphere select",
         "Place a sphere in the scene. Drag the coloured squares, or Alt+wheel, to resize it.",
         "U", 8, false},
        {icons::Icon::depth_front, "##splat_front", "Front surface",
         "The next gesture keeps only the nearest Gaussian in each small screen cell.",
         "N", 20, true},
        {icons::Icon::depth_through, "##splat_through", "All depths",
         "The next gesture selects every layer it covers, including Gaussians hidden behind the front one.",
         "N", 21, false},
        {icons::Icon::points, "##splat_centres", "Select by centre",
         "A Gaussian counts only when its centre is inside the gesture.",
         nullptr, 22, true},
        {icons::Icon::rings, "##splat_rings", "Select by ring",
         "A Gaussian counts when its ring crosses the gesture. Wide splats are easier to grab, using the same ellipse as Rings view.",
         nullptr, 23, false},
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
        Tool::pick, Tool::circle, Tool::polygon, Tool::brush, Tool::box, Tool::sphere};
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
            : item.action >= 3 && item.action <= 8
                ? tool_ == tools[item.action - 3]
            : item.action == 20 ? front_only_
            : item.action == 21 ? !front_only_
            : item.action == 22 ? !rings_hit_
            : item.action == 23 ? rings_hit_
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
        else if (item.action >= 3 && item.action <= 8)
            toggle_tool(app, tools[item.action - 3]);
        else if (item.action == 20) front_only_ = true;
        else if (item.action == 21) front_only_ = false;
        else if (item.action == 22) {
            rings_hit_ = false;
            hit_chosen_ = true;
        } else if (item.action == 23) {
            rings_hit_ = true;
            hit_chosen_ = true;
        }
    }
    const bool over_volume = draw_volume_bar(view_min, view_max);
    return over_volume ||
           ImGui::IsMouseHoveringRect(toolbar_min_, toolbar_max_, false);
}

void SplatEdit::draw_overlay(
    ImDrawList* draw, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera) const {
    if (count_ == 0 || draw == nullptr) return;
    draw->PushClipRect(view_min, view_max, true);
    if (volume_tool() && volume_placed_)
        draw_volume(draw, view_min, view_max, camera);
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
        if (rings_hit_) {
            ScreenEllipse ellipse;
            if (project_ellipse(camera, index, ellipse) && ellipse.valid) {
                const float scale_x = (view_max.x - view_min.x) /
                                      std::max(1.F, static_cast<float>(camera.width));
                const float scale_y = (view_max.y - view_min.y) /
                                      std::max(1.F, static_cast<float>(camera.height));
                const int segments = std::clamp(
                    static_cast<int>(ellipse.rx * 0.5F), 16, 64);
                draw->AddEllipse(
                    screen, {ellipse.rx * scale_x, ellipse.ry * scale_y}, mark,
                    ellipse.rotation, segments, 1.5F);
            }
        }
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool in_view = mouse.x >= view_min.x && mouse.x < view_max.x &&
                         mouse.y >= view_min.y && mouse.y < view_max.y;
    const bool over_bar = toolbar_visible_ &&
                          mouse.x >= toolbar_min_.x && mouse.x <= toolbar_max_.x &&
                          mouse.y >= toolbar_min_.y && mouse.y <= toolbar_max_.y;
    const bool over_volume_bar =
        volume_bar_max_.x > volume_bar_min_.x &&
        mouse.x >= volume_bar_min_.x && mouse.x <= volume_bar_max_.x &&
        mouse.y >= volume_bar_min_.y && mouse.y <= volume_bar_max_.y;
    // Left mode rail: pad 10, top 52, 44px wide, through the scene toggle.
    const bool over_rail = mouse.x <= view_min.x + 62.F &&
                           mouse.y >= view_min.y + 48.F &&
                           mouse.y <= view_min.y + 340.F;
    const ImU32 stroke = gesture_stroke();
    const ImU32 fill = gesture_fill();
    const bool show_cursor = tool_ != Tool::none && !volume_tool() && in_view &&
                             !over_bar && !over_volume_bar && !over_rail;
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
    if (!synced_) return;
    const ImGuiIO& io = ImGui::GetIO();
    const bool keys_hot = viewport_hot && !io.WantTextInput;
    const bool hot = keys_hot && !over_toolbar;
    if (keys_hot) handle_keys(app);

    if (tool_ == Tool::none) {
        if (stroking_) abort_stroke(app);
        return;
    }
    if (volume_tool()) {
        handle_volume(app, view_min, view_max, camera, hot);
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

namespace {

void view_direction(
    const splat_render::Camera& camera, const float point[3], float out[3]) {
    float right[3];
    float up[3];
    float forward[3];
    camera_basis(camera, right, up, forward);
    if (camera.model == splat_render::k_camera_orthographic) {
        out[0] = forward[0];
        out[1] = forward[1];
        out[2] = forward[2];
        normalize3(out);
        return;
    }
    out[0] = point[0] - camera.position[0];
    out[1] = point[1] - camera.position[1];
    out[2] = point[2] - camera.position[2];
    if (!normalize3(out)) {
        out[0] = forward[0];
        out[1] = forward[1];
        out[2] = forward[2];
        normalize3(out);
    }
}

void axis_plane_normal(const float axis[3], const float view[3], float normal[3]) {
    normal[0] = view[0] - axis[0] * dot3(view, axis);
    normal[1] = view[1] - axis[1] * dot3(view, axis);
    normal[2] = view[2] - axis[2] * dot3(view, axis);
    if (!normalize3(normal)) {
        const float helper[3] = {
            std::fabs(axis[1]) < 0.9F ? 0.F : 1.F,
            std::fabs(axis[1]) < 0.9F ? 1.F : 0.F, 0.F};
        cross3(axis, helper, normal);
        normalize3(normal);
    }
}

}  // namespace

void SplatEdit::place_volume(const App& app) {
    float lo[3] = {1e30F, 1e30F, 1e30F};
    float hi[3] = {-1e30F, -1e30F, -1e30F};
    bool any = false;
    if (selected_count_ > 0) {
        for (std::uint32_t index = 0; index < count_; ++index) {
            if (!selected_[index]) continue;
            const float* point = centers_.data() + static_cast<std::size_t>(index) * 4U;
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = std::min(lo[axis], point[axis]);
                hi[axis] = std::max(hi[axis], point[axis]);
            }
            any = true;
        }
    }
    volume_quat_[0] = 1.F;
    volume_quat_[1] = volume_quat_[2] = volume_quat_[3] = 0.F;
    volume_gesture_ = VolumeGesture::move;
    if (any) {
        for (int axis = 0; axis < 3; ++axis) {
            volume_center_[axis] = 0.5F * (lo[axis] + hi[axis]);
            volume_size_[axis] = std::max(0.05F, (hi[axis] - lo[axis]) * 1.12F);
        }
        if (tool_ == Tool::sphere) {
            const float radius = 0.5F * std::max(
                volume_size_[0], std::max(volume_size_[1], volume_size_[2]));
            volume_size_[0] = std::max(0.02F, radius);
        }
    } else {
        volume_center_[0] = app.camera.target.x;
        volume_center_[1] = app.camera.target.y;
        volume_center_[2] = app.camera.target.z;
        const float span = std::max(0.25F, app.camera.distance * 0.22F);
        if (tool_ == Tool::sphere) volume_size_[0] = span * 0.5F;
        else
            volume_size_[0] = volume_size_[1] = volume_size_[2] = span;
    }
    volume_placed_ = true;
}

void SplatEdit::adopt_volume_kind(const Tool tool) {
    if (tool == Tool::sphere) {
        const float radius = 0.5F * std::max(
            volume_size_[0], std::max(volume_size_[1], volume_size_[2]));
        volume_size_[0] = std::max(0.02F, radius);
    } else {
        const float side = std::max(0.05F, volume_size_[0] * 2.F);
        volume_size_[0] = volume_size_[1] = volume_size_[2] = side;
    }
}

void SplatEdit::apply_volume(const VolumeOp op) {
    if (!volume_placed_ || count_ == 0 || stroking_) return;
    float axis[3][3];
    if (tool_ == Tool::sphere) {
        axis[0][0] = 1.F;
        axis[0][1] = axis[0][2] = 0.F;
        axis[1][1] = 1.F;
        axis[1][0] = axis[1][2] = 0.F;
        axis[2][2] = 1.F;
        axis[2][0] = axis[2][1] = 0.F;
    } else
        quat_axes(volume_quat_, axis);
    const bool sphere = tool_ == Tool::sphere;
    const float half[3] = {
        sphere ? volume_size_[0] : volume_size_[0] * 0.5F,
        sphere ? volume_size_[0] : volume_size_[1] * 0.5F,
        sphere ? volume_size_[0] : volume_size_[2] * 0.5F};
    const auto contains = [&](const std::uint32_t index) {
        const float* point = centers_.data() + static_cast<std::size_t>(index) * 4U;
        const float rel[3] = {
            point[0] - volume_center_[0], point[1] - volume_center_[1],
            point[2] - volume_center_[2]};
        const float local[3] = {
            dot3(rel, axis[0]), dot3(rel, axis[1]), dot3(rel, axis[2])};
        float closest[3];
        float distance = 0.F;
        if (sphere) {
            const float len = length3(local);
            if (len <= half[0]) return true;
            const float scale = half[0] / std::max(len, 1e-8F);
            for (int k = 0; k < 3; ++k)
                closest[k] = volume_center_[k] +
                             (axis[0][k] * local[0] + axis[1][k] * local[1] +
                              axis[2][k] * local[2]) *
                                 scale;
            distance = len - half[0];
        } else {
            float clamped[3];
            bool inside = true;
            for (int k = 0; k < 3; ++k) {
                clamped[k] = std::clamp(local[k], -half[k], half[k]);
                if (clamped[k] != local[k]) inside = false;
            }
            if (inside) return true;
            for (int k = 0; k < 3; ++k)
                closest[k] = volume_center_[k] + axis[0][k] * clamped[0] +
                             axis[1][k] * clamped[1] + axis[2][k] * clamped[2];
            const float delta[3] = {
                point[0] - closest[0], point[1] - closest[1], point[2] - closest[2]};
            distance = length3(delta);
        }
        if (!rings_hit_ || !rings_ready() || distance <= 1e-5F) return false;
        float toward[3] = {
            closest[0] - point[0], closest[1] - point[1], closest[2] - point[2]};
        if (!normalize3(toward)) return false;
        const float* rotation =
            quaternions_.data() + static_cast<std::size_t>(index) * 4U;
        const float* log_scale =
            log_scales_.data() + static_cast<std::size_t>(index) * 3U;
        float gaussian[3][3];
        quat_axes(rotation, gaussian);
        const float lx = dot3(toward, gaussian[0]);
        const float ly = dot3(toward, gaussian[1]);
        const float lz = dot3(toward, gaussian[2]);
        const float sx = std::exp(std::clamp(log_scale[0], -12.F, 8.F));
        const float sy = std::exp(std::clamp(log_scale[1], -12.F, 8.F));
        const float sz = std::exp(std::clamp(log_scale[2], -12.F, 8.F));
        const float support = ring_sigma_ * std::sqrt(
            sx * lx * sx * lx + sy * ly * sy * ly + sz * lz * sz * lz);
        return distance <= support;
    };

    std::vector<std::uint8_t> before = selected_;
    if (op == VolumeOp::replace) std::fill(selected_.begin(), selected_.end(), 0);
    for (std::uint32_t index = 0; index < count_; ++index) {
        const float opacity = centers_[static_cast<std::size_t>(index) * 4U + 3U];
        const bool inside = contains(index);
        if (op == VolumeOp::replace || op == VolumeOp::add) {
            if (inside && opacity >= k_min_opacity) selected_[index] = 1;
        } else if (op == VolumeOp::remove) {
            if (inside) selected_[index] = 0;
        } else if (!inside) {
            selected_[index] = 0;
        }
    }
    if (before == selected_) {
        if (op == VolumeOp::replace) selected_ = before;
        return;
    }
    Snapshot snap;
    snap.selected = std::move(before);
    push_undo(std::move(snap));
    recount();
}

bool SplatEdit::draw_volume_bar(const ImVec2 view_min, const ImVec2 view_max) {
    volume_bar_min_ = volume_bar_max_ = {};
    if (!volume_tool()) return false;
    struct Item {
        icons::Icon icon;
        const char* id;
        const char* tip;
        const char* detail;
        const char* shortcut;
        int action;
        bool gap;
        bool glyph;
    };
    const bool box = tool_ == Tool::box;
    const Item items[] = {
        {icons::Icon::translate, "##vol_move", "Move",
         "Drag an axis to slide the volume. Drag the volume itself to move it on the view.",
         "G", 0, false, false},
        {icons::Icon::rotate, "##vol_rotate", "Rotate",
         "Drag a ring to turn the box.", "T", 1, false, false},
        {icons::Icon::scale, "##vol_scale", "Scale",
         "Scale mode. The coloured squares on the volume always resize it, and Alt+wheel does too.",
         "S", 2, false, false},
        {icons::Icon::select, "##vol_set", "Replace",
         "Select only the Gaussians inside the volume.", "Enter", 3, true, true},
        {icons::Icon::select, "##vol_add", "Add",
         "Add the Gaussians inside the volume to the selection.", "Shift+Enter", 4,
         false, true},
        {icons::Icon::select, "##vol_remove", "Remove",
         "Remove the Gaussians inside the volume from the selection.", "Alt+Enter", 5,
         false, true},
        {icons::Icon::select, "##vol_intersect", "Intersect",
         "Keep only the selected Gaussians that are also inside the volume.",
         "Ctrl+Enter", 6, false, true},
    };
    constexpr float k_button = 34.F;
    constexpr float k_gap = 4.F;
    constexpr float k_pad = 6.F;
    constexpr float k_group = 16.F;
    float content_w = 0.F;
    for (const Item& item : items) {
        if (!box && item.action == 1) continue;
        content_w += k_button + k_gap + (item.gap ? k_group : 0.F);
    }
    content_w -= k_gap;
    const float bar_w = k_pad * 2.F + content_w;
    const float bar_h = k_pad * 2.F + k_button;
    const float left = view_min.x + (view_max.x - view_min.x - bar_w) * 0.5F;
    const float top = toolbar_max_.y + 8.F;
    volume_bar_min_ = {left, top};
    volume_bar_max_ = {left + bar_w, top + bar_h};
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        {volume_bar_min_.x + 1.F, volume_bar_min_.y + 3.F},
        {volume_bar_max_.x + 1.F, volume_bar_max_.y + 3.F}, IM_COL32(0, 0, 0, 70),
        9.F);
    draw->AddRectFilled(
        volume_bar_min_, volume_bar_max_, IM_COL32(22, 24, 28, 236), 9.F);
    draw->AddRect(
        volume_bar_min_, volume_bar_max_, IM_COL32(255, 255, 255, 18), 9.F);
    float cursor_x = left + k_pad;
    for (const Item& item : items) {
        if (!box && item.action == 1) continue;
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
        const bool active = item.action == 0
            ? volume_gesture_ == VolumeGesture::move
            : item.action == 1 ? volume_gesture_ == VolumeGesture::rotate
            : item.action == 2 ? volume_gesture_ == VolumeGesture::scale
                               : false;
        ImGui::SetCursorScreenPos(min);
        ImGui::PushID(item.id);
        const bool pressed = ImGui::InvisibleButton(item.id, {k_button, k_button});
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        ImU32 fill = active ? theme::u32(theme::fade(theme::accent, 0.24F))
                            : hovered ? IM_COL32(58, 62, 72, 255)
                                      : theme::u32(theme::surface_3);
        draw->AddRectFilled(min, max, fill, 6.F);
        if (active)
            draw->AddRect(min, max, theme::u32(theme::accent, 0.75F), 6.F, 0, 1.F);
        const ImU32 colour = theme::u32(
            active ? theme::accent : hovered ? theme::text_bright : theme::text_muted);
        if (!item.glyph) {
            icons::draw(
                draw, item.icon, {min.x + 7.F, min.y + 7.F}, {max.x - 7.F, max.y - 7.F},
                colour, 1.5F);
        } else {
            const ImVec2 c{(min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F};
            if (item.action == 3) {
                draw->AddRectFilled({c.x - 5.F, c.y - 5.F}, {c.x + 5.F, c.y + 5.F}, colour, 1.5F);
            } else if (item.action == 4) {
                draw->AddLine({c.x - 6.F, c.y}, {c.x + 6.F, c.y}, colour, 1.7F);
                draw->AddLine({c.x, c.y - 6.F}, {c.x, c.y + 6.F}, colour, 1.7F);
            } else if (item.action == 5) {
                draw->AddLine({c.x - 6.F, c.y}, {c.x + 6.F, c.y}, colour, 1.7F);
            } else {
                draw->AddCircle(c, 5.5F, colour, 16, 1.4F);
                draw->AddCircle({c.x + 4.F, c.y}, 5.5F, colour, 16, 1.4F);
            }
        }
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            show_tip(item.tip, item.shortcut, item.detail);
        }
        if (!pressed) continue;
        if (item.action == 0) volume_gesture_ = VolumeGesture::move;
        else if (item.action == 1) volume_gesture_ = VolumeGesture::rotate;
        else if (item.action == 2) volume_gesture_ = VolumeGesture::scale;
        else if (item.action == 3) apply_volume(VolumeOp::replace);
        else if (item.action == 4) apply_volume(VolumeOp::add);
        else if (item.action == 5) apply_volume(VolumeOp::remove);
        else apply_volume(VolumeOp::intersect);
    }
    return ImGui::IsMouseHoveringRect(volume_bar_min_, volume_bar_max_, false);
}

namespace {

bool volume_ray_hit(
    const WorldRay& ray, const float center[3], const float axis[3][3],
    const float half[3], const bool sphere) {
    if (!ray.valid) return false;
    const float rel[3] = {
        ray.origin[0] - center[0], ray.origin[1] - center[1],
        ray.origin[2] - center[2]};
    const float origin[3] = {
        dot3(rel, axis[0]), dot3(rel, axis[1]), dot3(rel, axis[2])};
    const float direction[3] = {
        dot3(ray.direction, axis[0]), dot3(ray.direction, axis[1]),
        dot3(ray.direction, axis[2])};
    if (sphere) {
        const float along = dot3(origin, direction);
        const float height = along * along - (dot3(origin, origin) - half[0] * half[0]);
        if (height < 0.F) return false;
        const float root = std::sqrt(height);
        float t = -along - root;
        if (t < 0.F) t = -along + root;
        return t >= 0.F;
    }
    float enter = -1e30F;
    float exit = 1e30F;
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
        if (std::fabs(direction[axis_index]) < 1e-8F) {
            if (origin[axis_index] < -half[axis_index] ||
                origin[axis_index] > half[axis_index])
                return false;
            continue;
        }
        float t_enter = (-half[axis_index] - origin[axis_index]) / direction[axis_index];
        float t_exit = (half[axis_index] - origin[axis_index]) / direction[axis_index];
        if (t_enter > t_exit) std::swap(t_enter, t_exit);
        enter = std::max(enter, t_enter);
        exit = std::min(exit, t_exit);
        if (enter > exit) return false;
    }
    return exit >= 0.F;
}

void volume_basis(const float normal[3], float u[3], float v[3]) {
    const float helper[3] = {
        std::fabs(normal[1]) < 0.9F ? 0.F : 1.F,
        std::fabs(normal[1]) < 0.9F ? 1.F : 0.F, 0.F};
    cross3(normal, helper, u);
    normalize3(u);
    cross3(normal, u, v);
    normalize3(v);
}

}  // namespace

void SplatEdit::draw_volume(
    ImDrawList* draw, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera) const {
    const bool sphere = tool_ == Tool::sphere;
    float axis[3][3];
    if (sphere) {
        axis[0][0] = axis[1][1] = axis[2][2] = 1.F;
        axis[0][1] = axis[0][2] = axis[1][0] = axis[1][2] = axis[2][0] = axis[2][1] =
            0.F;
    } else
        quat_axes(volume_quat_, axis);
    const float half[3] = {
        sphere ? volume_size_[0] : std::max(0.02F, volume_size_[0] * 0.5F),
        sphere ? volume_size_[0] : std::max(0.02F, volume_size_[1] * 0.5F),
        sphere ? volume_size_[0] : std::max(0.02F, volume_size_[2] * 0.5F)};
    ImVec2 center_screen;
    float center_depth = 1.F;
    const bool center_ok = project_world(
        camera, view_min, view_max, volume_center_, center_screen, center_depth);
    const float handle = center_ok
        ? world_from_screen(camera, view_min, view_max, center_depth, 70.F)
        : 0.F;
    const int hot = volume_drag_ >= 0
        ? volume_drag_
        : pick_volume(view_min, view_max, camera, ImGui::GetIO().MousePos);
    const ImU32 axis_colour[3] = {
        IM_COL32(232, 92, 82, 255), IM_COL32(78, 196, 122, 255),
        IM_COL32(86, 156, 236, 255)};
    const auto stroke = [&](const ImVec2 a, const ImVec2 b, const ImU32 colour,
                            const float width) {
        draw->AddLine(a, b, IM_COL32(6, 8, 12, 170), width + 2.2F);
        draw->AddLine(a, b, colour, width);
    };
    const auto ring = [&](const float origin[3], const float u[3], const float v[3],
                          const float radius, const ImU32 colour, const float width) {
        ImVec2 previous;
        bool have = false;
        constexpr int k_steps = 96;
        for (int step = 0; step <= k_steps; ++step) {
            const float angle = static_cast<float>(step) / static_cast<float>(k_steps) * 6.2831853F;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            float point[3];
            for (int k = 0; k < 3; ++k)
                point[k] = origin[k] + (u[k] * c + v[k] * s) * radius;
            ImVec2 screen;
            float depth = 0.F;
            const bool ok = project_world(camera, view_min, view_max, point, screen, depth);
            if (have && ok) stroke(previous, screen, colour, width);
            have = ok;
            previous = screen;
        }
    };

    if (sphere) {
        const ImU32 wire = IM_COL32(236, 242, 248, 210);
        // Eight meridians and seven parallels.
        for (int meridian = 0; meridian < 8; ++meridian) {
            const float turn = static_cast<float>(meridian) * 0.39269908F;
            const float turn_c = std::cos(turn);
            const float turn_s = std::sin(turn);
            float heading[3];
            for (int k = 0; k < 3; ++k)
                heading[k] = axis[0][k] * turn_c + axis[2][k] * turn_s;
            ring(volume_center_, heading, axis[1], half[0], wire, 1.35F);
        }
        const float latitudes[7] = {
            0.F, 0.39269908F, -0.39269908F, 0.78539816F, -0.78539816F,
            1.17809725F, -1.17809725F};
        for (const float latitude : latitudes) {
            const float parallel_radius = half[0] * std::cos(latitude);
            const float height = half[0] * std::sin(latitude);
            float origin[3];
            for (int k = 0; k < 3; ++k)
                origin[k] = volume_center_[k] + axis[1][k] * height;
            ring(origin, axis[0], axis[2], parallel_radius, wire, 1.35F);
        }
        float view[3];
        view_direction(camera, volume_center_, view);
        float u[3];
        float v[3];
        volume_basis(view, u, v);
        ring(volume_center_, u, v, half[0], IM_COL32(255, 255, 255, 235), 1.7F);
    } else {
        struct Corner {
            ImVec2 screen;
            float depth{};
            bool ok{};
        };
        Corner corner[8];
        for (int index = 0; index < 8; ++index) {
            const float sign[3] = {
                (index & 1) ? 1.F : -1.F, (index & 2) ? 1.F : -1.F,
                (index & 4) ? 1.F : -1.F};
            float point[3];
            for (int k = 0; k < 3; ++k)
                point[k] = volume_center_[k] + axis[0][k] * half[0] * sign[0] +
                           axis[1][k] * half[1] * sign[1] +
                           axis[2][k] * half[2] * sign[2];
            corner[index].ok = project_world(
                camera, view_min, view_max, point, corner[index].screen,
                corner[index].depth);
        }
        const int faces[6][4] = {
            {1, 3, 7, 5}, {0, 4, 6, 2}, {2, 6, 7, 3},
            {0, 1, 5, 4}, {4, 5, 7, 6}, {0, 2, 3, 1}};
        const int face_axis[6] = {0, 0, 1, 1, 2, 2};
        const float face_sign[6] = {1.F, -1.F, 1.F, -1.F, 1.F, -1.F};
        struct Face {
            ImVec2 screen[4];
            float depth{};
        };
        Face visible[6];
        int visible_count = 0;
        for (int face = 0; face < 6; ++face) {
            bool ok = true;
            float depth = 0.F;
            for (int vertex = 0; vertex < 4; ++vertex) {
                const Corner& sample = corner[faces[face][vertex]];
                if (!sample.ok) {
                    ok = false;
                    break;
                }
                visible[visible_count].screen[vertex] = sample.screen;
                depth += sample.depth;
            }
            if (!ok) continue;
            float outward[3];
            const int axis_index = face_axis[face];
            for (int k = 0; k < 3; ++k)
                outward[k] = axis[axis_index][k] * face_sign[face];
            const float to_camera[3] = {
                camera.position[0] - volume_center_[0],
                camera.position[1] - volume_center_[1],
                camera.position[2] - volume_center_[2]};
            if (dot3(outward, to_camera) <= 0.F) continue;
            visible[visible_count].depth = depth * 0.25F;
            ++visible_count;
        }
        std::sort(visible, visible + visible_count, [](const Face& a, const Face& b) {
            return a.depth > b.depth;
        });
        for (int face = 0; face < visible_count; ++face)
            draw->AddConvexPolyFilled(visible[face].screen, 4, IM_COL32(74, 181, 245, 32));
        struct Edge {
            ImVec2 a;
            ImVec2 b;
            float depth{};
        };
        Edge edges[12];
        int edge_count = 0;
        for (int index = 0; index < 8; ++index) {
            for (int bit = 0; bit < 3; ++bit) {
                if (index & (1 << bit)) continue;
                const int other = index | (1 << bit);
                if (!corner[index].ok || !corner[other].ok) continue;
                edges[edge_count].a = corner[index].screen;
                edges[edge_count].b = corner[other].screen;
                edges[edge_count].depth =
                    0.5F * (corner[index].depth + corner[other].depth);
                ++edge_count;
            }
        }
        std::sort(edges, edges + edge_count, [](const Edge& a, const Edge& b) {
            return a.depth > b.depth;
        });
        for (int edge = 0; edge < edge_count; ++edge)
            stroke(edges[edge].a, edges[edge].b, IM_COL32(236, 242, 248, 220), 1.45F);
    }

    if (!center_ok) return;
    const auto arrow = [&](const ImVec2 from, const ImVec2 to, const ImU32 colour) {
        const float dx = to.x - from.x;
        const float dy = to.y - from.y;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len < 2.F) return;
        const float nx = dx / len;
        const float ny = dy / len;
        const ImVec2 tip = to;
        const ImVec2 base{to.x - nx * 11.F, to.y - ny * 11.F};
        const ImVec2 side{-ny * 4.5F, nx * 4.5F};
        draw->AddTriangleFilled(
            tip, {base.x + side.x, base.y + side.y}, {base.x - side.x, base.y - side.y},
            colour);
    };
    if (volume_gesture_ == VolumeGesture::rotate && !sphere) {
        const float radius = world_from_screen(camera, view_min, view_max, center_depth, 58.F);
        for (int index = 0; index < 3; ++index) {
            float u[3];
            float v[3];
            volume_basis(axis[index], u, v);
            const bool active = hot == index;
            ImVec2 previous;
            bool have = false;
            for (int step = 0; step <= 28; ++step) {
                const float angle = 0.4F + static_cast<float>(step) / 28.F * 4.7F;
                float point[3];
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                for (int k = 0; k < 3; ++k)
                    point[k] = volume_center_[k] + (u[k] * c + v[k] * s) * radius;
                ImVec2 screen;
                float depth = 0.F;
                const bool ok =
                    project_world(camera, view_min, view_max, point, screen, depth);
                if (have && ok)
                    stroke(previous, screen, axis_colour[index], active ? 2.6F : 1.6F);
                have = ok;
                previous = screen;
            }
        }
    } else if (volume_gesture_ != VolumeGesture::scale) {
        const float gap = world_from_screen(camera, view_min, view_max, center_depth, 14.F);
        for (int index = 0; index < 3; ++index) {
            float from[3];
            float to[3];
            for (int k = 0; k < 3; ++k) {
                from[k] = volume_center_[k] + axis[index][k] * gap;
                to[k] = volume_center_[k] + axis[index][k] * handle;
            }
            ImVec2 a;
            ImVec2 b;
            float depth_a = 0.F;
            float depth_b = 0.F;
            if (!project_world(camera, view_min, view_max, from, a, depth_a) ||
                !project_world(camera, view_min, view_max, to, b, depth_b))
                continue;
            stroke(a, b, axis_colour[index], hot == index ? 2.8F : 1.8F);
            arrow(a, b, axis_colour[index]);
        }
        draw->AddCircleFilled(center_screen, 5.5F, IM_COL32(8, 10, 14, 220));
        draw->AddCircle(center_screen, 5.5F, IM_COL32(236, 242, 248, 230), 16, 1.4F);
    }
    for (int index = 0; index < 3; ++index) {
        float point[3];
        for (int k = 0; k < 3; ++k)
            point[k] = volume_center_[k] + axis[index][k] * half[index];
        ImVec2 screen;
        float depth = 0.F;
        if (!project_world(camera, view_min, view_max, point, screen, depth)) continue;
        const bool active = hot == 10 + index;
        const float extent = active ? 8.F : 6.F;
        draw->AddRectFilled(
            {screen.x - extent, screen.y - extent},
            {screen.x + extent, screen.y + extent}, axis_colour[index], 2.F);
        draw->AddRect(
            {screen.x - extent - 1.5F, screen.y - extent - 1.5F},
            {screen.x + extent + 1.5F, screen.y + extent + 1.5F},
            IM_COL32(8, 10, 14, 230), 2.F, 0, 1.6F);
    }
    if (hot >= 10) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    else if (hot >= 0) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
}

int SplatEdit::pick_volume(
    const ImVec2 view_min, const ImVec2 view_max, const splat_render::Camera& camera,
    const ImVec2 mouse) const {
    const bool sphere = tool_ == Tool::sphere;
    float axis[3][3];
    if (sphere) {
        axis[0][0] = axis[1][1] = axis[2][2] = 1.F;
        axis[0][1] = axis[0][2] = axis[1][0] = axis[1][2] = axis[2][0] = axis[2][1] =
            0.F;
    } else
        quat_axes(volume_quat_, axis);
    const float half[3] = {
        sphere ? volume_size_[0] : volume_size_[0] * 0.5F,
        sphere ? volume_size_[0] : volume_size_[1] * 0.5F,
        sphere ? volume_size_[0] : volume_size_[2] * 0.5F};
    ImVec2 center_screen;
    float center_depth = 1.F;
    const bool center_ok = project_world(
        camera, view_min, view_max, volume_center_, center_screen, center_depth);
    float best = 1.e9F;
    int best_axis = -1;
    const auto consider = [&](const int index, const float distance, const float limit) {
        if (distance <= limit * limit && distance < best) {
            best = distance;
            best_axis = index;
        }
    };
    for (int index = 0; index < 3; ++index) {
        float point[3];
        for (int k = 0; k < 3; ++k)
            point[k] = volume_center_[k] + axis[index][k] * half[index];
        ImVec2 screen;
        float depth = 0.F;
        if (!project_world(camera, view_min, view_max, point, screen, depth)) continue;
        const float dx = mouse.x - screen.x;
        const float dy = mouse.y - screen.y;
        consider(10 + index, dx * dx + dy * dy, 12.F);
    }
    if (volume_gesture_ == VolumeGesture::rotate && !sphere) {
        const float radius =
            world_from_screen(camera, view_min, view_max, center_depth, 58.F);
        for (int index = 0; index < 3; ++index) {
            float u[3];
            float v[3];
            volume_basis(axis[index], u, v);
            ImVec2 previous;
            bool have = false;
            for (int step = 0; step <= 28; ++step) {
                const float angle = 0.4F + static_cast<float>(step) / 28.F * 4.7F;
                float point[3];
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                for (int k = 0; k < 3; ++k)
                    point[k] = volume_center_[k] + (u[k] * c + v[k] * s) * radius;
                ImVec2 screen;
                float depth = 0.F;
                const bool ok =
                    project_world(camera, view_min, view_max, point, screen, depth);
                if (have && ok)
                    consider(index, dist2_segment(mouse, previous, screen), 9.F);
                have = ok;
                previous = screen;
            }
        }
    } else if (volume_gesture_ != VolumeGesture::scale) {
        const float gap = world_from_screen(camera, view_min, view_max, center_depth, 14.F);
        const float length = world_from_screen(camera, view_min, view_max, center_depth, 70.F);
        for (int index = 0; index < 3; ++index) {
            float from[3];
            float to[3];
            for (int k = 0; k < 3; ++k) {
                from[k] = volume_center_[k] + axis[index][k] * gap;
                to[k] = volume_center_[k] + axis[index][k] * length;
            }
            ImVec2 a;
            ImVec2 b;
            float depth_a = 0.F;
            float depth_b = 0.F;
            if (!project_world(camera, view_min, view_max, from, a, depth_a) ||
                !project_world(camera, view_min, view_max, to, b, depth_b))
                continue;
            consider(index, dist2_segment(mouse, a, b), 9.F);
        }
        if (center_ok) {
            const float dx = mouse.x - center_screen.x;
            const float dy = mouse.y - center_screen.y;
            if (dx * dx + dy * dy <= best) return 3;
        }
    }
    if (best_axis >= 0) return best_axis;
    const WorldRay ray = camera_ray(camera, view_min, view_max, mouse);
    if (volume_ray_hit(ray, volume_center_, axis, half, sphere)) return 3;
    return -1;
}

void SplatEdit::begin_volume_drag(
    const int handle, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera, const ImVec2 mouse) {
    std::copy(std::begin(volume_center_), std::end(volume_center_), volume_drag_center_);
    std::copy(std::begin(volume_quat_), std::end(volume_quat_), volume_drag_quat_);
    std::copy(std::begin(volume_size_), std::end(volume_size_), volume_drag_size_);
    volume_drag_ = handle;
    const WorldRay ray = camera_ray(camera, view_min, view_max, mouse);
    float axis[3][3];
    if (tool_ == Tool::sphere) {
        axis[0][0] = axis[1][1] = axis[2][2] = 1.F;
        axis[0][1] = axis[0][2] = axis[1][0] = axis[1][2] = axis[2][0] = axis[2][1] =
            0.F;
    } else
        quat_axes(volume_drag_quat_, axis);
    float view[3];
    view_direction(camera, volume_drag_center_, view);
    float hit[3];
    if (handle == 3) {
        if (!ray_plane(ray, volume_drag_center_, view, hit)) {
            volume_drag_ = -1;
            return;
        }
        std::copy(std::begin(hit), std::end(hit), volume_drag_vec_);
        return;
    }
    const bool scaling = handle >= 10;
    const int axis_index = scaling ? handle - 10 : handle;
    if (axis_index < 0 || axis_index > 2) {
        volume_drag_ = -1;
        return;
    }
    const float* direction = axis[axis_index];
    if (!scaling && volume_gesture_ == VolumeGesture::rotate) {
        if (!ray_plane(ray, volume_drag_center_, direction, hit)) {
            volume_drag_ = -1;
            return;
        }
        volume_drag_vec_[0] = hit[0] - volume_drag_center_[0];
        volume_drag_vec_[1] = hit[1] - volume_drag_center_[1];
        volume_drag_vec_[2] = hit[2] - volume_drag_center_[2];
        return;
    }
    float normal[3];
    axis_plane_normal(direction, view, normal);
    if (!ray_plane(ray, volume_drag_center_, normal, hit)) {
        volume_drag_ = -1;
        return;
    }
    const float rel[3] = {
        hit[0] - volume_drag_center_[0], hit[1] - volume_drag_center_[1],
        hit[2] - volume_drag_center_[2]};
    volume_drag_param_ = dot3(rel, direction);
}

void SplatEdit::update_volume_drag(
    const ImVec2 view_min, const ImVec2 view_max, const splat_render::Camera& camera,
    const ImVec2 mouse) {
    if (volume_drag_ < 0) return;
    const WorldRay ray = camera_ray(camera, view_min, view_max, mouse);
    float axis[3][3];
    if (tool_ == Tool::sphere) {
        axis[0][0] = axis[1][1] = axis[2][2] = 1.F;
        axis[0][1] = axis[0][2] = axis[1][0] = axis[1][2] = axis[2][0] = axis[2][1] =
            0.F;
    } else
        quat_axes(volume_drag_quat_, axis);
    float view[3];
    view_direction(camera, volume_drag_center_, view);
    float hit[3];
    if (volume_drag_ == 3) {
        if (!ray_plane(ray, volume_drag_center_, view, hit)) return;
        for (int k = 0; k < 3; ++k)
            volume_center_[k] =
                volume_drag_center_[k] + hit[k] - volume_drag_vec_[k];
        return;
    }
    const bool scaling = volume_drag_ >= 10;
    const int axis_index = scaling ? volume_drag_ - 10 : volume_drag_;
    if (axis_index < 0 || axis_index > 2) return;
    const float* direction = axis[axis_index];
    if (!scaling && volume_gesture_ == VolumeGesture::rotate) {
        if (!ray_plane(ray, volume_drag_center_, direction, hit)) return;
        const float current[3] = {
            hit[0] - volume_drag_center_[0], hit[1] - volume_drag_center_[1],
            hit[2] - volume_drag_center_[2]};
        float crossed[3];
        cross3(volume_drag_vec_, current, crossed);
        const float angle = std::atan2(dot3(crossed, direction), dot3(volume_drag_vec_, current));
        float delta[4];
        quat_from_axis(direction, angle, delta);
        quat_mul(delta, volume_drag_quat_, volume_quat_);
        const float qlen = std::sqrt(
            volume_quat_[0] * volume_quat_[0] + volume_quat_[1] * volume_quat_[1] +
            volume_quat_[2] * volume_quat_[2] + volume_quat_[3] * volume_quat_[3]);
        if (qlen > 1e-8F) {
            for (float& component : volume_quat_) component /= qlen;
        }
        return;
    }
    float normal[3];
    axis_plane_normal(direction, view, normal);
    if (!ray_plane(ray, volume_drag_center_, normal, hit)) return;
    const float rel[3] = {
        hit[0] - volume_drag_center_[0], hit[1] - volume_drag_center_[1],
        hit[2] - volume_drag_center_[2]};
    const float param = dot3(rel, direction);
    if (scaling) {
        const float ratio = std::fabs(volume_drag_param_) > 1e-4F
            ? std::fabs(param / volume_drag_param_)
            : 1.F;
        if (tool_ == Tool::sphere)
            volume_size_[0] = std::max(0.02F, volume_drag_size_[0] * ratio);
        else
            volume_size_[axis_index] =
                std::max(0.04F, volume_drag_size_[axis_index] * ratio);
        return;
    }
    for (int k = 0; k < 3; ++k)
        volume_center_[k] = volume_drag_center_[k] +
                            direction[k] * (param - volume_drag_param_);
}

void SplatEdit::end_volume_drag() {
    if (volume_drag_ < 0) return;
    volume_drag_ = -1;
    bool same = true;
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(volume_center_[k] - volume_drag_center_[k]) > 1e-5F) same = false;
        if (std::fabs(volume_size_[k] - volume_drag_size_[k]) > 1e-5F) same = false;
    }
    for (int k = 0; k < 4; ++k)
        if (std::fabs(volume_quat_[k] - volume_drag_quat_[k]) > 1e-5F) same = false;
    if (same) return;
    Snapshot snap;
    snap.kind = Snapshot::Kind::volume;
    std::copy(std::begin(volume_drag_center_), std::end(volume_drag_center_), snap.volume_center);
    std::copy(std::begin(volume_drag_quat_), std::end(volume_drag_quat_), snap.volume_quat);
    std::copy(std::begin(volume_drag_size_), std::end(volume_drag_size_), snap.volume_size);
    push_undo(std::move(snap));
}

void SplatEdit::handle_volume(
    App&, const ImVec2 view_min, const ImVec2 view_max,
    const splat_render::Camera& camera, const bool hot) {
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (volume_drag_ >= 0) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left))
            update_volume_drag(view_min, view_max, camera, mouse);
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) end_volume_drag();
        return;
    }
    if (!(hot && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) return;
    const int handle = pick_volume(view_min, view_max, camera, mouse);
    if (handle >= 0) begin_volume_drag(handle, view_min, view_max, camera, mouse);
}

}  // namespace editor