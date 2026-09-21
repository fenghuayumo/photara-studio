#include "image_qa_view.hpp"

#include "i18n.hpp"
#include "icons.hpp"
#include "theme.hpp"

#include "io/image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace editor {
namespace {

using i18n::tr;

constexpr float k_toolbar_h = 48.F;
constexpr float k_filmstrip_h = 104.F;
constexpr std::uint32_t k_qa_long_edge = 1600;
constexpr std::uint32_t k_metric_long_edge = k_image_qa_metric_extent;

bool is_image_extension(std::string extension) {
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".tif" ||
           extension == ".tiff" || extension == ".bmp";
}

aetherscan::io::RgbImage resize_rgb(
    const aetherscan::io::RgbImage& source, const std::uint32_t width,
    const std::uint32_t height) {
    aetherscan::io::RgbImage out;
    if (source.width == 0 || source.height == 0 || width == 0 || height == 0)
        return out;
    if (source.width == width && source.height == height) return source;
    out.width = width;
    out.height = height;
    out.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    aetherscan::io::resize_bilinear(
        source.pixels.data(), source.width, source.height, 3, out.pixels.data(),
        width, height);
    return out;
}

aetherscan::io::RgbImage load_qa_image(const std::filesystem::path& path) {
    // Decode straight to the largest DCT scale that still covers the QA
    // texture, so the full capture never has to be materialised just to be
    // scaled down. The texture size itself is unchanged.
    aetherscan::io::RgbImage rgb = aetherscan::io::load_rgb_with_minimum_size(
        path, k_qa_long_edge, k_qa_long_edge);
    const std::uint32_t long_edge = std::max(rgb.width, rgb.height);
    if (long_edge <= k_qa_long_edge || long_edge == 0) return rgb;
    const float scale =
        static_cast<float>(k_qa_long_edge) / static_cast<float>(long_edge);
    const std::uint32_t width = std::max(
        1U, static_cast<std::uint32_t>(
                std::lround(static_cast<float>(rgb.width) * scale)));
    const std::uint32_t height = std::max(
        1U, static_cast<std::uint32_t>(
                std::lround(static_cast<float>(rgb.height) * scale)));
    return resize_rgb(rgb, width, height);
}

float luma(const std::uint8_t* pixel) {
    return 0.2126F * static_cast<float>(pixel[0]) +
           0.7152F * static_cast<float>(pixel[1]) +
           0.0722F * static_cast<float>(pixel[2]);
}

void fit_metric_size(
    const std::uint32_t src_w, const std::uint32_t src_h, std::uint32_t& w,
    std::uint32_t& h) {
    const std::uint32_t long_edge = std::max(src_w, src_h);
    if (long_edge <= k_metric_long_edge || long_edge == 0) {
        w = src_w;
        h = src_h;
        return;
    }
    const float scale =
        static_cast<float>(k_metric_long_edge) / static_cast<float>(long_edge);
    w = std::max(
        1U, static_cast<std::uint32_t>(
                std::lround(static_cast<float>(src_w) * scale)));
    h = std::max(
        1U, static_cast<std::uint32_t>(
                std::lround(static_cast<float>(src_h) * scale)));
}

ImageQaMetrics compute_metrics(
    const aetherscan::io::RgbImage& gt, const aetherscan::io::RgbImage& render) {
    ImageQaMetrics metrics;
    if (gt.width == 0 || gt.height == 0 || render.width == 0 ||
        render.height == 0)
        return metrics;

    std::uint32_t mw{};
    std::uint32_t mh{};
    fit_metric_size(gt.width, gt.height, mw, mh);
    const aetherscan::io::RgbImage a = resize_rgb(gt, mw, mh);
    const aetherscan::io::RgbImage b = resize_rgb(render, mw, mh);
    if (a.pixels.size() != b.pixels.size() || a.pixels.empty()) return metrics;

    const std::size_t pixels = static_cast<std::size_t>(mw) * mh;
    double se = 0.0;
    double ae = 0.0;
    for (std::size_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double d =
                (static_cast<double>(a.pixels[3 * i + c]) -
                 static_cast<double>(b.pixels[3 * i + c])) /
                255.0;
            se += d * d;
            ae += std::abs(d);
        }
    }
    const double samples = static_cast<double>(pixels) * 3.0;
    const double mse = se / samples;
    metrics.mae = static_cast<float>(ae / samples);
    metrics.rmse = static_cast<float>(std::sqrt(mse));
    metrics.psnr = mse <= 1e-12
        ? 99.F
        : static_cast<float>(-10.0 * std::log10(mse));

    constexpr int k_win = 8;
    constexpr double k_c1 = 6.5025;
    constexpr double k_c2 = 58.5225;
    double ssim_sum = 0.0;
    int windows = 0;
    if (mw >= k_win && mh >= k_win) {
        for (std::uint32_t y = 0; y + k_win <= mh; y += k_win) {
            for (std::uint32_t x = 0; x + k_win <= mw; x += k_win) {
                double sum_a = 0.0;
                double sum_b = 0.0;
                double sum_aa = 0.0;
                double sum_bb = 0.0;
                double sum_ab = 0.0;
                for (int yy = 0; yy < k_win; ++yy) {
                    for (int xx = 0; xx < k_win; ++xx) {
                        const std::size_t index =
                            (static_cast<std::size_t>(y + yy) * mw +
                             static_cast<std::size_t>(x + xx));
                        const double la = luma(a.pixels.data() + index * 3);
                        const double lb = luma(b.pixels.data() + index * 3);
                        sum_a += la;
                        sum_b += lb;
                        sum_aa += la * la;
                        sum_bb += lb * lb;
                        sum_ab += la * lb;
                    }
                }
                constexpr double n = static_cast<double>(k_win * k_win);
                const double mu_a = sum_a / n;
                const double mu_b = sum_b / n;
                const double var_a = sum_aa / n - mu_a * mu_a;
                const double var_b = sum_bb / n - mu_b * mu_b;
                const double cov = sum_ab / n - mu_a * mu_b;
                const double num =
                    (2.0 * mu_a * mu_b + k_c1) * (2.0 * cov + k_c2);
                const double den =
                    (mu_a * mu_a + mu_b * mu_b + k_c1) *
                    (var_a + var_b + k_c2);
                if (den > 1e-12) {
                    ssim_sum += num / den;
                    ++windows;
                }
            }
        }
    }
    metrics.ssim = windows > 0
        ? static_cast<float>(std::clamp(ssim_sum / windows, 0.0, 1.0))
        : 0.F;
    metrics.valid = true;
    return metrics;
}

void turbo_colour(const float t, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
    static constexpr float stops[][3] = {
        {0.086F, 0.004F, 0.298F}, {0.113F, 0.435F, 0.816F},
        {0.145F, 0.749F, 0.624F}, {0.831F, 0.914F, 0.216F},
        {0.976F, 0.227F, 0.161F},
    };
    const float x = std::clamp(t, 0.F, 1.F) * 4.F;
    const int i = std::min(3, static_cast<int>(x));
    const float f = x - static_cast<float>(i);
    r = static_cast<std::uint8_t>(
        std::lround((stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f) * 255.F));
    g = static_cast<std::uint8_t>(
        std::lround((stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f) * 255.F));
    b = static_cast<std::uint8_t>(
        std::lround((stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f) * 255.F));
}

aetherscan::io::RgbImage make_error_map(
    const aetherscan::io::RgbImage& gt, const aetherscan::io::RgbImage& render) {
    aetherscan::io::RgbImage map;
    if (gt.width == 0 || gt.height == 0) return map;
    const aetherscan::io::RgbImage aligned =
        resize_rgb(render, gt.width, gt.height);
    if (aligned.pixels.size() != gt.pixels.size()) return map;

    const std::size_t pixels =
        static_cast<std::size_t>(gt.width) * gt.height;
    std::vector<float> error(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        float se = 0.F;
        for (int c = 0; c < 3; ++c) {
            const float d =
                (static_cast<float>(gt.pixels[3 * i + c]) -
                 static_cast<float>(aligned.pixels[3 * i + c])) /
                255.F;
            se += d * d;
        }
        error[i] = std::sqrt(se / 3.F);
    }

    std::vector<float> sorted = error;
    const std::size_t p95 = static_cast<std::size_t>(
        std::min<double>(sorted.size() - 1, sorted.size() * 0.95));
    std::nth_element(
        sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(p95),
        sorted.end());
    const float scale = std::max(1e-4F, sorted[p95]);

    map.width = gt.width;
    map.height = gt.height;
    map.pixels.resize(pixels * 3);
    for (std::size_t i = 0; i < pixels; ++i) {
        turbo_colour(
            error[i] / scale, map.pixels[3 * i], map.pixels[3 * i + 1],
            map.pixels[3 * i + 2]);
    }
    return map;
}

struct QaItem {
    std::string name;
    std::filesystem::path path;
    const ViewPose* pose{};
    bool registered{};
};

QaItem item_at(
    const ImageQaState& state, const SparseScene& scene, const int index) {
    QaItem item;
    if (index < 0) return item;
    if (!scene.views.empty()) {
        if (static_cast<std::size_t>(index) >= scene.views.size()) return item;
        const ViewPose& pose = scene.views[static_cast<std::size_t>(index)];
        item.name = pose.name.empty() ? pose.image_path.filename().string()
                                      : pose.name;
        item.path = pose.image_path;
        item.pose = &pose;
        item.registered = pose.registered;
        return item;
    }
    if (static_cast<std::size_t>(index) >= state.folder_images.size())
        return item;
    item.path = state.folder_images[static_cast<std::size_t>(index)];
    item.name = item.path.filename().string();
    return item;
}

void draw_empty(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max, const char* title,
    const char* hint) {
    const float title_w = ImGui::CalcTextSize(title).x;
    const float hint_w = ImGui::CalcTextSize(hint).x;
    const float cx = (min.x + max.x) * 0.5F;
    const float y = min.y + (max.y - min.y) * 0.38F;
    draw->AddText(
        {cx - title_w * 0.5F, y}, theme::u32(theme::text_muted), title);
    draw->AddText(
        {cx - hint_w * 0.5F, y + 22.F}, theme::u32(theme::text_faint), hint);
}

void draw_chip(
    ImDrawList* draw, const ImVec2 origin, const char* label,
    const ImVec4& text, const ImVec4& fill) {
    const ImVec2 size = ImGui::CalcTextSize(label);
    const ImVec2 max{
        origin.x + size.x + 16.F, origin.y + size.y + 8.F};
    draw->AddRectFilled(origin, max, theme::u32(fill), 5.F);
    draw->AddText({origin.x + 8.F, origin.y + 4.F}, theme::u32(text), label);
}

bool mode_chip(
    const char* id, const icons::Icon icon, const char* label, const bool active,
    const bool enabled) {
    const ImVec2 text = ImGui::CalcTextSize(label);
    const ImVec2 size{text.x + 38.F, 30.F};
    ImGui::PushID(id);
    if (!enabled) ImGui::BeginDisabled();
    const bool pressed = ImGui::InvisibleButton("##chip", size);
    if (!enabled) ImGui::EndDisabled();
    const ImVec2 chip_min = ImGui::GetItemRectMin();
    const ImVec2 chip_max = ImGui::GetItemRectMax();
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec4 fill = !enabled
        ? ImVec4(0, 0, 0, 0)
        : (active ? theme::fade(theme::accent, 0.16F)
                  : (hovered ? theme::surface_3 : ImVec4(0, 0, 0, 0)));
    draw->AddRectFilled(chip_min, chip_max, theme::u32(fill), 6.F);
    if (active)
        draw->AddRect(chip_min, chip_max, theme::u32(theme::accent, 0.45F), 6.F);
    const ImVec4 icon_colour = !enabled
        ? theme::text_faint
        : (active ? theme::accent : theme::text_bright);
    icons::draw(
        draw, icon, {chip_min.x + 6.F, chip_min.y + 6.F},
        {chip_min.x + 24.F, chip_min.y + 24.F}, theme::u32(icon_colour), 1.6F);
    draw->AddText(
        {chip_min.x + 28.F, chip_min.y + (size.y - text.y) * 0.5F},
        theme::u32(icon_colour), label);
    ImGui::PopID();
    return pressed && enabled;
}

ImVec4 metric_tone(const float value, const float good, const float ok) {
    if (value >= good) return theme::success;
    if (value >= ok) return theme::warning;
    return theme::danger;
}

void draw_metric_card(
    ImDrawList* draw, ImVec2& origin, const char* key, const char* value,
    const ImVec4& accent) {
    ImFont* small = theme::small_font();
    ImFont* mono = theme::mono_font();
    const float key_w = small->CalcTextSizeA(small->FontSize, 1e6F, 0.F, key).x;
    const float val_w = mono->CalcTextSizeA(mono->FontSize, 1e6F, 0.F, value).x;
    const float width = std::max(key_w, val_w) + 20.F;
    const ImVec2 max{origin.x + width, origin.y + 40.F};
    draw->AddRectFilled(origin, max, IM_COL32(16, 18, 23, 230), 6.F);
    draw->AddRect(origin, max, theme::u32(theme::border, 0.7F), 6.F);
    draw->AddRectFilled(
        {origin.x, origin.y + 8.F}, {origin.x + 2.F, max.y - 8.F},
        theme::u32(accent));
    draw->AddText(
        small, small->FontSize, {origin.x + 10.F, origin.y + 5.F},
        theme::u32(theme::text_faint), key);
    draw->AddText(
        mono, mono->FontSize, {origin.x + 10.F, origin.y + 19.F},
        theme::u32(theme::text_bright), value);
    origin.x = max.x + 8.F;
}

void draw_features(
    ImDrawList* draw, const ViewPose& pose, const ImVec2 img_min,
    const ImVec2 img_max, const ImageQaState& state) {
    const float w = img_max.x - img_min.x;
    const float h = img_max.y - img_min.y;
    if (w < 8.F || h < 8.F || pose.features.empty()) return;

    constexpr std::size_t k_budget = 5'000;
    const std::size_t stride = std::max<std::size_t>(
        1, pose.features.size() / k_budget);
    draw->PushClipRect(img_min, img_max, true);
    for (std::size_t i = 0; i < pose.features.size(); i += stride) {
        const ImageFeature& feature = pose.features[i];
        if (feature.triangulated && !state.show_triangulated) continue;
        if (!feature.triangulated && !state.show_untracked) continue;
        const ImVec2 p{
            img_min.x + feature.u * w, img_min.y + feature.v * h};
        const float radius = std::clamp(feature.scale * w * 0.55F, 1.6F, 7.F);
        if (feature.triangulated) {
            draw->AddCircle(
                p, radius, theme::u32(theme::accent, 0.95F), 8, 1.2F);
            draw->AddCircleFilled(p, 1.4F, theme::u32(theme::accent));
        } else {
            draw->AddCircle(
                p, std::max(1.4F, radius * 0.7F),
                theme::u32(theme::warning, 0.55F), 7, 1.F);
        }
    }
    draw->PopClipRect();
}

void draw_live_keypoints(
    ImDrawList* draw, const std::vector<aetherscan::sfm::AlignLiveKeypoint>& points,
    const ImVec2 img_min, const ImVec2 img_max, const ImU32 colour) {
    const float w = img_max.x - img_min.x;
    const float h = img_max.y - img_min.y;
    if (w < 8.F || h < 8.F || points.empty()) return;
    draw->PushClipRect(img_min, img_max, true);
    for (const auto& point : points) {
        const ImVec2 p{img_min.x + point.u * w, img_min.y + point.v * h};
        const float radius = std::clamp(point.scale * w * 0.55F, 1.6F, 6.F);
        draw->AddCircle(p, radius, colour, 7, 1.1F);
    }
    draw->PopClipRect();
}

ImVec4 lerp_colour(const ImVec4& a, const ImVec4& b, const float t) {
    const float x = std::clamp(t, 0.F, 1.F);
    return {
        a.x + (b.x - a.x) * x, a.y + (b.y - a.y) * x,
        a.z + (b.z - a.z) * x, a.w + (b.w - a.w) * x};
}

float dist2_to_segment(const ImVec2 p, const ImVec2 a, const ImVec2 b) {
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float apx = p.x - a.x;
    const float apy = p.y - a.y;
    const float ab2 = abx * abx + aby * aby;
    const float t = ab2 > 1e-8F
        ? std::clamp((apx * abx + apy * aby) / ab2, 0.F, 1.F)
        : 0.F;
    const float dx = apx - abx * t;
    const float dy = apy - aby * t;
    return dx * dx + dy * dy;
}

int pick_live_match(
    const std::vector<aetherscan::sfm::AlignLiveMatch>& matches,
    const ImVec2 a_min, const ImVec2 a_max, const ImVec2 b_min,
    const ImVec2 b_max, const ImVec2 mouse) {
    const float aw = a_max.x - a_min.x;
    const float ah = a_max.y - a_min.y;
    const float bw = b_max.x - b_min.x;
    const float bh = b_max.y - b_min.y;
    if (aw < 8.F || ah < 8.F || bw < 8.F || bh < 8.F || matches.empty())
        return -1;
    constexpr float k_max = 14.F * 14.F;
    int best = -1;
    float best_d = k_max;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const auto& match = matches[static_cast<std::size_t>(i)];
        const ImVec2 p0{a_min.x + match.u0 * aw, a_min.y + match.v0 * ah};
        const ImVec2 p1{b_min.x + match.u1 * bw, b_min.y + match.v1 * bh};
        const float d = dist2_to_segment(mouse, p0, p1);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

void draw_live_matches(
    ImDrawList* draw, const std::vector<aetherscan::sfm::AlignLiveMatch>& matches,
    const ImVec2 a_min, const ImVec2 a_max, const ImVec2 b_min,
    const ImVec2 b_max, const int hovered, const bool verified) {
    const float aw = a_max.x - a_min.x;
    const float ah = a_max.y - a_min.y;
    const float bw = b_max.x - b_min.x;
    const float bh = b_max.y - b_min.y;
    if (aw < 8.F || ah < 8.F || bw < 8.F || bh < 8.F || matches.empty()) return;
    float lo = matches.front().score;
    float hi = matches.front().score;
    for (const auto& match : matches) {
        lo = std::min(lo, match.score);
        hi = std::max(hi, match.score);
    }
    const float span = std::max(1e-5F, hi - lo);
    const bool isolate = hovered >= 0 &&
                         hovered < static_cast<int>(matches.size());
    const ImVec4 high = verified ? theme::success : theme::accent;
    const ImVec4 low = verified ? theme::fade(theme::success, 0.55F)
                                : theme::warning;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const auto& match = matches[static_cast<std::size_t>(i)];
        const ImVec2 p0{a_min.x + match.u0 * aw, a_min.y + match.v0 * ah};
        const ImVec2 p1{b_min.x + match.u1 * bw, b_min.y + match.v1 * bh};
        const float t = verified ? 1.F : (match.score - lo) / span;
        const bool hot = isolate && i == hovered;
        const bool dim = isolate && !hot;
        const ImVec4 colour = hot ? theme::text_bright : lerp_colour(low, high, t);
        const float alpha = dim ? 0.07F : (hot ? 1.F : 0.32F + 0.60F * t);
        draw->AddLine(p0, p1, theme::u32(colour, alpha), hot ? 2.4F : 1.25F);
        if (dim) continue;
        const float radius = hot ? 3.6F : 2.2F;
        const ImU32 dot = theme::u32(hot ? theme::text_bright : high, 0.95F);
        draw->AddCircleFilled(p0, radius, dot);
        draw->AddCircleFilled(p1, radius, dot);
    }
}

void draw_colour_key(
    ImDrawList* draw, ImVec2& origin, const ImVec4& colour, const char* label) {
    ImFont* small = theme::small_font();
    const float text_w =
        small->CalcTextSizeA(small->FontSize, 240.F, 0.F, label).x;
    draw->AddCircleFilled(
        {origin.x + 5.F, origin.y + small->FontSize * 0.55F}, 4.F,
        theme::u32(colour));
    draw->AddText(
        small, small->FontSize, {origin.x + 14.F, origin.y},
        theme::u32(theme::text_muted), label);
    origin.x += text_w + 28.F;
}

void draw_named_chip(
    ImDrawList* draw, const ImVec2 origin, const char* tag, const char* name,
    const ImVec4& mark) {
    char text[192];
    if (name != nullptr && name[0] != '\0')
        std::snprintf(text, sizeof(text), "%s  %s", tag, name);
    else
        std::snprintf(text, sizeof(text), "%s", tag);
    ImFont* small = theme::small_font();
    const ImVec2 size =
        small->CalcTextSizeA(small->FontSize, 260.F, 0.F, text);
    const ImVec2 max{origin.x + size.x + 18.F, origin.y + size.y + 10.F};
    draw->AddRectFilled(
        origin, max, theme::u32(theme::fade(theme::surface_0, 0.78F)), 5.F);
    draw->AddRectFilled(
        {origin.x, origin.y + 5.F}, {origin.x + 3.F, max.y - 5.F},
        theme::u32(mark));
    draw->PushClipRect(origin, max, true);
    draw->AddText(
        small, small->FontSize, {origin.x + 10.F, origin.y + 5.F},
        theme::u32(theme::text_bright), text);
    draw->PopClipRect();
}

void fit_image_rect(
    const ImVec2 canvas_min, const ImVec2 canvas_max, const float img_w,
    const float img_h, const float zoom, const ImVec2 pan, ImVec2& out_min,
    ImVec2& out_max) {
    const float avail_w = std::max(1.F, canvas_max.x - canvas_min.x);
    const float avail_h = std::max(1.F, canvas_max.y - canvas_min.y);
    const float fit = std::min(avail_w / img_w, avail_h / img_h);
    const float scale = fit * std::max(0.15F, zoom);
    const ImVec2 size{img_w * scale, img_h * scale};
    const ImVec2 centre{
        (canvas_min.x + canvas_max.x) * 0.5F + pan.x,
        (canvas_min.y + canvas_max.y) * 0.5F + pan.y};
    out_min = {centre.x - size.x * 0.5F, centre.y - size.y * 0.5F};
    out_max = {centre.x + size.x * 0.5F, centre.y + size.y * 0.5F};
}

}  // namespace

ImageQaSession::~ImageQaSession() { clear(); }

void ImageQaSession::clear() {
    if (pending_.valid()) pending_.wait();
    for (auto& retired : retired_)
        if (retired.second.valid()) retired.second.wait();
    if (metrics_pending_.valid()) metrics_pending_.wait();
    retired_.clear();
    gt_cache_.clear();
    gt_cache_order_.clear();
    gt_.reset();
    error_.reset();
    gt_cpu_ = {};
    render_cpu_ = {};
    pending_ = {};
    metrics_pending_ = {};
    pending_path_.clear();
    loaded_path_.clear();
    pending_view_ = -1;
    loaded_view_ = -1;
    render_view_ = -1;
    render_revision_ = 0;
    loading_ = false;
    metrics_busy_ = false;
    failed_ = false;
    metrics_ = {};
}

void ImageQaSession::harvest_retired() {
    for (auto it = retired_.begin(); it != retired_.end();) {
        if (it->second.wait_for(std::chrono::seconds(0)) ==
            std::future_status::timeout) {
            ++it;
            continue;
        }
        try {
            // A superseded decode cannot change what is on screen, but it is
            // still a usable capture: keep it for the next visit.
            const aetherscan::io::RgbImage image = it->second.get();
            remember_gt(it->first, image);
        } catch (...) {
        }
        it = retired_.erase(it);
    }
}

void ImageQaSession::publish_gt(const aetherscan::io::RgbImage& image) {
    if (image.width == 0 || image.height == 0 || image.pixels.empty())
        throw std::runtime_error("empty capture");
    gt_cpu_ = image;
    gt_.upload(gt_cpu_);
    // The previous heatmap belongs to another capture. Tear it down before this
    // frame records ImGui image draws.
    error_.reset();
    metrics_ = {};
    queue_metrics();
}

void ImageQaSession::remember_gt(
    const std::filesystem::path::string_type& key,
    const aetherscan::io::RgbImage& image) {
    constexpr std::size_t k_cached_captures = 8;
    if (image.width == 0 || image.height == 0 || image.pixels.empty()) return;
    gt_cache_[key] = image;
    gt_cache_order_.push_back(key);
    while (gt_cache_order_.size() > k_cached_captures) {
        const std::filesystem::path::string_type oldest =
            gt_cache_order_.front();
        gt_cache_order_.pop_front();
        if (std::find(
                gt_cache_order_.begin(), gt_cache_order_.end(), oldest) ==
            gt_cache_order_.end())
            gt_cache_.erase(oldest);
    }
}

bool ImageQaSession::has_gt() const {
    return gt_.descriptor != nullptr && gt_cpu_.width > 0;
}

bool ImageQaSession::has_error() const { return error_.descriptor != nullptr; }

bool ImageQaSession::has_render_pixels() const {
    return render_view_ == loaded_view_ && loaded_view_ >= 0 &&
           !render_cpu_.pixels.empty();
}

ImTextureID ImageQaSession::gt_id() const {
    return gt_.descriptor ? reinterpret_cast<ImTextureID>(gt_.descriptor)
                          : ImTextureID{};
}

ImTextureID ImageQaSession::error_id() const {
    return error_.descriptor ? reinterpret_cast<ImTextureID>(error_.descriptor)
                             : ImTextureID{};
}

void ImageQaSession::request_gt(
    const int view_index, const std::filesystem::path& path) {
    if (path.empty() || view_index < 0) return;
    if (loaded_view_ == view_index && loaded_path_ == path && has_gt()) return;
    if (loading_ && pending_view_ == view_index && pending_path_ == path) return;
    harvest_retired();
    // A capture that is already decoded in this session is published in place.
    const std::filesystem::path::string_type key = path.native();
    if (const auto found = gt_cache_.find(key); found != gt_cache_.end()) {
        if (pending_.valid())
            retired_.emplace_back(
                pending_path_.native(), std::move(pending_));
        loading_ = false;
        failed_ = false;
        pending_view_ = -1;
        pending_path_.clear();
        loaded_path_ = path;
        loaded_view_ = view_index;
        try {
            publish_gt(found->second);
        } catch (...) {
            failed_ = true;
            gt_.reset();
            gt_cpu_ = {};
            loaded_view_ = -1;
        }
        return;
    }
    // Never join the in-flight decode: it belongs to a capture the user has
    // already moved past, and the UI must stay responsive while it lands.
    if (pending_.valid())
        retired_.emplace_back(pending_path_.native(), std::move(pending_));
    pending_path_ = path;
    pending_view_ = view_index;
    failed_ = false;
    loading_ = true;
    pending_ = std::async(std::launch::async, [path] {
        return load_qa_image(path);
    });
}

void ImageQaSession::poll() {
    if (loading_ && pending_.valid() &&
        pending_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            aetherscan::io::RgbImage image = pending_.get();
            remember_gt(pending_path_.native(), image);
            loaded_path_ = pending_path_;
            loaded_view_ = pending_view_;
            publish_gt(image);
        } catch (...) {
            failed_ = true;
            gt_.reset();
            gt_cpu_ = {};
            loaded_view_ = -1;
        }
        loading_ = false;
    }

    if (metrics_busy_ && metrics_pending_.valid() &&
        metrics_pending_.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
        try {
            MetricsJob job = metrics_pending_.get();
            metrics_busy_ = false;
            if (job.view != render_view_ || job.revision != render_revision_) {
                queue_metrics();
            } else {
                metrics_ = job.metrics;
                if (job.error.width > 0) error_.upload(job.error);
            }
        } catch (...) {
            metrics_busy_ = false;
            metrics_ = {};
        }
    }
}

void ImageQaSession::set_render(
    aetherscan::io::RgbImage render, const int view,
    const std::uint64_t revision) {
    render_cpu_ = std::move(render);
    render_view_ = view;
    render_revision_ = revision;
    queue_metrics();
}

void ImageQaSession::queue_metrics() {
    if (gt_cpu_.width == 0 || render_cpu_.width == 0) return;
    if (loaded_view_ < 0 || render_view_ != loaded_view_) return;
    if (metrics_busy_) return;
    metrics_ = {};
    // Keep the previous error texture until poll() replaces it. capture_qa_render
    // calls set_render() after draw_image_qa has already recorded AddImage.
    metrics_busy_ = true;
    const int view = render_view_;
    const std::uint64_t revision = render_revision_;
    aetherscan::io::RgbImage gt = gt_cpu_;
    aetherscan::io::RgbImage render = render_cpu_;
    metrics_pending_ = std::async(
        std::launch::async,
        [gt = std::move(gt), render = std::move(render), view, revision] {
            MetricsJob job;
            job.metrics = compute_metrics(gt, render);
            job.error = make_error_map(gt, render);
            job.view = view;
            job.revision = revision;
            return job;
        });
}

bool image_qa_needs_render(const ImageQaMode mode) {
    return mode == ImageQaMode::compare || mode == ImageQaMode::error;
}

void refresh_image_qa_folder(
    ImageQaState& state, const std::filesystem::path& directory) {
    const std::string key = directory.string();
    if (state.folder_dir == key) return;
    state.folder_dir = key;
    state.folder_images.clear();
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        if (is_image_extension(entry.path().extension().string()))
            state.folder_images.push_back(entry.path());
    }
    std::sort(state.folder_images.begin(), state.folder_images.end());
}

int image_qa_count(const ImageQaState& state, const SparseScene& scene) {
    if (!scene.views.empty())
        return static_cast<int>(scene.views.size());
    return static_cast<int>(state.folder_images.size());
}

int image_qa_index_for_path(
    const ImageQaState& state, const SparseScene& scene,
    const std::filesystem::path& path, const int fallback) {
    if (path.empty()) return fallback;
    if (!scene.views.empty()) {
        for (int i = 0; i < static_cast<int>(scene.views.size()); ++i) {
            if (scene.views[static_cast<std::size_t>(i)].image_path == path)
                return i;
        }
        return fallback;
    }
    for (int i = 0; i < static_cast<int>(state.folder_images.size()); ++i) {
        if (state.folder_images[static_cast<std::size_t>(i)] == path)
            return i;
    }
    const auto name = path.filename();
    for (int i = 0; i < static_cast<int>(state.folder_images.size()); ++i) {
        if (state.folder_images[static_cast<std::size_t>(i)].filename() == name)
            return i;
    }
    return fallback;
}

void select_image_qa_view(
    ImageQaState& state, const int index, const int count) {
    if (count <= 0) {
        state.selected = -1;
        return;
    }
    const int next = std::clamp(index, 0, count - 1);
    if (next == state.selected) return;
    state.selected = next;
    state.zoom = 1.F;
    state.pan = {};
    state.dragging_wipe = false;
    state.filmstrip_reselect = true;
    state.metrics_dirty = true;
}

void draw_image_qa(
    ImageQaState& state, ImageQaSession& session, const ImageQaDrawInput& input,
    const ImVec2 min, const ImVec2 max) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg));
    session.poll();
    if (input.pair_session != nullptr) input.pair_session->poll();

    const SparseScene empty_scene;
    const SparseScene& scene = input.scene ? *input.scene : empty_scene;
    const aetherscan::sfm::AlignLiveFrame* live = input.live;
    const bool live_features =
        live != nullptr && live->kind == aetherscan::sfm::AlignLiveKind::features;
    const bool live_matching =
        live != nullptr && aetherscan::sfm::is_live_pair_kind(live->kind);
    const bool live_inliers =
        live != nullptr && live->kind == aetherscan::sfm::AlignLiveKind::inliers;
    const int count = image_qa_count(state, scene);
    if (state.selected < 0 && count > 0) state.selected = 0;
    if (state.selected >= count) state.selected = count > 0 ? count - 1 : -1;
    const QaItem item = item_at(state, scene, state.selected);
    const std::filesystem::path live_a =
        live_features || live_matching ? live->path_a : std::filesystem::path{};
    if (!live_a.empty())
        session.request_gt(
            live->index_a >= 0 ? live->index_a : state.selected, live_a);
    else if (!item.path.empty())
        session.request_gt(state.selected, item.path);
    if (live_matching && input.pair_session != nullptr && !live->path_b.empty())
        input.pair_session->request_gt(
            live->index_b >= 0 ? live->index_b : 0, live->path_b);

    const ImVec2 toolbar_max{max.x, min.y + k_toolbar_h};
    draw->AddRectFilled(
        min, toolbar_max, theme::u32(theme::surface_1));
    draw->AddLine(
        {min.x, toolbar_max.y - 1.F}, {max.x, toolbar_max.y - 1.F},
        theme::u32(theme::border));

    ImGui::SetCursorScreenPos({min.x + 10.F, min.y + 9.F});
    struct ModeItem {
        const char* id;
        icons::Icon icon;
        const char* label;
        ImageQaMode mode;
        bool enabled;
        const char* tooltip;
    };
    const bool compare_ok = input.has_model || input.render_live;
    const bool live_align = live_features || live_matching;
    if (live_align) {
        ImFont* small = theme::small_font();
        char status[160];
        if (live_matching) {
            std::snprintf(
                status, sizeof(status), "%s",
                live_inliers ? tr("Verified pairs") : tr("Matching views"));
        } else {
            std::snprintf(
                status, sizeof(status), "%s  ·  %u %s",
                tr("Extracting features"), live->total_keypoints_a,
                tr("keypoints"));
        }
        draw->AddText(
            small, small->FontSize, {min.x + 14.F, min.y + 16.F},
            theme::u32(theme::accent), status);
        if (live_matching) {
            const float status_w = small->CalcTextSizeA(
                small->FontSize, 1e6F, 0.F, status).x;
            ImGui::SetCursorScreenPos({min.x + 22.F + status_w, min.y + 9.F});
            if (mode_chip(
                    "##qa_live_kp", icons::Icon::features, tr("Keypoints"),
                    state.show_live_keypoints, true))
                state.show_live_keypoints = !state.show_live_keypoints;
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                ImGui::SetTooltip("%s", tr("Show unmatched keypoints"));
        }
    } else {
        const ModeItem modes[] = {
            {"##qa_photo", icons::Icon::photo, tr("Photo"), ImageQaMode::photo,
             true, tr("Capture image")},
            {"##qa_feat", icons::Icon::features, tr("Features"),
             ImageQaMode::features, true,
             tr("Detected keypoints and triangulated tracks")},
            {"##qa_cmp", icons::Icon::compare, tr("Compare"), ImageQaMode::compare,
             true, tr("Slide to compare 3DGS against the training view")},
            {"##qa_err", icons::Icon::heatmap, tr("Error"), ImageQaMode::error,
             true, tr("Per-pixel photometric error map")},
        };
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {4.F, 0.F});
        for (const ModeItem& mode : modes) {
            if (mode_chip(
                    mode.id, mode.icon, mode.label, state.mode == mode.mode,
                    mode.enabled)) {
                if (state.mode != mode.mode) {
                    state.mode = mode.mode;
                    if (image_qa_needs_render(state.mode))
                        state.metrics_dirty = true;
                }
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                ImGui::SetTooltip("%s", mode.tooltip);
            ImGui::SameLine();
        }
        ImGui::PopStyleVar();
    }

    const ImVec2 nav_size{28.F, 28.F};
    char index_label[32];
    if (count > 0)
        std::snprintf(
            index_label, sizeof(index_label), "%d / %d", state.selected + 1,
            count);
    else
        std::snprintf(index_label, sizeof(index_label), "—");
    const ImVec2 label_size = ImGui::CalcTextSize(index_label);
    const float label_w = std::max(label_size.x, 56.F);
    const float nav_y = min.y + (k_toolbar_h - nav_size.y) * 0.5F;
    const float next_x = max.x - 12.F - nav_size.x;
    const float label_x = live_matching
        ? max.x - 16.F - label_w
        : next_x - 8.F - label_w;
    const float prev_x = label_x - 8.F - nav_size.x;

    ImGui::SetCursorScreenPos(
        {label_x, nav_y + (nav_size.y - label_size.y) * 0.5F});
    ImGui::PushStyleColor(ImGuiCol_Text, theme::text_muted);
    ImGui::TextUnformatted(index_label);
    ImGui::PopStyleColor();
    if (!live_matching) {
        ImGui::SetCursorScreenPos({prev_x, nav_y});
        if (icons::ghost_button(
                "##qa_prev", icons::Icon::chevron_left, nav_size, false, count > 1,
                tr("Previous image")))
            select_image_qa_view(state, state.selected - 1, count);
        ImGui::SetCursorScreenPos({next_x, nav_y});
        if (icons::ghost_button(
                "##qa_next", icons::Icon::chevron_right, nav_size, false,
                count > 1, tr("Next image")))
            select_image_qa_view(state, state.selected + 1, count);
    }

    if (!live_matching && !item.name.empty() && (max.x - min.x) > 780.F) {
        ImFont* small = theme::small_font();
        const float name_w =
            small->CalcTextSizeA(small->FontSize, 220.F, 0.F, item.name.c_str()).x;
        draw->AddText(
            small, small->FontSize,
            {prev_x - 12.F - name_w, min.y + (k_toolbar_h - small->FontSize) * 0.5F},
            theme::u32(theme::text_faint), item.name.c_str());
    }

    const bool show_strip = count > 0 && !live_matching;
    const ImVec2 canvas_min{min.x, toolbar_max.y};
    const ImVec2 canvas_max{
        max.x, show_strip ? max.y - k_filmstrip_h : max.y};

    ImGui::SetCursorScreenPos(canvas_min);
    ImGui::InvisibleButton(
        "##qa_canvas",
        {std::max(1.F, canvas_max.x - canvas_min.x),
         std::max(1.F, canvas_max.y - canvas_min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool canvas_hovered = ImGui::IsItemHovered();
    const ImGuiIO& io = ImGui::GetIO();

    const bool has_gt = session.has_gt();
    const float img_w = has_gt ? static_cast<float>(session.gt_width()) : 4.F;
    const float img_h = has_gt ? static_cast<float>(session.gt_height()) : 3.F;
    ImVec2 img_min{};
    ImVec2 img_max{};
    fit_image_rect(
        canvas_min, canvas_max, img_w, img_h, state.zoom, state.pan, img_min,
        img_max);

    if (canvas_hovered && !io.WantTextInput && io.MouseWheel != 0.F) {
        const float old = std::max(0.15F, state.zoom);
        const float next = std::clamp(
            old * std::exp(io.MouseWheel * 0.16F), 0.2F, 12.F);
        const ImVec2 mouse = io.MousePos;
        const ImVec2 centre{
            (canvas_min.x + canvas_max.x) * 0.5F + state.pan.x,
            (canvas_min.y + canvas_max.y) * 0.5F + state.pan.y};
        const float ratio = next / old;
        state.pan.x = (centre.x - mouse.x) * (ratio - 1.F) + state.pan.x;
        state.pan.y = (centre.y - mouse.y) * (ratio - 1.F) + state.pan.y;
        state.zoom = next;
        fit_image_rect(
            canvas_min, canvas_max, img_w, img_h, state.zoom, state.pan, img_min,
            img_max);
    }

    const bool compare = !live_align && state.mode == ImageQaMode::compare;
    const bool show_render = compare && input.has_render && has_gt;
    const float image_span = img_max.x - img_min.x;
    const float wipe_x =
        img_min.x + image_span * std::clamp(state.wipe, 0.F, 1.F);
    const bool over_image =
        io.MousePos.x >= img_min.x && io.MousePos.x <= img_max.x &&
        io.MousePos.y >= img_min.y && io.MousePos.y <= img_max.y;

    // Compare: left-drag anywhere on the photo moves the wipe. Middle-drag pans.
    if (show_render && canvas_hovered && over_image)
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    if (show_render && canvas_hovered && over_image &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        state.dragging_wipe = true;
    if (state.dragging_wipe && ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
        image_span > 1.F)
        state.wipe = std::clamp((io.MousePos.x - img_min.x) / image_span, 0.F, 1.F);
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) state.dragging_wipe = false;

    const bool panning =
        canvas_hovered && !state.dragging_wipe &&
        (ImGui::IsMouseDown(ImGuiMouseButton_Middle) ||
         (!show_render && ImGui::IsMouseDown(ImGuiMouseButton_Left)));
    if (panning) {
        state.pan.x += io.MouseDelta.x;
        state.pan.y += io.MouseDelta.y;
        fit_image_rect(
            canvas_min, canvas_max, img_w, img_h, state.zoom, state.pan, img_min,
            img_max);
    }
    if (canvas_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) &&
        !state.dragging_wipe) {
        state.zoom = 1.F;
        state.pan = {};
        fit_image_rect(
            canvas_min, canvas_max, img_w, img_h, state.zoom, state.pan, img_min,
            img_max);
    }
    if (canvas_hovered && !io.WantTextInput) {
        if (!live_matching) {
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
                select_image_qa_view(state, state.selected - 1, count);
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
                select_image_qa_view(state, state.selected + 1, count);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_F)) {
            state.zoom = 1.F;
            state.pan = {};
        }
    }

    draw->PushClipRect(canvas_min, canvas_max, true);
    if (live_matching) {
        const float gap = 8.F;
        const float mid = (canvas_min.x + canvas_max.x) * 0.5F;
        const ImVec2 left_max{mid - gap * 0.5F, canvas_max.y};
        const ImVec2 right_min{mid + gap * 0.5F, canvas_min.y};
        const bool has_b =
            input.pair_session != nullptr && input.pair_session->has_gt();
        ImVec2 a_min{};
        ImVec2 a_max{};
        ImVec2 b_min{};
        ImVec2 b_max{};
        const float aw = has_gt ? static_cast<float>(session.gt_width()) : 4.F;
        const float ah = has_gt ? static_cast<float>(session.gt_height()) : 3.F;
        const float bw = has_b
            ? static_cast<float>(input.pair_session->gt_width())
            : 4.F;
        const float bh = has_b
            ? static_cast<float>(input.pair_session->gt_height())
            : 3.F;
        fit_image_rect(
            canvas_min, left_max, aw, ah, state.zoom, state.pan, a_min, a_max);
        fit_image_rect(
            right_min, canvas_max, bw, bh, state.zoom, state.pan, b_min, b_max);
        if (has_gt) draw->AddImage(session.gt_id(), a_min, a_max);
        else
            draw_empty(
                draw, canvas_min, left_max, tr("Loading capture…"),
                tr("Reading the selected image"));
        if (has_b) draw->AddImage(input.pair_session->gt_id(), b_min, b_max);
        else
            draw_empty(
                draw, right_min, canvas_max, tr("Loading pair…"),
                tr("Reading the matched image"));
        const int hovered_match = has_gt && has_b && canvas_hovered && !panning
            ? pick_live_match(
                  live->matches, a_min, a_max, b_min, b_max, io.MousePos)
            : -1;
        if (state.show_live_keypoints) {
            if (has_gt)
                draw_live_keypoints(
                    draw, live->keypoints_a, a_min, a_max,
                    theme::u32(theme::warning, 0.28F));
            if (has_b)
                draw_live_keypoints(
                    draw, live->keypoints_b, b_min, b_max,
                    theme::u32(theme::warning, 0.28F));
        }
        if (has_gt && has_b)
            draw_live_matches(
                draw, live->matches, a_min, a_max, b_min, b_max, hovered_match,
                live_inliers);
        if (has_gt) {
            const std::string name_a = live->path_a.empty()
                ? std::string{}
                : live->path_a.filename().string();
            draw_named_chip(
                draw, {a_min.x + 8.F, a_min.y + 8.F}, "A", name_a.c_str(),
                theme::accent);
        }
        if (has_b) {
            const std::string name_b = live->path_b.empty()
                ? std::string{}
                : live->path_b.filename().string();
            draw_named_chip(
                draw, {b_min.x + 8.F, b_min.y + 8.F}, "B", name_b.c_str(),
                theme::warning);
        }
        if (hovered_match >= 0 &&
            hovered_match < static_cast<int>(live->matches.size())) {
            if (live_inliers) {
                ImGui::SetTooltip("%s", tr("line = verified correspondence"));
            } else {
                ImGui::SetTooltip(
                    "%s  %.3f", tr("score"),
                    live->matches[static_cast<std::size_t>(hovered_match)].score);
            }
        }
        char pair_legend[192];
        if (live_inliers) {
            std::snprintf(
                pair_legend, sizeof(pair_legend), "%u %s    ·    %s",
                live->total_matches, tr("inliers"),
                tr("line = verified correspondence"));
        } else {
            std::snprintf(
                pair_legend, sizeof(pair_legend), "%u %s    ·    %s",
                live->total_matches, tr("matches"),
                tr("brighter line = higher score"));
        }
        draw->AddText(
            {canvas_min.x + 16.F, canvas_max.y - 22.F},
            theme::u32(theme::text_muted), pair_legend);
    } else if (count <= 0 && !live_features) {
        draw_empty(
            draw, canvas_min, canvas_max, tr("No images to inspect"),
            tr("Select an image folder or align photos to open the 2D viewer"));
    } else if (session.loading()) {
        draw_empty(
            draw, canvas_min, canvas_max, tr("Loading capture…"),
            item.name.empty() ? tr("Reading the selected image")
                              : item.name.c_str());
    } else if (!has_gt) {
        draw_empty(
            draw, canvas_min, canvas_max, tr("Capture unavailable"),
            item.path.empty() ? tr("This view has no image path")
                              : tr("Could not decode the selected file"));
    } else if (!live_align && state.mode == ImageQaMode::error) {
        const bool error_current =
            session.has_error() && session.loaded_view() == state.selected &&
            session.render_view() == state.selected;
        if (error_current) {
            draw->AddImage(session.error_id(), img_min, img_max);
        } else if (!compare_ok) {
            draw->AddImage(session.gt_id(), img_min, img_max);
            draw_empty(
                draw, canvas_min, canvas_max, tr("Train 3DGS to build an error map"),
                tr("The heatmap compares the live splat against this capture"));
        } else if (!input.has_render) {
            draw->AddImage(session.gt_id(), img_min, img_max);
            draw_empty(
                draw, canvas_min, canvas_max, tr("Waiting for a rendered frame…"),
                tr("The live splat preview will appear here once it is ready"));
        } else {
            draw->AddImage(session.gt_id(), img_min, img_max);
            draw_empty(
                draw, canvas_min, canvas_max, tr("Computing error map…"),
                tr("PSNR / SSIM update as soon as both images are aligned"));
        }
    } else {
        draw->AddImage(session.gt_id(), img_min, img_max);
        if (show_render) {
            draw->PushClipRect({wipe_x, img_min.y}, img_max, true);
            draw->AddImage(input.render, img_min, img_max);
            draw->PopClipRect();
            draw->AddRectFilled(
                {wipe_x - 1.5F, img_min.y}, {wipe_x + 1.5F, img_max.y},
                theme::u32(theme::text_bright, 0.95F));
            draw->AddCircleFilled(
                {wipe_x, (img_min.y + img_max.y) * 0.5F}, 11.F,
                theme::u32(theme::accent));
            draw->AddCircle(
                {wipe_x, (img_min.y + img_max.y) * 0.5F}, 11.F,
                theme::u32(theme::text_bright, 0.95F), 20, 1.6F);
            draw->AddLine(
                {wipe_x - 4.F, (img_min.y + img_max.y) * 0.5F},
                {wipe_x + 4.F, (img_min.y + img_max.y) * 0.5F},
                theme::u32(theme::text_bright), 1.6F);
            draw_chip(
                draw, {img_min.x + 10.F, img_min.y + 10.F}, "GT",
                theme::text_bright, theme::fade(theme::surface_0, 0.72F));
            const ImVec2 gs_size = ImGui::CalcTextSize("3DGS");
            draw_chip(
                draw, {img_max.x - gs_size.x - 26.F, img_min.y + 10.F}, "3DGS",
                theme::text_bright, theme::fade(theme::surface_0, 0.72F));
            draw->AddText(
                {canvas_min.x + 16.F, canvas_max.y - 22.F},
                theme::u32(theme::text_faint),
                tr("Drag to wipe GT / 3DGS    ·    MMB pan    ·    wheel zoom"));
        } else if (compare && !compare_ok) {
            draw_empty(
                draw, canvas_min, canvas_max, tr("No splat to compare yet"),
                tr("Train 3DGS, then drag the vertical handle to wipe GT vs render"));
        } else if (compare && !input.has_render) {
            draw_empty(
                draw, canvas_min, canvas_max, tr("Waiting for the live splat…"),
                tr("The renderer is snapping to this training camera"));
        }
        if (live_features) {
            draw_live_keypoints(
                draw, live->keypoints_a, img_min, img_max,
                theme::u32(theme::warning, 0.85F));
            const std::string name_a = live->path_a.empty()
                ? std::string{}
                : live->path_a.filename().string();
            if (!name_a.empty())
                draw_named_chip(
                    draw, {img_min.x + 8.F, img_min.y + 8.F}, tr("Photo"),
                    name_a.c_str(), theme::accent);
        } else if (state.mode == ImageQaMode::features && item.pose)
            draw_features(draw, *item.pose, img_min, img_max, state);
    }
    draw->PopClipRect();

    if (live_features && has_gt) {
        char legend[128];
        std::snprintf(
            legend, sizeof(legend), "%u %s", live->total_keypoints_a,
            tr("keypoints"));
        draw->AddText(
            {canvas_min.x + 16.F, canvas_max.y - 22.F},
            theme::u32(theme::text_muted), legend);
    } else if (!live_align && has_gt && state.mode == ImageQaMode::features &&
               item.pose) {
        const std::size_t untracked =
            item.pose->features.size() > item.pose->triangulated_features
                ? item.pose->features.size() - item.pose->triangulated_features
                : 0;
        char tri_label[64];
        char untracked_label[64];
        std::snprintf(
            tri_label, sizeof(tri_label), "%zu %s",
            item.pose->triangulated_features, tr("triangulated"));
        std::snprintf(
            untracked_label, sizeof(untracked_label), "%zu %s", untracked,
            tr("untracked"));
        ImVec2 key{canvas_min.x + 16.F, canvas_max.y - 22.F};
        if (state.show_triangulated)
            draw_colour_key(draw, key, theme::accent, tri_label);
        if (state.show_untracked)
            draw_colour_key(draw, key, theme::warning, untracked_label);
        if (item.pose->features.empty())
            draw->AddText(
                {canvas_min.x + 16.F, canvas_max.y - 40.F},
                theme::u32(theme::warning),
                input.external_alignment
                    ? tr("No triangulated keypoints in this imported alignment")
                    : tr("No keypoints in this reconstruction — re-align to inspect features"));
    } else if (!live_align && state.mode == ImageQaMode::features &&
               !item.pose && input.external_alignment) {
        draw->AddText(
            {canvas_min.x + 16.F, canvas_max.y - 22.F},
            theme::u32(theme::warning),
            tr("Loading imported cameras for the feature overlay…"));
    }

    if (!live_align && image_qa_needs_render(state.mode) &&
        session.metrics_busy() && !session.metrics().valid) {
        ImVec2 card{canvas_min.x + 14.F, canvas_max.y - 58.F};
        draw_metric_card(draw, card, "PSNR", "…", theme::text_muted);
        draw_metric_card(draw, card, "SSIM", "…", theme::text_muted);
        draw_metric_card(draw, card, "MAE", "…", theme::text_muted);
    } else if (!live_align && image_qa_needs_render(state.mode) &&
               session.metrics().valid) {
        const ImageQaMetrics& m = session.metrics();
        ImVec2 card{canvas_min.x + 14.F, canvas_max.y - 58.F};
        char psnr[32];
        char ssim[32];
        char mae[32];
        std::snprintf(psnr, sizeof(psnr), "%.2f dB", m.psnr);
        std::snprintf(ssim, sizeof(ssim), "%.4f", m.ssim);
        std::snprintf(mae, sizeof(mae), "%.4f", m.mae);
        draw_metric_card(
            draw, card, "PSNR", psnr, metric_tone(m.psnr, 30.F, 24.F));
        draw_metric_card(
            draw, card, "SSIM", ssim, metric_tone(m.ssim, 0.90F, 0.80F));
        draw_metric_card(draw, card, "MAE", mae, theme::accent);
    }

    char zoom_label[16];
    std::snprintf(zoom_label, sizeof(zoom_label), "%d%%",
                  static_cast<int>(std::lround(state.zoom * 100.F)));
    const ImVec2 zoom_size = ImGui::CalcTextSize(zoom_label);
    draw_chip(
        draw,
        {canvas_max.x - zoom_size.x - 28.F, canvas_max.y - 32.F},
        zoom_label, theme::text_muted, theme::fade(theme::surface_0, 0.7F));

    if (!show_strip) return;

    const ImVec2 strip_min{min.x, canvas_max.y};
    draw->AddRectFilled(strip_min, max, theme::u32(theme::surface_1));
    draw->AddLine(
        strip_min, {max.x, strip_min.y}, theme::u32(theme::border));

    ImGui::SetCursorScreenPos({strip_min.x + 10.F, strip_min.y + 10.F});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild(
        "##qa_filmstrip", {max.x - min.x - 20.F, k_filmstrip_h - 16.F}, false,
        ImGuiWindowFlags_HorizontalScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    constexpr float k_thumb = 76.F;
    constexpr float k_gap = 8.F;
    const float strip_h = ImGui::GetContentRegionAvail().y;
    state.filmstrip_first = count;
    state.filmstrip_last = 0;
    for (int i = 0; i < count; ++i) {
        if (i) ImGui::SameLine(0.F, k_gap);
        ImGui::PushID(i);
        const ImVec2 thumb_min = ImGui::GetCursorScreenPos();
        const QaItem thumb = item_at(state, scene, i);
        const bool pair_a = live_matching &&
            (live->index_a == i ||
             (!live->path_a.empty() && !thumb.path.empty() &&
              (thumb.path == live->path_a ||
               thumb.path.filename() == live->path_a.filename())));
        const bool pair_b = live_matching && !pair_a &&
            (live->index_b == i ||
             (!live->path_b.empty() && !thumb.path.empty() &&
              (thumb.path == live->path_b ||
               thumb.path.filename() == live->path_b.filename())));
        const bool selected = live_matching ? pair_a : (i == state.selected);
        if (ImGui::InvisibleButton("##thumb", {k_thumb, strip_h}) &&
            !live_matching) {
            select_image_qa_view(state, i, count);
            state.filmstrip_reselect = false;
        }
        if (selected && state.filmstrip_reselect)
            ImGui::SetScrollHereX(0.5F);
        const bool hovered = ImGui::IsItemHovered();
        const ImVec2 thumb_max{thumb_min.x + k_thumb, thumb_min.y + strip_h};
        ImDrawList* strip = ImGui::GetWindowDrawList();
        strip->AddRectFilled(
            thumb_min, thumb_max,
            selected || pair_b ? IM_COL32(28, 36, 48, 255)
                               : IM_COL32(18, 20, 26, 255),
            6.F);
        const ImU32 thumb_border = selected
            ? theme::u32(theme::accent, 0.9F)
            : (pair_b ? theme::u32(theme::warning, 0.9F)
                      : theme::u32(theme::border, hovered ? 0.9F : 0.55F));
        strip->AddRect(
            thumb_min, thumb_max, thumb_border, 6.F, 0,
            (selected || pair_b) ? 1.8F : 1.F);

        const ImVec2 pic_min{thumb_min.x + 4.F, thumb_min.y + 4.F};
        const ImVec2 pic_max{thumb_max.x - 4.F, thumb_max.y - 18.F};
        ImTextureID photo{};
        if (input.photos && !scene.views.empty())
            photo = input.photos->id(static_cast<std::size_t>(i));
        if (photo) {
            strip->AddImage(photo, pic_min, pic_max);
        } else {
            strip->AddRectFilled(pic_min, pic_max, IM_COL32(12, 13, 16, 255), 4.F);
            char n[8];
            std::snprintf(n, sizeof(n), "%d", i + 1);
            const ImVec2 ns = ImGui::CalcTextSize(n);
            strip->AddText(
                {(pic_min.x + pic_max.x - ns.x) * 0.5F,
                 (pic_min.y + pic_max.y - ns.y) * 0.5F},
                theme::u32(theme::text_faint), n);
        }
        if (pair_a || pair_b) {
            ImFont* small = theme::small_font();
            strip->AddText(
                small, small->FontSize, {pic_min.x + 4.F, pic_min.y + 2.F},
                theme::u32(pair_a ? theme::accent : theme::warning),
                pair_a ? "A" : "B");
        }
        if (thumb.registered || (!scene.views.empty() && thumb.pose)) {
            strip->AddCircleFilled(
                {pic_max.x - 6.F, pic_min.y + 6.F}, 3.F,
                theme::u32(thumb.registered ? theme::success : theme::warning));
        }
        ImFont* small = theme::small_font();
        const char* label = thumb.name.empty() ? "—" : thumb.name.c_str();
        strip->PushClipRect(
            {thumb_min.x + 3.F, thumb_max.y - 16.F},
            {thumb_max.x - 3.F, thumb_max.y - 2.F}, true);
        strip->AddText(
            small, small->FontSize, {thumb_min.x + 5.F, thumb_max.y - 15.F},
            theme::u32((selected || pair_b) ? theme::text_bright : theme::text_faint),
            label);
        strip->PopClipRect();
        if (hovered && !thumb.name.empty())
            ImGui::SetTooltip("%s", thumb.name.c_str());

        if (ImGui::IsItemVisible()) {
            state.filmstrip_first = std::min(state.filmstrip_first, i);
            state.filmstrip_last = std::max(state.filmstrip_last, i);
        }
        ImGui::PopID();
    }
    if (state.filmstrip_reselect) state.filmstrip_reselect = false;
    ImGui::EndChild();

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
        io.MouseWheel != 0.F && io.MousePos.y >= strip_min.y) {
        ImGui::BeginChild("##qa_filmstrip");
        ImGui::SetScrollX(
            ImGui::GetScrollX() - io.MouseWheel * (k_thumb + k_gap) * 1.6F);
        ImGui::EndChild();
    }
    (void)compare_ok;
}

}  // namespace editor
