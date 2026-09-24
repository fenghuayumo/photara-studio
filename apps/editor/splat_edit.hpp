#pragma once

#include "splat_render/renderer.hpp"

#include "imgui.h"

#include <cstdint>
#include <string>
#include <vector>

namespace photara::splat {
struct GaussianModel;
}

namespace editor {

struct App;

// Screen-space Gaussian editing for the trained model. The floating toolbar
// matches the inspector projection ids used by the Vulkan preview.
class SplatEdit {
public:
    enum class Tool { none, pick, move, circle, polygon, brush };

    // Host arrays copied at load. Pointers are only read during sync.
    struct Host {
        const float* means{};
        const float* log_scales{};
        const float* quaternions{};
        const float* opacity_logits{};
        const float* sh{};
        const float* normals{};
        const float* filter_3d{};
        std::uint32_t count{};
        std::uint32_t sh_degree{};
        std::uint32_t sh_bases{1};
    };

    void clear();
    void sync(const std::string& key, const Host& host);
    // True after a model is loaded, including after every Gaussian is deleted.
    [[nodiscard]] bool bound_to(const std::string& key) const noexcept {
        return synced_ && key_ == key;
    }
    [[nodiscard]] bool dirty() const noexcept { return dirty_; }
    // CPU copy of the splats currently in the viewport, including deletions.
    bool copy_model(photara::splat::GaussianModel& model) const;
    [[nodiscard]] std::uint32_t selected_count() const noexcept {
        return selected_count_;
    }
    [[nodiscard]] std::uint32_t gaussian_count() const noexcept { return count_; }
    void select_all();
    void clear_selection();
    void invert_selection();
    void delete_selected(App& app);
    // Rings view selects by the covariance ellipse until the user picks a mode.
    void note_view(bool rings_view, float ring_scale = 2.828427F);
    [[nodiscard]] Tool tool() const noexcept { return tool_; }
    // A selection tool is armed. Otherwise the left button orbits.
    [[nodiscard]] bool tool_armed() const noexcept {
        return count_ > 0 && tool_ != Tool::none;
    }
    [[nodiscard]] bool front_only() const noexcept { return front_only_; }

    // Alt+wheel over the brush resizes it. True when the wheel should not zoom.
    bool consume_alt_wheel(bool pointer_in_view);
    void write_status(char* buffer, std::size_t size) const;

    // Draws the pill and handles its buttons. True while the pointer is over it.
    bool draw_toolbar(App& app, ImVec2 view_min, ImVec2 view_max);
    void draw_overlay(
        ImDrawList* draw, ImVec2 view_min, ImVec2 view_max,
        const splat_render::Camera& camera) const;
    // Left button edits only while a tool is armed. Right button stays free for orbit.
    void handle_pointer(
        App& app, ImVec2 view_min, ImVec2 view_max,
        const splat_render::Camera& camera, bool viewport_hot, bool over_toolbar);

private:
    struct Snapshot {
        enum class Kind { selection, deletion };
        Kind kind{Kind::selection};
        std::vector<std::uint8_t> selected;
        std::vector<float> xyz;
        // Indices into the model from before this deletion, ascending.
        std::vector<std::uint32_t> removed_index;
        std::vector<float> removed_centers;
        std::vector<float> removed_scales;
        std::vector<float> removed_quats;
        std::vector<float> removed_opacity;
        std::vector<float> removed_sh;
        std::vector<float> removed_normals;
        std::vector<float> removed_filter;
    };

    void push_undo(Snapshot snapshot);
    void restore(App& app, const Snapshot& snapshot);
    void begin_stroke(bool capture_positions);
    void end_stroke(App& app);
    void select_at(
        const splat_render::Camera& camera, ImVec2 view_min, ImVec2 view_max,
        ImVec2 mouse, bool replace_first);
    void apply_region(
        const splat_render::Camera& camera, ImVec2 view_min, ImVec2 view_max,
        int mode, ImVec2 a, ImVec2 b);
    void focus_selection(App& app) const;
    void upload(App& app);
    void recount();
    [[nodiscard]] std::vector<float> capture_xyz() const;
    void undo(App& app);
    void redo(App& app);
    void abort_stroke(App& app);
    void set_tool(App& app, Tool tool);
    void toggle_tool(App& app, Tool tool);
    void upload_model(App& app);
    void reinsert(const Snapshot& snapshot);
    void erase_recorded(const Snapshot& snapshot);
    void finish_gesture(
        App& app, const splat_render::Camera& camera, ImVec2 view_min,
        ImVec2 view_max, ImVec2 mouse);
    void handle_keys(App& app);

    struct ScreenEllipse {
        float u{};
        float v{};
        float rx{};
        float ry{};
        float rotation{};
        float depth{};
        bool valid{};
    };
    [[nodiscard]] bool rings_ready() const noexcept;
    bool project_ellipse(
        const splat_render::Camera& camera, std::uint32_t index,
        ScreenEllipse& ellipse) const;
    [[nodiscard]] float ellipse_reach(
        const splat_render::Camera& camera, float depth, std::uint32_t index) const;
    [[nodiscard]] static bool ellipse_contains(
        const ScreenEllipse& ellipse, float x, float y);
    [[nodiscard]] bool ellipse_hits(
        const ScreenEllipse& ellipse, int mode, ImVec2 ra, ImVec2 rb,
        const std::vector<ImVec2>* polygon, float radius) const;

    std::string key_;
    bool synced_{};
    bool dirty_{};
    std::uint32_t count_{};
    std::uint32_t selected_count_{};
    std::uint32_t sh_degree_{};
    std::uint32_t sh_bases_{1};
    std::vector<float> centers_;
    std::vector<float> log_scales_;
    std::vector<float> quaternions_;
    std::vector<float> opacity_;
    std::vector<float> sh_;
    std::vector<float> normals_;
    std::vector<float> filter_;
    std::vector<std::uint8_t> selected_;
    Tool tool_{Tool::none};
    // True: only the front surface in each screen cell. False: every layer.
    bool front_only_{true};
    // Hit the projected covariance ellipse instead of the centre.
    bool rings_hit_{};
    // Contour in projected sigmas. 2*sqrt(2) is SuperSplat's exp(-4) edge.
    float ring_sigma_{2.828427F};
    bool hit_chosen_{};
    float brush_radius_{26.F};
    std::vector<ImVec2> polygon_;
    bool stroking_{};
    bool stroke_changed_{};
    bool stroke_moved_{};
    ImVec2 stroke_origin_{};
    Snapshot stroke_before_;
    std::vector<Snapshot> undo_;
    std::vector<Snapshot> redo_;
    ImVec2 toolbar_min_{};
    ImVec2 toolbar_max_{};
    bool toolbar_visible_{};
};

}  // namespace editor
