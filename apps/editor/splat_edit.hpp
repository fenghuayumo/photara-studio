#pragma once

#include "splat_render/renderer.hpp"

#include "imgui.h"

#include <cstdint>
#include <string>
#include <vector>

namespace editor {

struct App;

// Screen-space Gaussian editing for the trained model. The floating toolbar
// matches the inspector projection ids used by the Vulkan preview.
class SplatEdit {
public:
    enum class Tool { none, pick, move, circle, polygon, brush };

    void clear();
    void sync(
        const std::string& key, const float* means, const float* opacity_logits,
        std::uint32_t count);
    [[nodiscard]] bool bound_to(const std::string& key) const noexcept {
        return !key_.empty() && key_ == key && count_ > 0;
    }
    [[nodiscard]] std::uint32_t selected_count() const noexcept {
        return selected_count_;
    }
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
        std::vector<std::uint8_t> selected;
        std::vector<float> xyz;
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
    void clear_selection();
    void select_all();
    void finish_gesture(
        App& app, const splat_render::Camera& camera, ImVec2 view_min,
        ImVec2 view_max, ImVec2 mouse);
    void handle_keys(App& app);

    std::string key_;
    std::uint32_t count_{};
    std::uint32_t selected_count_{};
    std::vector<float> centers_;
    std::vector<std::uint8_t> selected_;
    Tool tool_{Tool::none};
    // True: only the front surface in each screen cell. False: every layer.
    bool front_only_{true};
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
