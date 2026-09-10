#pragma once

#include "sparse_view.hpp"

namespace editor {

struct BoxDragState {
    bool dragging{};
    // 0-5 face resize, 6-8 axis translate, 9-11 plane translate.
    int part{-1};
    Vec3 grab_point{};
    ReconstructionBox start{};
};

struct ViewportGizmoState {
    bool visible{true};
    BoxDragState box;
};

// Top-right camera-orientation widget, plus a RealityScan-style region
// gizmo: center arrows move the box, face handles resize it.
bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, ImVec2 min, ImVec2 max,
    ReconstructionBox* region = nullptr, bool region_enabled = false,
    float scene_radius = 1.F);

}  // namespace editor
