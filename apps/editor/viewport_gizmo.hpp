#pragma once

#include "sparse_view.hpp"

namespace editor {

enum class TransformTool { orbit, translate, rotate, scale };
enum class GizmoSpace { world, local };

struct BoxDragState {
    bool dragging{};
    int face{-1};
    Vec3 grab_point{};
    ReconstructionBox start{};
};

struct ViewportGizmoState {
    bool visible{true};
    TransformTool tool{TransformTool::orbit};
    GizmoSpace space{GizmoSpace::world};
    BoxDragState box;
};

// Top-right camera-orientation widget, ImGuizmo for the reconstruction
// transform, and face handles to resize the reconstruction region box.
bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, ImVec2 min, ImVec2 max,
    ReconstructionTransform* transform = nullptr, float scene_radius = 1.F,
    bool object_enabled = false, ReconstructionBox* region = nullptr,
    bool region_enabled = false);

}  // namespace editor
