#pragma once

#include "sparse_view.hpp"

namespace editor {

enum class TransformTool { orbit, translate, rotate, scale };
enum class GizmoSpace { world, local };

struct ViewportGizmoState {
    bool visible{true};
    TransformTool tool{TransformTool::orbit};
    GizmoSpace space{GizmoSpace::world};
};

// Top-right camera-orientation widget, plus ImGuizmo for the reconstruction
// transform (Move / Rotate / Scale) when a transform tool is active.
bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, ImVec2 min, ImVec2 max,
    ReconstructionTransform* transform = nullptr, float scene_radius = 1.F,
    bool object_enabled = false);

}  // namespace editor
