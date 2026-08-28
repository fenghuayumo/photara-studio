#pragma once

#include "sparse_view.hpp"

namespace editor {

struct ViewportGizmoState {
    bool visible{true};
};

// Draws only the top-right camera-orientation widget. It deliberately has no
// scene transform controls: AetherScan's reconstruction viewport is for
// inspection, not editing the solved SfM coordinate system.
bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, ImVec2 min, ImVec2 max);

}  // namespace editor
