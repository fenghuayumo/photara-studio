#pragma once

#include "app.hpp"

namespace editor {

Action draw_menu_bar(App& app);
Action draw_toolbar(App& app);
Action draw_inspector(App& app);
void build_dock_space(App& app);
void draw_scene_panel(App& app);
void draw_viewport_panel(App& app);
void draw_console_panel(App& app);
void draw_status_bar(const App& app);
void draw_controls_window(App& app);
void draw_about_window(App& app);
void draw_mesh_export_modal(App& app);
void draw_alignment_export_modal(App& app);
void draw_splat_export_modal(App& app);
ClearResultsAction draw_clear_results_modal(App& app);

}  // namespace editor
