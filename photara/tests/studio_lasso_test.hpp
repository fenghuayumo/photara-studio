#pragma once

#include "ui.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

// Real ImGui input drives the production edit gesture without a GPU device.
inline void test_studio_lasso() {
    const auto check = [](const bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    };
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {800.F, 600.F};
    io.DeltaTime = 1.F / 60.F;
    io.ConfigInputTrickleEventQueue = false;
    io.Fonts->AddFontDefault();
    io.Fonts->Build();
    editor::App app;
    auto& edit = app.splat_edit;
    splat_render::Camera camera;
    camera.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    camera.model = splat_render::k_camera_orthographic;
    camera.width = 800; camera.height = 600;
    camera.fx = camera.fy = 100.F;
    camera.cx = 400.F; camera.cy = 300.F;
    const auto frame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0.F, 0.F});
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Lasso gesture test");
        handle_view_camera_shortcuts(app, true);
        if (edit.handle_depth_shortcuts(camera,
                io.MousePos.x >= 0.F && io.MousePos.x < 800.F &&
                io.MousePos.y >= 0.F && io.MousePos.y < 600.F)) io.MouseWheel = 0.F;
        edit.handle_pointer(app, {0.F, 0.F}, io.DisplaySize, camera,
                            io.MousePos.x >= 0.F && io.MousePos.x < 800.F &&
                            io.MousePos.y >= 0.F && io.MousePos.y < 600.F, false);
        ImGui::End();
        ImGui::Render();
    };
    const auto key = [&](const ImGuiKey key_code) {
        io.AddKeyEvent(key_code, true); frame();
        io.AddKeyEvent(key_code, false); frame();
    };
    const auto move = [&](const ImVec2 p) {
        io.AddMousePosEvent(p.x, p.y); frame();
    };
    const auto loop = [&](const std::vector<ImVec2>& points) {
        move(points.front());
        io.AddMouseButtonEvent(0, true); frame();
        for (const ImVec2 p : points) move(p);
        io.AddMouseButtonEvent(0, false); frame();
    };
    const ImVec2 pixels[] = {{140, 140}, {240, 140}, {140, 240}, {240, 240}, {140, 140}};
    float means[15], scales[15], quats[20]{}, opacity[5]{1, 1, 1, 1, 1};
    const auto bind = [&](const char* name, const unsigned count) {
        for (unsigned i = 0; i < 5; ++i) {
            const float depth = i == 4 ? 4.F : 2.F;
            const float scale = camera.model == splat_render::k_camera_pinhole ? depth : 1.F;
            means[3 * i] = (pixels[i].x - camera.cx) / camera.fx * scale;
            means[3 * i + 1] = (pixels[i].y - camera.cy) / camera.fy * scale;
            means[3 * i + 2] = depth;
            scales[3 * i] = scales[3 * i + 1] = scales[3 * i + 2] = std::log(.15F);
            quats[4 * i] = 1.F;
        }
        editor::SplatEdit::Host host;
        host.means = means; host.log_scales = scales; host.quaternions = quats;
        host.opacity_logits = opacity; host.count = count;
        edit.sync(name, host);
        move({200, 200});
        key(ImGuiKey_L);
        check(edit.tool() == editor::SplatEdit::Tool::lasso, "L must arm the lasso");
    };
    const std::vector<ImVec2> concave = {
        {100, 100}, {300, 100}, {300, 180}, {180, 180},
        {180, 300}, {100, 300}, {100, 100}};
    bind("lasso-ortho", 5);
    loop(concave);
    check(edit.selected_count() == 3, "Concave closed lasso must exclude its notch and hidden layer");
    edit.undo_edit(app);
    check(edit.selected_count() == 0, "A lasso gesture must undo as one edit");
    io.AddKeyEvent(ImGuiMod_Ctrl, true); key(ImGuiKey_Y);
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 3, "Redo must restore the lasso selection");
    key(ImGuiKey_N);
    loop(concave);
    check(edit.selected_count() == 4, "All-depth lasso must include the hidden layer");
    io.AddKeyEvent(ImGuiMod_Shift, true);
    loop({{220, 220}, {270, 220}, {270, 270}, {220, 270}});
    io.AddKeyEvent(ImGuiMod_Shift, false); frame();
    check(edit.selected_count() == 5, "Shift lasso must add to the existing selection");
    io.AddKeyEvent(ImGuiMod_Alt, true);
    loop({{120, 120}, {160, 120}, {160, 160}, {120, 160}});
    io.AddKeyEvent(ImGuiMod_Alt, false); frame();
    check(edit.selected_count() == 3, "Alt lasso must remove both selected depths");
    loop({{400, 400}});
    loop({{400, 400}, {500, 500}});
    check(edit.selected_count() == 3, "Clicks and zero-area drags must preserve selection");
    move({100, 100}); io.AddMouseButtonEvent(0, true); frame();
    move({300, 100}); move({300, 300}); key(ImGuiKey_Escape);
    io.AddMouseButtonEvent(0, false); frame();
    check(edit.selected_count() == 3, "Esc must cancel a pending lasso without changing selection");
    loop({{100, 100}, {900, 100}, {900, 300}, {100, 300}, {-20, 300}});
    check(edit.selected_count() == 5, "Releasing outside the viewport must finish the captured gesture");
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    loop(concave);
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 4, "Ctrl lasso must retain only selected points inside the region");
    edit.undo_edit(app);
    check(edit.selected_count() == 5, "Intersection must undo as one gesture");
    key(ImGuiKey_R);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    loop({{120, 120}, {160, 160}});
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 2, "Ctrl rectangle must preserve both selected depths inside");
    key(ImGuiKey_L);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    loop({{220, 220}, {270, 220}, {270, 270}, {220, 270}});
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 0, "Intersection must never add previously unselected points");
    loop(concave);
    check(edit.selected_count() == 4, "Fixture must restore the initial subset");
    key(ImGuiKey_B);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    loop({{140, 140}, {240, 140}});
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 3, "Brush intersection must retain the union of dabs, not their intersection");
    edit.undo_edit(app);
    check(edit.selected_count() == 4, "Brush intersection must undo as one gesture");
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    move({140, 140}); io.AddMouseButtonEvent(0, true); frame();
    key(ImGuiKey_Escape);
    io.AddMouseButtonEvent(0, false); io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 4, "Cancelled brush intersection must restore original selection");
    key(ImGuiKey_C);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    loop({{140, 140}, {160, 140}});
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 2, "Ctrl circle must intersect the existing selection");
    edit.undo_edit(app);
    key(ImGuiKey_P);
    for (const ImVec2 p : std::vector<ImVec2>{{220, 120}, {270, 120}, {270, 170}, {220, 170}}) {
        move(p); io.AddMouseButtonEvent(0, true); frame();
        io.AddMouseButtonEvent(0, false); frame();
    }
    io.AddKeyEvent(ImGuiMod_Ctrl, true); key(ImGuiKey_Enter);
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 1, "Ctrl+Enter must intersect a completed polygon");
    bind("lasso-ring", 1);
    const std::vector<ImVec2> strip = {{168, 50}, {171, 50}, {171, 230}, {168, 230}};
    loop(strip);
    check(edit.selected_count() == 0, "Centre selection must exclude a ring-only hit");
    edit.note_view(true);
    loop(strip);
    check(edit.selected_count() == 1, "Thin lasso must select a ring crossed between rim samples");
    camera.model = splat_render::k_camera_pinhole;
    bind("lasso-perspective", 5);
    if (!edit.front_only()) key(ImGuiKey_N);
    loop(concave);
    check(edit.selected_count() == 3, "Perspective lasso must use the same raster coordinates as rendering");
    key(ImGuiKey_P);
    move({100, 100}); io.AddMouseButtonEvent(0, true); frame();
    io.AddMouseButtonEvent(0, false); frame();
    key(ImGuiKey_Keypad5);
    check(camera.model == splat_render::k_camera_pinhole &&
          app.camera.projection == editor::EditorProjection::perspective,
          "Pending polygon points must block camera projection shortcuts");
    camera.model = splat_render::k_camera_orthographic;
    bind("selection-depth", 5);
    if (!edit.front_only()) key(ImGuiKey_N);
    edit.set_selection_depth_range(true, 3.F, 5.F);
    loop(concave);
    check(edit.selected_count() == 1,
          "Depth clipping must run before front-surface filtering so the rear layer can be selected");
    key(ImGuiKey_R);
    loop({{140, 140}});
    check(edit.selected_count() == 1, "Point pick must respect the same depth limits");
    key(ImGuiKey_L);
    edit.set_selection_depth_range(true, 2.F, 2.F);
    loop(concave);
    check(edit.selected_count() == 3, "Exact near/far boundaries must be inclusive");
    check(edit.fit_selection_depth_range(camera, true), "Fit depth to current selection failed");
    if (edit.front_only()) key(ImGuiKey_N);
    loop(concave);
    check(edit.selected_count() == 3, "Fitted range must exclude the deeper layer even with all depths enabled");
    edit.set_selection_depth_range(true, 5.F, 3.F);
    loop(concave);
    check(edit.selected_count() == 1, "Reversed depth limits must be normalized");
    edit.set_selection_depth_range(true, std::numeric_limits<float>::quiet_NaN(), 8.F);
    loop(concave);
    check(edit.selected_count() == 1, "Invalid ranges must leave the prior filter intact");
    check(edit.fit_selection_depth_range(camera, false), "Fit depth to model failed");
    loop(concave);
    check(edit.selected_count() == 4, "Model depth fit must cover both layers");
    edit.set_selection_depth_range(true, 3.F, 5.F);
    io.AddKeyEvent(ImGuiMod_Alt, true); loop(concave);
    io.AddKeyEvent(ImGuiMod_Alt, false); frame();
    check(edit.selected_count() == 3, "Depth-limited subtraction must preserve out-of-range selected Gaussians");
    edit.undo_edit(app);
    check(edit.selected_count() == 4, "Depth-limited selection must undo as one edit");
    io.AddKeyEvent(ImGuiMod_Ctrl, true); loop(concave);
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    check(edit.selected_count() == 1, "Intersection must combine shape and depth limits");
    key(ImGuiKey_B);
    loop({{140, 140}, {240, 140}});
    check(edit.selected_count() == 1, "Brush must exclude out-of-range centres");
    edit.set_selection_depth_range(false, 3.F, 5.F);
    key(ImGuiKey_L); loop(concave);
    check(edit.selected_count() == 4, "Disabling range must restore unrestricted selection");
    edit.set_selection_depth_range(true, 20.F, 30.F);
    loop(concave);
    check(edit.selected_count() == 0, "Empty depth interval must select nothing");
    check(!edit.fit_selection_depth_range(camera, true), "Fit to an empty selection must fail without changing settings");
    bind("depth-reset", 5); loop(concave);
    check(edit.selected_count() == 4, "Loading a different model must reset depth limits");
    edit.set_selection_depth_range(true, 3.F, 5.F);
    key(ImGuiKey_R);
    move({120, 120}); io.AddMouseButtonEvent(0, true); frame();
    edit.set_selection_depth_range(false, 0.F, 100.F);
    move({160, 160}); io.AddMouseButtonEvent(0, false); frame();
    check(edit.selected_count() == 1, "Depth settings must remain stable during a gesture");
    camera.model = splat_render::k_camera_pinhole;
    bind("depth-perspective", 5);
    edit.set_selection_depth_range(true, 3.F, 5.F); loop(concave);
    check(edit.selected_count() == 1, "Perspective depth selection must use camera Z");
    camera.model = splat_render::k_camera_equirectangular;
    edit.set_selection_depth_range(true, 4.5F, 6.F);
    // The near layer has camera Z=2, but two centres are 4.5..6 units away.
    loop({{0, 0}, {800, 0}, {800, 600}, {0, 600}});
    check(edit.selected_count() == 2, "Panorama depth selection must use radial distance rather than camera Z");
    camera.model = splat_render::k_camera_orthographic;
    bind("depth-hotkeys", 5);
    edit.set_selection_depth_range(true, 3.F, 5.F); loop(concave);
    io.AddKeyEvent(ImGuiMod_Ctrl, true); key(ImGuiKey_D);
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    splat_render::DisplayOptions cue;
    edit.apply_depth_preview(cue);
    check(cue.selection_depth_enabled && cue.selection_depth_near < 4.F && cue.selection_depth_far > 4.F &&
          cue.selection_depth_far - cue.selection_depth_near < .01F, "Ctrl+D must fit and enable selected depth");
    edit.set_selection_depth_range(true, 2.F, 4.F);
    io.AddKeyEvent(ImGuiMod_Ctrl, true); io.AddMouseWheelEvent(0.F, -1.F); frame();
    edit.apply_depth_preview(cue);
    check(cue.selection_depth_near > 2.F && std::abs(cue.selection_depth_far - cue.selection_depth_near - 2.F) < 1e-4F,
          "Ctrl+wheel must move range without changing thickness");
    const float centre = (cue.selection_depth_near + cue.selection_depth_far) * .5F;
    io.AddKeyEvent(ImGuiMod_Shift, true); io.AddMouseWheelEvent(0.F, 1.F); frame();
    edit.apply_depth_preview(cue);
    check(cue.selection_depth_far - cue.selection_depth_near > 2.F &&
          std::abs((cue.selection_depth_near + cue.selection_depth_far) * .5F - centre) < 1e-4F,
          "Ctrl+Shift+wheel must adjust thickness while preserving centre");
    io.AddKeyEvent(ImGuiMod_Shift, false); io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    edit.set_selection_depth_range(true, .1F, 1.F);
    io.AddKeyEvent(ImGuiMod_Ctrl, true); io.AddKeyEvent(ImGuiMod_Shift, true);
    io.AddMouseWheelEvent(0.F, 20.F); frame(); edit.apply_depth_preview(cue);
    check(cue.selection_depth_near >= 0.F && std::abs((cue.selection_depth_near + cue.selection_depth_far) * .5F - .55F) < 1e-4F,
          "Thickness growth near the camera must clamp at zero without shifting centre");
    io.AddKeyEvent(ImGuiMod_Shift, false); io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    move({900, 300});
    io.AddKeyEvent(ImGuiMod_Ctrl, true); io.AddMouseWheelEvent(0.F, -1.F); frame();
    splat_render::DisplayOptions after; edit.apply_depth_preview(after);
    check(cue.selection_depth_near == after.selection_depth_near && cue.selection_depth_far == after.selection_depth_far,
          "Wheel outside the viewport must not adjust depth");
    io.AddKeyEvent(ImGuiMod_Ctrl, false); frame();
    edit.clear();
    ImGui::DestroyContext();
}
