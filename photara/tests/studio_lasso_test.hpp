#pragma once

#include "app.hpp"
#include <cmath>
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
    edit.clear();
    ImGui::DestroyContext();
}
