#pragma once
#include "ui.hpp"
#include <cmath>
#include <stdexcept>

inline void test_studio_view_shortcuts() {
    using namespace editor;
    const auto check = [](const bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    };
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {800, 600}; io.DeltaTime = 1.F / 60.F;
    io.ConfigInputTrickleEventQueue = false;
    io.Fonts->AddFontDefault(); io.Fonts->Build();
    App app;
    app.workspace = ViewportWorkspace::scene_3d;
    app.camera.target = {1, 2, 3};
    bool hot = true, typing = false, consumed = false;
    const auto frame = [&] {
        ImGui::NewFrame(); ImGui::Begin("Camera shortcut test");
        io.WantTextInput = typing;
        consumed = handle_view_camera_shortcuts(app, hot);
        ImGui::End(); ImGui::Render();
    };
    const auto press = [&](const ImGuiKey key, const bool ctrl = false, const bool alt = false) {
        io.AddKeyEvent(ImGuiMod_Ctrl, ctrl); io.AddKeyEvent(ImGuiMod_Alt, alt);
        io.AddKeyEvent(key, true); frame();
        const bool accepted = consumed;
        io.AddKeyEvent(key, false); frame();
        io.AddKeyEvent(ImGuiMod_Ctrl, false); io.AddKeyEvent(ImGuiMod_Alt, false); frame();
        return accepted;
    };
    const auto view = [&](const ImGuiKey key, const bool opposite, const int axis, const float sign) {
        app.preview_follow_view = true;
        check(press(key, opposite), "Standard-view shortcut not accepted");
        check(app.camera.projection == EditorProjection::orthographic && !app.preview_follow_view,
              "Standard view must enter orthographic orbit and stop following captures");
        const auto camera = make_preview_camera(app.camera, 800, 600);
        const float target[] = {1, 2, 3};
        for (int i = 0; i < 3; ++i)
            check(std::abs(camera.position[i] - target[i] - (i == axis ? sign * app.camera.distance : 0.F)) < 1e-4F,
                  "Standard-view shortcut points from the wrong side of the model");
    };
    view(ImGuiKey_Keypad1, false, 2, 1.F); view(ImGuiKey_Keypad1, true, 2, -1.F);
    view(ImGuiKey_Keypad3, false, 0, 1.F); view(ImGuiKey_Keypad3, true, 0, -1.F);
    view(ImGuiKey_Keypad7, false, 1, 1.F); view(ImGuiKey_Keypad7, true, 1, -1.F);
    const float height = app.camera.ortho_height;
    check(press(ImGuiKey_Keypad5) && app.camera.projection == EditorProjection::perspective,
          "Num 5 must switch orthographic to perspective");
    check(press(ImGuiKey_Keypad5) && app.camera.projection == EditorProjection::orthographic &&
          std::abs(app.camera.ortho_height - height) < 1e-4F,
          "Num 5 round trip must retain the view scale");
    check(press(ImGuiKey_1, false, true) && app.camera.projection == EditorProjection::perspective,
          "Alt+1 must select perspective");
    check(press(ImGuiKey_2, false, true) && app.camera.projection == EditorProjection::fisheye,
          "Alt+2 must select fisheye");
    check(press(ImGuiKey_3, false, true) && app.camera.projection == EditorProjection::panorama,
          "Alt+3 must select panorama");
    check(!press(ImGuiKey_1) && !press(ImGuiKey_Keypad1, false, true) && !press(ImGuiKey_1, true, true),
          "Unassigned modifier combinations must not change the camera");
    hot = false; check(!press(ImGuiKey_Keypad1), "Shortcut leaked outside viewport"); hot = true;
    typing = true; check(!press(ImGuiKey_Keypad1), "Shortcut interrupted text entry"); typing = false;
    app.workspace = ViewportWorkspace::image_2d;
    check(!press(ImGuiKey_Keypad1), "Camera shortcut leaked into image QA");
    app.workspace = ViewportWorkspace::scene_3d;
    io.AddKeyEvent(ImGuiMod_Shift, true);
    check(!press(ImGuiKey_Keypad1), "Shift must not trigger standard views");
    io.AddKeyEvent(ImGuiMod_Shift, false); frame();
    io.AddMouseButtonEvent(0, true); frame();
    check(!press(ImGuiKey_Keypad1), "Shortcut interrupted a mouse drag");
    io.AddMouseButtonEvent(0, false); frame();
    check(app.camera.projection == EditorProjection::panorama, "Blocked shortcuts changed projection");
    io.AddKeyEvent(ImGuiKey_Keypad5, true); frame();
    check(consumed, "Initial toggle key press missing");
    frame(); check(!consumed, "Holding Num 5 must not toggle repeatedly");
    io.AddKeyEvent(ImGuiKey_Keypad5, false); frame();
    ImGui::DestroyContext();
}
