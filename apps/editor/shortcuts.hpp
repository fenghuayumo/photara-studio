#pragma once
#include "imgui.h"
#include <array>
#include <string>
#include <string_view>

namespace editor::shortcuts {
struct Binding { ImGuiKey key; int modifiers; };
struct Entry { const char* id; const char* group; const char* label; Binding defaults; Binding binding; std::string display; };
// Modifier bits: Ctrl=1, Shift=2, Alt=4, Super=8.
inline auto entries = std::to_array<Entry>({
    {"new_project", "Project", "New project", {ImGuiKey_N, 1}, {ImGuiKey_N, 1}},
    {"open_images", "Project", "Open image folder", {ImGuiKey_O, 1}, {ImGuiKey_O, 1}},
    {"open_project", "Project", "Open project folder", {ImGuiKey_O, 3}, {ImGuiKey_O, 3}},
    {"save", "Project", "Save project", {ImGuiKey_S, 1}, {ImGuiKey_S, 1}},
    {"save_as", "Project", "Save project as", {ImGuiKey_S, 3}, {ImGuiKey_S, 3}},
    {"preferences", "Project", "Preferences", {ImGuiKey_Comma, 1}, {ImGuiKey_Comma, 1}},
    {"workspace_2d", "Workspace", "2D workspace", {ImGuiKey_2, 0}, {ImGuiKey_2, 0}},
    {"workspace_3d", "Workspace", "3D workspace", {ImGuiKey_3, 0}, {ImGuiKey_3, 0}},
    {"perspective", "View", "Perspective", {ImGuiKey_1, 4}, {ImGuiKey_1, 4}},
    {"fisheye", "View", "Fisheye", {ImGuiKey_2, 4}, {ImGuiKey_2, 4}},
    {"panorama", "View", "Panorama", {ImGuiKey_3, 4}, {ImGuiKey_3, 4}},
    {"front", "View", "Front", {ImGuiKey_4, 4}, {ImGuiKey_4, 4}},
    {"back", "View", "Back", {ImGuiKey_4, 6}, {ImGuiKey_4, 6}},
    {"right", "View", "Right", {ImGuiKey_5, 4}, {ImGuiKey_5, 4}},
    {"left", "View", "Left", {ImGuiKey_5, 6}, {ImGuiKey_5, 6}},
    {"top", "View", "Top", {ImGuiKey_6, 4}, {ImGuiKey_6, 4}},
    {"bottom", "View", "Bottom", {ImGuiKey_6, 6}, {ImGuiKey_6, 6}},
    {"projection", "View", "Toggle orthographic", {ImGuiKey_0, 4}, {ImGuiKey_0, 4}},
    {"pick", "Gaussian", "Pick / rectangle", {ImGuiKey_R, 0}, {ImGuiKey_R, 0}},
    {"circle", "Gaussian", "Circle select", {ImGuiKey_C, 0}, {ImGuiKey_C, 0}},
    {"polygon", "Gaussian", "Polygon select", {ImGuiKey_P, 0}, {ImGuiKey_P, 0}},
    {"lasso", "Gaussian", "Lasso select", {ImGuiKey_L, 0}, {ImGuiKey_L, 0}},
    {"brush", "Gaussian", "Brush select", {ImGuiKey_B, 0}, {ImGuiKey_B, 0}},
    {"box", "Gaussian", "Box select", {ImGuiKey_Y, 0}, {ImGuiKey_Y, 0}},
    {"sphere", "Gaussian", "Sphere select", {ImGuiKey_U, 0}, {ImGuiKey_U, 0}},
    {"front_only", "Gaussian", "Front surface / all depths", {ImGuiKey_N, 0}, {ImGuiKey_N, 0}},
    {"select_all", "Gaussian", "Select all", {ImGuiKey_A, 1}, {ImGuiKey_A, 1}},
    {"clear", "Gaussian", "Clear selection", {ImGuiKey_A, 3}, {ImGuiKey_A, 3}},
    {"invert", "Gaussian", "Invert selection", {ImGuiKey_I, 1}, {ImGuiKey_I, 1}},
    {"delete_selection", "Gaussian", "Delete selected", {ImGuiKey_Delete, 0}, {ImGuiKey_Delete, 0}},
    {"undo", "Gaussian", "Undo", {ImGuiKey_Z, 1}, {ImGuiKey_Z, 1}},
    {"redo", "Gaussian", "Redo", {ImGuiKey_Z, 3}, {ImGuiKey_Z, 3}},
    {"redo_alt", "Gaussian", "Redo (alternate)", {ImGuiKey_Y, 1}, {ImGuiKey_Y, 1}},
    {"fit_depth", "Gaussian", "Fit depth to selection", {ImGuiKey_D, 1}, {ImGuiKey_D, 1}},
});
inline bool suspended = false;
inline int modifiers() {
    const auto& io = ImGui::GetIO();
    return (io.KeyCtrl ? 1 : 0) | (io.KeyShift ? 2 : 0) |
           (io.KeyAlt ? 4 : 0) | (io.KeySuper ? 8 : 0);
}
inline Entry* find(std::string_view id) {
    for (auto& entry : entries) if (id == entry.id) return &entry;
    return nullptr;
}
inline bool pressed(const char* id, bool repeat = false) {
    const auto* entry = find(id);
    return !suspended && entry && entry->binding.key != ImGuiKey_None &&
        modifiers() == entry->binding.modifiers &&
        ImGui::IsKeyPressed(entry->binding.key, repeat);
}
inline std::string label(Binding binding) {
    if (binding.key == ImGuiKey_None) return "Unbound";
    std::string result;
    if (binding.modifiers & 1) result += "Ctrl+";
    if (binding.modifiers & 2) result += "Shift+";
    if (binding.modifiers & 4) result += "Alt+";
    if (binding.modifiers & 8) result += "Super+";
    return result + ImGui::GetKeyName(binding.key);
}
inline std::string label(const char* id) { return label(find(id)->binding); }
inline const char* text(const char* id) {
    auto* entry = find(id);
    const auto value = label(entry->binding);
    if (entry->display != value) entry->display = value;
    return entry->display.c_str();
}
inline bool valid(Binding binding) {
    if (binding.modifiers < 0 || binding.modifiers > 7) return false;
    if (binding.key == ImGuiKey_None) return binding.modifiers == 0;
    if (binding.key < ImGuiKey_NamedKey_BEGIN || binding.key >= ImGuiKey_MouseLeft ||
        (binding.key >= ImGuiKey_LeftCtrl && binding.key <= ImGuiKey_RightSuper) ||
        binding.key == ImGuiKey_Menu) return false;
    // Fixed contextual commands inspect some keys regardless of modifiers.
    if (binding.key == ImGuiKey_Escape || binding.key == ImGuiKey_F ||
        binding.key == ImGuiKey_Enter || binding.key == ImGuiKey_Backspace ||
        (binding.key >= ImGuiKey_LeftArrow && binding.key <= ImGuiKey_DownArrow) ||
        (binding.key >= ImGuiKey_Keypad0 && binding.key <= ImGuiKey_KeypadEqual)) return false;
    if (binding.modifiers == 0 &&
        (binding.key == ImGuiKey_G || binding.key == ImGuiKey_T || binding.key == ImGuiKey_S ||
         binding.key == ImGuiKey_LeftBracket || binding.key == ImGuiKey_RightBracket)) return false;
    return !(binding.key == ImGuiKey_C && binding.modifiers == 1);
}
} // namespace editor::shortcuts
