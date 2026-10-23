#include "shortcuts.hpp"
#include <cstdio>
#include <cstdlib>
using namespace editor::shortcuts;
void check(bool condition) { if (!condition) std::abort(); }
int main() {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {800, 600};
    io.DeltaTime = 1.F / 60.F;
    unsigned char* pixels; int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    for (auto& entry : entries) {
        check(valid(entry.defaults));
        check(find(entry.id) == &entry);
        for (auto& other : entries)
            check(&entry == &other || entry.defaults.key != other.defaults.key ||
                  entry.defaults.modifiers != other.defaults.modifiers);
        int encoded = int(entry.binding.key) * 16 + entry.binding.modifiers;
        check(encoded / 16 == entry.binding.key && encoded % 16 == entry.binding.modifiers);
    }
    check(!valid({ImGuiKey_F, 4}));
    check(!valid({ImGuiKey_LeftArrow, 1}));
    check(!valid({ImGuiKey_A, 8}));
    check(!valid({ImGuiKey_LeftCtrl, 0}));
    check(!valid({ImGuiKey_Keypad1, 0}));
    check(valid({ImGuiKey_None, 0}));
    check(!valid({ImGuiKey_None, 1}));
    check(valid({ImGuiKey_K, 3}));
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_D, true);
    ImGui::NewFrame();
    check(pressed("fit_depth"));
    check(!pressed("front"));
    suspended = true;
    check(!pressed("fit_depth"));
    suspended = false;
    auto* action = find("fit_depth");
    action->binding = {ImGuiKey_K, 1};
    check(!pressed("fit_depth"));
    ImGui::EndFrame();
    io.AddKeyEvent(ImGuiKey_D, false);
    io.AddKeyEvent(ImGuiKey_K, true);
    ImGui::NewFrame();
    check(pressed("fit_depth"));
    check(label("fit_depth") == "Ctrl+K");
    action->binding = {ImGuiKey_None, 0};
    check(!pressed("fit_depth"));
    ImGui::EndFrame();
    ImGui::DestroyContext();
    std::puts("Shortcut defaults, validation, persistence encoding, remapping and dispatch passed.");
}
