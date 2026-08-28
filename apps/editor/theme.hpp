#pragma once

#include "imgui.h"

namespace editor::theme {

// Neutral dark surfaces with a single cyan-blue accent. Kept as ImVec4 so the
// same constants feed both ImGui style colours and raw ImDrawList calls.
inline constexpr ImVec4 accent{0.29F, 0.71F, 0.96F, 1.F};
inline constexpr ImVec4 accent_deep{0.02F, 0.38F, 0.62F, 1.F};
inline constexpr ImVec4 accent_hover{0.05F, 0.49F, 0.77F, 1.F};
inline constexpr ImVec4 success{0.30F, 0.80F, 0.49F, 1.F};
inline constexpr ImVec4 warning{0.96F, 0.72F, 0.29F, 1.F};
inline constexpr ImVec4 danger{0.94F, 0.38F, 0.36F, 1.F};
inline constexpr ImVec4 inactive{0.36F, 0.37F, 0.41F, 1.F};

inline constexpr ImVec4 text_bright{0.91F, 0.92F, 0.93F, 1.F};
inline constexpr ImVec4 text_muted{0.60F, 0.61F, 0.65F, 1.F};
inline constexpr ImVec4 text_faint{0.42F, 0.43F, 0.47F, 1.F};

inline constexpr ImVec4 surface_0{0.043F, 0.045F, 0.051F, 1.F};  // window
inline constexpr ImVec4 surface_1{0.071F, 0.073F, 0.082F, 1.F};  // panels
inline constexpr ImVec4 surface_2{0.106F, 0.110F, 0.122F, 1.F};  // inputs
inline constexpr ImVec4 surface_3{0.145F, 0.149F, 0.165F, 1.F};  // raised
inline constexpr ImVec4 border{0.180F, 0.186F, 0.204F, 1.F};
inline constexpr ImVec4 viewport_bg{0.027F, 0.031F, 0.043F, 1.F};
inline constexpr ImVec4 console_bg{0.022F, 0.024F, 0.029F, 1.F};

struct Fonts {
    ImFont* regular{};
    // Latin-only 12px face used for panel headers and small chrome. Chinese
    // glyphs are baked only into the regular face to keep the atlas small.
    ImFont* small{};
    ImFont* mono{};
};

Fonts load_fonts(ImGuiIO& io);
void apply_style();

ImFont* small_font();
ImFont* mono_font();

ImU32 u32(const ImVec4& colour, float alpha_scale = 1.F);
ImVec4 fade(const ImVec4& colour, float alpha);

// A bordered card with a header strip. The body is a nested child window, so
// `-1` item widths and per-panel scrolling both behave. Always pair with
// end_panel().
void begin_panel(
    const char* id, ImVec2 size, const char* title,
    const char* trailing = nullptr, ImVec2 body_padding = ImVec2(12.F, 10.F));
void end_panel();

// Header strip on its own, for stacking several sections inside one panel.
void section_header(const char* title, const char* trailing = nullptr);

// Inline status dot that advances the cursor like a small widget.
void status_dot(const ImVec4& colour, float diameter = 8.F);

void pill(const char* text, const ImVec4& text_colour, const ImVec4& fill);

// Slim rounded track. A negative `fraction` draws an animated indeterminate
// sweep instead of a filled bar.
void progress_track(ImVec2 size, float fraction, const ImVec4& fill);

bool primary_button(const char* label, ImVec2 size, bool enabled = true);
bool toolbar_button(
    const char* label, ImVec2 size, bool enabled = true, bool active = false);
bool danger_button(const char* label, ImVec2 size, bool enabled = true);

void caption(const char* text);
void metric(const char* key, const char* value);
void metric_coloured(const char* key, const char* value, const ImVec4& colour);

}  // namespace editor::theme
