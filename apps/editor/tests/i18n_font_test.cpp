#include "../i18n.cpp"  // Inspect every catalog entry without a public test-only API.
#include "theme.hpp"
#include "imgui_internal.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Run on a desktop with CJK system fonts installed. Exercise the real atlas,
// including the small face used by Studio chrome, rather than just font paths.
int main() {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    const auto fonts = editor::theme::load_fonts(io);
    io.Fonts->Build();
    const auto settings = std::filesystem::temp_directory_path() /
        ("photara-i18n-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) /
        "editor.language";
    editor::i18n::load(settings);
    for (const auto language : {editor::i18n::Language::en, editor::i18n::Language::zh_cn,
                               editor::i18n::Language::ja,
                               editor::i18n::Language::ko}) {
        editor::i18n::set_language(language);
        editor::i18n::load(settings);
        if (editor::i18n::language() != language) return EXIT_FAILURE;
        if (std::string(editor::i18n::id("Change shortcut", "###ChangeShortcut")) !=
            std::string(editor::i18n::tr("Change shortcut")) + "###ChangeShortcut")
            return EXIT_FAILURE;
        std::vector<const char*> texts{editor::i18n::native_name(language)};
        for (const auto& entry : editor::i18n::k_entries)
            texts.push_back(editor::i18n::tr(entry.en));
        for (const char* text : texts) {
            while (*text) {
                unsigned int codepoint{};
                if (*text == '\n' || *text == '\r' || *text == '\t') { ++text; continue; }
                text += ImTextCharFromUtf8(&codepoint, text, nullptr);
                for (ImFont* font : {fonts.regular, fonts.small}) {
                    if (!font->FindGlyphNoFallback(static_cast<ImWchar>(codepoint))) {
                        std::fprintf(stderr, "%s: missing U+%04X\n",
                                     editor::i18n::code(language), codepoint);
                        return EXIT_FAILURE;
                    }
                }
            }
        }
    }
    std::filesystem::remove(settings);
    std::filesystem::remove(settings.parent_path());
    ImGui::DestroyContext();
}
