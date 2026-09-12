#pragma once

#include <filesystem>

namespace editor::i18n {

enum class Language { en, zh_cn, ja, ko };

[[nodiscard]] Language language();
void set_language(Language language);
void load(const std::filesystem::path& path);
void save(const std::filesystem::path& path);

[[nodiscard]] const char* code(Language language);
[[nodiscard]] const char* native_name(Language language);

// Looks up `english` in the active table. English is the identity mapping and
// the fallback when a translation is missing.
[[nodiscard]] const char* tr(const char* english);

// Visible label plus a stable ImGui id suffix (`##foo` / `###Panel`).
[[nodiscard]] const char* id(const char* english, const char* imgui_suffix);

}  // namespace editor::i18n
