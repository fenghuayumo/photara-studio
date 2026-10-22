#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

namespace editor::control {

using json = nlohmann::json;

inline constexpr int k_schema_version = 1;

inline json success(json value = json::object()) {
    value["ok"] = true;
    return value;
}

inline json failure(
    const std::string& code, const std::string& message,
    json details = json::object()) {
    return json{
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message},
            {"details", std::move(details)},
        }},
    };
}

inline bool ok(const json& value) {
    return value.is_object() && value.value("ok", false);
}

inline std::string error_code(const json& value) {
    if (!value.is_object() || !value.contains("error")) return {};
    const json& error = value["error"];
    return error.is_object() ? error.value("code", std::string{}) : std::string{};
}

inline std::string error_message(const json& value) {
    if (!value.is_object() || !value.contains("error")) return {};
    const json& error = value["error"];
    return error.is_object() ? error.value("message", std::string{}) : std::string{};
}

}  // namespace editor::control
