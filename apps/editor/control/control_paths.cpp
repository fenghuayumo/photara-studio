#include "control/control_paths.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <random>

namespace editor::control {
namespace {

std::filesystem::path platform_config_root() {
    if (const char* override_dir = std::getenv("PHOTARA_CONTROL_DIR");
        override_dir != nullptr && override_dir[0] != '\0')
        return std::filesystem::path(override_dir);
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA");
        appdata != nullptr && appdata[0] != '\0')
        return std::filesystem::path(appdata) / "Photara" / "control";
    if (const char* local = std::getenv("LOCALAPPDATA");
        local != nullptr && local[0] != '\0')
        return std::filesystem::path(local) / "Photara" / "control";
#else
    if (const char* config = std::getenv("XDG_CONFIG_HOME");
        config != nullptr && config[0] != '\0')
        return std::filesystem::path(config) / "Photara" / "control";
    if (const char* home = std::getenv("HOME");
        home != nullptr && home[0] != '\0')
        return std::filesystem::path(home) / ".config" / "Photara" / "control";
#endif
    return std::filesystem::current_path() / "photara-control";
}

std::string random_hex(const std::size_t bytes) {
    std::random_device device;
    std::mt19937_64 generator(
        (static_cast<std::uint64_t>(device()) << 32U) ^ device());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string text;
    text.reserve(bytes * 2);
    for (std::size_t i = 0; i < bytes; ++i) {
        const auto value = static_cast<unsigned>(generator() & 0xFFU);
        text.push_back(kHex[value >> 4U]);
        text.push_back(kHex[value & 0x0FU]);
    }
    return text;
}

}  // namespace

std::filesystem::path discovery_dir() {
    return platform_config_root();
}

std::string make_session_id() {
    return random_hex(8);
}

std::string make_session_token() {
    return random_hex(24);
}

}  // namespace editor::control
