#include "sam/model_cache.hpp"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace photara::sam {
namespace {

std::filesystem::path env_path(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return {};
    return std::filesystem::path(value);
}

std::filesystem::path local_app_data() {
#if defined(_WIN32)
    return env_path("LOCALAPPDATA");
#else
    if (const auto xdg = env_path("XDG_CACHE_HOME"); !xdg.empty()) return xdg;
    const auto home = env_path("HOME");
    return home.empty() ? std::filesystem::path{} : home / ".cache";
#endif
}

std::filesystem::path roaming_app_data() {
#if defined(_WIN32)
    return env_path("APPDATA");
#else
    if (const auto xdg = env_path("XDG_CONFIG_HOME"); !xdg.empty()) return xdg;
    const auto home = env_path("HOME");
    return home.empty() ? std::filesystem::path{} : home / ".config";
#endif
}

}  // namespace

std::filesystem::path user_model_path() {
    const auto base = local_app_data();
    if (base.empty()) return std::filesystem::path(k_model_file);
    return base / "Photara" / "models" / k_model_file;
}

bool model_file_ready(const std::filesystem::path& path) {
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error))
        return false;
    return std::filesystem::file_size(path, error) > k_model_bytes / 2;
}

std::filesystem::path locate_model() {
    if (const auto from_env = env_path("PHOTARA_SAM_MODEL");
        model_file_ready(from_env))
        return from_env;
    if (const auto own = user_model_path(); model_file_ready(own)) return own;
    return {};
}

std::filesystem::path license_file() {
    const auto base = roaming_app_data();
    if (base.empty()) return std::filesystem::path("sam3-license-accepted");
    return base / "Photara" / "sam3-license-accepted";
}

bool license_accepted() {
    std::error_code error;
    return std::filesystem::is_regular_file(license_file(), error);
}

void accept_license() {
    const auto path = license_file();
    std::error_code error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream output(path, std::ios::trunc);
    output << "sam3\n";
    if (!output)
        throw std::runtime_error(
            "Could not record SAM 3 licence acceptance");
}

}  // namespace photara::sam
