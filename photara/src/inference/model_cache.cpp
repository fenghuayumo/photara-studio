#include "inference/model_cache.hpp"
#include <cstdlib>
#include <stdexcept>
#include <system_error>

namespace photara::inference {
namespace {
std::filesystem::path environment_path(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? std::filesystem::path(value) : std::filesystem::path{};
}
}

std::filesystem::path model_cache_root() {
#if defined(_WIN32)
    const auto base = environment_path("LOCALAPPDATA");
#else
    auto base = environment_path("XDG_CACHE_HOME");
    if (base.empty()) {
        const auto home = environment_path("HOME");
        if (!home.empty()) base = home / ".cache";
    }
#endif
    return base.empty() ? std::filesystem::path{} : base / "Photara" / "models";
}

std::filesystem::path cached_model_path(const std::string& filename) {
    const std::filesystem::path name(filename);
    if (name.empty() || name.has_parent_path() || name.is_absolute() ||
        name == "." || name == "..")
        throw std::invalid_argument("Cached model name must be a filename");
    return model_cache_root() / name;
}

bool model_file_ready(const std::filesystem::path& path, std::uintmax_t minimum_bytes) {
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error) || error)
        return false;
    const auto bytes = std::filesystem::file_size(path, error);
    return !error && bytes >= minimum_bytes;
}
} // namespace photara::inference
