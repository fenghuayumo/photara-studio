#pragma once

#include <filesystem>
#include <string>

namespace editor::control {

// Directory that holds one JSON discovery record per running Studio process.
// PHOTARA_CONTROL_DIR overrides the platform default.
std::filesystem::path discovery_dir();

std::string make_session_id();
std::string make_session_token();

}  // namespace editor::control
