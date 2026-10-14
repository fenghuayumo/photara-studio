#pragma once
#include <filesystem>
#include <string>

namespace photara::inference {
struct ModelInfo {
    std::string id;
    std::string revision; // Checkpoint revision or digest, not just a filename.
    std::filesystem::path weights;
};
} // namespace photara::inference
