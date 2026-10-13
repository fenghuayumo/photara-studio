#pragma once
#include "splat/types.hpp"
#include <filesystem>
#include <functional>
#include <vector>

namespace photara::splat {
struct ProcessingControl {
    bool vulkan{true};
    std::function<void(float)> progress;
    std::function<bool()> cancelled;
};
struct LodOptions : ProcessingControl {
    unsigned levels{4};
    float keep_ratio{0.5F};
    unsigned chunk_count{512000};
    unsigned chunk_min{8000};
    float chunk_extent{16.F};
};
// Writes PlayCanvas Streamed SOG v1 (unbundled SOG chunks).
// Destination must be new; publishing happens only after successful processing.
void export_streamed_sog(const GaussianModel &, const std::filesystem::path &,
                         const LodOptions & = {});

} // namespace photara::splat
