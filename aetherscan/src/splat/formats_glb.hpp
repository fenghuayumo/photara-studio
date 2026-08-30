#pragma once

#include "splat/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace aetherscan::splat::detail {

struct DecodedGlbGaussians {
    std::size_t count{};
    unsigned degree{};
    std::vector<float> means;
    std::vector<float> log_scales;
    std::vector<float> rotations;
    std::vector<float> opacity_logits;
    std::vector<float> sh;
};

std::vector<std::uint8_t> save_glb(const GaussianModel& model);
DecodedGlbGaussians load_glb(const std::filesystem::path& path);

}  // namespace aetherscan::splat::detail
