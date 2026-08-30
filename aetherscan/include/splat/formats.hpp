#pragma once

#include "splat/types.hpp"

#include <filesystem>
#include <string_view>

namespace aetherscan::splat {

// Final on-disk representation of a trained Gaussian model. This is separate
// from DatasetFormat, which describes external camera/SfM input.
enum class GaussianFormat {
    auto_detect,
    ply,
    sog,
    spz,
    glb,
};

GaussianFormat parse_gaussian_format(std::string_view value);
const char* gaussian_format_name(GaussianFormat format) noexcept;
const char* gaussian_format_extension(GaussianFormat format) noexcept;

void save_gaussians(
    const GaussianModel& model, const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect);

GaussianModel load_gaussians(
    const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect);

}  // namespace aetherscan::splat
