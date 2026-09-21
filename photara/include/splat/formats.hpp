#pragma once

#include "splat/types.hpp"

#include <filesystem>
#include <string_view>

namespace photara::splat {

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
GaussianFormat gaussian_format_from_path(const std::filesystem::path& path) noexcept;
const char* gaussian_format_name(GaussianFormat format) noexcept;
const char* gaussian_format_extension(GaussianFormat format) noexcept;

void save_gaussians(
    const GaussianModel& model, const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect);

// Keep bands 0..degree and drop the rest. No-op when `degree` is already
// greater than or equal to the model degree. Degree must be 0 to 3.
void restrict_sh_degree(GaussianModel& model, unsigned degree);

GaussianModel load_gaussians(
    const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect);

}  // namespace photara::splat
