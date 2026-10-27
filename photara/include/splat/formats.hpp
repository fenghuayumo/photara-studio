#pragma once

#include "splat/types.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>
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
    usdc,
};

GaussianFormat parse_gaussian_format(std::string_view value);
GaussianFormat gaussian_format_from_path(const std::filesystem::path& path) noexcept;
const char* gaussian_format_name(GaussianFormat format) noexcept;
const char* gaussian_format_extension(GaussianFormat format) noexcept;

void save_gaussians(
    const GaussianModel& model, const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect);

void save_sog_unbundled(const GaussianModel&, const std::filesystem::path& directory);

// Keep bands 0..degree and drop the rest. No-op when `degree` is already
// greater than or equal to the model degree. Degree must be 0 to 3.
void restrict_sh_degree(GaussianModel& model, unsigned degree);

// Uniform placement applied to every Gaussian of a model. The scale is
// positive and composes in log space, so anisotropy is preserved.
struct GaussianTransform {
    std::array<float, 3> translation{};
    float scale{1.F};
    // Euler XYZ angles in degrees, applied X then Y then Z about the origin.
    std::array<float, 3> rotation_degrees{};

    [[nodiscard]] bool identity() const noexcept {
        return translation[0] == 0.F && translation[1] == 0.F &&
               translation[2] == 0.F && scale == 1.F &&
               rotation_degrees[0] == 0.F && rotation_degrees[1] == 0.F &&
               rotation_degrees[2] == 0.F;
    }
};

GaussianModel transform_gaussians(
    const GaussianModel& model, const GaussianTransform& transform);

// Concatenates models in order into one. SH coefficients are zero-padded to
// the widest input band count; `normal_features` and `filter_3d` are kept
// only when every input carries them. Every model must be non-empty.
[[nodiscard]] GaussianModel merge_gaussians(
    const std::vector<GaussianModel>& models);

GaussianModel load_gaussians(
    const std::filesystem::path& path,
    GaussianFormat format = GaussianFormat::auto_detect,
    tinytensor::Device device = tinytensor::Device::CUDA);

inline GaussianModel load_gaussians(
    const std::filesystem::path& path, const tinytensor::Device device) {
    return load_gaussians(path, GaussianFormat::auto_detect, device);
}

}  // namespace photara::splat
