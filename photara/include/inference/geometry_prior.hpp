#pragma once
#include "inference/model_info.hpp"
#include "inference/runtime.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace photara::inference {
enum class DepthRepresentation { camera_z, inverse_camera_z, ray_distance };
enum class GeometryScale { relative, meters };
enum class NormalSource { absent, predicted, derived };

// Input-image pinhole intrinsics in pixels. Integer pixel centers;
// camera axes: x right, y down, z forward.
struct PinholeIntrinsics {
    float fx = 0, fy = 0, cx = 0, cy = 0;
};
struct GeometryPrior {
    std::uint32_t width = 0, height = 0;
    std::uint32_t input_width = 0, input_height = 0;
    // Input pixel center = output pixel center * scale + offset.
    std::array<float, 2> input_pixel_scale{1.F, 1.F};
    std::array<float, 2> input_pixel_offset{0.F, 0.F};
    DepthRepresentation depth_representation = DepthRepresentation::camera_z;
    GeometryScale scale = GeometryScale::relative;
    NormalSource normal_source = NormalSource::absent;
    // Row-major host maps: empty or width*height elements. Points and unit
    // normals use the camera frame above. All geometry shares the scale field.
    // Invalid pixels are ignored. Raw affine point maps must be recovered
    // before being published as geometry.
    std::vector<float> depth;
    std::vector<std::array<float, 3>> points;
    std::vector<std::array<float, 3>> normals;
    std::vector<std::uint8_t> valid; // Required: 0 invalid, 1 valid.
    std::vector<float> confidence; // Optional model score in [0, 1].
    std::optional<PinholeIntrinsics> intrinsics; // Input-image coordinates.
    ModelInfo model;
    std::string preprocessing_revision;
    DeviceInfo device; // Actual execution backend, including fallback.
    // Throws std::invalid_argument on malformed metadata or valid pixels.
    void validate() const;
};
} // namespace photara::inference
