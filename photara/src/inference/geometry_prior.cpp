#include "inference/geometry_prior.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace photara::inference {
void GeometryPrior::validate() const {
    const auto require = [](bool condition, const char* message) {
        if (!condition) throw std::invalid_argument(message);
    };
    require(width > 0 && height > 0 && input_width > 0 && input_height > 0,
        "Geometry prior requires nonzero output and input dimensions");
    require(static_cast<std::size_t>(width) <=
        std::numeric_limits<std::size_t>::max() / height, "Geometry prior size overflows");
    const auto count = static_cast<std::size_t>(width) * height;
    require(valid.size() == count, "Geometry prior requires a full validity mask");
    require(!depth.empty() || !points.empty(), "Geometry prior has no depth or points");
    require(depth.empty() || depth.size() == count, "Depth map size mismatch");
    require(points.empty() || points.size() == count, "Point map size mismatch");
    require(normals.empty() || normals.size() == count, "Normal map size mismatch");
    require(confidence.empty() || confidence.size() == count, "Confidence map size mismatch");
    require((normal_source == NormalSource::absent && normals.empty()) ||
        ((normal_source == NormalSource::predicted || normal_source == NormalSource::derived) &&
         !normals.empty()), "Normal source does not match normal map");
    require(depth_representation == DepthRepresentation::camera_z ||
        depth_representation == DepthRepresentation::inverse_camera_z ||
        depth_representation == DepthRepresentation::ray_distance, "Unknown depth representation");
    require(scale == GeometryScale::relative || scale == GeometryScale::meters,
        "Unknown geometry scale");
    for (int axis = 0; axis < 2; ++axis) {
        require(std::isfinite(input_pixel_scale[axis]) && input_pixel_scale[axis] > 0 &&
            std::isfinite(input_pixel_offset[axis]), "Invalid input pixel mapping");
    }
    if (intrinsics) {
        require(std::isfinite(intrinsics->fx) && intrinsics->fx > 0 &&
            std::isfinite(intrinsics->fy) && intrinsics->fy > 0 &&
            std::isfinite(intrinsics->cx) && std::isfinite(intrinsics->cy),
            "Invalid pinhole intrinsics");
    }
    for (std::size_t i = 0; i < count; ++i) {
        require(valid[i] <= 1, "Validity mask must contain only zero or one");
        if (!valid[i]) continue;
        if (!depth.empty())
            require(std::isfinite(depth[i]) && depth[i] > 0, "Invalid depth at valid pixel");
        if (!points.empty()) {
            for (const float value : points[i])
                require(std::isfinite(value), "Nonfinite point at valid pixel");
            require(points[i][2] > 0, "Point must be in front of the camera");
        }
        if (!normals.empty()) {
            float length_squared = 0;
            for (const float value : normals[i]) {
                require(std::isfinite(value), "Nonfinite normal at valid pixel");
                length_squared += value * value;
            }
            require(std::abs(length_squared - 1.F) <= 0.01F,
                "Normal must have unit length at valid pixel");
        }
        if (!confidence.empty())
            require(std::isfinite(confidence[i]) && confidence[i] >= 0 && confidence[i] <= 1,
                "Confidence must be in [0, 1] at valid pixel");
    }
}
} // namespace photara::inference
