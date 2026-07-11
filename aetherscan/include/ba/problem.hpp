#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aetherscan::ba {

using Index = std::uint32_t;

// World-to-camera pose: p_camera = R(q) * (p_world - center).
struct Pose {
    double qw{1.0};
    double qx{0.0};
    double qy{0.0};
    double qz{0.0};
    double cx{0.0};
    double cy{0.0};
    double cz{0.0};
};

struct PinholeIntrinsics {
    double fx{1.0};
    double fy{1.0};
    double cx{0.0};
    double cy{0.0};
    double k1{0.0};
    double k2{0.0};
    double p1{0.0};
    double p2{0.0};
};

struct Point3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

// Structure-of-arrays keeps each field contiguous for coalesced GPU reads and
// allows later sorting by point or camera without changing the public model.
struct Observations {
    std::vector<Index> camera;
    std::vector<Index> point;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> weight;

    [[nodiscard]] std::size_t size() const noexcept { return camera.size(); }

    void reserve(const std::size_t count) {
        camera.reserve(count);
        point.reserve(count);
        x.reserve(count);
        y.reserve(count);
        weight.reserve(count);
    }

    void push_back(
        const Index camera_index,
        const Index point_index,
        const double observed_x,
        const double observed_y,
        const double observation_weight = 1.0) {
        camera.push_back(camera_index);
        point.push_back(point_index);
        x.push_back(observed_x);
        y.push_back(observed_y);
        weight.push_back(observation_weight);
    }

    void validate(const std::size_t camera_count, const std::size_t point_count) const {
        const auto count = size();
        if (point.size() != count || x.size() != count || y.size() != count ||
            weight.size() != count) {
            throw std::invalid_argument("Observation arrays have different lengths");
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (camera[i] >= camera_count || point[i] >= point_count) {
                throw std::out_of_range("Observation references an invalid parameter block");
            }
            if (weight[i] < 0.0) {
                throw std::invalid_argument("Observation weight must be non-negative");
            }
        }
    }
};

struct Problem {
    std::vector<Pose> poses;
    std::vector<PinholeIntrinsics> intrinsics;
    std::vector<Point3> points;
    Observations observations;

    void validate() const {
        if (poses.empty() || points.empty() || observations.size() == 0) {
            throw std::invalid_argument("BA problem must contain poses, points and observations");
        }
        if (intrinsics.size() != poses.size()) {
            throw std::invalid_argument("The first backend version requires one intrinsics block per pose");
        }
        observations.validate(poses.size(), points.size());
    }
};

}  // namespace aetherscan::ba
