#include "inference/models/moge.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace photara::inference {
namespace {
struct Sample { double x, y, z, u, v; };
struct Evaluation { double cost = std::numeric_limits<double>::infinity(), focal = 0; };

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
bool finite(const std::array<float, 3>& point) {
    return std::all_of(point.begin(), point.end(), [](float value) { return std::isfinite(value); });
}
std::vector<float> resize(const std::vector<float>& source, int channels,
    std::uint32_t width, std::uint32_t height, std::uint32_t target_width, std::uint32_t target_height) {
    if (width == target_width && height == target_height) return source;
    std::vector<float> result(static_cast<std::size_t>(target_width) * target_height * channels);
    for (std::uint32_t y = 0; y < target_height; ++y) {
        const float sy = std::max(0.F, static_cast<float>((y + 0.5) * height / target_height - 0.5));
        const auto y0 = static_cast<std::uint32_t>(sy), y1 = std::min(y0 + 1, height - 1);
        const float dy = sy - y0;
        for (std::uint32_t x = 0; x < target_width; ++x) {
            const float sx = std::max(0.F, static_cast<float>((x + 0.5) * width / target_width - 0.5));
            const auto x0 = static_cast<std::uint32_t>(sx), x1 = std::min(x0 + 1, width - 1);
            const float dx = sx - x0;
            for (int c = 0; c < channels; ++c) {
                const auto at = [&](std::uint32_t xx, std::uint32_t yy) {
                    return source[(static_cast<std::size_t>(yy) * width + xx) * channels + c];
                };
                const float top = at(x0, y0) * (1.F - dx) + at(x1, y0) * dx;
                const float bottom = at(x0, y1) * (1.F - dx) + at(x1, y1) * dx;
                result[(static_cast<std::size_t>(y) * target_width + x) * channels + c] =
                    top * (1.F - dy) + bottom * dy;
            }
        }
    }
    return result;
}
Evaluation evaluate(const std::vector<Sample>& samples, double shift,
    const std::optional<PinholeIntrinsics>& known, double diagonal) {
    double focal = 0;
    if (!known) {
        double numerator = 0, denominator = 0;
        for (const auto& sample : samples) {
            const double depth = sample.z + shift;
            if (!(depth > 0)) return {};
            const double x = sample.x / depth, y = sample.y / depth;
            numerator += x * sample.u + y * sample.v;
            denominator += x * x + y * y;
        }
        if (!(denominator > 0)) return {};
        focal = numerator / denominator;
        if (!(focal > 0) || !std::isfinite(focal)) return {};
    }
    const double fx = known ? 2.0 * known->fx / diagonal : focal;
    const double fy = known ? 2.0 * known->fy / diagonal : focal;
    double cost = 0;
    for (const auto& sample : samples) {
        const double depth = sample.z + shift;
        if (!(depth > 0)) return {};
        const double rx = fx * sample.x / depth - sample.u;
        const double ry = fy * sample.y / depth - sample.v;
        cost += rx * rx + ry * ry;
    }
    return {cost / samples.size(), focal};
}

MogeRecoveryInfo recover(const std::vector<std::array<float, 3>>& points,
    const std::vector<std::uint8_t>& valid, const MogeGeometryOptions& options) {
    MogeRecoveryInfo result;
    result.used_known_intrinsics = options.intrinsics.has_value();
    const double diagonal = std::hypot(static_cast<double>(options.width), options.height);
    const double cx = options.intrinsics ? options.intrinsics->cx : (options.width - 1) / 2.0;
    const double cy = options.intrinsics ? options.intrinsics->cy : (options.height - 1) / 2.0;
    std::vector<Sample> samples;
    std::unordered_set<std::size_t> distinct_pixels;
    double min_z = std::numeric_limits<double>::infinity(), max_z = 0, extent = 0;
    // Match the official nearest-neighbor 64x64 fit grid, including repeated
    // source pixels for small images. No image resampling changes the rays.
    for (std::uint32_t y = 0; y < 64; ++y)
        for (std::uint32_t x = 0; x < 64; ++x) {
            const auto sx = static_cast<std::uint32_t>(static_cast<std::uint64_t>(x) * options.width / 64);
            const auto sy = static_cast<std::uint32_t>(static_cast<std::uint64_t>(y) * options.height / 64);
            const auto index = static_cast<std::size_t>(sy) * options.width + sx;
            if (!valid[index]) continue;
            distinct_pixels.insert(index);
            const auto& point = points[index];
            samples.push_back({point[0], point[1], point[2], 2.0 * (sx - cx) / diagonal, 2.0 * (sy - cy) / diagonal});
            min_z = std::min(min_z, static_cast<double>(point[2]));
            max_z = std::max(max_z, static_cast<double>(point[2]));
            extent = std::max(extent, std::hypot(static_cast<double>(point[0]), point[1]));
        }
    result.sample_count = samples.size();
    if (distinct_pixels.size() < 4) return result;
    // Unknown focal and shift are indistinguishable on constant-Z maps. The
    // supplied K removes that ambiguity, so known-focal planes remain usable.
    if (!options.intrinsics && max_z - min_z <= std::max(1e-12, max_z * 1e-6)) {
        result.status = MogeRecoveryStatus::degenerate;
        return result;
    }
    const double scale = std::max({max_z, extent, 1e-12});
    const double log_low = std::log(scale * 1e-9), log_high = std::log(scale * 1e6);
    constexpr int steps = 192;
    int best = -1;
    Evaluation optimum;
    double best_log = 0;
    const auto objective = [&](double log_distance) {
        return evaluate(samples, std::exp(log_distance) - min_z, options.intrinsics, diagonal);
    };
    for (int i = 0; i <= steps; ++i) {
        const double parameter = log_low + (log_high - log_low) * i / steps;
        const auto value = objective(parameter);
        if (value.cost < optimum.cost) { best = i; optimum = value; best_log = parameter; }
    }
    if (best <= 0 || best >= steps) {
        result.status = MogeRecoveryStatus::degenerate;
        return result;
    }
    double a = log_low + (log_high - log_low) * (best - 1) / steps;
    double b = log_low + (log_high - log_low) * (best + 1) / steps;
    constexpr double ratio = 0.6180339887498948482;
    double c = b - ratio * (b - a), d = a + ratio * (b - a);
    auto fc = objective(c), fd = objective(d);
    for (int iteration = 0; iteration < 96; ++iteration) {
        if (fc.cost < fd.cost) {
            b = d; d = c; fd = fc; c = b - ratio * (b - a); fc = objective(c);
        } else {
            a = c; c = d; fc = fd; d = a + ratio * (b - a); fd = objective(d);
        }
    }
    if (fc.cost < optimum.cost) { optimum = fc; best_log = c; }
    if (fd.cost < optimum.cost) { optimum = fd; best_log = d; }
    if (!std::isfinite(optimum.cost)) { result.status = MogeRecoveryStatus::degenerate; return result; }
    result.status = MogeRecoveryStatus::recovered;
    result.z_shift = std::exp(best_log) - min_z;
    result.normalized_focal = options.intrinsics ? 2.0 * options.intrinsics->fx / diagonal : optimum.focal;
    result.reprojection_rms = std::sqrt(optimum.cost);
    return result;
}
}

MogeGeometryResult recover_moge_geometry(const MogeRawGeometry& raw,
    const MogeGeometryOptions& options, const ModelInfo& model, const DeviceInfo& device) {
    require(raw.width > 0 && raw.height > 0 && options.width > 0 && options.height > 0,
        "MoGe postprocess requires nonzero dimensions");
    const auto source_count = static_cast<std::uint64_t>(raw.width) * raw.height;
    const auto count64 = static_cast<std::uint64_t>(options.width) * options.height;
    require(source_count <= std::numeric_limits<std::size_t>::max() / 3 &&
        count64 <= std::numeric_limits<std::size_t>::max() / (3 * sizeof(float)), "MoGe map size overflows");
    require(raw.points.size() == source_count * 3 && raw.normals.size() == source_count * 3 &&
        raw.mask_logits.size() == source_count, "MoGe head map sizes do not match dimensions");
    require(std::isfinite(options.mask_threshold) && options.mask_threshold > 0 && options.mask_threshold < 1,
        "MoGe mask threshold must lie strictly between zero and one");
    require(std::isfinite(raw.log_metric_scale), "MoGe metric scale is nonfinite");
    const double metric_scale = std::exp(static_cast<double>(raw.log_metric_scale));
    require(metric_scale > 0 && metric_scale <= std::numeric_limits<float>::max() &&
        static_cast<float>(metric_scale) > 0, "MoGe metric scale is outside float range");
    if (options.intrinsics) {
        const auto& k = *options.intrinsics;
        require(std::isfinite(k.fx) && k.fx > 0 && std::isfinite(k.fy) && k.fy > 0 &&
            std::isfinite(k.cx) && std::isfinite(k.cy), "Invalid known MoGe intrinsics");
    }
    auto point_values = resize(raw.points, 3, raw.width, raw.height, options.width, options.height);
    auto normal_values = resize(raw.normals, 3, raw.width, raw.height, options.width, options.height);
    auto logits = resize(raw.mask_logits, 1, raw.width, raw.height, options.width, options.height);
    const auto count = static_cast<std::size_t>(count64);
    std::vector<std::array<float, 3>> affine(count), normals(count);
    std::vector<std::uint8_t> valid(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const double z = std::exp(static_cast<double>(point_values[3 * i + 2]));
        affine[i] = {static_cast<float>(point_values[3 * i] * z),
            static_cast<float>(point_values[3 * i + 1] * z), static_cast<float>(z)};
        const double length = std::hypot(std::hypot(static_cast<double>(normal_values[3 * i]),
            normal_values[3 * i + 1]), normal_values[3 * i + 2]);
        const float exponential = std::exp(-std::abs(logits[i]));
        const float probability = logits[i] >= 0 ? 1.F / (1.F + exponential) : exponential / (1.F + exponential);
        if (!finite(affine[i]) || affine[i][2] <= 0 || !std::isfinite(length) || length <= 1e-12 ||
            !std::isfinite(logits[i]) || !(probability > options.mask_threshold)) continue;
        for (int axis = 0; axis < 3; ++axis) normals[i][axis] = static_cast<float>(normal_values[3 * i + axis] / length);
        valid[i] = 1;
    }
    MogeGeometryResult result;
    auto& prior = result.prior;
    prior.width = prior.input_width = options.width;
    prior.height = prior.input_height = options.height;
    prior.scale = GeometryScale::meters;
    prior.normal_source = NormalSource::predicted;
    prior.model = model;
    prior.device = device;
    prior.preprocessing_revision = "moge2-native-geometry-v1";
    prior.depth.resize(count, 0);
    prior.points.resize(count);
    prior.normals.resize(count);
    prior.valid.resize(count, 0);
    prior.intrinsics = options.intrinsics;
    result.recovery = recover(affine, valid, options);
    if (result.recovery.status != MogeRecoveryStatus::recovered) { prior.validate(); return result; }
    if (!prior.intrinsics) {
        const double pixels_focal = result.recovery.normalized_focal *
            std::hypot(static_cast<double>(options.width), options.height) / 2.0;
        require(std::isfinite(pixels_focal) && pixels_focal > 0 && pixels_focal <= std::numeric_limits<float>::max(),
            "Recovered MoGe focal is outside float range");
        prior.intrinsics = PinholeIntrinsics{static_cast<float>(pixels_focal), static_cast<float>(pixels_focal),
            (options.width - 1) / 2.F, (options.height - 1) / 2.F};
    }
    const auto& k = *prior.intrinsics;
    for (std::uint32_t y = 0; y < options.height; ++y)
        for (std::uint32_t x = 0; x < options.width; ++x) {
            const auto i = static_cast<std::size_t>(y) * options.width + x;
            if (!valid[i]) continue;
            const float depth = static_cast<float>((affine[i][2] + result.recovery.z_shift) * metric_scale);
            const std::array<float, 3> point{static_cast<float>((x - static_cast<double>(k.cx)) / k.fx * depth),
                static_cast<float>((y - static_cast<double>(k.cy)) / k.fy * depth), depth};
            if (!(depth > 0) || !finite(point)) continue;
            prior.depth[i] = depth;
            prior.points[i] = point;
            prior.normals[i] = normals[i];
            prior.valid[i] = 1;
        }
    prior.validate();
    return result;
}
} // namespace photara::inference
