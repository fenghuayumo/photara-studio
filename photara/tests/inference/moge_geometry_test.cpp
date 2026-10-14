#include "inference/models/moge.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace photara::inference;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void invalid(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Malformed geometry input was accepted");
}
MogeRawGeometry synthetic(const MogeGeometryOptions& options, const PinholeIntrinsics& k,
    double shift, bool flat = false) {
    MogeRawGeometry raw;
    raw.width = options.width; raw.height = options.height;
    const auto count = static_cast<std::size_t>(raw.width) * raw.height;
    raw.points.resize(count * 3); raw.normals.resize(count * 3);
    raw.mask_logits.resize(count, 5.F); raw.log_metric_scale = std::log(2.F);
    for (std::uint32_t y = 0; y < raw.height; ++y)
        for (std::uint32_t x = 0; x < raw.width; ++x) {
            const auto i = static_cast<std::size_t>(y) * raw.width + x;
            const double depth = flat ? 3.0 : 2.0 + 0.013 * x + 0.021 * y + 0.0003 * x * y;
            const double z = depth - shift;
            const double px = (x - static_cast<double>(k.cx)) / k.fx * depth;
            const double py = (y - static_cast<double>(k.cy)) / k.fy * depth;
            raw.points[3 * i] = static_cast<float>(px / z);
            raw.points[3 * i + 1] = static_cast<float>(py / z);
            raw.points[3 * i + 2] = static_cast<float>(std::log(z));
            raw.normals[3 * i + 2] = -3.F;
        }
    return raw;
}
void analytic(bool known, double shift, bool off_center = false, bool flat = false) {
    MogeGeometryOptions options;
    options.width = 40; options.height = 32;
    const PinholeIntrinsics k{50.F, off_center ? 63.F : 50.F,
        off_center ? 16.F : 19.5F, off_center ? 18.F : 15.5F};
    if (known) options.intrinsics = k;
    auto raw = synthetic(options, k, shift, flat);
    const ModelInfo model{"analytic", "test", {}};
    const DeviceInfo device{Backend::cpu, 0, "test", 0, 0};
    auto result = recover_moge_geometry(raw, options, model, device);
    check(result.recovery.status == MogeRecoveryStatus::recovered, "Analytic fit failed");
    check(std::abs(result.recovery.z_shift - shift) < 2e-5, "Incorrect recovered z-shift");
    check(result.prior.intrinsics.has_value(), "Recovered intrinsics missing");
    const auto& actual_k = *result.prior.intrinsics;
    check(std::abs(actual_k.fx - k.fx) < 5e-4 && std::abs(actual_k.fy - k.fy) < 5e-4,
        "Incorrect recovered focal length");
    check(actual_k.cx == k.cx && actual_k.cy == k.cy, "Principal point changed");
    check(result.prior.model.id == model.id && result.prior.device.name == device.name, "Geometry metadata lost");
    check(result.prior.scale == GeometryScale::meters && result.prior.normal_source == NormalSource::predicted,
        "Geometry output semantics incorrect");
    check(result.prior.confidence.empty(), "Mask probabilities became geometry confidence");
    for (std::size_t i = 0; i < result.prior.depth.size(); ++i) {
        const auto x = i % options.width, y = i / options.width;
        const double depth = 2 * (flat ? 3.0 : 2.0 + 0.013 * x + 0.021 * y + 0.0003 * x * y);
        check(result.prior.valid[i] == 1 && std::abs(result.prior.depth[i] - depth) < 5e-5, "Invalid metric depth");
        check(result.prior.normals[i][2] == -1.F, "Normal normalization failed");
        const auto& point = result.prior.points[i];
        check(std::abs(k.fx * point[0] / point[2] + k.cx - x) < 1e-5 &&
            std::abs(k.fy * point[1] / point[2] + k.cy - y) < 1e-5, "Point projection mismatch");
    }
    result.prior.validate();
}
void failure_cases() {
    MogeGeometryOptions options;
    options.width = 40; options.height = 32;
    PinholeIntrinsics k{50, 50, 19.5F, 15.5F};
    auto raw = synthetic(options, k, 0.4);
    raw.mask_logits[0] = 0;
    raw.points[3] = std::numeric_limits<float>::quiet_NaN();
    raw.normals[8] = 0;
    raw.mask_logits[3] = std::numeric_limits<float>::denorm_min(); // Sigmoid rounds to the exact threshold.
    auto partial = recover_moge_geometry(raw, options, {}, {});
    check(partial.recovery.status == MogeRecoveryStatus::recovered && !partial.prior.valid[0] &&
        !partial.prior.valid[1] && !partial.prior.valid[2] && !partial.prior.valid[3], "Malformed pixel validity incorrect");
    std::fill(raw.mask_logits.begin(), raw.mask_logits.end(), -10.F);
    auto empty = recover_moge_geometry(raw, options, {}, {});
    check(empty.recovery.status == MogeRecoveryStatus::no_support && !empty.prior.intrinsics, "Empty map guessed intrinsics");
    check(std::all_of(empty.prior.valid.begin(), empty.prior.valid.end(), [](auto v) { return v == 0; }), "Empty map has valid geometry");
    raw.mask_logits[20] = 5.F;
    options.intrinsics = k;
    auto sparse = recover_moge_geometry(raw, options, {}, {});
    check(sparse.recovery.status == MogeRecoveryStatus::no_support, "Repeated samples overstated sparse support");
    options.intrinsics.reset();
    auto plane = synthetic(options, k, 0.4, true);
    auto degenerate = recover_moge_geometry(plane, options, {}, {});
    check(degenerate.recovery.status == MogeRecoveryStatus::degenerate && !degenerate.prior.intrinsics,
        "Unknown-focal plane should be degenerate");
    options.intrinsics = PinholeIntrinsics{0, 50, 0, 0};
    invalid([&] { (void)recover_moge_geometry(raw, options, {}, {}); });
    options.intrinsics.reset(); raw.log_metric_scale = 1000;
    invalid([&] { (void)recover_moge_geometry(raw, options, {}, {}); });
    raw.log_metric_scale = 0; raw.normals.clear();
    invalid([&] { (void)recover_moge_geometry(raw, options, {}, {}); });
}
}
int main() {
    try {
        for (const double shift : {-0.8, 0.0, 0.7}) { analytic(false, shift); analytic(true, shift); }
        analytic(true, 0.7, true); analytic(true, -0.8, false, true);
        failure_cases();
        std::cout << "MoGe geometry recovery contracts passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
