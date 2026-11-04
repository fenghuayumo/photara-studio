#include "splat/rasterizer.hpp"
#include "../src/splat/sh_adam_quant.hpp"
#include "../src/splat/cuda_ops.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>

using namespace photara::splat;
using tinytensor::Tensor;
constexpr auto cuda = tinytensor::Device::CUDA;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void sync() {
    require(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
}
GaussianModel fixture(std::size_t n, unsigned degree, bool filter) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    const std::size_t bases = (degree + 1) * (degree + 1);
    std::vector<float> xyz(n * 3), scales(n * 3), q(n * 4), sh(n * bases * 3), opacity(n);
    for (std::size_t i = 0; i < n; ++i) {
        xyz[3 * i] = uniform(rng) * 4.F;
        xyz[3 * i + 1] = uniform(rng) * 2.5F;
        xyz[3 * i + 2] = 4.F + uniform(rng);
        // Include culled rows; fused Adam must still decay their moments.
        if (i % 13 == 0)
            xyz[3 * i + 2] = -2.F;
        for (unsigned c = 0; c < 3; ++c)
            scales[3 * i + c] = -6.5F + .5F * uniform(rng);
        for (unsigned c = 0; c < 4; ++c)
            q[4 * i + c] = uniform(rng);
        if (i % 17 == 0)
            std::fill(q.begin() + 4 * i, q.begin() + 4 * i + 4, 0.F);
        opacity[i] = uniform(rng) * 3.F;
    }
    for (auto& v : sh)
        v = uniform(rng) * .1F;
    GaussianModel m;
    m.sh_degree = degree;
    m.means = Tensor::from_vector(xyz, {n, 3}, cuda);
    m.log_scales = Tensor::from_vector(scales, {n, 3}, cuda);
    m.quaternions = Tensor::from_vector(q, {n, 4}, cuda);
    m.opacity_logits = Tensor::from_vector(opacity, {n, 1}, cuda);
    m.sh = Tensor::from_vector(sh, {n, bases, 3}, cuda);
    if (filter)
        m.filter_3d = Tensor::full({n, 1}, .001F, cuda);
    return m;
}
GaussianModel clone(GaussianModel m) {
    m.means = m.means.clone();
    m.log_scales = m.log_scales.clone();
    m.quaternions = m.quaternions.clone();
    m.opacity_logits = m.opacity_logits.clone();
    m.sh = m.sh.clone();
    return m;
}
Camera camera(unsigned width = 320, unsigned height = 240) {
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = width * .7F;
    c.fy = width * .7F;
    c.cx = width * .5F;
    c.cy = height * .5F;
    c.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    return c;
}
struct ErrorSummary {
    double absolute{}, relative{}, ulps{}, relative_l2{};
};
bool audit_errors = false;
std::map<std::string, ErrorSummary> error_summaries;
void record_error(const char* name, double expected, double actual, double ulp) {
    if (!audit_errors)
        return;
    auto& summary = error_summaries[name];
    const double error = std::abs(expected - actual);
    summary.absolute = std::max(summary.absolute, error);
    if (std::max(std::abs(expected), std::abs(actual)) > 1e-12)
        summary.relative =
            std::max(summary.relative, error / std::max(std::abs(expected), std::abs(actual)));
    summary.ulps = std::max(summary.ulps, ulp > 0 ? error / ulp : 0.);
}
double float_spacing(float value) {
    const float magnitude = std::abs(value);
    return double(std::nextafter(magnitude, std::numeric_limits<float>::infinity())) - magnitude;
}
bool near(const std::vector<float>& x, const std::vector<float>& y, float atol, float rtol) {
    if (x.size() != y.size())
        return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
            std::abs(x[i] - y[i]) > atol + rtol * std::max(std::abs(x[i]), std::abs(y[i])))
            return false;
    return true;
}
void close(
    const Tensor& a,
    const Tensor& b,
    const char* name,
    float atol = 2e-7F,
    float rtol = 1e-6F) {
    require(a.is_valid() == b.is_valid(), name);
    if (!a.is_valid())
        return;
    const auto x = a.to_vector(), y = b.to_vector();
    if (audit_errors)
        for (std::size_t i = 0; i < x.size(); ++i)
            record_error(name, x[i], y[i], std::max(float_spacing(x[i]), float_spacing(y[i])));
    require(near(x, y, atol, rtol), name);
}
// Compare changes, not parameter magnitudes. Allow only float32 subtraction
// one parameter ULP plus a small fraction of the actual update. Require
// updates above that noise; SH also crosses quantized optimizer state bins.
void close_update(
    const Tensor& before,
    const Tensor& reference,
    const Tensor& actual,
    const char* name,
    float rtol = 1e-4F,
    float atol = 0.F) {
    const auto base = before.to_vector(), ref = reference.to_vector(), got = actual.to_vector();
    bool strong_signal = false;
    bool rejects_missing = false;
    for (std::size_t i = 0; i < base.size(); ++i) {
        const float expected = ref[i] - base[i], observed = got[i] - base[i];
        const double roundoff = std::max(float_spacing(base[i]), float_spacing(ref[i]));
        const double tolerance = roundoff + atol + rtol * std::abs(expected);
        record_error(
            name, expected, observed, std::max(float_spacing(base[i]), float_spacing(ref[i])));
        require(
            std::isfinite(observed) && std::isfinite(expected) &&
                std::abs(expected - observed) <= tolerance,
            name);
        strong_signal |= std::abs(expected) > 16.F * roundoff;
        rejects_missing |= std::abs(expected) > tolerance;
    }
    require(strong_signal && rejects_missing, "Update comparison cannot detect missing updates");
}
void close_gradient(
    const Tensor& reference,
    const Tensor& actual,
    const char* name,
    float atol,
    float rtol) {
    // Per-component allowances cover cancellation in geometry derivatives;
    // the independent norm bound prevents those allowances hiding a bias.
    close(reference, actual, name, atol, rtol);
    const auto values = reference.to_vector(), observed = actual.to_vector();
    require(
        !near(values, std::vector<float>(values.size()), atol, rtol),
        "Gradient comparison cannot detect zeroed gradients");
    double squared_error = 0., squared_signal = 0.;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double error = double(values[i]) - observed[i];
        squared_error += error * error;
        squared_signal += double(values[i]) * values[i];
    }
    const double relative_l2 = std::sqrt(squared_error / squared_signal);
    if (audit_errors)
        error_summaries[name].relative_l2 =
            std::max(error_summaries[name].relative_l2, relative_l2);
    require(relative_l2 <= 2e-6, "Gradient relative L2 error");
}
// Exercise the comparators themselves: otherwise an over-wide tolerance can
// make parity pass even if an entire kernel stops writing useful results.
void comparison_guards() {
    const auto must_reject = [](auto compare) {
        bool rejected = false;
        try {
            compare();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "Comparator accepted an injected regression");
    };
    const auto reference = Tensor::from_vector({1e-6F, -2e-6F, 3e-6F}, {3}, cuda);
    close_gradient(reference, reference, "guard gradient", 1e-9F, 5e-4F);
    const auto zero = Tensor::zeros({3}, cuda);
    const auto wrong = Tensor::from_vector({1.01e-6F, -2.02e-6F, 3.03e-6F}, {3}, cuda);
    must_reject([&] { close_gradient(reference, zero, "zeroed gradient", 1e-9F, 5e-4F); });
    must_reject([&] { close_gradient(reference, wrong, "biased gradient", 1e-9F, 5e-4F); });
    const auto before = Tensor::from_vector({1.F, -1.F, .5F}, {3}, cuda);
    const auto after = Tensor::from_vector({1.001F, -.999F, .501F}, {3}, cuda);
    const auto wrong_after = Tensor::from_vector({1.00101F, -.99899F, .50101F}, {3}, cuda);
    for (const float rtol : {1e-4F, 1e-3F}) {
        close_update(before, after, after, "guard update", rtol);
        must_reject([&] { close_update(before, after, before, "missing update", rtol); });
        must_reject([&] { close_update(before, after, wrong_after, "biased update", rtol); });
    }
}
struct Optimizer {
    StructureAdamUpdate structure;
    SHAdamUpdate sh;
    explicit Optimizer(const GaussianModel& m) {
        structure.means_first = Tensor::full(m.means.shape(), .001F, cuda);
        structure.means_second = Tensor::full(m.means.shape(), .01F, cuda);
        structure.scales_first = Tensor::full(m.log_scales.shape(), .001F, cuda);
        structure.scales_second = Tensor::full(m.log_scales.shape(), .01F, cuda);
        structure.rotations_first = Tensor::full(m.quaternions.shape(), .001F, cuda);
        structure.rotations_second = Tensor::full(m.quaternions.shape(), .01F, cuda);
        structure.opacity_first = Tensor::full(m.opacity_logits.shape(), .001F, cuda);
        structure.opacity_second = Tensor::full(m.opacity_logits.shape(), .01F, cuda);
        structure.means_lr = 1e-5F;
        structure.scales_lr = 1e-4F;
        structure.quaternions_lr = 1e-4F;
        structure.opacities_lr = 1e-3F;
        structure.opacity_reg = 1e-5F;
        structure.scale_reg = 1e-5F;
        structure.minimum_log_scale = -12.F;
        structure.maximum_log_scale = 0.F;
        auto state = detail::make_sh_adam_quant(m.size(), m.sh.shape()[1] * 3, cuda);
        sh.first = state.first;
        sh.packed = state.packed;
        sh.bounds = state.bounds;
        sh.lr = .0025F;
        sh.rest_lr = .000125F;
    }
};
void diagnostic_dump() {
    const char* prior = std::getenv("PHOTARA_SPLAT_GRAD_DUMP");
    const bool had_prior = prior != nullptr;
    const std::string previous = prior ? prior : "";
    const auto folder =
        std::filesystem::temp_directory_path() /
        ("photara-activation-dump-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    // Own only a directory we actually created, including on test failures.
    require(
        std::filesystem::create_directory(folder), "Cannot create diagnostic fixture directory");
    const auto set_dump = [](const char* value) {
#ifdef _WIN32
        require(
            _putenv_s("PHOTARA_SPLAT_GRAD_DUMP", value ? value : "") == 0, "Set dump environment");
#else
        require(
            (value ? setenv("PHOTARA_SPLAT_GRAD_DUMP", value, 1)
                   : unsetenv("PHOTARA_SPLAT_GRAD_DUMP")) == 0,
            "Set dump environment");
#endif
    };
    struct Cleanup {
        std::filesystem::path folder;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(folder, error);
        }
    } cleanup{folder};
    const auto prefix = (folder / "dump-").string();
    set_dump(prefix.c_str());
    try {
        auto m = fixture(137, 3, true);
        auto before = detail::activate_parameters(m);
        const auto scales = before.scales.to_vector(), rotations = before.quaternions.to_vector(),
                   opacities = before.opacities.to_vector();
        Rasterizer raster;
        RasterizeOptions o;
        o.fuse_activation = true;
        o.require_depth = false;
        o.active_sh_degree = 3;
        configure_rasterization(o, TrainingOptions{});
        Optimizer optimizer(m);
        optimizer.structure.scales_lr = .1F;
        optimizer.structure.opacities_lr = .1F;
        const auto frame = raster.forward(m, camera(), o);
        const auto gc = Tensor::full(frame.color.shape(), .001F, cuda);
        const auto ga = Tensor::full(frame.alpha.shape(), .001F, cuda);
        (void)raster.backward(m, frame, gc, ga, {}, {}, {}, &optimizer.sh, &optimizer.structure);
        const auto check_file = [&](const char* suffix, const std::vector<float>& expected) {
            const auto path = prefix + "1." + suffix;
            require(
                std::filesystem::file_size(path) == expected.size() * sizeof(float),
                "Dump size mismatch");
            std::ifstream input(path, std::ios::binary);
            std::vector<float> observed(expected.size());
            input.read(reinterpret_cast<char*>(observed.data()), observed.size() * sizeof(float));
            require(
                bool(input) && near(expected, observed, 0.F, 0.F),
                "Dump must contain pre-update activation");
        };
        check_file("inputscales", scales);
        check_file("inputrot", rotations);
        check_file("inputopa", opacities);
        const auto after = detail::activate_parameters(m);
        require(
            !near(scales, after.scales.to_vector(), 0.F, 0.F),
            "Dump fixture did not update scales");
        require(
            !near(opacities, after.opacities.to_vector(), 0.F, 0.F),
            "Dump fixture did not update opacity");
        set_dump(had_prior ? previous.c_str() : nullptr);
    } catch (...) {
        set_dump(had_prior ? previous.c_str() : nullptr);
        throw;
    }
}
void correctness() {
    diagnostic_dump();
    Rasterizer raster;
    for (const auto n : {137U, 140003U})
        for (bool geometry : {false, true})
            for (bool filter : {false, true})
                for (const auto mode :
                     {photara::CameraModel::pinhole, photara::CameraModel::opencv_fisheye,
                      photara::CameraModel::equirectangular}) {
                    if (n > 137 && mode != photara::CameraModel::pinhole)
                        continue;
                    auto original = fixture(n, 3, filter);
                    auto c = camera();
                    c.model = mode;
                    RasterizeOptions reference;
                    configure_rasterization(reference, TrainingOptions{});
                    reference.active_sh_degree = 3;
                    reference.require_depth = geometry;
                    reference.fuse_activation = false;
                    auto expected = raster.forward(original, c, reference);
                    auto gc = Tensor::full(expected.color.shape(), 1e-5F, cuda);
                    auto ga = Tensor::full(expected.alpha.shape(), 1e-5F, cuda);
                    auto gd = geometry ? Tensor::full(expected.median_depth.shape(), 1e-5F, cuda)
                                       : Tensor{};
                    auto gn =
                        geometry ? Tensor::full(expected.normal.shape(), 1e-5F, cuda) : Tensor{};
                    const auto gradients = raster.backward(original, expected, gc, ga, gd, gn);
                    {
                        auto options = reference;
                        options.fuse_activation = true;
                        auto m = clone(original), ref = clone(original);
                        auto frame = raster.forward(m, c, options);
                        require(
                            frame.rendered_instances == expected.rendered_instances,
                            "instance count changed");
                        close(expected.color, frame.color, "RGB", 2e-7F, 1e-6F);
                        close(expected.alpha, frame.alpha, "alpha", 1e-7F, 1e-6F);
                        close(expected.median_depth, frame.median_depth, "depth", 1e-8F, 5e-7F);
                        close(expected.normal, frame.normal, "normal");
                        close(expected.visibility, frame.visibility, "visibility", 0, 0);
                        require(
                            expected.radii.to_vector_int() == frame.radii.to_vector_int(),
                            "radii changed");
                        const auto g = raster.backward(m, frame, gc, ga, gd, gn);
                        close_gradient(gradients.means, g.means, "mean gradients", 1e-9F, 5e-4F);
                        close_gradient(gradients.sh, g.sh, "SH gradients", 1e-13F, 1e-6F);
                        close_gradient(
                            gradients.log_scales, g.log_scales, "scale gradients", 5e-11F, 1e-5F);
                        close_gradient(
                            gradients.quaternions, g.quaternions, "quaternion gradients", 5e-11F,
                            1e-5F);
                        close_gradient(
                            gradients.opacity_logits, g.opacity_logits, "opacity gradients", 5e-10F,
                            1e-4F);
                        Optimizer a(m), b(ref);
                        // Correctness cases need observable steps at float32 precision.
                        // The benchmark retains production-scale learning rates.
                        a.structure.means_lr = b.structure.means_lr = .01F;
                        a.structure.scales_lr = b.structure.scales_lr = .02F;
                        a.structure.quaternions_lr = b.structure.quaternions_lr = .02F;
                        for (unsigned step = 0; step < 3; ++step) {
                            const auto before = clone(ref);
                            const auto x = raster.forward(m, c, options),
                                       y = raster.forward(ref, c, reference);
                            (void)raster.backward(m, x, gc, ga, gd, gn, {}, &a.sh, &a.structure);
                            (void)raster.backward(ref, y, gc, ga, gd, gn, {}, &b.sh, &b.structure);
                            close_update(before.means, ref.means, m.means, "Adam mean increments");
                            close_update(
                                before.log_scales, ref.log_scales, m.log_scales,
                                "Adam scale increments");
                            close_update(
                                before.quaternions, ref.quaternions, m.quaternions,
                                "Adam rotation increments", 1e-4F, 1e-9F);
                            close_update(
                                before.opacity_logits, ref.opacity_logits, m.opacity_logits,
                                "Adam opacity increments");
                            close_update(before.sh, ref.sh, m.sh, "Adam SH increments", 1e-3F);
                            const Tensor first_a[]{
                                a.structure.means_first, a.structure.scales_first,
                                a.structure.rotations_first, a.structure.opacity_first};
                            const Tensor first_b[]{
                                b.structure.means_first, b.structure.scales_first,
                                b.structure.rotations_first, b.structure.opacity_first};
                            const Tensor second_a[]{
                                a.structure.means_second, a.structure.scales_second,
                                a.structure.rotations_second, a.structure.opacity_second};
                            const Tensor second_b[]{
                                b.structure.means_second, b.structure.scales_second,
                                b.structure.rotations_second, b.structure.opacity_second};
                            for (unsigned state = 0; state < 4; ++state) {
                                close(
                                    first_a[state], first_b[state], "Adam first moments", 2e-9F,
                                    1e-4F);
                                close(
                                    second_a[state], second_b[state], "Adam second moments", 1e-10F,
                                    1e-6F);
                            }
                        }
                    }
                }
    // Empty visibility must remain safe in both activation paths.
    auto m = fixture(140003, 0, false);
    auto c = camera();
    c.world_to_camera[14] = -20.F;
    for (bool fused : {false, true}) {
        RasterizeOptions o;
        o.require_depth = false;
        o.fuse_activation = fused;
        const auto frame = raster.forward(m, c, o);
        require(frame.rendered_instances == 0, "empty camera emitted instances");
    }
    sync();
    std::cout << "CUDA frontend forward/gradient/Adam parity passed\n";
}
void benchmark(std::size_t n, unsigned iterations) {
    Rasterizer raster;
    const auto c = camera(960, 540);
    // Alternate order across rounds to expose clock/temperature bias.
    for (unsigned round = 0; round < 3; ++round)
        for (unsigned arm = 0; arm < 2; ++arm) {
            const unsigned variant = round % 2 ? 1 - arm : arm;
            auto m = fixture(n, 3, false);
            Optimizer optimizer(m);
            RasterizeOptions o;
            configure_rasterization(o, TrainingOptions{});
            o.active_sh_degree = 3;
            o.require_depth = false;
            o.fuse_activation = variant != 0;
            const auto gc = Tensor::full({3, c.height, c.width}, 1e-6F, cuda);
            const auto ga = Tensor::zeros({c.height, c.width}, cuda);
            const auto step = [&] {
                const auto frame = raster.forward(m, c, o);
                (void)raster.backward(
                    m, frame, gc, ga, {}, {}, {}, &optimizer.sh, &optimizer.structure);
            };
            for (unsigned warm = 0; warm < 5; ++warm)
                step();
            sync();
            cudaMemPool_t pool{};
            require(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess, "default pool");
            std::uint64_t zero = 0;
            require(
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero) == cudaSuccess,
                "reset pool peak");
            const auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < iterations; ++i)
                step();
            sync();
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count() /
                iterations;
            std::uint64_t peak = 0;
            require(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &peak) == cudaSuccess,
                "pool peak");
            std::cout << "frontend_bench count=" << n << " round=" << round
                      << " variant=" << variant << " ms=" << ms << " peak_bytes=" << peak
                      << std::endl;
        }
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench")
            benchmark(
                argc > 2 ? std::stoul(argv[2]) : 1000000, argc > 3 ? std::stoul(argv[3]) : 40);
        else {
            comparison_guards();
            audit_errors = argc > 1 && std::string(argv[1]) == "--audit-errors";
            correctness();
            for (const auto& [name, error] : error_summaries)
                std::cout << name << " max_abs=" << error.absolute << " max_rel=" << error.relative
                          << " max_ulps=" << error.ulps << " relative_l2=" << error.relative_l2
                          << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
