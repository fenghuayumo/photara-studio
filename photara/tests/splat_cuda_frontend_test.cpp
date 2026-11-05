#include "splat/rasterizer.hpp"
#include "../src/splat/sh_adam_quant.hpp"
#include "../src/splat/cuda_ops.hpp"
#include "../src/splat/optimizer.hpp"
#include <cuda_runtime_api.h>
#include "vulkan/backend.hpp"
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
GaussianModel fixture(
    std::size_t n,
    unsigned degree,
    bool filter,
    tinytensor::Device device = cuda) {
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
    m.means = Tensor::from_vector(xyz, {n, 3}, device);
    m.log_scales = Tensor::from_vector(scales, {n, 3}, device);
    m.quaternions = Tensor::from_vector(q, {n, 4}, device);
    m.opacity_logits = Tensor::from_vector(opacity, {n, 1}, device);
    m.sh = Tensor::from_vector(sh, {n, bases, 3}, device);
    if (filter)
        m.filter_3d = Tensor::full({n, 1}, .001F, device);
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
        const auto device = m.means.device();
        structure.means_first = Tensor::full(m.means.shape(), .001F, device);
        structure.means_second = Tensor::full(m.means.shape(), .01F, device);
        structure.scales_first = Tensor::full(m.log_scales.shape(), .001F, device);
        structure.scales_second = Tensor::full(m.log_scales.shape(), .01F, device);
        structure.rotations_first = Tensor::full(m.quaternions.shape(), .001F, device);
        structure.rotations_second = Tensor::full(m.quaternions.shape(), .01F, device);
        structure.opacity_first = Tensor::full(m.opacity_logits.shape(), .001F, device);
        structure.opacity_second = Tensor::full(m.opacity_logits.shape(), .01F, device);
        structure.means_lr = 1e-5F;
        structure.scales_lr = 1e-4F;
        structure.quaternions_lr = 1e-4F;
        structure.opacities_lr = 1e-3F;
        structure.opacity_reg = 1e-5F;
        structure.scale_reg = 1e-5F;
        structure.minimum_log_scale = -12.F;
        structure.maximum_log_scale = 0.F;
        auto state = detail::make_sh_adam_quant(m.size(), m.sh.shape()[1] * 3, device);
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
    for (const auto method : {RenderMethod::gaussian_3dgs, RenderMethod::gaussian_3dgut})
        for (const auto n : {137U, 140003U})
            for (bool geometry : {false, true})
                for (bool filter : {false, true})
                    for (const auto mode :
                         {photara::CameraModel::pinhole, photara::CameraModel::opencv_fisheye,
                          photara::CameraModel::equirectangular}) {
                        if (method == RenderMethod::gaussian_3dgut && n > 137)
                            continue;
                        if (n > 137 && mode != photara::CameraModel::pinhole)
                            continue;
                        auto original = fixture(n, 3, filter);
                        original.render_method = method;
                        auto c = camera();
                        c.model = mode;
                        RasterizeOptions reference;
                        configure_rasterization(reference, TrainingOptions{});
                        reference.active_sh_degree = 3;
                        reference.require_depth = geometry;
                        auto m = clone(original), ref = clone(original);
                        const auto gc = Tensor::full({3, c.height, c.width}, 1e-5F, cuda);
                        const auto ga = Tensor::full({c.height, c.width}, 1e-5F, cuda);
                        const auto gd =
                            geometry ? Tensor::full({c.height, c.width}, 1e-5F, cuda) : Tensor{};
                        const auto gn =
                            geometry ? Tensor::full({3, c.height, c.width}, 1e-5F, cuda) : Tensor{};
                        {
                            const auto options = reference;
                            Optimizer a(m), b(ref);
                            a.structure.capture_opacity_gradient = true;
                            // Correctness cases need observable steps at float32 precision.
                            // The benchmark retains production-scale learning rates.
                            a.structure.means_lr = b.structure.means_lr = .01F;
                            a.structure.scales_lr = b.structure.scales_lr = .02F;
                            a.structure.quaternions_lr = b.structure.quaternions_lr = .02F;
                            for (unsigned step = 0; step < 3; ++step) {
                                const auto before = clone(ref);
                                const auto x = raster.forward(m, c, options),
                                           y = raster.forward(ref, c, reference);
                                const unsigned iteration = step + 1;
                                a.structure.correction1 = a.sh.correction1 =
                                    1.F - std::pow(.9F, float(iteration));
                                a.structure.correction2 = a.sh.correction2 =
                                    1.F - std::pow(.999F, float(iteration));
                                auto gradients = raster.backward(ref, y, gc, ga, gd, gn);
                                detail::add_geometry_regularization(
                                    ref, gradients, b.structure.opacity_reg * float(ref.size()),
                                    b.structure.scale_reg * float(ref.size()) * 3.F);
                                const auto fused =
                                    raster.backward(m, x, gc, ga, gd, gn, {}, &a.sh, &a.structure);
                                close_gradient(
                                    gradients.opacity_logits, fused.opacity_logits,
                                    "fused opacity gradient", 5e-10F, 1e-4F);
                                TrainingOptions training;
                                training.scales_lr = b.structure.scales_lr;
                                training.quaternions_lr = b.structure.quaternions_lr;
                                training.opacities_lr = b.structure.opacities_lr;
                                training.adam_epsilon = b.structure.epsilon;
                                detail::AdamState means{
                                    b.structure.means_first, b.structure.means_second};
                                detail::AdamState scales{
                                    b.structure.scales_first, b.structure.scales_second};
                                detail::AdamState rotations{
                                    b.structure.rotations_first, b.structure.rotations_second};
                                detail::AdamState opacity{
                                    b.structure.opacity_first, b.structure.opacity_second};
                                detail::adam_step_structure(
                                    ref, gradients, means, scales, rotations, opacity,
                                    b.structure.means_lr, iteration, training,
                                    b.structure.minimum_log_scale, b.structure.maximum_log_scale);
                                detail::ShAdamQuant sh{
                                    b.sh.first, b.sh.packed, b.sh.bounds,
                                    int(ref.sh.shape()[1] * 3)};
                                detail::sh_adam_quant_step(
                                    ref.sh, gradients.sh, sh, sh.stride, b.sh.lr, b.sh.rest_lr, .9F,
                                    .999F, a.sh.correction1, a.sh.correction2, b.sh.epsilon, 0.F);
                                close_update(
                                    before.means, ref.means, m.means, "Adam mean increments");
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
                                        second_a[state], second_b[state], "Adam second moments",
                                        1e-10F, 1e-6F);
                                }
                            }
                        }
                    }
    // Empty visibility remains safe with the single activation path.
    auto m = fixture(140003, 0, false);
    auto c = camera();
    c.world_to_camera[14] = -20.F;
    RasterizeOptions o;
    o.require_depth = false;
    require(raster.forward(m, c, o).rendered_instances == 0, "empty camera emitted instances");
    sync();
    std::cout << "CUDA frontend forward/gradient/Adam parity passed\n";
}

void reused_gradient_close(const Tensor& a, const Tensor& b, const char* name,
                           float atol, float rtol) {
    close(a, b, name, atol, rtol);
    const auto x = a.to_vector(), y = b.to_vector();
    double norm = 0, error = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        norm += double(x[i]) * x[i];
        error += double(x[i] - y[i]) * (x[i] - y[i]);
    }
    require(error <= norm * 1e-10, name); // Relative L2 <= 1e-5; rejects missing small gradients.
}

void normal_color_reuse() {
    Rasterizer raster;
    for (int camera_type = 0; camera_type < 4; ++camera_type)
        for (bool compact : {false, true})
            for (bool depth : {false, true}) {
                auto m = fixture(1031, 3, compact);
                auto scales = m.log_scales.to_vector();
                for (auto& v : scales)
                    v += 4.5F;
                m.log_scales = Tensor::from_vector(scales, m.log_scales.shape(), cuda);
                std::vector<float> features(m.size() * 4);
                for (std::size_t i = 0; i < m.size(); ++i) {
                    features[4 * i] = .3F;
                    features[4 * i + 1] = -.7F;
                    features[4 * i + 2] = 1.F;
                    features[4 * i + 3] = i % 2 ? -.5F : .8F;
                }
                m.normal_features = Tensor::from_vector(features, {m.size(), 4}, cuda);
                auto c = camera(97, 65);
                if (camera_type == 1) {
                    c.model = photara::CameraModel::opencv_fisheye;
                    c.k1 = .02F;
                } else if (camera_type == 2) {
                    c.model = photara::CameraModel::equirectangular;
                    c.fx = c.width / (2.F * 3.14159265F);
                    c.fy = c.height / 3.14159265F;
                } else if (camera_type == 3) {
                    c.orthographic = true;
                    c.fx = c.fy = 10.F;
                }
                RasterizeOptions o;
                configure_rasterization(o, TrainingOptions{});
                o.require_depth = depth;
                o.active_sh_degree = 3;
                o.background = {.03F, -.02F, .04F};
                o.rasterization.pixel_center_offset = .5F;
                o.rasterization.compensate_2d = compact;
                o.compact_instance_storage = compact;
                o.snapshot_compaction_threshold = compact ? 0 : std::size_t(-1);
                auto primary = raster.forward(m, c, o);
                require(primary.rendered_instances > 0, "Reuse fixture has no instances");
                const auto gc = Tensor::full(primary.color.shape(), .0003F, cuda);
                const auto ga = Tensor::full(primary.alpha.shape(), -.0002F, cuda);
                const auto gd =
                    depth ? Tensor::full(primary.alpha.shape(), .0001F, cuda) : Tensor{};
                const auto gn =
                    depth ? Tensor::full(primary.color.shape(), -.0001F, cuda) : Tensor{};
                (void)raster.backward(m, primary, gc, ga, gd, gn);
                const auto colors = detail::normal_features_to_normals(m.normal_features);
                o.colors_precomp = colors;
                auto reference = raster.forward(m, c, o);
                const auto expected = raster.backward(m, reference, gc, ga, gd, gn);
                const auto reused = raster.forward_colors(m, primary, colors);
                require(!primary.context.impl, "Reuse did not consume the primary context");
                require(
                    reused.alpha.data_ptr() == primary.alpha.data_ptr() &&
                        reused.median_depth.data_ptr() == primary.median_depth.data_ptr() &&
                        reused.normal.data_ptr() == primary.normal.data_ptr(),
                    "Reuse allocated duplicate geometry attachments");
                close(reference.color, reused.color, "Reused signed feature render", 0, 0);
                close(reference.alpha, reused.alpha, "Reused alpha", 0, 0);
                close(reference.median_depth, reused.median_depth, "Reused median depth", 0, 0);
                close(reference.normal, reused.normal, "Reused normal", 0, 0);
                close(reference.visibility, reused.visibility, "Reused visibility", 0, 0);
                require(
                    reference.radii.to_vector_int() == reused.radii.to_vector_int(),
                    "Reused radii differ");
                auto actual = raster.backward(m, reused, gc, ga, gd, gn);
                const auto compare_gradients = [&](const ModelGradients& a,
                                                   const ModelGradients& b) {
                    reused_gradient_close(a.means, b.means, "Reused means gradient", 2e-7F, 2e-5F);
                    reused_gradient_close(a.log_scales, b.log_scales, "Reused scales gradient", 2e-7F, 2e-5F);
                    reused_gradient_close(a.quaternions, b.quaternions, "Reused quaternion gradient", 2e-7F, 2e-5F);
                    reused_gradient_close(
                        a.opacity_logits, b.opacity_logits, "Reused opacity gradient", 2e-7F,
                        2e-5F);
                    reused_gradient_close(
                        a.colors_precomp, b.colors_precomp, "Reused colors gradient", 2e-7F, 2e-5F);
                    reused_gradient_close(
                        detail::normal_features_backward(m.normal_features, a.colors_precomp),
                        detail::normal_features_backward(m.normal_features, b.colors_precomp),
                        "Reused normal feature gradient", 2e-7F, 2e-5F);
                    require(!b.sh.is_valid(), "Feature render differentiated SH");
                };
                compare_gradients(expected, actual);
                if (depth) {
                    const auto target_camera = raster_sample_camera(c, o);
                    const auto loss_a =
                        detail::compute_normal_field_loss(reference, target_camera, .03F, true);
                    const auto loss_b =
                        detail::compute_normal_field_loss(reused, target_camera, .03F, true);
                    require(std::abs(loss_a.total - loss_b.total) <= 1e-7F + 1e-6F * std::abs(loss_a.total),
                            "Reused normal-field loss differs");
                    if (camera_type == 0) {
                        require(loss_a.total > 0.F, "Normal field fixture has no loss");
                        const auto values = loss_a.color.to_vector();
                        require(
                            std::any_of(
                                values.begin(), values.end(),
                                [](float v) { return std::abs(v) > 1e-8F; }),
                            "Normal field fixture has no feature gradients");
                    }
                    compare_gradients(
                        raster.backward(
                            m, reference, loss_a.color, loss_a.alpha, loss_a.depth, loss_a.normal),
                        raster.backward(
                            m, reused, loss_b.color, loss_b.alpha, loss_b.depth, loss_b.normal));
                }
            }
    // Zero instances still blend the background and preserve empty geometry.
    auto m = fixture(137, 0, false);
    auto c = camera(33, 17);
    c.world_to_camera[14] = -20.F;
    RasterizeOptions o;
    o.background = {.1F, .2F, -.3F};
    auto source = raster.forward(m, c, o);
    auto gc = Tensor::zeros(source.color.shape(), cuda);
    auto ga = Tensor::zeros(source.alpha.shape(), cuda);
    (void)raster.backward(m, source, gc, ga, ga, gc);
    o.colors_precomp = Tensor::full({m.size(), 3}, -.5F, cuda);
    auto reused = raster.forward_colors(m, source, o.colors_precomp);
    auto reference = raster.forward(m, c, o);
    close(reference.color, reused.color, "Reused empty draw list", 0, 0);
    close(reference.median_depth, reused.median_depth, "Reused empty depth", 0, 0);
    auto g = raster.backward(m, reused, gc, ga, ga, gc);
    close(g.means, Tensor::zeros(g.means.shape(), cuda), "Empty reused gradient", 0, 0);

    const auto rejects = [&](RenderResult& frame, const GaussianModel& model) {
        bool rejected = false;
        try {
            (void)raster.forward_colors(model, frame, o.colors_precomp);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && frame.context.impl, "Unsafe reuse was accepted or consumed source");
    };
    source = raster.forward(m, c, o);
    rejects(source, m); // Primary backward must run first.
    (void)raster.backward(m, source, gc, ga, ga, gc);
    auto alias = source;
    rejects(source, m);
    alias.context.impl.reset();
    auto changed = clone(m);
    rejects(source, changed);
    c = camera(33, 17);
    source = raster.forward(m, c, RasterizeOptions{});
    Optimizer optimizer(m);
    (void)raster.backward(m, source, gc, ga, ga, gc, {}, nullptr, &optimizer.structure);
    rejects(source, m); // Fused structure Adam invalidates the projection.
    sync();
    std::cout << "CUDA normal-field color reuse parity and lifecycle passed\n";
}

void benchmark_normal_reuse(std::size_t n, unsigned iterations) {
    const auto c = camera(960, 540);
    auto m = fixture(n, 3, true);
    std::vector<float> features(n * 4);
    for (std::size_t i = 0; i < n; ++i) {
        features[4 * i] = .3F;
        features[4 * i + 1] = -.7F;
        features[4 * i + 2] = 1.F;
        features[4 * i + 3] = i % 2 ? -.5F : .8F;
    }
    m.normal_features = Tensor::from_vector(features, {n, 4}, cuda);
    Rasterizer raster;
    RasterizeOptions o;
    configure_rasterization(o, TrainingOptions{});
    o.active_sh_degree = 3;
    const auto gc = Tensor::full({3, c.height, c.width}, 1e-6F, cuda);
    const auto ga = Tensor::zeros({c.height, c.width}, cuda);
    const auto gn = Tensor::zeros({3, c.height, c.width}, cuda);
    for (unsigned round = 0; round < 3; ++round)
        for (unsigned arm = 0; arm < 2; ++arm) {
            const bool reuse = (arm + round) % 2;
            const auto step = [&] {
                auto primary = raster.forward(m, c, o);
                (void)raster.backward(m, primary, gc, ga, ga, gn);
                auto field_options = o;
                field_options.colors_precomp =
                    detail::normal_features_to_normals(m.normal_features);
                if (!reuse)
                    primary.context.impl.reset();
                auto field = reuse ? raster.forward_colors(m, primary, field_options.colors_precomp)
                                   : raster.forward(m, c, field_options);
                const auto loss = detail::compute_normal_field_loss(field, c, .03F, false);
                auto gradients =
                    raster.backward(m, field, loss.color, loss.alpha, loss.depth, loss.normal);
                gradients.normal_features =
                    detail::normal_features_backward(m.normal_features, gradients.colors_precomp);
            };
            for (unsigned warm = 0; warm < 5; ++warm)
                step();
            sync();
            cudaMemPool_t pool{};
            require(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess, "default pool");
            std::uint64_t peak = 0;
            require(
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &peak) == cudaSuccess,
                "reset pool peak");
            const auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < iterations; ++i)
                step();
            sync();
            const auto ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count() /
                iterations;
            require(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &peak) == cudaSuccess,
                "pool peak");
            std::cout << "normal_reuse_bench count=" << n << " round=" << round
                      << " reuse=" << reuse << " ms=" << ms << " peak_bytes=" << peak << std::endl;
        }
}

void benchmark(std::size_t n, unsigned iterations, RenderMethod method, bool vulkan) {
    const auto device = vulkan ? tinytensor::Device::Vulkan : cuda;
    const auto wait_gpu = [&] {
        if (vulkan) {
            tinytensor::vulkan::synchronize();
            require(
                vkDeviceWaitIdle(tinytensor::vulkan::device_handles().device) == VK_SUCCESS,
                "Vulkan benchmark synchronization failed");
        } else
            sync();
    };
    Rasterizer raster;
    const auto c = camera(960, 540);
    for (unsigned round = 0; round < 3; ++round) {
        auto m = fixture(n, 3, false, device);
        m.render_method = method;
        Optimizer optimizer(m);
        RasterizeOptions o;
        configure_rasterization(o, TrainingOptions{});
        o.active_sh_degree = 3;
        o.require_depth = false;
        const auto gc = Tensor::full({3, c.height, c.width}, 1e-6F, device);
        const auto ga = Tensor::zeros({c.height, c.width}, device);
        const auto step = [&] {
            const auto frame = raster.forward(m, c, o);
            (void)raster.backward(
                m, frame, gc, ga, {}, {}, {}, &optimizer.sh, &optimizer.structure);
        };
        for (unsigned warm = 0; warm < 5; ++warm)
            step();
        wait_gpu();
        cudaMemPool_t pool{};
        if (!vulkan) {
            require(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess, "default pool");
            std::uint64_t zero = 0;
            require(
                cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero) == cudaSuccess,
                "reset pool peak");
        }
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < iterations; ++i)
            step();
        wait_gpu();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count() /
            iterations;
        std::uint64_t peak = 0;
        if (!vulkan)
            require(
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &peak) == cudaSuccess,
                "pool peak");
        std::cout << "frontend_bench backend=" << (vulkan ? "vulkan" : "cuda")
                  << " method=" << (method == RenderMethod::gaussian_3dgut ? "3dgut" : "3dgs")
                  << " count=" << n << " round=" << round << " ms=" << ms;
        if (!vulkan)
            std::cout << " peak_bytes=" << peak;
        std::cout << std::endl;
    }
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench-normal-reuse") {
            benchmark_normal_reuse(
                argc > 2 ? std::stoul(argv[2]) : 1000000, argc > 3 ? std::stoul(argv[3]) : 120);
        } else if (argc > 1 && std::string(argv[1]) == "--normal-reuse") {
            normal_color_reuse();
        } else if (
            argc > 1 &&
            (std::string(argv[1]) == "--bench" || std::string(argv[1]) == "--bench-vulkan"))
            benchmark(
                argc > 2 ? std::stoul(argv[2]) : 1000000, argc > 3 ? std::stoul(argv[3]) : 40,
                argc > 4 && std::string(argv[4]) == "3dgut" ? RenderMethod::gaussian_3dgut
                                                            : RenderMethod::gaussian_3dgs,
                std::string(argv[1]) == "--bench-vulkan");
        else {
            comparison_guards();
            audit_errors = argc > 1 && std::string(argv[1]) == "--audit-errors";
            correctness();
            normal_color_reuse();
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
