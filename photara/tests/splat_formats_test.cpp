#include "splat/formats.hpp"
#include "splat/deployment.hpp"
#include "../src/splat/sog_quantization.hpp"
#include "../src/splat/sog_vulkan.hpp"
#include "../src/splat/sog_palette.hpp"
#include "../src/splat/formats_sog_xt.hpp"
#include "splat/rasterizer.hpp"
#include "splat/trainer.hpp"
#include "sfm/asfm.hpp"
#include "io/image.hpp"
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#if defined(PHOTARA_HAS_OPENMP)
#include <omp.h>
#endif

using namespace photara::splat;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
GaussianModel fixture(unsigned n, unsigned degree) {
    const std::size_t bases = (degree + 1) * (degree + 1);
    std::vector<float> positions(n * 3), scales(n * 3, -2), q(n * 4), sh(n * bases * 3);
    for (unsigned i = 0; i < n; ++i) {
        positions[i * 3] = float(i % 256) * .01F;
        positions[i * 3 + 1] = float(i / 256) * .01F;
        positions[i * 3 + 2] = 2;
        q[i * 4] = 1;
        if (degree) sh[(i * bases + 1) * 3] = i % 2 ? -.5F : .5F;
    }
    constexpr auto cpu = tinytensor::Device::CPU;
    GaussianModel m;
    m.sh_degree = degree;
    m.means = tinytensor::Tensor::from_vector(positions, {n, 3}, cpu);
    m.log_scales = tinytensor::Tensor::from_vector(scales, {n, 3}, cpu);
    m.quaternions = tinytensor::Tensor::from_vector(q, {n, 4}, cpu);
    m.opacity_logits = tinytensor::Tensor::zeros({n, 1}, cpu);
    m.sh = tinytensor::Tensor::from_vector(sh, {n, bases, 3}, cpu);
    return m;
}
// Reordered properties, optional fields and a partial batch after 4 MiB.
void ply_loading(const std::filesystem::path& folder) {
    for (bool learned : {false, true}) {
        constexpr std::size_t n = 18001, bases = 16;
        std::vector<std::string> names{"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "nx"};
        for (unsigned i = 0; i < 45; ++i) names.push_back("f_rest_" + std::to_string(i));
        if (learned) {
            names.push_back("filter_3D");
            for (unsigned i = 0; i < 4; ++i) names.push_back("gaussian_features_" + std::to_string(i));
        }
        std::reverse(names.begin(), names.end());
        const auto path = folder / "batched.ply";
        std::ofstream output(path, std::ios::binary);
        output << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << '\n';
        for (const auto& name : names) output << "property float " << name << '\n';
        output << "end_header\n";
        const auto payload = output.tellp();
        const auto value = [](std::size_t row, const std::string& name) {
            if (name == "rot_0") return 1.F;
            if (name.starts_with("rot_")) return 0.F;
            if (name.starts_with("scale_")) return -2.F;
            if (name == "opacity" && row < 2)
                return row == 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
            return float(row % 101) * .01F + float(name.size()) * .001F;
        };
        for (std::size_t row = 0; row < n; ++row)
            for (const auto& name : names) {
                const float v = value(row, name);
                output.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        output.close();
        const auto m = load_gaussians(path, tinytensor::Device::CPU);
        require(m.size() == n && m.sh_degree == 3, "Batched PLY changed shape");
        for (std::size_t i = 0; i < n; ++i) {
            require(m.means.ptr<float>()[i * 3] == value(i, "x"), "Batched PLY changed position");
            for (unsigned c = 0; c < 3; ++c) {
                require(m.sh.ptr<float>()[i * bases * 3 + c] == value(i, "f_dc_" + std::to_string(c)),
                        "Batched PLY changed DC");
                for (unsigned b = 1; b < bases; ++b)
                    require(m.sh.ptr<float>()[(i * bases + b) * 3 + c] ==
                            value(i, "f_rest_" + std::to_string(c * 15 + b - 1)), "Batched PLY changed SH order");
            }
            for (unsigned c = 0; c < 4; ++c)
                require(m.normal_features.ptr<float>()[i * 4 + c] ==
                        (learned ? value(i, "gaussian_features_" + std::to_string(c)) : ((c == 0 || c == 3) ? 1.F : 0.F)),
                        "Batched PLY changed normal features");
            if (learned) require(m.filter_3d.ptr<float>()[i] == value(i, "filter_3D"), "Batched PLY changed filter");
        }
        require(m.opacity_logits.ptr<float>()[0] == 104.F && m.opacity_logits.ptr<float>()[1] == -104.F,
                "Batched PLY lost saturated opacity handling");
        // A non-finite attribute must still be rejected after direct tensor decoding.
        std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(payload);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        corrupt.write(reinterpret_cast<const char*>(&nan), sizeof(nan));
        corrupt.close();
        bool rejected = false;
        try { (void)load_gaussians(path, tinytensor::Device::CPU); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "PLY accepted non-finite parameters");
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        rejected = false;
        try { (void)load_gaussians(path, tinytensor::Device::CPU); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "PLY accepted a truncated final batch");
    }
}
// Reproducible CPU loading benchmark; the generated PLY is about 248 MB.
void ply_load_benchmark(const std::filesystem::path& folder) {
    const auto path = folder / "load-benchmark.ply";
    save_gaussians(fixture(1'000'000, 3), path);
    for (int run = 0; run < 3; ++run) {
        const auto start = std::chrono::steady_clock::now();
        const auto model = load_gaussians(path, tinytensor::Device::CPU);
        require(model.size() == 1'000'000, "Benchmark lost Gaussians");
        std::cout << "PLY load run=" << run << " seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << std::endl;
    }
}
void antialias_metadata(const std::filesystem::path& folder) {
    for (auto mode : {SplatFilterMode::classic, SplatFilterMode::mip_2d}) {
        auto model = fixture(17, 0);
        model.filter_mode = mode;
        const bool aa = mode == SplatFilterMode::mip_2d;
        const auto sog = folder / (aa ? "aa.sog" : "classic.sog");
        const auto spz = folder / (aa ? "aa.spz" : "classic.spz");
        const auto unbundled = folder / (aa ? "aa" : "classic");
        const auto glb = folder / (aa ? "aa.glb" : "classic.glb");
        save_gaussians(model, sog, GaussianFormat::sog);
        save_gaussians(model, spz, GaussianFormat::spz);
        save_gaussians(model, glb, GaussianFormat::glb);
        save_sog_unbundled(model, unbundled);
        std::ifstream meta_file(unbundled / "meta.json", std::ios::binary);
        const std::string meta((std::istreambuf_iterator<char>(meta_file)), {});
        require((meta.find("\"model\":\"antialiased\"") != std::string::npos) == aa,
                "SOG antialias metadata is wrong");
        std::ifstream spz_file(spz, std::ios::binary);
        spz_file.seekg(14);
        const int flags = spz_file.get();
        require(flags != EOF && ((flags & 1) != 0) == aa, "SPZ antialias flag is wrong");
        std::ifstream glb_file(glb, std::ios::binary);
        const std::string glb_bytes((std::istreambuf_iterator<char>(glb_file)), {});
        require((glb_bytes.find("\"photara\":{\"antialias\":true}") != std::string::npos) == aa,
                "GLB antialias metadata is wrong");
        for (const auto& file : {sog, spz, unbundled, glb})
            require(load_gaussians(file, tinytensor::Device::CPU).filter_mode == mode,
                    "Antialias filter mode did not survive round trip");
    }
    auto baked = fixture(1, 0);
    const auto original_scales = baked.log_scales;
    baked.filter_mode = SplatFilterMode::mip;
    baked.filter_3d = tinytensor::Tensor::from_vector({.2F}, {1, 1}, tinytensor::Device::CPU);
    bake_gaussian_3d_filter(baked);
    const double expected_scale = std::sqrt(std::exp(-4.0) + .04);
    require(std::abs(std::exp(baked.log_scales.to_vector()[0]) - expected_scale) < 1e-6,
            "Mip3D bake changed effective scale");
    const double expected_alpha = .5 * std::pow(std::exp(-2.0) / expected_scale, 3);
    require(std::abs(1.0 / (1.0 + std::exp(-baked.opacity_logits.to_vector()[0])) - expected_alpha) < 1e-6,
            "Mip3D bake changed effective opacity");
    require(original_scales.to_vector()[0] == -2.F && !baked.filter_3d.is_valid() &&
            baked.filter_mode == SplatFilterMode::mip_2d, "Bake changed source storage or lost Mip2D");
    const auto baked_ply = folder / "baked.ply";
    save_gaussians(baked, baked_ply);
    const auto restored_baked = load_gaussians(baked_ply, tinytensor::Device::CPU);
    require(!restored_baked.filter_3d.is_valid() &&
            restored_baked.filter_mode == SplatFilterMode::mip_2d &&
            std::abs(std::exp(restored_baked.log_scales.to_vector()[0]) - expected_scale) < 1e-6 &&
            std::abs(1.0 / (1.0 + std::exp(-restored_baked.opacity_logits.to_vector()[0])) - expected_alpha) < 1e-6,
            "Baked PLY did not preserve effective scale, opacity and filter mode");
    auto lod_model = fixture(17, 0);
    lod_model.filter_mode = SplatFilterMode::mip_2d;
    LodOptions lod_options;
    lod_options.levels = 2;
    lod_options.vulkan = false;
    const auto lod = folder / "aa-lod";
    export_streamed_sog(lod_model, lod, lod_options);
    require(load_gaussians(lod / "0_0", tinytensor::Device::CPU).filter_mode == SplatFilterMode::mip_2d,
            "Streamed SOG lost antialias metadata");
    auto model = fixture(17, 0);
    for (auto mode : {SplatFilterMode::mip_3d, SplatFilterMode::mip}) {
        model.filter_mode = mode;
        for (auto format : {GaussianFormat::sog, GaussianFormat::spz, GaussianFormat::glb, GaussianFormat::usdc}) {
            bool rejected = false;
            try { save_gaussians(model, folder / "rejected", format); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Export discarded a Mip3D filter");
        }
    }

}
void gut_exports(const std::filesystem::path& folder) {
    auto model = fixture(17, 1);
    model.render_method = RenderMethod::gaussian_3dgut;
    for (auto mode : {SplatFilterMode::classic, SplatFilterMode::mip_2d}) {
        model.filter_mode = mode;
        for (auto format : {GaussianFormat::sog, GaussianFormat::spz, GaussianFormat::glb}) {
            const auto path = folder / (std::string("gut-") + filter_mode_name(mode) + "." + gaussian_format_extension(format));
            save_gaussians(model, path, format);
            const auto restored = load_gaussians(path, tinytensor::Device::CPU);
            require(restored.size() == model.size() && restored.sh_degree == model.sh_degree &&
                    restored.filter_mode == mode, "3DGUT export lost shape or filter mode");
            const auto values = restored.sh.to_vector();
            const auto expected = model.sh.to_vector();
            for (std::size_t i = 0; i < values.size(); ++i)
                require(std::abs(values[i] - expected[i]) < .04F, "3DGUT export changed SH data");
        }
    }
    LodOptions options;
    options.vulkan = false;
    options.levels = 2;
    export_streamed_sog(model, folder / "gut-lod", options);
    require(load_gaussians(folder / "gut-lod" / "0_0", tinytensor::Device::CPU).filter_mode == SplatFilterMode::mip_2d,
            "3DGUT streamed SOG lost antialias metadata");
}
void scalar_quantization() {
    std::vector<float> sparse{-.5F, 0, .5F, 0, .5F};
    auto book = detail::sog_codebook(sparse, "test");
    require(book.size() == 256 && std::is_sorted(book.begin(), book.end()), "Invalid scalar codebook");
    for (float v : sparse)
        require(book[detail::sog_nearest_codebook(book, v)] == v, "Sparse values must remain exact");
    require(detail::sog_nearest_codebook({0, 2}, 1) == 0, "Midpoint tie must choose lower entry");
    std::vector<float> values;
    for (unsigned i = 0; i < 10000; ++i) values.push_back(.02F * std::sin(float(i) * 1.37F));
    values.push_back(-3); values.push_back(3);
    book = detail::sog_codebook(values, "test");
    double adaptive_error = 0, uniform_error = 0;
    for (float v : values) {
        const double delta = book[detail::sog_nearest_codebook(book, v)] - v;
        adaptive_error += delta * delta;
        const double uniform = -3 + std::round((double(v) + 3) / 6 * 255) / 255 * 6;
        uniform_error += (uniform - v) * (uniform - v);
    }
    require(adaptive_error < uniform_error * .1, "Adaptive quantization failed dense-center/outlier fixture");
    for (float v : values) {
        const auto nearest = std::min_element(book.begin(), book.end(), [v](float a, float b) {
            return std::abs(double(a) - v) < std::abs(double(b) - v);
        });
        require(std::abs(double(book[detail::sog_nearest_codebook(book, v)]) - v) ==
                std::abs(double(*nearest) - v), "Binary codebook query is not nearest");
    }
    std::cout << "Scalar SSE adaptive=" << adaptive_error << " uniform=" << uniform_error << '\n';
}
void roundtrip(const std::filesystem::path& folder, unsigned n, unsigned degree, GaussianFormat format) {
    auto m = fixture(n, degree);
    const auto file = folder / (std::to_string(n) + "." + gaussian_format_extension(format));
    const auto start = std::chrono::steady_clock::now();
    float progress = 0;
    unsigned updates = 0;
    save_gaussians_with_progress(m, file, format, [&](float value) {
        require(value >= progress && value <= 1.F, "Export progress is not monotonic");
        progress = value;
        ++updates;
    });
    require(progress == 1.F && updates >= 2, "Export progress did not complete");
    if (format == GaussianFormat::sog || format == GaussianFormat::spz ||
        (format == GaussianFormat::ply && n > 4096))
        require(updates >= 6, "Export omitted incremental progress updates");
    auto loaded = load_gaussians(file, tinytensor::Device::CPU);
    require(loaded.size() == n && loaded.sh_degree == degree, "Format changed shape or SH degree");
    const auto original = m.sh.to_vector(), decoded = loaded.sh.to_vector();
    for (std::size_t i = 0; i < original.size(); ++i)
        require(std::abs(original[i] - decoded[i]) < (format == GaussianFormat::spz ? .04F : .005F),
                "Format changed an alternating SH class");
    const auto means = m.means.to_vector(), restored = loaded.means.to_vector();
    for (std::size_t i = 0; i < means.size(); ++i)
        require(std::abs(means[i] - restored[i]) < .02F, "Format changed positions");
    std::cout << gaussian_format_name(format) << " n=" << n << " degree=" << degree
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " bytes=" << std::filesystem::file_size(file) << '\n';
}
Eigen::Matrix3f covariance(const std::vector<float>& q, const std::vector<float>& s, std::size_t i) {
    Eigen::Quaternionf rotation(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
    rotation.normalize();
    Eigen::Vector3f variance;
    for (unsigned d = 0; d < 3; ++d) variance[d] = std::exp(2 * s[i * 3 + d]);
    return rotation.toRotationMatrix() * variance.asDiagonal() * rotation.toRotationMatrix().transpose();
}

void sog_xt(const std::filesystem::path& folder, unsigned n = 17) {
    require(parse_gaussian_format("SOG-XT") == GaussianFormat::sog_xt &&
            gaussian_format_from_path("test.SOG-XT") == GaussianFormat::sog_xt &&
            std::string(gaussian_format_name(GaussianFormat::sog_xt)) == "sog-xt" &&
            std::string(gaussian_format_extension(GaussianFormat::sog_xt)) == "sog-xt",
            "SOG-XT format dispatch failed");
    for (unsigned degree = 0; degree <= 3; ++degree) {
        auto model = fixture(n, degree);
        const unsigned bases = (degree + 1) * (degree + 1);
        auto positions = model.means.to_vector(), scales = model.log_scales.to_vector();
        auto rotations = model.quaternions.to_vector(), sh = model.sh.to_vector();
        std::vector<float> logits(n);
        for (unsigned i = 0; i < n; ++i) {
            // Stable row IDs recover attribute correspondence after arbitrary sorting.
            positions[i * 3] = float(i) * .01F;
            positions[i * 3 + 1] = -1.F;
            rotations[i * 4] = std::cos(float(i % 13) * .1F);
            rotations[i * 4 + 1] = std::sin(float(i % 13) * .1F);
            if (i % 2) for (unsigned d = 0; d < 4; ++d) rotations[i * 4 + d] *= -1;
            scales[i * 3] = -2.2F;
            scales[i * 3 + 1] = -1.6F;
            scales[i * 3 + 2] = -3.F;
            logits[i] = float(i % 7) - 3;
            for (unsigned b = 0; b < bases; ++b) for (unsigned c = 0; c < 3; ++c)
                sh[(std::size_t(i) * bases + b) * 3 + c] = .03F * float(b * 3 + c) + (i % 2 ? -.2F : .2F);
        }
        constexpr auto cpu = tinytensor::Device::CPU;
        model.means = tinytensor::Tensor::from_vector(positions, {n, 3}, cpu);
        model.log_scales = tinytensor::Tensor::from_vector(scales, {n, 3}, cpu);
        model.quaternions = tinytensor::Tensor::from_vector(rotations, {n, 4}, cpu);
        model.opacity_logits = tinytensor::Tensor::from_vector(logits, {n, 1}, cpu);
        model.sh = tinytensor::Tensor::from_vector(sh, {n, bases, 3}, cpu);
        model.filter_mode = degree % 2 ? SplatFilterMode::mip_2d : SplatFilterMode::classic;
        const auto path = folder / ("roundtrip-" + std::to_string(degree) + ".sog-xt");
        const auto start = std::chrono::steady_clock::now();
        save_gaussians(model, path);
        const auto restored = load_gaussians(path, cpu);
        SogXtExportOptions options;
        float progress = 0;
        options.progress = [&](float value) {
            require(value >= progress && value <= 1.F, "SOG-XT progress is not monotonic");
            progress = value;
        };
        const auto repeated_path = folder / ("repeated-" + std::to_string(degree) + ".sog-xt");
        save_sog_xt(model, repeated_path, options);
        require(progress == 1.F, "SOG-XT progress did not complete");
        const auto repeated_model = load_gaussians(repeated_path, cpu);
        require(repeated_model.means.to_vector() == restored.means.to_vector() &&
                repeated_model.log_scales.to_vector() == restored.log_scales.to_vector() &&
                repeated_model.quaternions.to_vector() == restored.quaternions.to_vector() &&
                repeated_model.opacity_logits.to_vector() == restored.opacity_logits.to_vector() &&
                repeated_model.sh.to_vector() == restored.sh.to_vector(),
                "SOG-XT parallel encoding changed decoded attributes");
        require(restored.size() == n && restored.sh_degree == degree &&
                restored.filter_mode == model.filter_mode, "SOG-XT changed shape or metadata");
        const auto p = restored.means.to_vector(), s = restored.log_scales.to_vector();
        const auto q = restored.quaternions.to_vector(), h = restored.sh.to_vector();
        const auto a = restored.opacity_logits.to_vector();
        float log_min = std::numeric_limits<float>::max(), log_max = std::numeric_limits<float>::lowest();
        for (float value : positions) {
            const float encoded = std::copysign(std::log1p(std::abs(value)), value);
            log_min = std::min(log_min, encoded); log_max = std::max(log_max, encoded);
        }
        const double half_step = (double(log_max) - log_min) / (2 * 65535);
        std::vector<bool> seen(n);
        for (unsigned row = 0; row < n; ++row) {
            const auto original = unsigned(std::lround(p[row * 3] / .01F));
            require(original < n && !seen[original], "SOG-XT lost, duplicated or misplaced a Gaussian");
            seen[original] = true;
            for (unsigned d = 0; d < 3; ++d) {
                const float value = positions[original * 3 + d];
                const double bound = (1 + std::abs(value)) * std::expm1(half_step) * 1.01 + 2e-6;
                require(std::abs(p[row * 3 + d] - value) <= bound, "SOG-XT position quantization exceeds half-step bound");
            }
            require((covariance(q, s, row) - covariance(rotations, scales, original)).norm() /
                    covariance(rotations, scales, original).norm() < .055F, "SOG-XT changed Gaussian covariance");
            require(std::abs(a[row] - logits[original]) < .025F, "SOG-XT detached opacity");
            for (unsigned b = 0; b < bases * 3; ++b)
                require(std::abs(h[std::size_t(row) * bases * 3 + b] - sh[std::size_t(original) * bases * 3 + b]) < .006F,
                        "SOG-XT detached or corrupted SH coefficients");
        }
        std::cout << "sog-xt n=" << n << " degree=" << degree << " bytes=" << std::filesystem::file_size(path)
                  << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
        if (n <= 17) {
            const auto assets = detail::encode_sog_xt(model);
            if (degree == 3) {
                auto external = assets;
                auto meta = nlohmann::json::parse(assets.at("container_meta.yaml"));
                meta.erase("format"); meta.erase("version");
                meta["meta"].erase("count"); meta["meta"].erase("sh_degree");
                for (auto& op : meta["ops"]) for (auto& transform : op["transforms"])
                    if (transform.contains("simple_quantize")) transform["simple_quantize"].erase("levels");
                const std::string yaml = "meta:\n  grid_side: " + meta["meta"]["grid_side"].dump() +
                    "\n  f_rest_centroids_side: " + meta["meta"]["f_rest_centroids_side"].dump() +
                    "\n  image_codec: webp\nops: " + meta["ops"].dump() + "\n";
                external["container_meta.yaml"].assign(yaml.begin(), yaml.end());
                const auto expected = detail::decode_sog_xt(assets);
                const auto actual = detail::decode_sog_xt(external);
                require(actual.count == n && actual.degree == 3, "Reference YAML inference failed");
                for (std::size_t i = 0; i < actual.sh.size(); ++i)
                    require(std::abs(actual.sh[i] - expected.sh[i]) < 1e-5F, "Reference SH ordering differs");
                for (std::size_t i = 0; i < actual.means.size(); ++i)
                    require(std::abs(actual.means[i] - expected.means[i]) < 1e-5F, "Reference means differ");
                const auto directory = folder / ("external-" + std::to_string(n));
                std::filesystem::create_directories(directory);
                for (const auto& [name, bytes] : external) {
                    std::ofstream out(directory / name, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                }
                require(load_gaussians(directory, cpu).size() == n &&
                        load_gaussians(directory / "container_meta.yaml", cpu).size() == n,
                        "Reference SOG-XT directory detection failed");
            }
            const auto rejects = [&](detail::SogXtFiles invalid) {
                bool rejected = false;
                try { (void)detail::decode_sog_xt(invalid); } catch (const std::exception&) { rejected = true; }
                require(rejected, "SOG-XT accepted corrupt assets");
            };
            auto invalid = assets;
            invalid.erase("active_mask.webp"); rejects(std::move(invalid));
            invalid = assets;
            invalid["scales.webp"] = {0, 1, 2}; rejects(std::move(invalid));
            invalid = assets;
            auto& text = invalid["container_meta.yaml"];
            std::string json(text.begin(), text.end());
            const auto key = json.find("\"count\":17");
            if (n == 17) {
                require(key != std::string::npos, "SOG-XT metadata omitted count");
                json.replace(key, 10, "\"count\":18");
                text.assign(json.begin(), json.end()); rejects(std::move(invalid));
            }
        }
    }
}

void sog_xt_covariance_symmetry() {
    constexpr unsigned side = 8, n = side * side;
    std::vector<float> q(n * 4), scales(n * 3), features(n * 3);
    std::vector<std::uint8_t> active(n, 1);
    for (unsigned i = 0; i < n; ++i) {
        const Eigen::Quaternionf rotation(Eigen::AngleAxisf(float(i) * .21F, Eigen::Vector3f(1, 2, 3).normalized()));
        q[i * 4] = rotation.w(); q[i * 4 + 1] = rotation.x(); q[i * 4 + 2] = rotation.y(); q[i * 4 + 3] = rotation.z();
        for (unsigned d = 0; d < 3; ++d) { scales[i * 3 + d] = -float((i + d) % 5) * .7F; features[i * 3 + d] = float((i * 17 + d) % 19); }
    }
    const auto before_q = q, before_scales = scales;
    detail::sog_xt_pras(q, scales, active, side);
    for (unsigned i = 0; i < n; ++i)
        require((covariance(q, scales, i) - covariance(before_q, before_scales, i)).norm() < 2e-6F,
                "PRAS changed covariance before quantization");
    auto order = detail::sog_xt_layout(features, side, 3);
    std::sort(order.begin(), order.end());
    for (unsigned i = 0; i < n; ++i) require(order[i] == i, "SOG-XT layout is not a permutation");
}

void sog_xt_parallel_parity() {
#if defined(PHOTARA_HAS_OPENMP)
    const int previous = omp_get_max_threads();
    // Include a non-power-of-two grid with incomplete edge tiles.
    for (unsigned side : {64U, 72U}) {
        const unsigned n = side * side;
        std::vector<float> features(n * 6), q(n * 4), scales(n * 3);
        std::vector<std::uint8_t> active(n, 1);
        std::mt19937 random(731);
        std::uniform_real_distribution<float> values(-1.F, 1.F);
        for (auto& v : features) v = values(random);
        for (unsigned i = 0; i < n; ++i) {
            Eigen::Quaternionf rotation(1.F, values(random), values(random), values(random));
            rotation.normalize();
            q[i*4] = rotation.w(); q[i*4+1] = rotation.x();
            q[i*4+2] = rotation.y(); q[i*4+3] = rotation.z();
            for (unsigned d = 0; d < 3; ++d) scales[i*3+d] = values(random);
            active[i] = i % 7 != 0;
        }
        auto parallel_q = q, parallel_s = scales;
        omp_set_num_threads(1);
        const auto serial = detail::sog_xt_layout(features, side, 6);
        detail::sog_xt_pras(q, scales, active, side);
        omp_set_num_threads(4);
        require(serial == detail::sog_xt_layout(features, side, 6),
                "SOG-XT parallel layout differs from serial");
        detail::sog_xt_pras(parallel_q, parallel_s, active, side);
        require(q == parallel_q && scales == parallel_s,
                "SOG-XT parallel PRAS differs from serial");
    }
    omp_set_num_threads(previous);
#endif
}

void sog_xt_benchmark(const std::filesystem::path& folder, const std::filesystem::path& source) {
    std::filesystem::create_directories(folder);
    auto model = load_gaussians(source, tinytensor::Device::CPU);
    bake_gaussian_3d_filter(model);
    std::vector<Camera> cameras;
    const auto sfm_path = source.parent_path() / "sfm.bin";
    if (std::filesystem::exists(sfm_path)) {
        const auto scene = photara::sfm::load_asfm(sfm_path);
        std::vector<unsigned> registered;
        for (unsigned i = 0; i < scene.images.size(); ++i) if (scene.images[i].registered) registered.push_back(i);
        for (unsigned v = 0; v < std::min<std::size_t>(8, registered.size()); ++v) {
            const auto& image = scene.images[registered[std::size_t(v) * registered.size() / std::min<std::size_t>(8, registered.size())]];
            const auto& intrinsic = scene.camera_of(image);
            const double factor = std::min(1.0, 800.0 / std::max(intrinsic.width, intrinsic.height));
            Camera camera;
            camera.width = unsigned(std::lround(intrinsic.width * factor));
            camera.height = unsigned(std::lround(intrinsic.height * factor));
            camera.fx = float(intrinsic.fx * factor); camera.fy = float(intrinsic.fy * factor);
            camera.cx = float(intrinsic.cx * factor); camera.cy = float(intrinsic.cy * factor);
            camera.model = intrinsic.model;
            camera.k1 = float(intrinsic.k1); camera.k2 = float(intrinsic.k2);
            camera.k3 = float(intrinsic.p1); camera.k4 = float(intrinsic.p2);
            const photara::sfm::Vec3 translation = image.pose.translation();
            for (unsigned r = 0; r < 4; ++r) for (unsigned c = 0; c < 4; ++c)
                camera.world_to_camera[c * 4 + r] = r < 3 ? float(c < 3 ? image.pose.R(r,c) : translation[r]) : float(c == 3);
            for (unsigned d = 0; d < 3; ++d) camera.position[d] = float(image.pose.C[d]);
            cameras.push_back(camera);
        }
    }
    const auto render = [&](GaussianModel gpu, const std::string& name) {
        for (auto* tensor : {&gpu.means, &gpu.log_scales, &gpu.quaternions, &gpu.opacity_logits, &gpu.sh, &gpu.normal_features})
            if (tensor->is_valid()) *tensor = tensor->to(tinytensor::Device::CUDA);
        std::vector<std::vector<float>> images;
        RasterizeOptions options;
        options.active_sh_degree = gpu.sh_degree; options.require_depth = false; options.record_backward_state = false;
        options.kernel_size = gpu.filter_mode == SplatFilterMode::mip_2d ? .1F : .3F;
        options.rasterization.compensate_2d = gpu.filter_mode == SplatFilterMode::mip_2d;
        for (unsigned v = 0; v < cameras.size(); ++v) {
            const auto& camera = cameras[v];
            auto rgb = Rasterizer{}.forward(gpu, camera, options).color.to_vector();
            const auto pixels = std::size_t(camera.width) * camera.height;
            photara::io::RgbImage image{camera.width, camera.height, std::vector<std::uint8_t>(pixels * 3)};
            for (std::size_t i = 0; i < pixels; ++i) for (unsigned c = 0; c < 3; ++c)
                image.pixels[i * 3 + c] = std::uint8_t(std::lround(std::clamp(rgb[c * pixels + i], 0.F, 1.F) * 255));
            photara::io::save_rgb_png(image, folder / (name + "-view-" + std::to_string(v) + ".png"));
            images.push_back(std::move(rgb));
        }
        return images;
    };
    const auto reference_images = cameras.empty() ? std::vector<std::vector<float>>{} : render(model, "original");
    std::cout << "Encoding benchmark n=" << model.size() << " degree=" << model.sh_degree << std::endl;
    for (auto format : {GaussianFormat::ply, GaussianFormat::sog, GaussianFormat::sog_xt}) {
        const auto path = folder / (std::string("benchmark.") + gaussian_format_extension(format));
        const auto start = std::chrono::steady_clock::now();
        save_gaussians(model, path);
        const auto encoded = std::chrono::steady_clock::now();
        const auto restored = load_gaussians(path, tinytensor::Device::CPU);
        require(restored.size() == model.size() && restored.sh_degree == model.sh_degree,
                "Encoding benchmark changed model shape");
        std::cout << gaussian_format_name(format) << " bytes=" << std::filesystem::file_size(path)
                  << " encode_seconds=" << std::chrono::duration<double>(encoded - start).count()
                  << " decode_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - encoded).count()
                  << std::endl;
        if (format == GaussianFormat::sog_xt) save_gaussians(restored, folder / "decoded-sog-xt.ply");
        if (!cameras.empty() && format != GaussianFormat::ply) {
            const auto images = render(restored, gaussian_format_name(format));
            double mean_psnr = 0;
            for (unsigned v = 0; v < images.size(); ++v) {
                double mse = 0;
                for (std::size_t i = 0; i < images[v].size(); ++i) {
                    const double delta = double(images[v][i]) - reference_images[v][i]; mse += delta * delta;
                }
                mse /= images[v].size();
                const double psnr = -10 * std::log10(std::max(mse, 1e-15)); mean_psnr += psnr;
                std::cout << gaussian_format_name(format) << " view=" << v << " psnr_vs_original=" << psnr << " mse=" << mse << std::endl;
            }
            std::cout << gaussian_format_name(format) << " mean_psnr_vs_original=" << mean_psnr / images.size() << std::endl;
        }
    }
}

#if defined(TINYTENSOR_HAS_VULKAN)
void gpu_clustering(unsigned n, unsigned budget) {
    constexpr unsigned dims = 45;
    std::vector<float> points(std::size_t(n) * dims);
    for (unsigned i = 0; i < n; ++i)
        for (unsigned d = 0; d < dims; ++d)
            points[std::size_t(i) * dims + d] = float(i * 17U % 8U) * .1F +
                float(d) * .001F + .0001F * std::sin(float(i + d));
    const auto start = std::chrono::steady_clock::now();
    auto palette = detail::cluster_sog_vulkan(points, n, dims, budget);
    const auto k = palette.centroids.size() / dims;
    require(k > 1 && k <= budget && palette.labels.size() == n, "Invalid Vulkan palette size");
    std::vector<double> sums(k * dims);
    std::vector<unsigned> counts(k);
    double error = 0;
    for (unsigned i = 0; i < n; ++i) {
        const auto label = palette.labels[i];
        require(label < k, "Vulkan cluster label out of range");
        ++counts[label];
        for (unsigned d = 0; d < dims; ++d) {
            const auto v = points[std::size_t(i) * dims + d];
            sums[label * dims + d] += v;
            const double delta = v - palette.centroids[label * dims + d];
            error += delta * delta;
        }
    }
    require(error / points.size() < 1e-6, "Vulkan clustering mixed separated SH classes");
    for (std::size_t c = 0; c < k; ++c) {
        require(counts[c] > 0, "Unoccupied palette entry escaped compaction");
        for (unsigned d = 0; d < dims; ++d)
            require(std::abs(sums[c * dims + d] / counts[c] - palette.centroids[c * dims + d]) < 2e-4,
                    "Vulkan center is not the mean of its assigned members");
    }
    std::cout << "Vulkan SH clustering n=" << n << " palette=" << k << " mse=" << error / points.size()
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
}
void gpu_clustering_quality() {
    constexpr unsigned n = 131073, dims = 45, k = 65536;
    std::vector<float> points(std::size_t(n) * dims);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-.25F, .25F);
    for (auto& v : points) v = distribution(random);
    const auto start = std::chrono::steady_clock::now();
    const auto palette = detail::cluster_sog_vulkan(points, n, dims, k);
    double old_error = 0, new_error = 0;
    for (unsigned i = 0; i < n; ++i) {
        const std::size_t old_label = std::size_t(i) * k / n;
        const std::size_t old_source = old_label * n / k;
        for (unsigned d = 0; d < dims; ++d) {
            const double a = points[std::size_t(i) * dims + d];
            const double old_delta = a - points[old_source * dims + d];
            const double new_delta = a - palette.centroids[std::size_t(palette.labels[i]) * dims + d];
            old_error += old_delta * old_delta;
            new_error += new_delta * new_delta;
        }
    }
    require(new_error < old_error * .8, "Vulkan vector clustering did not improve SH reconstruction");
    std::cout << "Random degree-3 SH n=" << n << " old_mse=" << old_error / points.size()
              << " Vulkan_mse=" << new_error / points.size() << " seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << '\n';
}
void streamed_palette(const std::filesystem::path& folder) {
    constexpr unsigned n = 65537;
    auto model = fixture(n, 1);
    LodOptions options;
    options.levels = 2;
    options.keep_ratio = .999999F; // Ceil preserves the count; exercise both SOG writes.
    const auto output = folder / "lod";
    export_streamed_sog(model, output, options);
    auto decoded = load_gaussians(output / "0_0", tinytensor::Device::CPU);
    require(decoded.size() == n, "Streamed SOG lost points");
    const auto means = decoded.means.to_vector(), sh = decoded.sh.to_vector();
    for (unsigned i = 0; i < n; ++i) {
        const int column = static_cast<int>(std::lround(means[i * 3] / .01F));
        const float expected = column % 2 ? -.5F : .5F;
        require(std::abs(sh[(std::size_t(i) * 4 + 1) * 3] - expected) < .005F,
                "Morton ordering detached streamed SH from its position");
    }
}
#endif
} // namespace
void sog_xt_training_test(const std::filesystem::path& folder) {
    using tinytensor::Tensor;using tinytensor::Device;
    auto model=fixture(129,3);
    std::vector<float> opacity_values(model.size());
    for(std::size_t i=0;i<model.size();++i)opacity_values[i]=float(int(i%11)-5)*.1F;
    model.opacity_logits=Tensor::from_vector(opacity_values,model.opacity_logits.shape(),Device::CPU);
    for(auto* t:{&model.means,&model.log_scales,&model.quaternions,&model.opacity_logits,&model.sh})*t=t->to(Device::CUDA);
    detail::SogXtTrainingCodec codec;
    detail::encode_sog_xt(model,&codec);
    auto quantized=detail::sog_xt_training_forward(codec);
    const auto files=detail::encode_sog_xt_frozen(codec,quantized);
    const auto decoded=detail::decode_sog_xt(files);
    for(const auto& pair:{std::pair{quantized.means.to_vector(),decoded.means},
        std::pair{quantized.log_scales.to_vector(),decoded.log_scales},
        std::pair{quantized.opacity_logits.to_vector(),decoded.opacity_logits},
        std::pair{quantized.sh.to_vector(),decoded.sh}}) {
        require(pair.first.size()==pair.second.size(),"Frozen codec shape mismatch");
        for(std::size_t i=0;i<pair.first.size();++i)
            require(std::abs(pair.first[i]-pair.second[i])<3e-5F*(1+std::abs(pair.first[i])),"Training/export codec disagreement");
    }
    const auto q=quantized.quaternions.to_vector();
    for(std::size_t i=0;i<model.size();++i){float norm=0;for(unsigned d=0;d<4;++d)norm+=q[i*4+d]*q[i*4+d];
        for(unsigned d=0;d<4;++d)require(std::abs(q[i*4+d]/std::sqrt(norm)-decoded.rotations[i*4+d])<2e-5F,"Frozen quaternion normalization mismatch");}
    require(files.at("f_rest_labels.webp")==codec.files.at("f_rest_labels.webp")&&
        files.at("active_mask.webp")==codec.files.at("active_mask.webp"),"Training changed discrete codec layout");
    codec.latent.opacity_logits=Tensor::zeros_like(codec.latent.opacity_logits);
    auto rebased_forward=detail::sog_xt_training_forward(codec);
    const auto rebased=detail::decode_sog_xt(detail::encode_sog_xt_frozen(codec,rebased_forward));
    const auto opacity_expected=rebased_forward.opacity_logits.to_vector();
    for(std::size_t i=0;i<model.size();++i)
        require(std::abs(opacity_expected[i]-rebased.opacity_logits[i])<1e-5F,"Frozen range rebasing changed opacity");
    // The following gradient check uses the most recent forward's centers.
    quantized=rebased_forward;
    ModelGradients g;
    g.means=Tensor::ones_like(quantized.means);g.log_scales=Tensor::ones_like(quantized.log_scales);
    g.quaternions=Tensor::ones_like(quantized.quaternions);g.opacity_logits=Tensor::ones_like(quantized.opacity_logits);g.sh=Tensor::ones_like(quantized.sh);
    auto incoming=g.sh.to_vector();
    const auto sh_stride=model.sh.shape()[1]*3;
    for(std::size_t i=0;i<model.size();++i)for(std::size_t d=3;d<sh_stride;++d)incoming[i*sh_stride+d]=float(i%7+1);
    g.sh=Tensor::from_vector(incoming,g.sh.shape(),Device::CUDA);
    detail::sog_xt_training_backward(codec,quantized,g,0,0);
    const auto grads=g.sh.to_vector(),raw=codec.raw_centers.to_vector(),lo=codec.sh_low.to_vector(),hi=codec.sh_high.to_vector();
    const auto labels=codec.labels.to_vector_int();
    const auto counts=codec.counts.to_vector_int();
    const unsigned dims=codec.dimensions,stride=dims+3;
    std::vector<double> sums(counts.size()*dims);
    for(std::size_t i=0;i<model.size();++i)for(unsigned d=0;d<dims;++d)sums[std::size_t(labels[i])*dims+d]+=incoming[i*stride+3+d];
    for(std::size_t i=0;i<model.size();++i)for(unsigned d=0;d<dims;++d){const auto c=std::size_t(labels[i])*dims+d;
        const float expected=raw[c]>=lo[d]&&raw[c]<=hi[d]&&hi[d]>lo[d]?float(sums[c]/counts[labels[i]]):0.F;
        require(std::abs(grads[i*stride+3+d]-expected)<1e-5F,"SH centroid STE did not conserve the shared gradient");}
    const auto image=folder/"adapt-frame.png";
    photara::io::save_rgb_png({32,32,std::vector<std::uint8_t>(32*32*3,128)},image);
    photara::mvs::MvsScene scene;
    for(unsigned i=0;i<4;++i){photara::mvs::MvsView view;view.id=i;view.path=i%2?image:folder/"heldout-missing.png";
        view.width=view.src_width=view.height=view.src_height=32;view.fx=view.fy=view.src_fx=view.src_fy=20;
        view.cx=view.cy=view.src_cx=view.src_cy=15.5F;scene.views.push_back(view);}
    TrainingOptions options;options.sog_xt_finetune_iterations=4;options.sog_xt_batch_size=2;
    options.evaluation_split_every=2;options.use_ppisp=options.use_bilateral_grid=true;
    options.use_mvs_depth=options.use_mvs_normals=options.use_normal_field=options.use_depth_normal_loss=false;
    const auto path=folder/"adapted.sog-xt";
    auto result=Trainer(options).finetune_sog_xt(model,scene,path,[](const TrainingProgress& p){
        require(std::isfinite(p.loss),"Non-finite encoding-aware loss");return true;});
    require(result.size()==model.size()&&std::filesystem::is_regular_file(path),"Adaptation failed to export its final codec");
    const auto cancelled=folder/"cancelled.sog-xt";
    Trainer(options).finetune_sog_xt(model,scene,cancelled,[](const TrainingProgress&){return false;});
    require(!std::filesystem::exists(cancelled),"Cancelled codec adaptation wrote an output");
    std::cout<<"SOG-XT training/export parity, shared gradients, holdout and cancellation passed\n";
}

int run_tests(int argc, char** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--vulkan";
        {
            constexpr unsigned n=65,dims=3,budget=23;
            std::vector<float> points(n*dims);
            for(unsigned i=0;i<n;++i)for(unsigned d=0;d<dims;++d)
                points[i*dims+d]=float((i*13+d*7)%29)/29;
            detail::SogPalette p;p.labels.assign(n,0);p.centroids.assign(dims,0);
            double before=0;
            detail::complete_sog_palette(points,n,dims,1,p);
            for(unsigned i=0;i<n;++i)for(unsigned d=0;d<dims;++d)before+=std::pow(double(points[i*dims+d])-p.centroids[d],2);
            if(gpu) {
#if defined(TINYTENSOR_HAS_VULKAN)
                p=detail::cluster_sog_vulkan(points,n,dims,budget);
#endif
            } else detail::refine_sog_palette_cpu(points,n,dims,budget,p);
            require(p.centroids.size()==budget*dims,"Non-power-of-two SOG budget was truncated");
            std::vector<unsigned> counts(budget);double after=0;
            for(unsigned i=0;i<n;++i){require(p.labels[i]<budget,"Invalid global SOG label");++counts[p.labels[i]];
                for(unsigned d=0;d<dims;++d)after+=std::pow(double(points[i*dims+d])-p.centroids[p.labels[i]*dims+d],2);}
            require(std::all_of(counts.begin(),counts.end(),[](auto v){return v>0;}),"Empty requested SOG codeword");
            require(after<=before,"Global SOG refinement increased SSE");
            std::fill(points.begin(),points.end(),.25F);
            if(gpu) {
#if defined(TINYTENSOR_HAS_VULKAN)
                p=detail::cluster_sog_vulkan(points,n,dims,budget);
#endif
            } else detail::refine_sog_palette_cpu(points,n,dims,budget,p);
            std::fill(counts.begin(),counts.end(),0);
            for(auto label:p.labels){require(label<budget,"Invalid constant SOG label");++counts[label];}
            require(p.centroids.size()==budget*dims&&std::all_of(counts.begin(),counts.end(),[](auto v){return v>0;}),"Constant SH collapsed the SOG budget");
        }
        const auto folder = std::filesystem::temp_directory_path() / ("photara-formats-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(folder);
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code e; std::filesystem::remove_all(p, e); } } cleanup{folder};
        if (argc > 1 && std::string(argv[1]) == "--sog-xt-export-perf") {
            auto model = argc > 2 ? load_gaussians(std::filesystem::u8path(argv[2]), tinytensor::Device::CPU)
                                  : fixture(65536, 3);
            bake_gaussian_3d_filter(model);
            std::cout << "SOG-XT export benchmark count=" << model.size() << " degree=" << model.sh_degree << '\n';
            // Warm GPU initialization/pipelines before timing repeated exports.
            SogXtExportOptions warmup;
            save_sog_xt(model, folder / "warmup.sog-xt", warmup);
            for (unsigned repeat = 0; repeat < 3; ++repeat)
            {
                SogXtExportOptions options;
                const auto output = folder / "export.sog-xt";
                const auto start = std::chrono::steady_clock::now();
                save_sog_xt(model, output, options);
                std::cout << "repeat=" << repeat << " elapsed_ms="
                          << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()
                          << " bytes=" << std::filesystem::file_size(output) << '\n';
            }
            return 0;
        }
        if(argc>1&&std::string(argv[1])=="--sog-xt-training-only"){sog_xt_training_test(folder);return 0;}
        if(argc==4&&std::string(argv[1])=="--decode-sog-xt") {
            save_gaussians_ply(load_gaussians(std::filesystem::u8path(argv[2]),tinytensor::Device::CPU),std::filesystem::u8path(argv[3]));return 0;
        }
        if (argc > 3 && std::string(argv[1]) == "--sog-xt-compare") {
            const auto actual = load_gaussians(std::filesystem::u8path(argv[2]), tinytensor::Device::CPU);
            const auto expected = load_gaussians(std::filesystem::u8path(argv[3]), tinytensor::Device::CPU);
            require(actual.size() == expected.size() && actual.sh_degree == expected.sh_degree, "External container shape mismatch");
            const tinytensor::Tensor* a[] = {&actual.means, &actual.log_scales, &actual.quaternions, &actual.opacity_logits, &actual.sh};
            const tinytensor::Tensor* b[] = {&expected.means, &expected.log_scales, &expected.quaternions, &expected.opacity_logits, &expected.sh};
            for (unsigned t = 0; t < 5; ++t) {
                const auto av = a[t]->to_vector(), bv = b[t]->to_vector();
                for (std::size_t i = 0; i < av.size(); ++i)
                    require(std::abs(av[i] - bv[i]) <= 2e-5F * (1 + std::abs(bv[i])), "External YAML decoded attributes differ");
            }
            std::cout << "External YAML comparison passed n=" << actual.size() << '\n';
            return 0;
        }
        if (argc > 2 && std::string(argv[1]) == "--sog-xt-benchmark") {
            sog_xt_benchmark(argc > 3 ? std::filesystem::u8path(argv[3]) : folder, std::filesystem::u8path(argv[2]));
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--ply-load-benchmark") {
            ply_load_benchmark(folder);
            return 0;
        }
        if (gpu) {
#if defined(TINYTENSOR_HAS_VULKAN)
            sog_xt(folder, 2049);
            gpu_clustering(2049, 64);
            gpu_clustering(262145, 1024); // GPU upload-batch boundary and degree-3 SH.
            gpu_clustering_quality();
            for (unsigned n : {65536U, 65537U, 131072U, 262145U})
                roundtrip(folder, n, 1, GaussianFormat::sog);
            streamed_palette(folder);
#else
            throw std::runtime_error("Vulkan test requires Vulkan build");
#endif
        } else {
            roundtrip(folder, 8193, 3, GaussianFormat::ply);
            sog_xt_parallel_parity();
            sog_xt_covariance_symmetry();
            sog_xt(folder);
            sog_xt(folder, 1);
            ply_loading(folder);
            antialias_metadata(folder);
            gut_exports(folder);
            scalar_quantization();
            for (unsigned d = 0; d <= 3; ++d) {
                roundtrip(folder, 17, d, GaussianFormat::ply);
                roundtrip(folder, 17, d, GaussianFormat::sog);
                roundtrip(folder, 17, d, GaussianFormat::glb);
                roundtrip(folder, 17, d, GaussianFormat::spz);
            }
        }
        std::cout << "Splat format tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> storage;
    for (int i = 0; i < argc; ++i) {
        const auto utf8 = std::filesystem::path(argv[i]).u8string();
        storage.emplace_back(reinterpret_cast<const char*>(utf8.data()), utf8.size());
    }
    std::vector<char*> args;
    for (auto& arg : storage) args.push_back(arg.data());
    return run_tests(argc, args.data());
}
#else
int main(int argc, char** argv) { return run_tests(argc, argv); }
#endif
