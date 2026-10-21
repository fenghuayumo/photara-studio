#include "splat/formats.hpp"
#include "splat/deployment.hpp"
#include "../src/splat/sog_quantization.hpp"
#include "../src/splat/sog_vulkan.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

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
    save_gaussians(m, file, format);
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
int main(int argc, char** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--vulkan";
        const auto folder = std::filesystem::temp_directory_path() / ("photara-formats-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(folder);
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code e; std::filesystem::remove_all(p, e); } } cleanup{folder};
        if (gpu) {
#if defined(TINYTENSOR_HAS_VULKAN)
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
            scalar_quantization();
            for (unsigned d = 0; d <= 3; ++d) {
                roundtrip(folder, 17, d, GaussianFormat::sog);
                roundtrip(folder, 17, d, GaussianFormat::glb);
                roundtrip(folder, 17, d, GaussianFormat::spz);
            }
        }
        std::cout << "Splat format tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
