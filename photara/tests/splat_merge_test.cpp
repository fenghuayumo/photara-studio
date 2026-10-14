#include "splat/formats.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

using namespace photara::splat;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

GaussianModel fixture(
    const std::size_t count, const unsigned degree,
    const std::vector<float>& means) {
    constexpr auto cpu = tinytensor::Device::CPU;
    GaussianModel model;
    model.sh_degree = degree;
    const std::size_t bases = (degree + 1U) * (degree + 1U);
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, cpu);
    model.log_scales =
        tinytensor::Tensor::from_vector(std::vector<float>(count * 3U, -1.F), {count, 3U}, cpu);
    std::vector<float> rotations(count * 4U, 0.F);
    for (std::size_t index = 0; index < count; ++index) rotations[index * 4U] = 1.F;
    model.quaternions = tinytensor::Tensor::from_vector(rotations, {count, 4U}, cpu);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>(count, -2.F), {count, 1U}, cpu);
    std::vector<float> sh(count * bases * 3U);
    for (std::size_t index = 0; index < sh.size(); ++index)
        sh[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03F;
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, cpu);
    return model;
}

void near(
    const std::vector<float>& actual, const std::vector<float>& expected,
    const float tolerance, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index)
        require(std::abs(actual[index] - expected[index]) <= tolerance, message);
}

void test_transform_and_merge() {
    // Degree-0 model (2 gaussians) merged after a scaled, translated
    // degree-1 model (3 gaussians).
    GaussianModel first = fixture(2, 0, {0.F, 0.F, 0.F, 1.F, 2.F, 3.F});
    const std::vector<float> second_means{
        5.F, 6.F, 7.F, -1.F, -2.F, -3.F, 0.5F, 1.5F, 2.5F};
    GaussianModel second = fixture(3, 1, second_means);
    const std::vector<float> filter{0.25F, 0.5F, 1.F};
    constexpr auto cpu = tinytensor::Device::CPU;
    second.filter_3d =
        tinytensor::Tensor::from_vector(filter, {3U, 1U}, cpu);

    GaussianTransform transform;
    transform.translation = {10.F, -2.F, 4.F};
    transform.scale = 2.F;
    const GaussianModel placed = transform_gaussians(second, transform);

    near(
        placed.means.to_vector(),
        {20.F, 10.F, 18.F, 8.F, -6.F, -2.F, 11.F, 1.F, 9.F}, 1e-5F,
        "Transformed means are wrong");
    const auto placed_scales = placed.log_scales.to_vector();
    for (const float value : placed_scales)
        require(std::abs(value - (-1.F + std::log(2.F))) <= 1e-5F,
                "Transformed log scales are wrong");
    near(
        placed.filter_3d.to_vector(), {0.5F, 1.F, 2.F}, 1e-5F,
        "Transformed filter radii are wrong");
    require(transform_gaussians(second, GaussianTransform{}).means.to_vector() ==
                second.means.to_vector(),
            "Identity transform copied instead of returning the model");

    const GaussianModel merged = merge_gaussians({first, placed});
    require(merged.size() == 5, "Merged Gaussian count is wrong");
    require(merged.sh_degree == 1, "Merged SH degree is wrong");
    require(merged.sh.shape()[1] == 4, "Merged SH was not padded to 4 bases");
    const auto merged_means = merged.means.to_vector();
    const auto merged_opacities = merged.opacity_logits.to_vector();
    const auto merged_sh = merged.sh.to_vector();
    require(merged_means[0] == 0.F && merged_means[3] == 1.F,
            "Merged means were reordered");
    require(std::abs(merged_means[6] - 20.F) <= 1e-4F,
            "Merged means were reordered");
    require(std::abs(merged_opacities[0] + 2.F) <= 1e-6F &&
                std::abs(merged_opacities[4] + 2.F) <= 1e-6F,
            "Merged opacities were reordered");
    // The degree-0 model keeps its DC colour in the padded layout while its
    // higher bands stay zero; the degree-1 rows follow unchanged.
    const auto first_sh = first.sh.to_vector();
    const auto placed_sh = placed.sh.to_vector();
    for (std::size_t gaussian = 0; gaussian < 2; ++gaussian) {
        for (std::size_t component = 0; component < 3U; ++component)
            require(std::abs(merged_sh[(gaussian * 4U) * 3U + component] -
                             first_sh[gaussian * 3U + component]) <= 1e-6F,
                    "Merged DC colour changed");
        for (std::size_t base = 1; base < 4U; ++base)
            for (std::size_t component = 0; component < 3U; ++component)
                require(merged_sh[(gaussian * 4U + base) * 3U + component] ==
                            0.F,
                        "Merged higher SH bands were not zero-padded");
    }
    near(placed_sh,
         std::vector<float>(
             merged_sh.begin() + static_cast<std::ptrdiff_t>(2 * 4U * 3U),
             merged_sh.end()),
         0.F, "Merged higher-degree rows changed");
    // Only one input carried filter_3d, so the merged model must drop it.
    require(!merged.filter_3d.is_valid(),
            "Merged model kept a partial filter field");

    bool rejected = false;
    try { (void)merge_gaussians({}); } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Merging an empty list was accepted");
    rejected = false;
    GaussianTransform bad;
    bad.scale = 0.F;
    try { (void)transform_gaussians(second, bad); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "A non-positive transform scale was accepted");
}

void test_merge_optional_fields_and_roundtrip() {
    constexpr auto cpu = tinytensor::Device::CPU;
    GaussianModel first = fixture(2, 1, {0.F, 0.F, 0.F, 1.F, 1.F, 1.F});
    GaussianModel second = fixture(3, 2, {2.F, 0.F, 0.F, 3.F, 0.F, 0.F, 4.F, 0.F, 0.F});
    first.normal_features = tinytensor::Tensor::from_vector(
        std::vector<float>(2 * 4U, 0.5F), {2U, 4U}, cpu);
    second.normal_features = tinytensor::Tensor::from_vector(
        std::vector<float>(3 * 4U, 0.25F), {3U, 4U}, cpu);
    first.filter_3d = tinytensor::Tensor::from_vector(
        std::vector<float>{0.1F, 0.2F}, {2U, 1U}, cpu);
    second.filter_3d = tinytensor::Tensor::from_vector(
        std::vector<float>(3, 0.3F), {3U, 1U}, cpu);

    const GaussianModel merged = merge_gaussians({first, second});
    require(merged.size() == 5, "Merged count with optional fields is wrong");
    require(merged.sh_degree == 2, "Merged degree with SH2 is wrong");
    require(merged.sh.shape()[1] == 9, "Merged bases were not padded to 9");
    require(merged.normal_features.is_valid() &&
                merged.normal_features.numel() == 5 * 4U,
            "Merged normal features were dropped unexpectedly");
    require(merged.filter_3d.is_valid() && merged.filter_3d.numel() == 5,
            "Merged filter radii were dropped unexpectedly");
    const auto normals = merged.normal_features.to_vector();
    require(normals[0] == 0.5F && normals[2 * 4U] == 0.25F,
            "Merged normal features were reordered");

    const auto folder = std::filesystem::temp_directory_path() /
                        ("photara-merge-" +
                         std::to_string(
                             std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count()));
    std::filesystem::create_directories(folder);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{folder};
    const auto file = folder / "merged.ply";
    save_gaussians(merged, file);
    const GaussianModel loaded = load_gaussians(file, GaussianFormat::ply, cpu);
    require(loaded.size() == 5, "Merged PLY round trip changed the count");
    require(loaded.sh_degree == 2, "Merged PLY round trip changed the degree");
    near(loaded.means.to_vector(), merged.means.to_vector(), 1e-5F,
         "Merged PLY round trip changed means");
}

// Two visually distinct synthetic blobs for manual editor testing:
// `photara_splat_merge_test --write <folder>` dumps them as PLY files.
GaussianModel demo_model(
    const std::size_t count, const unsigned degree,
    const std::array<float, 3>& centre, const float radius,
    const std::array<float, 3>& colour) {
    constexpr auto cpu = tinytensor::Device::CPU;
    constexpr float c0 = 0.282094791773878F;
    const std::size_t bases = (degree + 1U) * (degree + 1U);
    std::vector<float> means, scales, rotations, opacities, sh;
    means.reserve(count * 3U);
    scales.reserve(count * 3U);
    rotations.reserve(count * 4U);
    opacities.reserve(count);
    sh.reserve(count * bases * 3U);
    // Golden-angle spiral: an even, deterministic shell of centres.
    const float golden = 3.14159265358979F * (3.F - std::sqrt(5.F));
    for (std::size_t index = 0; index < count; ++index) {
        const float y = 1.F - 2.F * (static_cast<float>(index) + 0.5F) /
                             static_cast<float>(count);
        const float r = std::sqrt(std::max(0.F, 1.F - y * y));
        const float theta = golden * static_cast<float>(index);
        means.push_back(centre[0] + radius * r * std::cos(theta));
        means.push_back(centre[1] + radius * y);
        means.push_back(centre[2] + radius * r * std::sin(theta));
        scales.insert(scales.end(), 3, std::log(0.035F));
        rotations.insert(rotations.end(), {1.F, 0.F, 0.F, 0.F});
        opacities.push_back(2.5F);
        for (std::size_t base = 0; base < bases; ++base)
            for (std::size_t component = 0; component < 3U; ++component) {
                if (base == 0) {
                    sh.push_back((colour[component] - 0.5F) / c0);
                } else {
                    // Small deterministic higher-band detail.
                    sh.push_back(
                        0.6F * std::sin(
                                   static_cast<float>(index % 31) +
                                   static_cast<float>(base * 3U + component)));
                }
            }
    }
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, cpu);
    model.log_scales =
        tinytensor::Tensor::from_vector(scales, {count, 3U}, cpu);
    model.quaternions =
        tinytensor::Tensor::from_vector(rotations, {count, 4U}, cpu);
    model.opacity_logits =
        tinytensor::Tensor::from_vector(opacities, {count, 1U}, cpu);
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, cpu);
    model.sh_degree = degree;
    return model;
}

void write_demo_models(const std::filesystem::path& folder) {
    std::filesystem::create_directories(folder);
    // Red SH0 blob at the origin and a teal SH1 blob beside it, so the two
    // models stay recognizable after import, placement, and merge.
    save_gaussians(
        demo_model(3000, 0, {0.F, 0.F, 0.F}, 0.8F, {0.92F, 0.16F, 0.10F}),
        folder / "demo_red_sh0.ply", GaussianFormat::ply);
    save_gaussians(
        demo_model(3000, 1, {2.2F, 0.3F, 0.F}, 0.5F, {0.10F, 0.62F, 0.92F}),
        folder / "demo_teal_sh1.ply", GaussianFormat::ply);
}

int main(const int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write") {
            const auto folder = std::filesystem::path(argv[2]);
            write_demo_models(folder);
            std::cout << "Demo models written to "
                      << std::filesystem::absolute(folder).string() << '\n';
            return 0;
        }
        test_transform_and_merge();
        test_merge_optional_fields_and_roundtrip();
        std::cout << "Gaussian transform/merge tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
