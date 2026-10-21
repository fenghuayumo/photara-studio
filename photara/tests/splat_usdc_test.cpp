#include "splat/formats.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace photara::splat;
void require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
GaussianModel fixture(unsigned degree) {
    GaussianModel m;
    m.sh_degree = degree;
    const std::size_t bases = (degree + 1) * (degree + 1);
    constexpr auto cpu = tinytensor::Device::CPU;
    m.means = tinytensor::Tensor::from_vector(std::vector<float>{1, 2, 3, -2, .5F, 4, 0, 1, -1},
                                              {3, 3}, cpu);
    m.log_scales = tinytensor::Tensor::from_vector({std::log(.1F), std::log(.2F), std::log(.3F),
                                                    std::log(.4F), std::log(.5F), std::log(.6F),
                                                    std::log(.7F), std::log(.8F), std::log(.9F)},
                                                   {3, 3}, cpu);
    m.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{2, 0, 0, 0, 1, 0, 0, 1, .5F, .5F, .5F, .5F}, {3, 4}, cpu);
    m.opacity_logits = tinytensor::Tensor::from_vector(std::vector<float>{-2, 0, 2}, {3, 1}, cpu);
    std::vector<float> sh(3 * bases * 3);
    for (std::size_t i = 0; i < sh.size(); ++i)
        sh[i] = (int(i % 19) - 9) * .037F;
    m.sh = tinytensor::Tensor::from_vector(sh, {3, bases, 3}, cpu);
    return m;
}
void near(const std::vector<float> &a, const std::vector<float> &b, float tolerance) {
    require(a.size() == b.size(), "USDC array size changed");
    for (std::size_t i = 0; i < a.size(); ++i)
        require(std::abs(a[i] - b[i]) <= tolerance, "USDC round trip changed data");
}
void verify_fixture(const GaussianModel& restored) {
    const auto source = fixture(3);
    near(source.means.to_vector(), restored.means.to_vector(), 1e-6F);
    near(source.log_scales.to_vector(), restored.log_scales.to_vector(), 1e-6F);
    near(source.sh.to_vector(), restored.sh.to_vector(), 1e-6F);
    near(restored.quaternions.to_vector(),
         {1, 0, 0, 0, std::sqrt(.5F), 0, 0, std::sqrt(.5F), .5F, .5F, .5F, .5F}, 1e-6F);
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write") {
            save_gaussians(fixture(3), std::filesystem::u8path(argv[2]), GaussianFormat::usdc);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--read-fixture") {
            verify_fixture(load_gaussians(std::filesystem::u8path(argv[2]), tinytensor::Device::CPU));
            return 0;
        }
        auto folder = std::filesystem::temp_directory_path() /
                      ("photara-usdc-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(folder);
        struct Cleanup {
            std::filesystem::path p;
            ~Cleanup() {
                std::error_code ec;
                std::filesystem::remove_all(p, ec);
            }
        } cleanup{folder};
        require(parse_gaussian_format("USDC") == GaussianFormat::usdc, "USDC parser");
        require(gaussian_format_from_path("model.USDC") == GaussianFormat::usdc,
                "USDC extension detection");
        for (unsigned degree = 0; degree <= 3; ++degree) {
            auto source = fixture(degree);
            auto file = folder / ("model" + std::to_string(degree) + ".usdc");
            save_gaussians(source, file);
            std::ifstream input(file, std::ios::binary);
            char header[8];
            input.read(header, 8);
            require(std::string(header, 8) == "PXR-USDC", "USDC file is not binary crate");
            auto restored = load_gaussians(file, tinytensor::Device::CPU);
            require(restored.size() == source.size() && restored.sh_degree == degree,
                    "USDC shape or SH degree changed");
            near(source.means.to_vector(), restored.means.to_vector(), 1e-6F);
            near(source.log_scales.to_vector(), restored.log_scales.to_vector(), 1e-6F);
            near(source.opacity_logits.to_vector(), restored.opacity_logits.to_vector(), 1e-5F);
            near(source.sh.to_vector(), restored.sh.to_vector(), 1e-6F);
            auto q = restored.quaternions.to_vector();
            near(q, {1, 0, 0, 0, std::sqrt(.5F), 0, 0, std::sqrt(.5F), .5F, .5F, .5F, .5F}, 1e-6F);
        }
        auto bad = fixture(0);
        bad.quaternions = tinytensor::Tensor::zeros({3, 4}, tinytensor::Device::CPU);
        bool rejected = false;
        try {
            save_gaussians(bad, folder / "invalid.usdc");
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected && !std::filesystem::exists(folder / "invalid.usdc"),
                "Invalid USDC model published");
        std::cout << "Native USDC tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
