#include "inference/models/grounding_dino.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

photara::inference::GroundingDinoImage read_ppm(
    const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    std::string magic;
    int width = 0;
    int height = 0;
    int maximum = 0;
    input >> magic >> width >> height >> maximum;
    input.get();
    if (magic != "P6" || width <= 0 || height <= 0 || maximum != 255)
        throw std::runtime_error("The smoke test expects an 8-bit binary PPM");
    photara::inference::GroundingDinoImage image;
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    image.rgb.resize(static_cast<std::size_t>(width) * height * 3U);
    input.read(reinterpret_cast<char*>(image.rgb.data()),
        static_cast<std::streamsize>(image.rgb.size()));
    if (!input) throw std::runtime_error("Truncated PPM");
    return image;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: grounding_dino_smoke weights vocab image.ppm\n";
        return 2;
    }
    try {
        photara::inference::GroundingDinoOptions options;
        options.runtime.backend = argc > 4 && std::string(argv[4]) == "cpu"
            ? photara::inference::Backend::cpu : photara::inference::Backend::vulkan;
        options.runtime.allow_cpu_fallback = false;
        options.runtime.cpu_threads = 4;
        options.box_threshold = 0.25F;
        photara::inference::GroundingDinoDetector detector(
            argv[1], argv[2], options);
        const auto detections = detector.detect(
            read_ppm(argv[3]), {"a cat"}, {0.25F, 0.20F, 0.10F});
        std::cout << "device=" << detector.device_info().name
                  << " detections=" << detections.size() << '\n';
        if (detections.empty()) throw std::runtime_error("No cat detected in smoke-test image");
        for (const auto& detection : detections)
            std::cout << detection.score << ' ' << detection.x0 << ' '
                      << detection.y0 << ' ' << detection.x1 << ' '
                      << detection.y1 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
