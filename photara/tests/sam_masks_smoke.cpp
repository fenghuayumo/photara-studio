#include "sam/masks.hpp"
#include "sam/model_cache.hpp"
#include "io/image.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

// Manual real-checkpoint gate. No downloads or licence acceptance.
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: sam_masks_smoke model image output_dir [language|sam3]\n";
        return 2;
    }
    try {
        photara::sam::GenerateOptions options;
        options.model = argv[1];
        options.images = {argv[2]};
        options.output_dir = argv[3];
        options.engine = argc > 4 ? argv[4] : "language";
        options.backend = "vulkan";
        options.text = "cat";
        options.video = false;
        options.require_target = true;
        std::size_t callbacks = 0;
        options.progress = [&](std::size_t n, std::size_t total) {
            if (n != 1 || total != 1) throw std::runtime_error("Invalid progress");
            ++callbacks;
        };
        const auto result = photara::sam::generate_masks(options);
        const auto mask = photara::io::load_gray(options.output_dir /
            (options.images.front().stem().string() + ".png"));
        const auto size = photara::io::load_image_size(options.images.front());
        const auto foreground = std::count(mask.pixels.begin(), mask.pixels.end(), 255);
        if (result.written != 1 || callbacks != 1 || mask.width != size.width ||
            mask.height != size.height || foreground == 0 || foreground == mask.pixels.size())
            throw std::runtime_error("Missing, mis-sized, empty or full mask");
        std::cout << "foreground_fraction=" << double(foreground) / mask.pixels.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
