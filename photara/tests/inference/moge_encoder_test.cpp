// Manual same-checkpoint parity tool. Inputs are prepared by the Python gate.
#include "models/moge/encoder.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace photara::inference;
namespace {
void save(const std::filesystem::path& directory, const std::string& name,
    const std::vector<float>& values) {
    std::ofstream output(directory / (name + ".bin"), std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!output) throw std::runtime_error("Cannot write encoder stage " + name);
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 6) {
            std::cerr << "Usage: moge_encoder_test model.gguf rows cols normalized-chw.bin output-dir [--vulkan]\n";
            return 1;
        }
        const auto rows = std::stoul(argv[2]), columns = std::stoul(argv[3]);
        if (rows == 0 || columns == 0 || rows > 4096 || columns > 4096 || rows * columns > 4096)
            throw std::invalid_argument("Invalid encoder token grid");
        std::vector<float> image(rows * columns * 3 * 14 * 14);
        std::ifstream input(argv[4], std::ios::binary);
        input.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(float));
        if (!input || input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Normalized image size mismatch");
        const std::filesystem::path output_directory(argv[5]);
        std::filesystem::create_directories(output_directory);
        RuntimeOptions options;
        options.backend = argc > 6 && std::string(argv[6]) == "--vulkan" ? Backend::vulkan : Backend::cpu;
        options.allow_cpu_fallback = false;
        options.cpu_threads = 8;
        detail::GgmlRuntime runtime(options);
        detail::MogeWeights weights(argv[1], runtime.backend());
        auto result = detail::encode_moge_vitb(weights, runtime,
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns), image, true);
        for (const auto& stage : result.stages) save(output_directory, stage.name, stage.values);
        save(output_directory, "features", result.features);
        save(output_directory, "class_token", result.class_token);
        // The allocator is reused with a new graph/context. Verify the normal
        // non-diagnostic path does not depend on retaining intermediate outputs.
        auto repeated = detail::encode_moge_vitb(weights, runtime,
            static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(columns), image, false);
        if (repeated.features != result.features || repeated.class_token != result.class_token)
            throw std::runtime_error("Repeated encoder execution differs from diagnostic execution");
        std::cout << "Encoder " << rows << 'x' << columns << " patches on "
            << backend_name(runtime.device().backend) << ", " << result.stages.size()
            << " stages, compute buffers " << runtime.compute_buffer_bytes() << " bytes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
