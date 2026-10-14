#include "inference/models/moge.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace photara::inference;
namespace {
std::vector<float> load(const std::filesystem::path& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!input || input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Invalid raw output file");
    return values;
}
template<class T> void save(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!output) throw std::runtime_error("Cannot write geometry output");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 11) {
            std::cerr << "Usage: moge_postprocess_compare raw-dir raw-width raw-height width height output-dir [fx fy cx cy]\n";
            return 1;
        }
        const std::filesystem::path directory(argv[1]), output(argv[6]);
        MogeRawGeometry raw;
        raw.width = static_cast<std::uint32_t>(std::stoul(argv[2]));
        raw.height = static_cast<std::uint32_t>(std::stoul(argv[3]));
        const auto count = static_cast<std::size_t>(raw.width) * raw.height;
        raw.points = load(directory / "points.bin", count * 3);
        raw.normals = load(directory / "normals.bin", count * 3);
        raw.mask_logits = load(directory / "mask_logits.bin", count);
        raw.log_metric_scale = load(directory / "log_metric_scale.bin", 1)[0];
        MogeGeometryOptions options;
        options.width = static_cast<std::uint32_t>(std::stoul(argv[4]));
        options.height = static_cast<std::uint32_t>(std::stoul(argv[5]));
        if (argc == 11) options.intrinsics = PinholeIntrinsics{
            std::stof(argv[7]), std::stof(argv[8]), std::stof(argv[9]), std::stof(argv[10])};
        auto result = recover_moge_geometry(raw, options, {"reference-fixture", "test", {}}, {});
        result.prior.validate();
        std::filesystem::create_directories(output);
        save(output / "depth.bin", result.prior.depth);
        save(output / "points.bin", result.prior.points);
        save(output / "normals.bin", result.prior.normals);
        save(output / "valid.bin", result.prior.valid);
        std::ofstream report(output / "recovery.txt");
        report << std::setprecision(17) << static_cast<int>(result.recovery.status) << ' '
            << result.recovery.z_shift << ' ' << result.recovery.normalized_focal << ' '
            << result.recovery.reprojection_rms << ' ' << result.recovery.sample_count;
        if (result.prior.intrinsics) {
            const auto& k = *result.prior.intrinsics;
            report << ' ' << k.fx << ' ' << k.fy << ' ' << k.cx << ' ' << k.cy;
        }
        std::cout << "MoGe geometry recovery status " << static_cast<int>(result.recovery.status)
            << ", shift " << result.recovery.z_shift << ", samples " << result.recovery.sample_count << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
