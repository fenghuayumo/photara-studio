#pragma once

#include "mvs/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aetherscan::splat {

enum class DatasetFormat {
    auto_detect,
    colmap,
    reality_capture,
    openmvs,
};

[[nodiscard]] DatasetFormat parse_dataset_format(std::string_view value);
[[nodiscard]] std::string_view dataset_format_name(DatasetFormat format) noexcept;

struct DatasetLoadRequest {
    // A directory for COLMAP/RealityCapture, or a .csv/.mvs file.
    std::filesystem::path source;
    // Image root. Empty uses conventional paths below source.
    std::filesystem::path image_directory;
    // Optional PLY replacing the source sparse cloud.
    std::filesystem::path initial_point_cloud;
    DatasetFormat format{DatasetFormat::auto_detect};
    // Used only when camera metadata has no initial points.
    std::size_t random_initial_point_count{10'000};
    std::uint32_t seed{42};
};

struct DatasetLoadResult {
    mvs::MvsScene scene;
    DatasetFormat format{DatasetFormat::auto_detect};
    std::filesystem::path resolved_source;
    std::filesystem::path initial_point_cloud;
    bool generated_initial_points{false};
    bool initial_points_dense{false};
    std::vector<std::string> warnings;
};

// Format adapters are intentionally independent of the trainer. Additional
// camera/dataset formats can be registered without changing GGGS code.
class DatasetReader {
public:
    virtual ~DatasetReader() = default;

    [[nodiscard]] virtual DatasetFormat format() const noexcept = 0;
    [[nodiscard]] virtual bool probe(
        const DatasetLoadRequest& request) const = 0;
    [[nodiscard]] virtual DatasetLoadResult load(
        const DatasetLoadRequest& request) const = 0;
};

class DatasetLoader {
public:
    DatasetLoader() = default;
    DatasetLoader(DatasetLoader&&) noexcept = default;
    DatasetLoader& operator=(DatasetLoader&&) noexcept = default;

    DatasetLoader(const DatasetLoader&) = delete;
    DatasetLoader& operator=(const DatasetLoader&) = delete;

    void register_reader(std::unique_ptr<DatasetReader> reader);
    [[nodiscard]] DatasetLoadResult load(
        const DatasetLoadRequest& request) const;

private:
    std::vector<std::unique_ptr<DatasetReader>> readers_;
};

[[nodiscard]] DatasetLoader make_default_dataset_loader();
[[nodiscard]] DatasetLoadResult load_splat_dataset(
    const DatasetLoadRequest& request);

}  // namespace aetherscan::splat
