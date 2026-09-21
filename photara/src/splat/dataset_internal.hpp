#pragma once

#include "splat/dataset.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace photara::splat::dataset_detail {

inline std::string lower_ascii(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

inline std::filesystem::path source_directory(
    const std::filesystem::path& source) {
    return std::filesystem::is_directory(source)
        ? source
        : source.parent_path();
}

inline bool has_colmap_model(const std::filesystem::path& path) {
    const auto has_files = [](const std::filesystem::path& directory) {
        return (std::filesystem::is_regular_file(directory / "cameras.bin") &&
                std::filesystem::is_regular_file(directory / "images.bin") &&
                std::filesystem::is_regular_file(directory / "points3D.bin")) ||
               (std::filesystem::is_regular_file(directory / "cameras.txt") &&
                std::filesystem::is_regular_file(directory / "images.txt") &&
                std::filesystem::is_regular_file(directory / "points3D.txt"));
    };
    return has_files(path) || has_files(path / "sparse" / "0") ||
           has_files(path / "sparse") || has_files(path / "0");
}

inline std::filesystem::path resolve_image(
    const std::filesystem::path& name,
    const DatasetLoadRequest& request,
    const std::filesystem::path& source_root) {
    std::vector<std::filesystem::path> candidates;
    if (name.is_absolute()) candidates.push_back(name);
    if (!request.image_directory.empty()) {
        candidates.push_back(request.image_directory / name);
        candidates.push_back(request.image_directory / name.filename());
    }
    candidates.push_back(source_root / name);
    candidates.push_back(source_root / "images" / name);
    candidates.push_back(source_root / "Images" / name);
    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate)) return candidate;

    const auto search_root = !request.image_directory.empty()
        ? request.image_directory
        : source_root;
    std::error_code error;
    if (std::filesystem::is_directory(search_root, error)) {
        for (std::filesystem::recursive_directory_iterator iterator(
                 search_root,
                 std::filesystem::directory_options::skip_permission_denied,
                 error),
             end;
             iterator != end && !error; iterator.increment(error)) {
            if (iterator->is_regular_file(error) &&
                iterator->path().filename() == name.filename())
                return iterator->path();
        }
    }
    throw std::runtime_error(
        "Dataset image does not exist: " + name.string());
}

[[nodiscard]] std::unique_ptr<DatasetReader> make_colmap_reader();
[[nodiscard]] std::unique_ptr<DatasetReader> make_reality_capture_reader();
[[nodiscard]] std::unique_ptr<DatasetReader> make_openmvs_reader();

}  // namespace photara::splat::dataset_detail
