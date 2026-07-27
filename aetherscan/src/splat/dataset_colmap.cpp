#include "dataset_internal.hpp"

#include "splat/colmap.hpp"

#include <utility>

namespace aetherscan::splat::dataset_detail {
namespace {

class ColmapReader final : public DatasetReader {
public:
    [[nodiscard]] DatasetFormat format() const noexcept override {
        return DatasetFormat::colmap;
    }

    [[nodiscard]] bool probe(
        const DatasetLoadRequest& request) const override {
        return has_colmap_model(request.source);
    }

    [[nodiscard]] DatasetLoadResult load(
        const DatasetLoadRequest& request) const override {
        std::filesystem::path images = request.image_directory;
        if (images.empty()) {
            auto root = source_directory(request.source);
            for (unsigned level = 0; level < 4 && !root.empty(); ++level) {
                if (std::filesystem::is_directory(root / "images")) {
                    images = root / "images";
                    break;
                }
                root = root.parent_path();
            }
        }
        auto colmap = load_colmap_scene(
            request.source, images);
        DatasetLoadResult result;
        result.scene = std::move(colmap.scene);
        result.format = format();
        result.resolved_source = std::move(colmap.model_directory);
        return result;
    }
};

}  // namespace

std::unique_ptr<DatasetReader> make_colmap_reader() {
    return std::make_unique<ColmapReader>();
}

}  // namespace aetherscan::splat::dataset_detail
