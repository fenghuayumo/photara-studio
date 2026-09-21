#include "features/extractor.hpp"

#include "io/image.hpp"

#include <stdexcept>

namespace photara::features {

FeatureSet FeatureExtractor::extract_file(const std::filesystem::path& path) const {
    if (info().accepts_gray) {
        const io::GrayImage image = io::load_gray(path);
        return extract_gray(image.pixels, image.width, image.height);
    }
    if (info().accepts_rgb) {
        const io::RgbImage image = io::load_rgb(path);
        return extract_rgb(image.pixels, image.width, image.height);
    }
    throw std::runtime_error(
        std::string(name()) + " does not accept gray or RGB image input");
}

FeatureSet FeatureExtractor::extract_rgb(
    std::span<const std::uint8_t> /*pixels*/, std::uint32_t /*width*/,
    std::uint32_t /*height*/, std::size_t /*row_stride*/) const {
    throw std::runtime_error(std::string(name()) + " does not implement extract_rgb");
}

}  // namespace photara::features
