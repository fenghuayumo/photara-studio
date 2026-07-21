#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace aetherscan::io {

struct GrayImage {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;  // row-major, top-left origin

    [[nodiscard]] std::size_t row_stride() const noexcept {
        return static_cast<std::size_t>(width);
    }
};

// Loads any FreeImage-supported format as 8-bit grayscale, top-left origin.
GrayImage load_gray(const std::filesystem::path& path);

// Returns an empty image when the source has no alpha channel/transparency.
GrayImage load_alpha(const std::filesystem::path& path);

struct RgbImage {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;  // row-major RGB, 3 bytes/pixel
};

RgbImage load_rgb(const std::filesystem::path& path);

// Writes 8-bit RGB PNG (top-left origin).
void save_rgb_png(const RgbImage& image, const std::filesystem::path& path);

// Bilinear resize of grayscale or interleaved RGB (channels = 1 or 3).
void resize_bilinear(
    const std::uint8_t* source, std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t channels, std::uint8_t* destination, std::uint32_t destination_width,
    std::uint32_t destination_height);

}  // namespace aetherscan::io
