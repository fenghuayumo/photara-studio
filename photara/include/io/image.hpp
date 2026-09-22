#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <string>

namespace photara::io {

struct ImageSize {
    std::uint32_t width{};
    std::uint32_t height{};
};

// Reads image dimensions without converting or copying its pixel buffer.
ImageSize load_image_size(const std::filesystem::path& path);
// Probes whether the file carries an alpha channel without decoding pixels.
[[nodiscard]] bool image_has_alpha(const std::filesystem::path& path);
// Optional EXIF lens description; empty when absent or unreadable.
std::string load_lens_description(const std::filesystem::path& path);
// Stable physical-camera identity used to avoid sharing intrinsics between
// different bodies, lenses, or zoom settings. Empty when EXIF is unavailable.
std::string load_camera_identity(
	const std::filesystem::path& path,
	double* focal_length_mm = nullptr,
	std::uint32_t image_width = 0,
	std::uint32_t image_height = 0,
	double* focal_prior_px = nullptr);

struct GrayImage {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;  // row-major, top-left origin

    [[nodiscard]] std::size_t row_stride() const noexcept {
        return static_cast<std::size_t>(width);
    }
};

// Loads an image as 8-bit grayscale, top-left origin. JPEG/PNG use direct
// codec paths; other supported formats fall back to FreeImage.
GrayImage load_gray(const std::filesystem::path& path);

// Returns an empty image when the source has no alpha channel/transparency.
GrayImage load_alpha(const std::filesystem::path& path);

struct RgbImage {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;  // row-major RGB, 3 bytes/pixel
};

RgbImage load_rgb(const std::filesystem::path& path);
// Decodes JPEG with the largest libjpeg DCT scale (1/2, 1/4, or 1/8) whose
// output still covers minimum_width x minimum_height. Other formats use the
// ordinary full-resolution RGB decoder.
[[nodiscard]] RgbImage load_rgb_with_minimum_size(
    const std::filesystem::path& path,
    std::uint32_t minimum_width,
    std::uint32_t minimum_height);

// Writes 8-bit RGB PNG (top-left origin).
void save_rgb_png(const RgbImage& image, const std::filesystem::path& path);

// Writes 8-bit grayscale PNG (top-left origin). Used for foreground masks.
void save_gray_png(const GrayImage& image, const std::filesystem::path& path);

// Bilinear resize of grayscale or interleaved RGB (channels = 1 or 3).
void resize_bilinear(
    const std::uint8_t* source, std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t channels, std::uint8_t* destination, std::uint32_t destination_width,
    std::uint32_t destination_height);

}  // namespace photara::io
