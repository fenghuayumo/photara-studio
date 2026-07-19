#include "io/image.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <FreeImage.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace aetherscan::io {
namespace {

struct FreeImageRuntime {
    FreeImageRuntime() { FreeImage_Initialise(FALSE); }
    ~FreeImageRuntime() { FreeImage_DeInitialise(); }
};

std::mutex& freeimage_mutex() {
    static std::mutex mutex;
    return mutex;
}

void ensure_freeimage() {
    static FreeImageRuntime runtime;
    (void)runtime;
}

FREE_IMAGE_FORMAT detect_format(const std::filesystem::path& path) {
    const auto utf8 = path.string();
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(utf8.c_str(), 0);
    if (format == FIF_UNKNOWN) format = FreeImage_GetFIFFromFilename(utf8.c_str());
    if (format == FIF_UNKNOWN)
        throw std::runtime_error("Unsupported image format: " + utf8);
    return format;
}

void copy_top_left(
    FIBITMAP* bitmap, std::vector<std::uint8_t>& pixels, const std::uint32_t width,
    const std::uint32_t height, const std::uint32_t channels) {
    pixels.resize(static_cast<std::size_t>(width) * height * channels);
    const bool top_down = FreeImage_GetImageType(bitmap) != FIT_BITMAP
                              ? true
                              : (FreeImage_GetInfoHeader(bitmap)->biHeight < 0);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t source_y = top_down ? y : (height - 1 - y);
        const BYTE* row = FreeImage_GetScanLine(bitmap, static_cast<int>(source_y));
        std::uint8_t* destination =
            pixels.data() + static_cast<std::size_t>(y) * width * channels;
        if (channels == 1) {
            std::copy_n(row, width, destination);
        } else {
            for (std::uint32_t x = 0; x < width; ++x) {
                destination[3 * x + 0] = row[3 * x + FI_RGBA_RED];
                destination[3 * x + 1] = row[3 * x + FI_RGBA_GREEN];
                destination[3 * x + 2] = row[3 * x + FI_RGBA_BLUE];
            }
        }
    }
}

}  // namespace

GrayImage load_gray(const std::filesystem::path& path) {
    std::lock_guard lock(freeimage_mutex());
    ensure_freeimage();
    const auto format = detect_format(path);
    FIBITMAP* loaded = FreeImage_Load(format, path.string().c_str(), 0);
    if (!loaded) throw std::runtime_error("Failed to load image: " + path.string());
    FIBITMAP* gray = FreeImage_ConvertToGreyscale(loaded);
    FreeImage_Unload(loaded);
    if (!gray) throw std::runtime_error("Failed to convert image to grayscale: " + path.string());
    GrayImage image;
    image.width = FreeImage_GetWidth(gray);
    image.height = FreeImage_GetHeight(gray);
    if (image.width == 0 || image.height == 0) {
        FreeImage_Unload(gray);
        throw std::runtime_error("Empty grayscale image: " + path.string());
    }
    copy_top_left(gray, image.pixels, image.width, image.height, 1);
    FreeImage_Unload(gray);
    return image;
}

RgbImage load_rgb(const std::filesystem::path& path) {
    std::lock_guard lock(freeimage_mutex());
    ensure_freeimage();
    const auto format = detect_format(path);
    FIBITMAP* loaded = FreeImage_Load(format, path.string().c_str(), 0);
    if (!loaded) throw std::runtime_error("Failed to load image: " + path.string());
    FIBITMAP* rgb = FreeImage_ConvertTo24Bits(loaded);
    FreeImage_Unload(loaded);
    if (!rgb) throw std::runtime_error("Failed to convert image to RGB: " + path.string());
    RgbImage image;
    image.width = FreeImage_GetWidth(rgb);
    image.height = FreeImage_GetHeight(rgb);
    if (image.width == 0 || image.height == 0) {
        FreeImage_Unload(rgb);
        throw std::runtime_error("Empty RGB image: " + path.string());
    }
    copy_top_left(rgb, image.pixels, image.width, image.height, 3);
    FreeImage_Unload(rgb);
    return image;
}

void save_rgb_png(const RgbImage& image, const std::filesystem::path& path) {
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() !=
            static_cast<std::size_t>(image.width) * image.height * 3U)
        throw std::invalid_argument("Invalid RGB image for PNG export");
    std::lock_guard lock(freeimage_mutex());
    ensure_freeimage();
    FIBITMAP* bitmap = FreeImage_Allocate(
        static_cast<int>(image.width), static_cast<int>(image.height), 24);
    if (!bitmap)
        throw std::runtime_error("Failed to allocate PNG bitmap");
    for (std::uint32_t y = 0; y < image.height; ++y) {
        // FreeImage stores bottom-up scanlines for FIT_BITMAP.
        BYTE* row = FreeImage_GetScanLine(
            bitmap, static_cast<int>(image.height - 1U - y));
        const std::uint8_t* source =
            image.pixels.data() +
            static_cast<std::size_t>(y) * image.width * 3U;
        for (std::uint32_t x = 0; x < image.width; ++x) {
            row[3 * x + FI_RGBA_RED] = source[3 * x + 0];
            row[3 * x + FI_RGBA_GREEN] = source[3 * x + 1];
            row[3 * x + FI_RGBA_BLUE] = source[3 * x + 2];
        }
    }
    if (!FreeImage_Save(FIF_PNG, bitmap, path.string().c_str(), 0)) {
        FreeImage_Unload(bitmap);
        throw std::runtime_error("Failed to save PNG: " + path.string());
    }
    FreeImage_Unload(bitmap);
}

void resize_bilinear(
    const std::uint8_t* source, const std::uint32_t source_width,
    const std::uint32_t source_height, const std::uint32_t channels, std::uint8_t* destination,
    const std::uint32_t destination_width, const std::uint32_t destination_height) {
    if (source == nullptr || destination == nullptr || source_width == 0 || source_height == 0 ||
        destination_width == 0 || destination_height == 0 || (channels != 1 && channels != 3))
        throw std::invalid_argument("Invalid bilinear resize arguments");
    const double scale_x =
        static_cast<double>(source_width) / static_cast<double>(destination_width);
    const double scale_y =
        static_cast<double>(source_height) / static_cast<double>(destination_height);
    for (std::uint32_t y = 0; y < destination_height; ++y) {
        const double source_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
        const std::int32_t y0 = static_cast<std::int32_t>(std::floor(source_y));
        const std::int32_t y1 = y0 + 1;
        const double fy = source_y - y0;
        const std::uint32_t iy0 = static_cast<std::uint32_t>(std::clamp(y0, 0, static_cast<std::int32_t>(source_height - 1)));
        const std::uint32_t iy1 = static_cast<std::uint32_t>(std::clamp(y1, 0, static_cast<std::int32_t>(source_height - 1)));
        for (std::uint32_t x = 0; x < destination_width; ++x) {
            const double source_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
            const std::int32_t x0 = static_cast<std::int32_t>(std::floor(source_x));
            const std::int32_t x1 = x0 + 1;
            const double fx = source_x - x0;
            const std::uint32_t ix0 = static_cast<std::uint32_t>(std::clamp(x0, 0, static_cast<std::int32_t>(source_width - 1)));
            const std::uint32_t ix1 = static_cast<std::uint32_t>(std::clamp(x1, 0, static_cast<std::int32_t>(source_width - 1)));
            for (std::uint32_t c = 0; c < channels; ++c) {
                const auto sample = [&](std::uint32_t sx, std::uint32_t sy) {
                    return static_cast<double>(
                        source[(static_cast<std::size_t>(sy) * source_width + sx) * channels + c]);
                };
                const double v00 = sample(ix0, iy0);
                const double v10 = sample(ix1, iy0);
                const double v01 = sample(ix0, iy1);
                const double v11 = sample(ix1, iy1);
                const double value =
                    (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 +
                    (1.0 - fx) * fy * v01 + fx * fy * v11;
                destination[(static_cast<std::size_t>(y) * destination_width + x) * channels + c] =
                    static_cast<std::uint8_t>(std::clamp(value + 0.5, 0.0, 255.0));
            }
        }
    }
}

}  // namespace aetherscan::io
