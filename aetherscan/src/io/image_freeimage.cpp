#include "io/image.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <FreeImage.h>
#include <jpeglib.h>
#include <png.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <csetjmp>
#include <cstdlib>
#include <fstream>
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

bool has_jpeg_signature(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::array<char, 2> signature{};
    return stream.read(signature.data(), signature.size()) &&
           signature[0] == '\xff' && signature[1] == '\xd8';
}

bool has_png_signature(const std::filesystem::path& path) {
    static constexpr std::array<unsigned char, 8> signature{
        0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::ifstream stream(path, std::ios::binary);
    std::array<unsigned char, 8> actual{};
    return stream.read(
               reinterpret_cast<char*>(actual.data()), actual.size()) &&
           actual == signature;
}

enum class DirectImageFormat {
    jpeg,
    png,
    other,
};

DirectImageFormat direct_image_format(const std::filesystem::path& path) {
    if (has_jpeg_signature(path)) return DirectImageFormat::jpeg;
    if (has_png_signature(path)) return DirectImageFormat::png;
    return DirectImageFormat::other;
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("Failed to open image: " + path.string());
    const std::streampos end = stream.tellg();
    if (end < 0)
        throw std::runtime_error(
            "Failed to determine image size: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!bytes.empty() &&
        !stream.read(reinterpret_cast<char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Failed to read image: " + path.string());
    return bytes;
}

struct JpegErrorManager {
    jpeg_error_mgr public_error{};
    jmp_buf jump{};
    char message[JMSG_LENGTH_MAX]{};
};

void jpeg_error_exit(j_common_ptr context) {
    auto* error = reinterpret_cast<JpegErrorManager*>(context->err);
    (*context->err->format_message)(context, error->message);
    std::longjmp(error->jump, 1);
}

std::uint32_t largest_safe_jpeg_denominator(
    const jpeg_decompress_struct& context,
    const std::uint32_t minimum_width,
    const std::uint32_t minimum_height) {
    for (const std::uint32_t denominator : {8U, 4U, 2U}) {
        if (context.image_width / denominator >= minimum_width &&
            context.image_height / denominator >= minimum_height)
            return denominator;
    }
    return 1U;
}

[[noreturn]] void throw_jpeg_error(
    const JpegErrorManager& error, const std::string& action) {
    throw std::runtime_error(action + ": " + error.message);
}

void read_jpeg_scanlines(
    jpeg_decompress_struct& context, std::uint8_t* pixels,
    const std::uint32_t width, const std::uint32_t components) {
    constexpr JDIMENSION k_batch = 32;
    JSAMPROW rows[k_batch];
    const std::size_t stride =
        static_cast<std::size_t>(width) * components;
    while (context.output_scanline < context.output_height) {
        const JDIMENSION start = context.output_scanline;
        const JDIMENSION batch = std::min(
            k_batch, context.output_height - start);
        for (JDIMENSION i = 0; i < batch; ++i)
            rows[i] = pixels + (static_cast<std::size_t>(start + i) * stride);
        jpeg_read_scanlines(&context, rows, batch);
    }
}

RgbImage decode_jpeg_rgb(
    const std::vector<std::uint8_t>& file,
    const std::uint32_t minimum_width, const std::uint32_t minimum_height) {
    jpeg_decompress_struct context{};
    JpegErrorManager error;
    context.err = jpeg_std_error(&error.public_error);
    error.public_error.error_exit = jpeg_error_exit;

    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&context);
        throw_jpeg_error(error, "JPEG decode failed");
    }

    jpeg_create_decompress(&context);
    jpeg_mem_src(
        &context, const_cast<unsigned char*>(file.data()),
        static_cast<unsigned long>(file.size()));
    jpeg_read_header(&context, TRUE);
    context.scale_num = 1;
    context.scale_denom = largest_safe_jpeg_denominator(
        context, std::max(minimum_width, 1U),
        std::max(minimum_height, 1U));
    context.out_color_space = JCS_RGB;
    jpeg_start_decompress(&context);

    if (context.output_components != 3)
        throw std::runtime_error("JPEG decoder did not produce RGB output");

    RgbImage image;
    image.width = static_cast<std::uint32_t>(context.output_width);
    image.height = static_cast<std::uint32_t>(context.output_height);
    image.pixels.resize(
        static_cast<std::size_t>(image.width) * image.height * 3U);
    read_jpeg_scanlines(context, image.pixels.data(), image.width, 3U);
    jpeg_finish_decompress(&context);
    jpeg_destroy_decompress(&context);
    return image;
}

GrayImage decode_jpeg_gray(const std::vector<std::uint8_t>& file) {
    jpeg_decompress_struct context{};
    JpegErrorManager error;
    context.err = jpeg_std_error(&error.public_error);
    error.public_error.error_exit = jpeg_error_exit;

    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&context);
        throw_jpeg_error(error, "JPEG grayscale decode failed");
    }

    jpeg_create_decompress(&context);
    jpeg_mem_src(
        &context, const_cast<unsigned char*>(file.data()),
        static_cast<unsigned long>(file.size()));
    jpeg_read_header(&context, TRUE);
    context.scale_num = 1;
    context.scale_denom = 1;
    context.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&context);
    if (context.output_components != 1)
        throw std::runtime_error("JPEG decoder did not produce grayscale");

    GrayImage image;
    image.width = static_cast<std::uint32_t>(context.output_width);
    image.height = static_cast<std::uint32_t>(context.output_height);
    image.pixels.resize(
        static_cast<std::size_t>(image.width) * image.height);
    read_jpeg_scanlines(context, image.pixels.data(), image.width, 1U);
    jpeg_finish_decompress(&context);
    jpeg_destroy_decompress(&context);
    return image;
}

ImageSize jpeg_image_size(const std::vector<std::uint8_t>& file) {
    jpeg_decompress_struct context{};
    JpegErrorManager error;
    context.err = jpeg_std_error(&error.public_error);
    error.public_error.error_exit = jpeg_error_exit;

    if (setjmp(error.jump) != 0) {
        jpeg_destroy_decompress(&context);
        throw_jpeg_error(error, "JPEG header probe failed");
    }

    jpeg_create_decompress(&context);
    jpeg_mem_src(
        &context, const_cast<unsigned char*>(file.data()),
        static_cast<unsigned long>(file.size()));
    jpeg_read_header(&context, TRUE);
    const ImageSize size{
        static_cast<std::uint32_t>(context.image_width),
        static_cast<std::uint32_t>(context.image_height)};
    jpeg_destroy_decompress(&context);
    if (size.width == 0 || size.height == 0)
        throw std::runtime_error("Empty JPEG image");
    return size;
}

class PngImage {
public:
    PngImage() noexcept { image_.version = PNG_IMAGE_VERSION; }
    ~PngImage() { png_image_free(&image_); }

    png_image& get() noexcept { return image_; }
    [[nodiscard]] const png_image& get() const noexcept { return image_; }

    [[noreturn]] static void fail(
        const png_image& image, const std::string& action) {
        std::string message = image.message[0] != '\0'
            ? image.message
            : "unknown libpng error";
        throw std::runtime_error(action + ": " + message);
    }

private:
    png_image image_{};
};

void begin_png_read(
    PngImage& image, const std::vector<std::uint8_t>& file) {
    if (png_image_begin_read_from_memory(
            &image.get(), file.data(), file.size()) == 0)
        PngImage::fail(image.get(), "PNG header decode failed");
}

void finish_png_read(
    PngImage& image, std::vector<std::uint8_t>& pixels) {
    pixels.resize(PNG_IMAGE_SIZE(image.get()));
    if (png_image_finish_read(
            &image.get(), nullptr, pixels.data(), 0, nullptr) == 0)
        PngImage::fail(image.get(), "PNG decode failed");
}

ImageSize png_image_size(const std::vector<std::uint8_t>& file) {
    PngImage image;
    begin_png_read(image, file);
    const ImageSize size{
        static_cast<std::uint32_t>(image.get().width),
        static_cast<std::uint32_t>(image.get().height)};
    if (size.width == 0 || size.height == 0)
        throw std::runtime_error("Empty PNG image");
    return size;
}

bool png_has_alpha(const std::vector<std::uint8_t>& file) {
    PngImage image;
    begin_png_read(image, file);
    return (image.get().format & PNG_FORMAT_FLAG_ALPHA) != 0;
}

RgbImage decode_png_rgb(const std::vector<std::uint8_t>& file) {
    PngImage image;
    begin_png_read(image, file);
    image.get().format = PNG_FORMAT_RGB;
    RgbImage result;
    result.width = static_cast<std::uint32_t>(image.get().width);
    result.height = static_cast<std::uint32_t>(image.get().height);
    finish_png_read(image, result.pixels);
    return result;
}

GrayImage decode_png_gray(const std::vector<std::uint8_t>& file) {
    PngImage image;
    begin_png_read(image, file);
    image.get().format = PNG_FORMAT_GRAY;
    GrayImage result;
    result.width = static_cast<std::uint32_t>(image.get().width);
    result.height = static_cast<std::uint32_t>(image.get().height);
    finish_png_read(image, result.pixels);
    return result;
}

GrayImage decode_png_alpha(const std::vector<std::uint8_t>& file) {
    PngImage image;
    begin_png_read(image, file);
    if ((image.get().format & PNG_FORMAT_FLAG_ALPHA) == 0) return {};
    image.get().format = PNG_FORMAT_RGBA;
    GrayImage result;
    result.width = static_cast<std::uint32_t>(image.get().width);
    result.height = static_cast<std::uint32_t>(image.get().height);
    finish_png_read(image, result.pixels);
    const std::size_t pixels = static_cast<std::size_t>(result.width) *
        result.height;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        result.pixels[pixel] = result.pixels[4 * pixel + 3];
    result.pixels.resize(pixels);
    return result;
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

std::string load_lens_description(const std::filesystem::path& path) {
    std::lock_guard lock(freeimage_mutex());
    ensure_freeimage();
    try {
        const auto format = detect_format(path);
        FIBITMAP* bitmap = FreeImage_Load(format, path.string().c_str(), FIF_LOAD_NOPIXELS);
        if (!bitmap) return {};
        std::string description;
        for (const auto metadata : {FIMD_EXIF_EXIF, FIMD_EXIF_MAIN}) {
            for (const char* key : {"LensModel", "LensMake"}) {
                FITAG* tag = nullptr;
                if (FreeImage_GetMetadata(metadata, bitmap, key, &tag) && tag) {
                    const char* value = FreeImage_TagToString(metadata, tag);
                    if (value) { description += value; description += ' '; }
                }
            }
        }
        FreeImage_Unload(bitmap);
        return description;
    } catch (const std::exception&) { return {}; }
}

ImageSize load_image_size(const std::filesystem::path& path) {
    const DirectImageFormat direct_format = direct_image_format(path);
    if (direct_format == DirectImageFormat::jpeg)
        return jpeg_image_size(read_file_bytes(path));
    if (direct_format == DirectImageFormat::png)
        return png_image_size(read_file_bytes(path));

    ensure_freeimage();
    const auto format = detect_format(path);
    FIBITMAP* loaded = FreeImage_Load(
        format, path.string().c_str(), FIF_LOAD_NOPIXELS);
    if (!loaded)
        loaded = FreeImage_Load(format, path.string().c_str(), 0);
    if (!loaded)
        throw std::runtime_error("Failed to load image: " + path.string());
    const ImageSize size{
        FreeImage_GetWidth(loaded), FreeImage_GetHeight(loaded)};
    FreeImage_Unload(loaded);
    if (size.width == 0 || size.height == 0)
        throw std::runtime_error("Empty image: " + path.string());
    return size;
}

namespace {

double metadata_number(FITAG* tag, FREE_IMAGE_MDMODEL metadata) {
    const char* text = FreeImage_TagToString(metadata, tag);
    if (!text || !*text) return 0.0;
    char* end = nullptr;
    const double numerator = std::strtod(text, &end);
    if (end == text) return 0.0;
    if (*end == '/') {
        const char* denominator_text = end + 1;
        const double denominator = std::strtod(denominator_text, &end);
        if (end == denominator_text || denominator == 0.0) return 0.0;
        return numerator / denominator;
    }
    return numerator;
}

}

std::string load_camera_identity(
    const std::filesystem::path& path, double* focal_length_mm,
    const std::uint32_t image_width, const std::uint32_t image_height,
    double* focal_prior_px) {
    std::lock_guard lock(freeimage_mutex());
    ensure_freeimage();
    try {
        const auto format = detect_format(path);
        FIBITMAP* bitmap = FreeImage_Load(
            format, path.string().c_str(), FIF_LOAD_NOPIXELS);
        if (!bitmap) return {};
        std::string identity;
        double plane_x_resolution = 0.0;
        double plane_resolution_unit = 0.0;
        double sensor_pixel_width = 0.0;
        for (const auto metadata : {FIMD_EXIF_MAIN, FIMD_EXIF_EXIF}) {
            FITAG* tag = nullptr;
            if (FreeImage_GetMetadata(
                    metadata, bitmap, "FocalLength", &tag) &&
                tag && focal_length_mm && *focal_length_mm <= 0.0) {
                const double parsed = metadata_number(tag, metadata);
                if (parsed > 0.0) *focal_length_mm = parsed;
            }
        }
        for (const auto metadata : {FIMD_EXIF_MAIN, FIMD_EXIF_EXIF}) {
            for (const char* key :
                 {"Make", "Model", "BodySerialNumber", "LensMake", "LensModel",
                  "FocalPlaneXResolution", "FocalPlaneResolutionUnit",
                  "PixelXDimension"}) {
                FITAG* tag = nullptr;
                if (FreeImage_GetMetadata(metadata, bitmap, key, &tag) && tag) {
                    if (focal_prior_px && *focal_prior_px <= 0.0 &&
                        std::string(key) == "FocalLengthIn35mmFormat") {
                        const double focal_35mm = metadata_number(tag, metadata);
                        if (focal_35mm > 0.0 && image_width > 0 && image_height > 0) {
                            const double diagonal = std::sqrt(
                                static_cast<double>(image_width) * image_width +
                                static_cast<double>(image_height) * image_height);
                            *focal_prior_px = focal_35mm / 43.27 * diagonal;
                        }
                    }
                    if (std::string(key) == "FocalPlaneXResolution") {
                        plane_x_resolution = metadata_number(tag, metadata);
                    } else if (std::string(key) == "FocalPlaneResolutionUnit") {
                        plane_resolution_unit = metadata_number(tag, metadata);
                    } else if (std::string(key) == "PixelXDimension") {
                        sensor_pixel_width = metadata_number(tag, metadata);
                    }
                    const char* value = FreeImage_TagToString(metadata, tag);
                    if (value && *value) {
                        identity += value;
                        identity += '\n';
                    }
                }
            }
        }
        for (const auto metadata : {FIMD_EXIF_MAIN, FIMD_EXIF_EXIF}) {
            FITAG* tag = nullptr;
            if (FreeImage_GetMetadata(
                    metadata, bitmap, "FocalLength", &tag) &&
                tag && focal_length_mm && *focal_length_mm <= 0.0) {
                const double parsed = metadata_number(tag, metadata);
                if (parsed > 0.0) *focal_length_mm = parsed;
            }
        }
        if (focal_prior_px && focal_length_mm && *focal_prior_px <= 0.0 && *focal_length_mm > 0.0 &&
            plane_x_resolution > 0.0 && plane_resolution_unit >= 2.0 &&
            plane_resolution_unit <= 5.0) {
            double pixels_per_mm = 0.0;
            switch (static_cast<int>(plane_resolution_unit)) {
                case 2: pixels_per_mm = plane_x_resolution / 25.4; break;
                case 3: pixels_per_mm = plane_x_resolution / 10.0; break;
                case 4: pixels_per_mm = plane_x_resolution; break;
                case 5: pixels_per_mm = plane_x_resolution * 1000.0; break;
                default: break;
            }
            if (sensor_pixel_width > 0.0 && image_width > 0)
                pixels_per_mm *= static_cast<double>(image_width) / sensor_pixel_width;
            const double focal_px = *focal_length_mm * pixels_per_mm;
            const double long_edge = std::max(image_width, image_height);
            if (focal_px > 0.1 * long_edge && focal_px < 100.0 * long_edge)
                *focal_prior_px = focal_px;
        }
        FreeImage_Unload(bitmap);
        return identity;
    } catch (const std::exception&) {
        return {};
    }
}

bool image_has_alpha(const std::filesystem::path& path) {
    const DirectImageFormat direct_format = direct_image_format(path);
    if (direct_format == DirectImageFormat::jpeg) return false;
    if (direct_format == DirectImageFormat::png)
        return png_has_alpha(read_file_bytes(path));

    ensure_freeimage();
    const auto format = detect_format(path);
    FIBITMAP* loaded = FreeImage_Load(
        format, path.string().c_str(), FIF_LOAD_NOPIXELS);
    if (!loaded)
        loaded = FreeImage_Load(format, path.string().c_str(), 0);
    if (!loaded)
        throw std::runtime_error("Failed to load image: " + path.string());
    const bool has_alpha = FreeImage_GetBPP(loaded) == 32 ||
                           FreeImage_IsTransparent(loaded);
    FreeImage_Unload(loaded);
    return has_alpha;
}

GrayImage load_gray(const std::filesystem::path& path) {
    const DirectImageFormat direct_format = direct_image_format(path);
    if (direct_format == DirectImageFormat::jpeg)
        return decode_jpeg_gray(read_file_bytes(path));
    if (direct_format == DirectImageFormat::png)
        return decode_png_gray(read_file_bytes(path));

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

GrayImage load_alpha(const std::filesystem::path& path) {
    const DirectImageFormat direct_format = direct_image_format(path);
    if (direct_format == DirectImageFormat::jpeg) return {};
    if (direct_format == DirectImageFormat::png)
        return decode_png_alpha(read_file_bytes(path));

    ensure_freeimage();
    const auto format = detect_format(path);
    FIBITMAP* loaded = FreeImage_Load(format, path.string().c_str(), 0);
    if (!loaded) throw std::runtime_error("Failed to load image: " + path.string());
    if (!FreeImage_IsTransparent(loaded) && FreeImage_GetBPP(loaded) != 32) {
        FreeImage_Unload(loaded);
        return {};
    }
    FIBITMAP* rgba = FreeImage_ConvertTo32Bits(loaded);
    FreeImage_Unload(loaded);
    if (!rgba)
        throw std::runtime_error("Failed to read image alpha: " + path.string());
    GrayImage image;
    image.width = FreeImage_GetWidth(rgba);
    image.height = FreeImage_GetHeight(rgba);
    image.pixels.resize(static_cast<std::size_t>(image.width) * image.height);
    const bool top_down = FreeImage_GetInfoHeader(rgba)->biHeight < 0;
    for (std::uint32_t y = 0; y < image.height; ++y) {
        const std::uint32_t source_y = top_down ? y : image.height - 1 - y;
        const BYTE* row = FreeImage_GetScanLine(rgba, static_cast<int>(source_y));
        for (std::uint32_t x = 0; x < image.width; ++x)
            image.pixels[static_cast<std::size_t>(y) * image.width + x] =
                row[4 * x + FI_RGBA_ALPHA];
    }
    FreeImage_Unload(rgba);
    return image;
}

RgbImage load_rgb(const std::filesystem::path& path) {
    const DirectImageFormat direct_format = direct_image_format(path);
    if (direct_format == DirectImageFormat::jpeg)
        return decode_jpeg_rgb(read_file_bytes(path), 1U, 1U);
    if (direct_format == DirectImageFormat::png)
        return decode_png_rgb(read_file_bytes(path));

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

RgbImage load_rgb_with_minimum_size(
    const std::filesystem::path& path, const std::uint32_t minimum_width,
    const std::uint32_t minimum_height) {
    if (direct_image_format(path) != DirectImageFormat::jpeg)
        return load_rgb(path);

    const std::vector<std::uint8_t> file = read_file_bytes(path);
    return decode_jpeg_rgb(file, minimum_width, minimum_height);
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
