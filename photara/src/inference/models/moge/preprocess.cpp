#include "preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace photara::inference::detail {
namespace {
std::uint32_t round_even(double value) {
    const auto lower = static_cast<std::uint32_t>(std::floor(value));
    const double fraction = value - lower;
    return lower + (fraction > 0.5 || (fraction == 0.5 && (lower & 1U)));
}
struct Tap { std::uint32_t start; std::vector<float> weights; };
std::vector<Tap> taps(std::uint32_t input, std::uint32_t output) {
    // Matches ATen's separable antialiased bilinear filter, including edge
    // truncation and renormalization. Float weights/accumulation match F32 RGB.
    const float scale = static_cast<float>(input) / output;
    const float support = std::max(scale, 1.F);
    const float inverse_scale = scale >= 1.F ? static_cast<float>(1.0 / scale) : 1.F;
    std::vector<Tap> result(output);
    for (std::uint32_t i = 0; i < output; ++i) {
        const float center = static_cast<float>(scale * (i + 0.5));
        const auto first = static_cast<std::uint32_t>(std::max(0.0, std::floor(center - support + 0.5)));
        const auto end = static_cast<std::uint32_t>(std::min<double>(input, std::floor(center + support + 0.5)));
        auto& tap = result[i];
        tap.start = first;
        float total = 0;
        for (auto j = first; j < end; ++j) {
            const auto distance = static_cast<float>((j - center + 0.5) * inverse_scale);
            const float value = std::max(0.F, 1.F - std::abs(distance));
            tap.weights.push_back(value);
            total += value;
        }
        for (auto& value : tap.weights) value /= total;
    }
    return result;
}
}
MogeImage preprocess_moge(const io::RgbImage& image, std::uint32_t num_tokens) {
    const auto count = static_cast<std::uint64_t>(image.width) * image.height;
    if (!image.width || !image.height || count > std::numeric_limits<std::size_t>::max() / 3 ||
        image.pixels.size() != count * 3)
        throw std::invalid_argument("MoGe requires a nonempty tightly packed RGB image");
    if (!num_tokens || num_tokens > 4096)
        throw std::invalid_argument("MoGe token budget must lie in 1..4096");
    const double aspect = static_cast<double>(image.width) / image.height;
    MogeImage result;
    result.rows = round_even(std::sqrt(num_tokens / aspect));
    result.columns = round_even(std::sqrt(num_tokens * aspect));
    if (!result.rows || !result.columns || static_cast<std::uint64_t>(result.rows) * result.columns > 4096)
        throw std::invalid_argument("MoGe rounded grid must contain 1..4096 patches");
    const auto width = result.columns * 14, height = result.rows * 14;
    const auto plane = static_cast<std::size_t>(width) * height;
    const auto horizontal = taps(image.width, width), vertical = taps(image.height, height);
    result.normalized_rgb.resize(plane * 3);
    std::vector<float> intermediate(static_cast<std::size_t>(width) * image.height);
    constexpr float mean[]{0.485F, 0.456F, 0.406F}, deviation[]{0.229F, 0.224F, 0.225F};
    for (int channel = 0; channel < 3; ++channel) {
        for (std::uint32_t y = 0; y < image.height; ++y)
            for (std::uint32_t x = 0; x < width; ++x) {
                const auto& tap = horizontal[x];
                float value = 0;
                for (std::size_t j = 0; j < tap.weights.size(); ++j)
                    value += (image.pixels[(static_cast<std::size_t>(y) * image.width + tap.start + j) * 3 + channel] / 255.F) * tap.weights[j];
                intermediate[static_cast<std::size_t>(y) * width + x] = value;
            }
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x) {
                const auto& tap = vertical[y];
                float value = 0;
                for (std::size_t j = 0; j < tap.weights.size(); ++j)
                    value += intermediate[(tap.start + j) * width + x] * tap.weights[j];
                result.normalized_rgb[channel * plane + static_cast<std::size_t>(y) * width + x] =
                    (value - mean[channel]) / deviation[channel];
            }
    }
    return result;
}
} // namespace photara::inference::detail
