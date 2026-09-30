#include "sift_cpu.hpp"

#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace photara::features {
namespace {

constexpr int kFilterWidthFactor = 4;
constexpr int kKernelMaxWidth = 33;
constexpr int kKernelMinWidth = 5;
constexpr double kTwoPi = 6.283185307179586476925286766559;

#if defined(PHOTARA_HAS_OPENMP)
inline bool use_openmp(const std::size_t work_items) {
    return parallel::allow_inner_parallelism() && work_items > 64;
}
#else
inline bool use_openmp(const std::size_t) { return false; }
#endif

inline int clampi(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

inline float fetch(const float* data, int index, int count) {
    return index >= 0 && index < count ? data[index] : 0.0F;
}

inline float inv_sqrt(float value) {
    return value > 0.0F ? 1.0F / std::sqrt(value) : 0.0F;
}

struct Vec4 {
    float x, y, z, w;
};

struct DetKey {
    int x;
    int y;
    float dx;
    float dy;
    float ds;
};

struct Grad {
    float mag;
    float rot;
};

struct OctaveShape {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pixels;
};

struct LevelSlot {
    int octave = 0;
    int level = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<DetKey> keys;
    std::vector<float> values;
    std::vector<float> descriptors;
};

std::uint32_t aligned_width(std::uint32_t width) {
    return (width + 3U) / 4U * 4U;
}

std::vector<float> gaussian_kernel(float sigma) {
    const auto radius = static_cast<int>(
        std::ceil(static_cast<float>(kFilterWidthFactor) * sigma - 0.5F));
    int width = 2 * radius + 1;
    if (width > kKernelMaxWidth) width = kKernelMaxWidth;
    if (width < kKernelMinWidth) width = kKernelMinWidth;
    std::vector<float> kernel(static_cast<std::size_t>(width));
    const int half = width / 2;
    float sum = 0.0F;
    for (int i = 0; i <= half; ++i) {
        const float value =
            std::exp(-0.5F * static_cast<float>(i * i) / (sigma * sigma));
        kernel[static_cast<std::size_t>(half - i)] = value;
        kernel[static_cast<std::size_t>(half + i)] = value;
        sum += i == 0 ? value : 2.0F * value;
    }
    const float inverse = 1.0F / sum;
    for (auto& value : kernel) value *= inverse;
    return kernel;
}

void convert_u8(
    const std::uint8_t* input, float* output, int height, int source_stride,
    int destination_stride) {
    const bool omp = use_openmp(static_cast<std::size_t>(height) * destination_stride);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int row = 0; row < height; ++row) {
        const std::uint8_t* source = input + static_cast<std::size_t>(row) * source_stride;
        float* destination =
            output + static_cast<std::size_t>(row) * destination_stride;
        int col = 0;
#if defined(__AVX2__)
        const __m256 scale = _mm256_set1_ps(1.0F / 255.0F);
        for (; col + 8 <= destination_stride; col += 8) {
            const __m128i bytes =
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + col));
            const __m256i ints = _mm256_cvtepu8_epi32(bytes);
            _mm256_storeu_ps(destination + col, _mm256_mul_ps(_mm256_cvtepi32_ps(ints), scale));
        }
#endif
        for (; col < destination_stride; ++col)
            destination[col] = static_cast<float>(source[col]) * (1.0F / 255.0F);
    }
}

void upsample(
    const float* source, float* destination, int src_width, int src_height,
    int dst_width, int dst_height) {
    const int count = src_width * src_height;
    const bool omp = use_openmp(static_cast<std::size_t>(dst_height) * src_width);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int y = 0; y < dst_height; ++y) {
        const int row = y >> 1;
        for (int x = 0; x < src_width; ++x) {
            if (x * 2 + 1 >= dst_width) break;
            const int index = row * src_width + x;
            const int dst_index = y * dst_width + x * 2;
            if ((y & 1) != 0) {
                const float v11 = fetch(source, index, count);
                const float v12 = fetch(source, index + 1, count);
                const float v21 = fetch(source, index + src_width, count);
                const float v22 = fetch(source, index + src_width + 1, count);
                const float first = v21 * 0.5F + 0.5F * v11;
                const float second = v22 * 0.5F + 0.5F * v12;
                destination[dst_index] = first;
                destination[dst_index + 1] = first * 0.5F + second * 0.5F;
            } else {
                const float v1 = fetch(source, index, count);
                const float v2 = fetch(source, index + 1, count);
                destination[dst_index] = v1;
                destination[dst_index + 1] = v1 * 0.5F + v2 * 0.5F;
            }
        }
    }
}

void downsample(
    const float* source, float* destination, int src_width, int src_height,
    int dst_width, int dst_height, int scale) {
    const bool omp = use_openmp(static_cast<std::size_t>(dst_width) * dst_height);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int y = 0; y < dst_height; ++y) {
        const int src_row = y * scale;
        float* row = destination + static_cast<std::size_t>(y) * dst_width;
        for (int x = 0; x < dst_width; ++x) {
            int src_col = x * scale;
            if (src_col >= src_width) src_col = src_width - 1;
            float value = 0.0F;
            if (src_row >= 0 && src_row < src_height && src_width > 0)
                value = source[src_row * src_width + src_col];
            row[x] = value;
        }
    }
}

void blur_horizontal(
    const float* source, float* destination, int width, int height, int radius,
    const float* kernel) {
    const bool omp = use_openmp(static_cast<std::size_t>(width) * height);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int y = 0; y < height; ++y) {
        const float* row = source + static_cast<std::size_t>(y) * width;
        float* out = destination + static_cast<std::size_t>(y) * width;
        int x = 0;
#if defined(__AVX2__)
        for (; x + 8 <= width; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                alignas(32) float taps[8];
                for (int i = 0; i < 8; ++i)
                    taps[i] = row[clampi(x + i + k, 0, width - 1)];
                acc = _mm256_fmadd_ps(
                    _mm256_load_ps(taps), _mm256_set1_ps(kernel[k + radius]), acc);
            }
            _mm256_storeu_ps(out + x, acc);
        }
#endif
        for (; x < width; ++x) {
            float value = 0.0F;
            for (int k = -radius; k <= radius; ++k)
                value += row[clampi(x + k, 0, width - 1)] * kernel[k + radius];
            out[x] = value;
        }
    }
}

void blur_vertical(
    const float* source, float* destination, int width, int height, int radius,
    const float* kernel) {
    const bool omp = use_openmp(static_cast<std::size_t>(width) * height);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int y = 0; y < height; ++y) {
        float* out = destination + static_cast<std::size_t>(y) * width;
        int x = 0;
#if defined(__AVX2__)
        for (; x + 8 <= width; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                const int sample = clampi(y + k, 0, height - 1);
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(
                        source + static_cast<std::size_t>(sample) * width + x),
                    _mm256_set1_ps(kernel[k + radius]), acc);
            }
            _mm256_storeu_ps(out + x, acc);
        }
#endif
        for (; x < width; ++x) {
            float value = 0.0F;
            for (int k = -radius; k <= radius; ++k)
                value += source[static_cast<std::size_t>(clampi(y + k, 0, height - 1)) *
                                    width +
                                x] *
                         kernel[k + radius];
            out[x] = value;
        }
    }
}

void blur(
    float* destination, const float* source, float* scratch, int width, int height,
    float sigma) {
    if (!(sigma > 0.0F)) {
        if (destination != source) {
            std::memcpy(
                destination, source,
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                    sizeof(float));
        }
        return;
    }
    const std::vector<float> weights = gaussian_kernel(sigma);
    const int radius = static_cast<int>(weights.size() / 2);
    blur_horizontal(source, scratch, width, height, radius, weights.data());
    blur_vertical(scratch, destination, width, height, radius, weights.data());
}

void dog_map(
    const float* current, const float* previous, float* dog, int width, int height) {
    const int pixels = width * height;
    const bool omp = use_openmp(static_cast<std::size_t>(pixels));
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int index = 0; index < pixels; ++index)
        dog[index] = current[index] - previous[index];
}

void gradient_map(const float* image, Grad* gradient, int width, int height) {
    const int count = width * height;
    const bool omp = use_openmp(static_cast<std::size_t>(count));
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (omp)
#else
    (void)omp;
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = y * width + x;
            const float dx = fetch(image, index + 1, count) - fetch(image, index - 1, count);
            const float dy =
                fetch(image, index + width, count) - fetch(image, index - width, count);
            const float magnitude = 0.5F * std::sqrt(dx * dx + dy * dy);
            gradient[index].mag = magnitude;
            gradient[index].rot = magnitude == 0.0F ? 0.0F : std::atan2(dy, dx);
        }
    }
}

bool detect_extremum(
    const float* previous, const float* current, const float* next, int width,
    int height, int x, int y, float dog_threshold0, float dog_threshold,
    float edge_threshold, float& dx, float& dy, float& ds) {
    dx = 0.0F;
    dy = 0.0F;
    ds = 0.0F;
    if (y <= 0 || x <= 0 || y >= height - 1 || x >= width - 1) return false;
    const int index = y * width + x;
    const int row_above = index - width;
    const int row_center = index;
    const int row_below = index + width;

    const float center = current[row_center];
    if (std::fabs(center) <= dog_threshold0) return false;

    float data[3][3];
    float datap[3][3];
    float datan[3][3];
    data[1][1] = center;
    data[1][0] = current[row_center - 1];
    data[1][2] = current[row_center + 1];
    float nmax = std::fmax(data[1][0], data[1][2]);
    float nmin = std::fmin(data[1][0], data[1][2]);
    if (center <= nmax && center >= nmin) return false;

    for (int pass = 0; pass < 2; ++pass) {
        const int r = pass == 0 ? 0 : 2;
        const int base = r == 0 ? row_above : row_below;
        for (int c = 0; c < 3; ++c) data[r][c] = current[base + c - 1];
        if (center > nmax) {
            nmax = std::fmax(nmax, std::fmax(data[r][0], std::fmax(data[r][1], data[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = std::fmin(nmin, std::fmin(data[r][0], std::fmin(data[r][1], data[r][2])));
            if (center > nmin) return false;
        }
    }

    const float doubled = center * 2.0F;
    const float fxx = data[1][0] + data[1][2] - doubled;
    const float fyy = data[0][1] + data[2][1] - doubled;
    const float fxy = 0.25F * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
    const float det = fxx * fyy - fxy * fxy;
    const float trace = fxx + fyy;
    if (det <= 0.0F || trace * trace > edge_threshold * det) return false;

    for (int r = 0; r < 3; ++r) {
        const int base = r == 0 ? row_above : (r == 1 ? row_center : row_below);
        for (int c = 0; c < 3; ++c) datap[r][c] = previous[base + c - 1];
        if (center > nmax) {
            nmax = std::fmax(
                nmax, std::fmax(datap[r][0], std::fmax(datap[r][1], datap[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = std::fmin(
                nmin, std::fmin(datap[r][0], std::fmin(datap[r][1], datap[r][2])));
            if (center > nmin) return false;
        }
    }
    for (int r = 0; r < 3; ++r) {
        const int base = r == 0 ? row_above : (r == 1 ? row_center : row_below);
        for (int c = 0; c < 3; ++c) datan[r][c] = next[base + c - 1];
        if (center > nmax) {
            nmax = std::fmax(
                nmax, std::fmax(datan[r][0], std::fmax(datan[r][1], datan[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = std::fmin(
                nmin, std::fmin(datan[r][0], std::fmin(datan[r][1], datan[r][2])));
            if (center > nmin) return false;
        }
    }

    const float fx = 0.5F * (data[1][2] - data[1][0]);
    const float fy = 0.5F * (data[2][1] - data[0][1]);
    const float fs = 0.5F * (datan[1][1] - datap[1][1]);
    const float fss = datan[1][1] + datap[1][1] - doubled;
    const float fxs =
        0.25F * (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
    const float fys =
        0.25F * (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

    Vec4 row0 = fxx > 0.0F ? Vec4{fxx, fxy, fxs, -fx} : Vec4{-fxx, -fxy, -fxs, fx};
    Vec4 row1 = fxy > 0.0F ? Vec4{fxy, fyy, fys, -fy} : Vec4{-fxy, -fyy, -fys, fy};
    Vec4 row2 = fxs > 0.0F ? Vec4{fxs, fys, fss, -fs} : Vec4{-fxs, -fys, -fss, fs};
    const float pivot = std::fmax(row0.x, std::fmax(row1.x, row2.x));
    bool offset_ok = true;
    if (pivot >= 1e-10F) {
        if (pivot == row1.x) std::swap(row0, row1);
        else if (pivot == row2.x) std::swap(row0, row2);
        row0.y /= row0.x;
        row0.z /= row0.x;
        row0.w /= row0.x;
        row1.y -= row1.x * row0.y;
        row1.z -= row1.x * row0.z;
        row1.w -= row1.x * row0.w;
        row2.y -= row2.x * row0.y;
        row2.z -= row2.x * row0.z;
        row2.w -= row2.x * row0.w;
        if (std::fabs(row2.y) > std::fabs(row1.y)) std::swap(row2, row1);
        if (std::fabs(row1.y) >= 1e-10F) {
            row1.z /= row1.y;
            row1.w /= row1.y;
            row2.z -= row2.y * row1.z;
            row2.w -= row2.y * row1.w;
            if (std::fabs(row2.z) >= 1e-10F) {
                ds = row2.w / row2.z;
                dy = row1.w - ds * row1.z;
                dx = row0.w - ds * row0.z - dy * row0.y;
                offset_ok = std::fabs(center + 0.5F * (dx * fx + dy * fy + ds * fs)) >
                                dog_threshold &&
                            std::fabs(ds) < 1.0F && std::fabs(dx) < 1.0F &&
                            std::fabs(dy) < 1.0F;
            }
        }
    }
    return offset_ok;
}

std::vector<DetKey> detect_level(
    const float* previous, const float* current, const float* next, int width,
    int height, float dog_threshold0, float dog_threshold, float edge_threshold) {
    std::vector<DetKey> keys;
    const bool omp = use_openmp(static_cast<std::size_t>(width) * height);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel if (omp)
#endif
    {
        std::vector<DetKey> local;
        local.reserve(256);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp for nowait schedule(static)
#endif
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float dx, dy, ds;
                if (!detect_extremum(
                        previous, current, next, width, height, x, y, dog_threshold0,
                        dog_threshold, edge_threshold, dx, dy, ds))
                    continue;
                local.push_back({x, y, dx, dy, ds});
            }
        }
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp critical
#endif
        {
            keys.insert(keys.end(), local.begin(), local.end());
        }
    }
    std::sort(keys.begin(), keys.end(), [](const DetKey& a, const DetKey& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
    return keys;
}

Grad sample_gradient(
    const Grad* gradient, int width, int height, float x, float y) {
    const float fx = x - 0.5F;
    const float fy = y - 0.5F;
    const int ix = static_cast<int>(std::floor(fx));
    const int iy = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(ix);
    const float ty = fy - static_cast<float>(iy);
    const int x0 = clampi(ix, 0, width - 1);
    const int x1 = clampi(ix + 1, 0, width - 1);
    const int y0 = clampi(iy, 0, height - 1);
    const int y1 = clampi(iy + 1, 0, height - 1);
    const Grad a = gradient[y0 * width + x0];
    const Grad b = gradient[y0 * width + x1];
    const Grad c = gradient[y1 * width + x0];
    const Grad d = gradient[y1 * width + x1];
    const Grad ab{a.mag + (b.mag - a.mag) * tx, a.rot + (b.rot - a.rot) * tx};
    const Grad cd{c.mag + (d.mag - c.mag) * tx, c.rot + (d.rot - c.rot) * tx};
    return {ab.mag + (cd.mag - ab.mag) * ty, ab.rot + (cd.rot - ab.rot) * ty};
}

void pack_orientation(
    const DetKey& key, const Grad* gradient, int width, int height,
    unsigned num_orientation, float sigma, float sigma_step, float* out) {
    constexpr float kTenDegrees = 5.7295779513082320876798154814105F;
    float x = static_cast<float>(key.x) + 0.5F + key.dx;
    float y = static_cast<float>(key.y) + 0.5F + key.dy;
    float scale = sigma * std::pow(sigma_step, key.ds);
    if (num_orientation == 0) {
        out[0] = x;
        out[1] = y;
        out[2] = scale;
        out[3] = 0.0F;
        return;
    }

    float vote[37];
    for (int i = 0; i < 37; ++i) vote[i] = 0.0F;
    const float gsigma = scale * 1.5F;
    const float win = std::fabs(scale) * 3.0F;
    const float dist_threshold = win * win + 0.5F;
    const float factor = -0.5F / (gsigma * gsigma);
    const float xmin = std::fmax(1.5F, std::floor(x - win) + 0.5F);
    const float ymin = std::fmax(1.5F, std::floor(y - win) + 0.5F);
    const float xmax = std::fmin(static_cast<float>(width) - 1.5F, std::floor(x + win) + 0.5F);
    const float ymax =
        std::fmin(static_cast<float>(height) - 1.5F, std::floor(y + win) + 0.5F);
    for (float sy = ymin; sy <= ymax; sy += 1.0F) {
        for (float sx = xmin; sx <= xmax; sx += 1.0F) {
            const float ddx = sx - x;
            const float ddy = sy - y;
            const float squared = ddx * ddx + ddy * ddy;
            if (squared >= dist_threshold) continue;
            const Grad sample = sample_gradient(gradient, width, height, sx, sy);
            const float weight = sample.mag * std::exp(squared * factor);
            int bin = static_cast<int>(std::floor(sample.rot * kTenDegrees));
            if (bin < 0) bin += 36;
            if (bin >= 0 && bin <= 36) vote[bin] += weight;
        }
    }
    constexpr float kOneThird = 1.0F / 3.0F;
    for (int pass = 0; pass < 6; ++pass) {
        vote[36] = vote[0];
        float previous = vote[35];
        for (int j = 0; j < 36; ++j) {
            const float smoothed = kOneThird * (previous + vote[j] + vote[j + 1]);
            previous = vote[j];
            vote[j] = smoothed;
        }
    }
    vote[36] = vote[0];

    unsigned packed = 0;
    if (num_orientation == 1) {
        int index_max = 0;
        float max_vote = vote[0];
        for (int i = 1; i < 36; ++i) {
            index_max = vote[i] > max_vote ? i : index_max;
            max_vote = std::fmax(max_vote, vote[i]);
        }
        const float pre = vote[index_max == 0 ? 35 : index_max - 1];
        const float next = vote[index_max + 1];
        const float off = 0.5F * ((next - pre) / (max_vote + max_vote - next - pre));
        float fraction = (static_cast<float>(index_max) + 0.5F + off) / 36.0F;
        if (fraction < 0.0F) fraction += 1.0F;
        packed = (65535U << 16) |
                 (static_cast<unsigned>(std::floor(fraction * 65535.0F)) & 0xFFFFU);
    } else {
        float max_vote = vote[0];
        for (int i = 1; i < 36; ++i) max_vote = std::fmax(max_vote, vote[i]);
        const float vote_threshold = max_vote * 0.8F;
        float previous = vote[35];
        float max_rot0 = 0.0F, max_rot1 = 0.0F;
        float max_vot0 = 0.0F, max_vot1 = 0.0F;
        int orientations = 0;
        for (int i = 0; i < 36; ++i) {
            const float next = vote[i + 1];
            if (vote[i] > vote_threshold && vote[i] > previous && vote[i] > next) {
                const float di =
                    0.5F * ((next - previous) / (vote[i] + vote[i] - next - previous));
                const float rotation = static_cast<float>(i) + di + 0.5F;
                const float weight = vote[i];
                if (weight > max_vot1) {
                    if (weight > max_vot0) {
                        max_vot1 = max_vot0;
                        max_rot1 = max_rot0;
                        max_vot0 = weight;
                        max_rot0 = rotation;
                    } else {
                        max_vot1 = weight;
                        max_rot1 = rotation;
                    }
                    ++orientations;
                }
            }
            previous = vote[i];
        }
        float first = max_rot0 / 36.0F;
        if (first < 0.0F) first += 1.0F;
        unsigned low = orientations == 0
                           ? 65535U
                           : static_cast<unsigned>(std::floor(first * 65535.0F));
        unsigned high = 65535U;
        if (orientations > 1) {
            float second = max_rot1 / 36.0F;
            if (second < 0.0F) second += 1.0F;
            high = static_cast<unsigned>(std::floor(second * 65535.0F));
        }
        packed = (high << 16) | low;
    }
    out[0] = x;
    out[1] = y;
    out[2] = scale;
    std::memcpy(out + 3, &packed, sizeof(packed));
}

void descriptor_block(
    const float* key, const Grad* gradient, int width, int height, unsigned bin,
    float* bins8) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwo = 6.28318530717958647692F;
    constexpr float kBinScale = 4.0F / 3.14159265358979323846F;
    const float ix = static_cast<float>(bin & 3U);
    const float iy = static_cast<float>(bin >> 2);
    const float key_x = key[0];
    const float key_y = key[1];
    const float key_scale = key[2];
    const float key_angle = key[3];
    const float spacing = std::fabs(key_scale * 3.0F);
    const float sine = std::sin(key_angle);
    const float cosine = std::cos(key_angle);
    const float angle = key_angle > kPi ? key_angle - kTwo : key_angle;
    const float cos_spacing = cosine * spacing;
    const float sin_spacing = sine * spacing;
    const float cos_over = cosine / spacing;
    const float sin_over = sine / spacing;
    const float offset_x = ix - 1.5F;
    const float offset_y = iy - 1.5F;
    const float point_x = cos_spacing * offset_x - sin_spacing * offset_y + key_x;
    const float point_y = cos_spacing * offset_y + sin_spacing * offset_x + key_y;
    const float box = std::fabs(cos_spacing) + std::fabs(sin_spacing);
    const float xmin = std::fmax(1.5F, std::floor(point_x - box) + 0.5F);
    const float ymin = std::fmax(1.5F, std::floor(point_y - box) + 0.5F);
    const float xmax =
        std::fmin(static_cast<float>(width) - 1.5F, std::floor(point_x + box) + 0.5F);
    const float ymax =
        std::fmin(static_cast<float>(height) - 1.5F, std::floor(point_y + box) + 0.5F);

    float bins[9];
    for (int i = 0; i < 9; ++i) bins[i] = 0.0F;
    for (float sy = ymin; sy <= ymax; sy += 1.0F) {
        for (float sx = xmin; sx <= xmax; sx += 1.0F) {
            const float ddx = sx - point_x;
            const float ddy = sy - point_y;
            const float nx = cos_over * ddx + sin_over * ddy;
            const float ny = cos_over * ddy - sin_over * ddx;
            const float ax = std::fabs(nx);
            const float ay = std::fabs(ny);
            if (ax < 1.0F && ay < 1.0F) {
                const Grad sample = sample_gradient(gradient, width, height, sx, sy);
                const float dnx = nx + offset_x;
                const float dny = ny + offset_y;
                const float weight = std::exp(-0.125F * (dnx * dnx + dny * dny)) *
                                     (1.0F - ax) * (1.0F - ay) * sample.mag;
                float theta = (angle - sample.rot) * kBinScale;
                if (theta < 0.0F) theta += 8.0F;
                const float lower = std::floor(theta);
                const unsigned start = static_cast<unsigned>(lower);
                const float weight1 = lower + 1.0F - theta;
                const float weight2 = theta - lower;
                for (unsigned k = 0; k < 8; ++k) {
                    if (k == start) {
                        bins[k] += weight1 * weight;
                        bins[k + 1] += weight2 * weight;
                    }
                }
            }
        }
    }
    bins[0] += bins[8];
    for (int i = 0; i < 8; ++i) bins8[i] = bins[i];
}

void normalize_descriptor(float* row) {
    float values[128];
    float norm1 = 0.0F;
    for (int i = 0; i < 32; ++i) {
        const float x = row[i * 4 + 0];
        const float y = row[i * 4 + 1];
        const float z = row[i * 4 + 2];
        const float w = row[i * 4 + 3];
        values[i * 4 + 0] = x;
        values[i * 4 + 1] = y;
        values[i * 4 + 2] = z;
        values[i * 4 + 3] = w;
        norm1 += x * x + y * y + z * z + w * w;
    }
    const float scale1 = inv_sqrt(norm1);
    float norm2 = 0.0F;
    for (int i = 0; i < 128; ++i) {
        values[i] = std::fmin(0.2F, values[i] * scale1);
        norm2 += values[i] * values[i];
    }
    const float scale2 = inv_sqrt(norm2);
    for (int i = 0; i < 128; ++i) row[i] = values[i] * scale2;
}

struct Pyramid {
    int dog_level_num = 0;
    int level_min = 0;
    int level_max = 0;
    int level_num = 0;
    int level_ds = 0;
    float sigma0 = 0;
    float sigman = 0;
    float sigmak = 0;
    float sigma_step = 0;
    float dsigma0 = 0;
    std::vector<float> level_sigma;
    std::vector<float> sigma_inc;

    explicit Pyramid(int octave_layers) {
        dog_level_num = octave_layers;
        level_min = -1;
        level_max = dog_level_num + 1;
        level_num = level_max - level_min + 1;
        level_ds = std::min(level_min + dog_level_num, level_max);
        sigma0 = 1.6F * std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigman = 0.5F;
        sigmak = std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigma_step = sigmak;
        dsigma0 = sigma0 * std::sqrt(1.0F - 1.0F / (sigmak * sigmak));
        level_sigma.resize(static_cast<std::size_t>(level_num));
        for (int level = level_min; level <= level_max; ++level)
            level_sigma[static_cast<std::size_t>(level - level_min)] =
                sigma0 * std::pow(
                             2.0F, static_cast<float>(level) /
                                       static_cast<float>(dog_level_num));
        sigma_inc.resize(static_cast<std::size_t>(level_num - 1));
        for (int level = level_min + 1; level <= level_max; ++level)
            sigma_inc[static_cast<std::size_t>(level - level_min - 1)] =
                dsigma0 * std::pow(sigmak, static_cast<float>(level));
    }

    [[nodiscard]] float initial_smooth_sigma(int octave_min) const {
        const float sa = sigma0 * std::pow(
                                     2.0F, static_cast<float>(level_min) /
                                               static_cast<float>(dog_level_num));
        const float sb = sigman / std::pow(2.0F, static_cast<float>(octave_min));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }

    [[nodiscard]] float skip_sigma() const {
        const float sa = sigma0 * std::pow(sigmak, static_cast<float>(level_min));
        const float sb =
            sigma0 * std::pow(sigmak, static_cast<float>(level_ds - dog_level_num));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }
};

struct Plan {
    int octave_min = 0;
    int octave_num = 0;
    std::uint32_t first_width = 0;
    std::uint32_t first_height = 0;
};

Plan make_plan(const SiftOptions& options, std::uint32_t width, std::uint32_t height) {
    Plan result;
    result.octave_min = options.first_octave;
    std::uint32_t working_width = width;
    std::uint32_t working_height = height;
    if (result.octave_min < 0) {
        working_width <<= static_cast<std::uint32_t>(-result.octave_min);
        working_height <<= static_cast<std::uint32_t>(-result.octave_min);
    } else {
        working_width >>= static_cast<std::uint32_t>(result.octave_min);
        working_height >>= static_cast<std::uint32_t>(result.octave_min);
    }
    const std::uint32_t maximum_dimension =
        options.maximum_image_dimension *
        (1U << static_cast<std::uint32_t>(std::max(0, -options.first_octave)));
    while (working_width > maximum_dimension || working_height > maximum_dimension) {
        ++result.octave_min;
        working_width = std::max(1U, working_width >> 1);
        working_height = std::max(1U, working_height >> 1);
    }
    const double smallest =
        static_cast<double>(std::min(working_width, working_height));
    result.octave_num = std::max(1, static_cast<int>(std::floor(std::log2(smallest))) - 3);
    result.first_width = working_width;
    result.first_height = working_height;
    return result;
}

template <class T>
void ensure(std::vector<T>& buffer, std::size_t count) {
    if (buffer.size() < count) buffer.resize(count);
}

}  // namespace

struct SiftCpuEngine::Impl {
    SiftOptions options;
    Pyramid pyramid;
    std::vector<float> gauss[2];
    std::vector<float> dogs[3];
    std::vector<float> scratch;
    std::vector<float> carry;
    std::vector<float> input_float;
    std::vector<Grad> gradient;
    std::vector<std::vector<float>> saved;

    explicit Impl(SiftOptions value)
        : options(std::move(value)),
          pyramid(static_cast<int>(options.octave_layers)) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations == 0 || options.octave_layers == 0 ||
            options.maximum_orientations > 4 || options.octave_layers > 16)
            throw std::invalid_argument("Invalid SIFT limits");
        if (options.first_octave < -1 || options.first_octave > 4)
            throw std::invalid_argument("SIFT first_octave must be in [-1, 4]");
        saved.resize(static_cast<std::size_t>(pyramid.dog_level_num));
    }

    void process_level(
        LevelSlot& slot, const float* gaussian, int width, int height) {
        if (slot.keys.empty()) return;
        ensure(gradient, static_cast<std::size_t>(width) * height);
        gradient_map(gaussian, gradient.data(), width, height);

        const int count = static_cast<int>(slot.keys.size());
        std::vector<float> packed(static_cast<std::size_t>(count) * 4);
        const bool omp = use_openmp(static_cast<std::size_t>(count));
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 16) if (omp)
#else
        (void)omp;
#endif
        for (int feature = 0; feature < count; ++feature) {
            pack_orientation(
                slot.keys[static_cast<std::size_t>(feature)], gradient.data(), width,
                height, options.maximum_orientations,
                pyramid.level_sigma[static_cast<std::size_t>(slot.level + 1)],
                pyramid.sigma_step, packed.data() + static_cast<std::size_t>(feature) * 4);
        }

        constexpr double orientation_factor = kTwoPi / 65535.0;
        auto& values = slot.values;
        values.clear();
        values.reserve(static_cast<std::size_t>(count) * 8);
        for (int feature = 0; feature < count; ++feature) {
            const float x = packed[feature * 4 + 0];
            const float y = packed[feature * 4 + 1];
            const float scale = packed[feature * 4 + 2];
            std::uint32_t bits = 0;
            std::memcpy(&bits, &packed[feature * 4 + 3], sizeof(bits));
            const std::uint32_t first = bits & 0xFFFFU;
            const std::uint32_t second = bits >> 16;
            if (first == 65535U) continue;
            values.insert(values.end(), {
                x, y, scale, static_cast<float>(orientation_factor * first)});
            if (second != 65535U && second != first) {
                values.insert(
                    values.end(),
                    {x, y, scale, static_cast<float>(orientation_factor * second)});
            }
        }
        const int oriented_count = static_cast<int>(values.size() / 4);
        if (oriented_count == 0) {
            slot.descriptors.clear();
            return;
        }
        slot.descriptors.assign(static_cast<std::size_t>(oriented_count) * 128, 0.0F);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 4) if (omp)
#endif
        for (int feature = 0; feature < oriented_count; ++feature) {
            const float* key = values.data() + static_cast<std::size_t>(feature) * 4;
            float* row = slot.descriptors.data() + static_cast<std::size_t>(feature) * 128;
            for (unsigned bin = 0; bin < 16; ++bin)
                descriptor_block(key, gradient.data(), width, height, bin, row + bin * 8);
            normalize_descriptor(row);
        }
    }

    FeatureSet extract(
        std::span<const std::uint8_t> pixels, std::uint32_t width, std::uint32_t height,
        std::size_t row_stride) {
        if (row_stride == 0) row_stride = width;
        if (width == 0 || height == 0 || row_stride < width ||
            pixels.size() < row_stride * static_cast<std::size_t>(height))
            throw std::invalid_argument("Invalid grayscale image view");

        FeatureSet result;
        result.image_width = width;
        result.image_height = height;
        result.descriptor_dimension = 128;
        result.metric = DescriptorMetric::l2;
        result.extractor_name = "sift";
        if (width < 8 || height < 8) return result;

        const std::uint32_t effective_width = std::max(4U, width & ~3U);
        const Plan plan = make_plan(options, effective_width, height);
        const int octave_num = plan.octave_num;
        const int dog_levels = pyramid.dog_level_num;
        const std::size_t level_total =
            static_cast<std::size_t>(octave_num) * static_cast<std::size_t>(dog_levels);

        std::vector<OctaveShape> shapes(static_cast<std::size_t>(octave_num));
        {
            std::uint32_t w = plan.first_width;
            std::uint32_t h = plan.first_height;
            for (int octave = 0; octave < octave_num; ++octave) {
                const std::uint32_t aligned = aligned_width(w);
                shapes[static_cast<std::size_t>(octave)] =
                    OctaveShape{aligned, h, aligned * h};
                w = std::max(1U, w >> 1);
                h = std::max(1U, h >> 1);
            }
        }
        const std::size_t max_pixels = shapes.front().pixels;
        ensure(gauss[0], max_pixels);
        ensure(gauss[1], max_pixels);
        ensure(dogs[0], max_pixels);
        ensure(dogs[1], max_pixels);
        ensure(dogs[2], max_pixels);
        ensure(scratch, max_pixels);
        ensure(carry, max_pixels);
        ensure(input_float, static_cast<std::size_t>(effective_width) * height);
        convert_u8(
            pixels.data(), input_float.data(), static_cast<int>(height),
            static_cast<int>(row_stride), static_cast<int>(effective_width));

        const float dog_threshold = static_cast<float>(options.contrast_threshold);
        const float dog_threshold0 = 0.8F * dog_threshold;
        const float edge_threshold =
            static_cast<float>(
                (options.edge_threshold + 1.0) * (options.edge_threshold + 1.0) /
                options.edge_threshold);
        const int source_index = pyramid.level_ds - pyramid.level_min;
        std::vector<LevelSlot> slots(level_total);

        for (int octave = 0; octave < octave_num; ++octave) {
            const auto& shape = shapes[static_cast<std::size_t>(octave)];
            const int w = static_cast<int>(shape.width);
            const int h = static_cast<int>(shape.height);
            float* base = gauss[0].data();
            if (octave == 0) {
                if (plan.octave_min < 0) {
                    upsample(
                        input_float.data(), base, static_cast<int>(effective_width),
                        static_cast<int>(height), w, h);
                } else {
                    downsample(
                        input_float.data(), base, static_cast<int>(effective_width),
                        static_cast<int>(height), w, h, 1 << plan.octave_min);
                }
                blur(base, base, scratch.data(), w, h,
                     pyramid.initial_smooth_sigma(plan.octave_min));
            } else {
                const auto& previous = shapes[static_cast<std::size_t>(octave - 1)];
                downsample(
                    carry.data(), base, static_cast<int>(previous.width),
                    static_cast<int>(previous.height), w, h, 2);
                blur(base, base, scratch.data(), w, h, pyramid.skip_sigma());
            }

            for (int i = 0; i < dog_levels; ++i)
                ensure(saved[static_cast<std::size_t>(i)], shape.pixels);

            for (int level = 1; level < pyramid.level_num; ++level) {
                float* destination = gauss[level & 1].data();
                const float* source = gauss[(level - 1) & 1].data();
                blur(destination, source, scratch.data(), w, h,
                     pyramid.sigma_inc[static_cast<std::size_t>(level - 1)]);
                const int dog = (level - 1) % 3;
                dog_map(destination, source, dogs[dog].data(), w, h);
                if (level == source_index) {
                    std::memcpy(
                        carry.data(), destination, shape.pixels * sizeof(float));
                }
                if (level <= dog_levels) {
                    std::memcpy(
                        saved[static_cast<std::size_t>(level - 1)].data(), destination,
                        shape.pixels * sizeof(float));
                }
                if (level >= 3 && level - 3 < dog_levels) {
                    const int feature_level = level - 3;
                    LevelSlot& slot = slots[
                        static_cast<std::size_t>(octave) * dog_levels +
                        static_cast<std::size_t>(feature_level)];
                    slot.octave = octave;
                    slot.level = feature_level;
                    slot.width = shape.width;
                    slot.height = shape.height;
                    slot.keys = detect_level(
                        dogs[feature_level % 3].data(),
                        dogs[(feature_level + 1) % 3].data(),
                        dogs[(feature_level + 2) % 3].data(), w, h, dog_threshold0,
                        dog_threshold, edge_threshold);
                }
            }
            for (int feature_level = 0; feature_level < dog_levels; ++feature_level) {
                LevelSlot& slot = slots[
                    static_cast<std::size_t>(octave) * dog_levels +
                    static_cast<std::size_t>(feature_level)];
                process_level(
                    slot, saved[static_cast<std::size_t>(feature_level)].data(), w, h);
            }
        }

        std::vector<LevelSlot*> kept;
        {
            long long accumulated = 0;
            const long long threshold = static_cast<long long>(options.maximum_features);
            for (std::size_t index = level_total; index-- > 0;) {
                if (threshold > 0 && accumulated > threshold) continue;
                accumulated += static_cast<long long>(slots[index].keys.size());
                if (!slots[index].keys.empty()) kept.push_back(&slots[index]);
            }
            std::reverse(kept.begin(), kept.end());
        }

        std::size_t total_features = 0;
        for (LevelSlot* slot : kept) total_features += slot->values.size() / 4;
        if (options.maximum_features > 0) {
            std::size_t total = total_features;
            std::size_t index = 0;
            while (total > options.maximum_features && index < kept.size()) {
                const std::size_t count = kept[index]->values.size() / 4;
                if (count > 0 && total - count > options.maximum_features) {
                    total -= count;
                    kept[index]->values.clear();
                    kept[index]->descriptors.clear();
                } else {
                    break;
                }
                ++index;
            }
            total_features = total;
        }
        if (total_features == 0) return result;

        const float octave_scale_base =
            plan.octave_min >= 0
                ? static_cast<float>(1 << plan.octave_min)
                : 1.0F / static_cast<float>(1 << -plan.octave_min);
        result.keypoints.reserve(total_features);
        result.descriptors.reserve(total_features * 128);
        for (const LevelSlot* level : kept) {
            const float octave_scale =
                octave_scale_base * static_cast<float>(1 << level->octave);
            for (std::size_t feature = 0; feature < level->values.size() / 4; ++feature) {
                const float x = level->values[feature * 4 + 0];
                const float y = level->values[feature * 4 + 1];
                const float scale = level->values[feature * 4 + 2];
                const float orientation = level->values[feature * 4 + 3];
                double mirrored =
                    std::fmod(kTwoPi - static_cast<double>(orientation), kTwoPi);
                if (mirrored < 0.0) mirrored += kTwoPi;
                result.keypoints.push_back(
                    {octave_scale * (x - 0.5F) + 0.5F,
                     octave_scale * (y - 0.5F) + 0.5F, octave_scale * scale,
                     static_cast<float>(mirrored), 0.0F});
            }
            result.descriptors.insert(
                result.descriptors.end(), level->descriptors.begin(),
                level->descriptors.end());
        }
        return result;
    }
};

SiftCpuEngine::SiftCpuEngine(SiftOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

SiftCpuEngine::~SiftCpuEngine() = default;

FeatureSet SiftCpuEngine::extract(
    std::span<const std::uint8_t> pixels, std::uint32_t width, std::uint32_t height,
    std::size_t row_stride) {
    return impl_->extract(pixels, width, height, row_stride);
}

}  // namespace photara::features
