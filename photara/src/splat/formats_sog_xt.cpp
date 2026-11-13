#include "formats_sog_xt.hpp"
#include "sog_vulkan.hpp"
#include "sog_palette.hpp"
#include "core/logging.hpp"
#include <chrono>
#include <deque>
#include <future>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace photara::splat::detail {
namespace {
using Json = nlohmann::json;
using Bytes = std::vector<std::uint8_t>;

Json yaml_json(const YAML::Node& node, unsigned depth = 0) {
    if (depth > 32)
        throw std::runtime_error("SOG-XT metadata nesting limit exceeded");
    if (node.IsMap()) {
        Json value = Json::object();
        for (const auto& item : node)
            value[item.first.as<std::string>()] = yaml_json(item.second, depth + 1);
        return value;
    }
    if (node.IsSequence()) {
        Json value = Json::array();
        for (const auto& item : node)
            value.push_back(yaml_json(item, depth + 1));
        return value;
    }
    if (node.IsNull())
        return nullptr;
    const auto text = node.Scalar();
    const auto parsed = Json::parse(text, nullptr, false);
    return parsed.is_discarded() ? Json(text) : parsed;
}

unsigned grid_side(std::size_t n) {
    const auto side = (static_cast<std::size_t>(std::ceil(std::sqrt(double(n)))) + 7) / 8 * 8;
    if (!n || side > WEBP_MAX_DIMENSION)
        throw std::runtime_error("SOG-XT count exceeds WebP grid limits");
    return static_cast<unsigned>(side);
}

struct Image {
    unsigned width{}, height{};
    Bytes rgba;
};

Image read_image(const SogXtFiles& files, const std::string& name) {
    const auto it = files.find(name);
    if (it == files.end())
        throw std::runtime_error("Missing SOG-XT asset: " + name);
    int w = 0, h = 0;
    if (!WebPGetInfo(it->second.data(), it->second.size(), &w, &h) || w <= 0 || h <= 0)
        throw std::runtime_error("Invalid SOG-XT WebP: " + name);
    std::unique_ptr<std::uint8_t, decltype(&WebPFree)> data(
        WebPDecodeRGBA(it->second.data(), it->second.size(), &w, &h), &WebPFree);
    if (!data)
        throw std::runtime_error("Cannot decode SOG-XT WebP: " + name);
    return {unsigned(w), unsigned(h), Bytes(data.get(), data.get() + std::size_t(w) * h * 4)};
}

// WebP's convenience RGBA encoder can discard RGB under alpha=0. Quaternion
// and UV planes use alpha as DATA, so exact lossless encoding is essential.
Bytes write_image(const Bytes& rgba, unsigned width, unsigned height) {
    WebPConfig config;
    if (!WebPConfigInit(&config))
        throw std::runtime_error("Cannot initialize WebP");
    config.lossless = 1;
    config.quality = 100;
    config.method = 6;
    config.exact = 1;
    config.thread_level = 1;
    WebPPicture pic;
    if (!WebPPictureInit(&pic))
        throw std::runtime_error("Cannot initialize WebP picture");
    struct Cleanup {
        WebPPicture* p;
        ~Cleanup() {
            WebPPictureFree(p);
        }
    } cleanup{&pic};
    pic.use_argb = 1;
    pic.width = int(width);
    pic.height = int(height);
    WebPMemoryWriter writer;
    WebPMemoryWriterInit(&writer);
    struct WriterCleanup {
        WebPMemoryWriter* w;
        ~WriterCleanup() {
            WebPMemoryWriterClear(w);
        }
    } wc{&writer};
    pic.writer = WebPMemoryWrite;
    pic.custom_ptr = &writer;
    if (!WebPPictureImportRGBA(&pic, rgba.data(), int(width * 4)) || !WebPEncode(&config, &pic))
        throw std::runtime_error("Cannot encode SOG-XT WebP");
    return Bytes(writer.mem, writer.mem + writer.size);
}

struct Quantizer {
    std::vector<float> low, high;
    unsigned levels{};
    Quantizer(
        const std::vector<float>& values,
        unsigned dims,
        unsigned q,
        const std::vector<std::uint8_t>* active = nullptr)
        : low(dims, std::numeric_limits<float>::max()),
          high(dims, std::numeric_limits<float>::lowest()), levels(q) {
        for (std::size_t i = 0; i < values.size() / dims; ++i) {
            if (active && !(*active)[i])
                continue;
            for (unsigned d = 0; d < dims; ++d) {
                const float v = values[i * dims + d];
                if (!std::isfinite(v))
                    throw std::runtime_error("Non-finite SOG-XT attribute");
                low[d] = std::min(low[d], v);
                high[d] = std::max(high[d], v);
            }
        }
    }
    explicit Quantizer(const Json& j) {
        const auto array = [](const Json& value) {
            return value.is_number() ? std::vector<float>{value.get<float>()}
                                     : value.get<std::vector<float>>();
        };
        low = array(j.at("min_values"));
        high = array(j.at("max_values"));
        levels = j.value("levels", 256U);
        if (low.empty() || low.size() != high.size() || levels < 2 || levels > 65536)
            throw std::runtime_error("Invalid SOG-XT quantizer");
        for (std::size_t d = 0; d < low.size(); ++d)
            if (!std::isfinite(low[d]) || !std::isfinite(high[d]) || low[d] > high[d])
                throw std::runtime_error("Invalid SOG-XT quantization range");
    }
    unsigned encode(float v, unsigned d) const {
        const double range = double(high[d]) - low[d];
        return range == 0 ? 0
                          : unsigned(std::lround(
                                std::clamp((double(v) - low[d]) / range, 0.0, 1.0) * (levels - 1)));
    }
    float decode(unsigned v, unsigned d) const {
        if (v >= levels)
            throw std::runtime_error("SOG-XT value exceeds quantization range");
        return float(double(low[d]) + (double(high[d]) - low[d]) * v / (levels - 1));
    }
    Json json() const {
        return {{"min_values", low}, {"max_values", high}, {"levels", levels}};
    }
};

// Separable box filter, O(N*D) independent of radius. Padded entries are kept
// throughout layout optimization; an explicit active mask prevents data loss.
std::vector<float> blur(
    const std::vector<float>& values,
    unsigned side,
    unsigned dims,
    unsigned radius) {
    std::vector<float> temp(values.size()), result(values.size());
    for (unsigned pass = 0; pass < 2; ++pass) {
        const auto& src = pass ? temp : values;
        auto& dst = pass ? result : temp;
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(side >= 64)
#endif
        for (int line = 0; line < int(side); ++line)
            for (unsigned d = 0; d < dims; ++d) {
                const auto offset = [&](unsigned p) -> std::size_t {
                    return (pass ? std::size_t(p) * side + line : std::size_t(line) * side + p) *
                               dims +
                           d;
                };
                double sum = 0;
                for (unsigned p = 0; p <= std::min(radius, side - 1); ++p)
                    sum += src[offset(p)];
                for (unsigned p = 0; p < side; ++p) {
                    const unsigned left = p > radius ? p - radius : 0;
                    const unsigned right = std::min(side - 1, p + radius);
                    dst[offset(p)] = float(sum / (right - left + 1));
                    if (p >= radius)
                        sum -= src[offset(p - radius)];
                    if (p + radius + 1 < side)
                        sum += src[offset(p + radius + 1)];
                }
            }
    }
    return result;
}

SogPalette cpu_palette(
    const std::vector<float>& points,
    unsigned n,
    unsigned dims,
    unsigned budget) {
    // Bounded-memory hierarchical two-means Lloyd clustering, also usable in
    // non-Vulkan builds. No N*K matrix or repeated per-point neural decoding.
    SogPalette out;
    out.labels.resize(n);
    std::vector<unsigned> rows(n);
    std::iota(rows.begin(), rows.end(), 0);
    const auto split = [&](auto&& self, std::size_t first, std::size_t end,
                           unsigned leaves) -> void {
        if (leaves <= 1 || end - first <= 1) {
            const auto label = unsigned(out.centroids.size() / dims);
            std::vector<double> sum(dims);
            for (auto i = first; i < end; ++i) {
                out.labels[rows[i]] = label;
                for (unsigned d = 0; d < dims; ++d)
                    sum[d] += points[std::size_t(rows[i]) * dims + d];
            }
            for (double v : sum)
                out.centroids.push_back(float(v / (end - first)));
            return;
        }
        unsigned axis = 0;
        double best = -1;
        for (unsigned d = 0; d < dims; ++d) {
            double sum = 0, sq = 0;
            for (auto i = first; i < end; ++i) {
                const double v = points[std::size_t(rows[i]) * dims + d];
                sum += v;
                sq += v * v;
            }
            const double variance = sq - sum * sum / (end - first);
            if (variance > best) {
                best = variance;
                axis = d;
            }
        }
        auto mid = first + (end - first) / 2;
        std::nth_element(
            rows.begin() + first, rows.begin() + mid, rows.begin() + end,
            [&](unsigned a, unsigned b) {
                return points[std::size_t(a) * dims + axis] < points[std::size_t(b) * dims + axis];
            });
        // Median seed prevents an empty branch for identical SH vectors.
        for (unsigned step = 0; step < 4; ++step) {
            std::vector<double> a(dims), b(dims);
            for (auto i = first; i < end; ++i)
                for (unsigned d = 0; d < dims; ++d)
                    (i < mid ? a[d] : b[d]) += points[std::size_t(rows[i]) * dims + d];
            for (unsigned d = 0; d < dims; ++d) {
                a[d] /= mid - first;
                b[d] /= end - mid;
            }
            auto pivot =
                std::partition(rows.begin() + first, rows.begin() + end, [&](unsigned row) {
                    double da = 0, db = 0;
                    for (unsigned d = 0; d < dims; ++d) {
                        const double v = points[std::size_t(row) * dims + d];
                        da += (v - a[d]) * (v - a[d]);
                        db += (v - b[d]) * (v - b[d]);
                    }
                    return da < db;
                });
            const auto next = std::size_t(pivot - rows.begin());
            if (next == first || next == end)
                break;
            mid = next;
        }
        self(self, first, mid, leaves / 2);
        self(self, mid, end, leaves - leaves / 2);
    };
    split(split, 0, n, std::min(n, budget));
    refine_sog_palette_cpu(points, n, dims, budget, out);
    return out;
}
} // namespace

std::vector<std::uint32_t> sog_xt_layout(
    const std::vector<float>& features,
    unsigned side,
    unsigned dims) {
    const std::size_t n = std::size_t(side) * side;
    if (!side || !dims || features.size() != n * dims || n > UINT32_MAX)
        throw std::invalid_argument("Invalid SOG-XT layout dimensions");
    std::vector<std::uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    // Coarse-to-fine assignment sorting: each disjoint group solves its exact
    // 4! assignment to low-pass targets. Shifts connect groups between passes.
    // This CPU PLAS-style implementation is independent of the authors' GPU code.
    std::vector<float> current(features);
    for (unsigned radius = std::max(1U, side / 2);; radius /= 2) {
        for (unsigned pass = 0; pass < 4; ++pass) {
            const auto target = blur(current, side, dims, radius);
            const unsigned shift_x = pass & 1, shift_y = (pass >> 1) & 1;
            const unsigned stride = std::max(1U, radius);
            // Each 2*stride tile pairs four subtiles; all assignments are disjoint.
            const unsigned tiles = (side + 2 * stride - 1) / (2 * stride);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(n >= 4096)
#endif
            for (std::int64_t row = 0; row < std::int64_t(tiles) * stride; ++row)
                for (unsigned bx = 0; bx < side; bx += 2 * stride)
                    for (unsigned dx = 0; dx < stride; ++dx) {
                        const unsigned by = unsigned(row / stride) * 2 * stride;
                        const unsigned dy = unsigned(row % stride);
                        if (bx + dx + stride >= side || by + dy + stride >= side)
                            continue;
                        std::array<std::size_t, 4> p{};
                        for (unsigned j = 0; j < 4; ++j)
                            p[j] = std::size_t((by + dy + (j / 2) * stride + shift_y) % side) *
                                       side +
                                   (bx + dx + (j % 2) * stride + shift_x) % side;
                        double cost[4][4]{};
                        for (unsigned a = 0; a < 4; ++a)
                            for (unsigned b = 0; b < 4; ++b)
                                for (unsigned d = 0; d < dims; ++d) {
                                    const double delta =
                                        current[p[b] * dims + d] - target[p[a] * dims + d];
                                    cost[a][b] += delta * delta;
                                }
                        std::array<unsigned, 4> perm{0, 1, 2, 3}, best = perm;
                        double minimum = cost[0][0] + cost[1][1] + cost[2][2] + cost[3][3];
                        while (std::next_permutation(perm.begin(), perm.end())) {
                            const double value = cost[0][perm[0]] + cost[1][perm[1]] +
                                                 cost[2][perm[2]] + cost[3][perm[3]];
                            if (value < minimum) {
                                minimum = value;
                                best = perm;
                            }
                        }
                        const std::array<std::uint32_t, 4> old{
                            order[p[0]], order[p[1]], order[p[2]], order[p[3]]};
                        for (unsigned j = 0; j < 4; ++j) {
                            order[p[j]] = old[best[j]];
                            std::copy_n(
                                features.data() + std::size_t(order[p[j]]) * dims, dims,
                                current.data() + p[j] * dims);
                        }
                    }
        }
        if (radius == 1)
            break;
    }
    return order;
}

void sog_xt_pras(
    std::vector<float>& rotations,
    std::vector<float>& scales,
    const Bytes& active,
    unsigned side) {
    const std::size_t n = std::size_t(side) * side;
    if (rotations.size() != n * 4 || scales.size() != n * 3 || active.size() != n)
        throw std::invalid_argument("Invalid SOG-XT PRAS dimensions");
    struct Symmetry {
        Eigen::Quaternionf q;
        std::array<unsigned, 3> axes;
    };
    std::vector<Symmetry> symmetries;
    std::array<unsigned, 3> axes{0, 1, 2};
    do {
        for (unsigned signs = 0; signs < 8; ++signs) {
            Eigen::Matrix3f m = Eigen::Matrix3f::Zero();
            for (unsigned j = 0; j < 3; ++j)
                m(axes[j], j) = signs & (1U << j) ? -1.F : 1.F;
            if (m.determinant() > 0)
                symmetries.push_back({Eigen::Quaternionf(m), axes});
        }
    } while (std::next_permutation(axes.begin(), axes.end()));
    float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < n; ++i)
        if (active[i])
            for (unsigned d = 0; d < 3; ++d) {
                lo = std::min(lo, scales[i * 3 + d]);
                hi = std::max(hi, scales[i * 3 + d]);
            }
    const float range = std::max(hi - lo, 1e-12F);
    for (unsigned radius = std::max(1U, side / 4);; radius /= 2) {
        std::vector<float> proxy(n * 7);
        for (std::size_t i = 0; i < n; ++i) {
            for (unsigned d = 0; d < 4; ++d)
                proxy[i * 7 + d] = rotations[i * 4 + d] * .5F + .5F;
            for (unsigned d = 0; d < 3; ++d)
                proxy[i * 7 + 4 + d] = (scales[i * 3 + d] - lo) / range;
        }
        const auto target = blur(proxy, side, 7, radius);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(n >= 4096)
#endif
        for (std::int64_t i = 0; i < std::int64_t(n); ++i) {
            if (!active[i])
                continue;
            const Eigen::Quaternionf q(
                rotations[i * 4], rotations[i * 4 + 1], rotations[i * 4 + 2], rotations[i * 4 + 3]);
            const std::array<float, 3> s{scales[i * 3], scales[i * 3 + 1], scales[i * 3 + 2]};
            float minimum = std::numeric_limits<float>::max();
            std::array<float, 4> best_q{};
            std::array<float, 3> best_s{};
            for (const auto& symmetry : symmetries) {
                const auto c = q * symmetry.q;
                const std::array<float, 4> candidate{c.w(), c.x(), c.y(), c.z()};
                for (float sign : {1.F, -1.F}) {
                    float cost = 0;
                    for (unsigned d = 0; d < 4; ++d)
                        cost += std::abs(candidate[d] * sign * .5F + .5F - target[i * 7 + d]);
                    for (unsigned d = 0; d < 3; ++d)
                        cost +=
                            std::abs((s[symmetry.axes[d]] - lo) / range - target[i * 7 + 4 + d]);
                    if (cost < minimum) {
                        minimum = cost;
                        for (unsigned d = 0; d < 4; ++d)
                            best_q[d] = candidate[d] * sign;
                        for (unsigned d = 0; d < 3; ++d)
                            best_s[d] = s[symmetry.axes[d]];
                    }
                }
            }
            std::copy(best_q.begin(), best_q.end(), rotations.begin() + i * 4);
            std::copy(best_s.begin(), best_s.end(), scales.begin() + i * 3);
        }
        if (radius == 1)
            break;
    }
}

SogXtFiles encode_sog_xt(const GaussianModel& model, SogXtTrainingCodec* training,
                          const SogXtExportOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    auto previous = started;
    const auto stage = [&](const char* name, float progress) {
        const auto now = std::chrono::steady_clock::now();
        core::Logger::instance().info("sog_xt_export stage=", name, " elapsed_ms=",
            std::chrono::duration<double, std::milli>(now - previous).count());
        previous = now;
        if (options.progress) options.progress(progress);
    };
    const auto n = model.size();
    const unsigned degree = model.sh_degree;
    const unsigned bases = (degree + 1) * (degree + 1), dims = (bases - 1) * 3;
    if (!n || degree > 3 || n > UINT32_MAX)
        throw std::runtime_error("Invalid SOG-XT model");
    auto means = model.means.to_vector(), scales = model.log_scales.to_vector();
    auto q = model.quaternions.to_vector(), opacity = model.opacity_logits.to_vector(),
         sh = model.sh.to_vector();
    if (means.size() != n * 3 || scales.size() != n * 3 || q.size() != n * 4 ||
        opacity.size() != n || sh.size() != n * bases * 3)
        throw std::runtime_error("Invalid SOG-XT tensor shapes");
    for (const auto* values : {&means, &scales, &q, &opacity, &sh})
        for (float v : *values)
            if (!std::isfinite(v))
                throw std::runtime_error("Non-finite SOG-XT model");
    const unsigned side = grid_side(n);
    const auto pixels = std::size_t(side) * side;
    for (float& v : means)
        v = std::copysign(std::log1p(std::abs(v)), v);
    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Quaternionf rot(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
        if (rot.norm() < 1e-12F)
            rot = Eigen::Quaternionf::Identity();
        else
            rot.normalize();
        q[i * 4] = rot.w();
        q[i * 4 + 1] = rot.x();
        q[i * 4 + 2] = rot.y();
        q[i * 4 + 3] = rot.z();
    }
    Quantizer position_quant(means, 1, 65536);
    // Both byte planes guide layout, emphasizing coarse spatial locality.
    std::vector<float> features(pixels * 6);
    for (std::size_t i = 0; i < n; ++i)
        for (unsigned d = 0; d < 3; ++d) {
            const unsigned v = position_quant.encode(means[i * 3 + d], 0);
            features[i * 6 + d] = float(v >> 8) / 255;
            features[i * 6 + 3 + d] = float(v & 255) / 255 * .1F;
        }
    stage("prepare", .05F);
    const auto order = sog_xt_layout(features, side, 6);
    stage("layout", .3F);
    std::vector<float>().swap(features);
    Bytes active(pixels);
    const auto reorder = [&](const std::vector<float>& src, unsigned channels) {
        std::vector<float> dst(pixels * channels);
        for (std::size_t i = 0; i < pixels; ++i)
            if (order[i] < n)
                std::copy_n(
                    src.data() + std::size_t(order[i]) * channels, channels,
                    dst.data() + i * channels);
        return dst;
    };
    std::vector<float> dc(n * 3), ac(n * dims);
    for (std::size_t i = 0; i < n; ++i) {
        std::copy_n(sh.data() + i * bases * 3, 3, dc.data() + i * 3);
        if (dims)
            std::copy_n(sh.data() + i * bases * 3 + 3, dims, ac.data() + i * dims);
    }
    std::vector<float>().swap(sh);
    means = reorder(means, 3);
    scales = reorder(scales, 3);
    q = reorder(q, 4);
    opacity = reorder(opacity, 1);
    dc = reorder(dc, 3);
    for (std::size_t i = 0; i < pixels; ++i)
        active[i] = order[i] < n;
    sog_xt_pras(q, scales, active, side);
    stage("reorder_pras", .55F);
    if (training) {
        training->side = side;
        training->dimensions = dims;
        training->pixels.clear();
        for (unsigned i = 0; i < pixels; ++i)
            if (active[i])
                training->pixels.push_back(i);
        const auto tensor = [&](const std::vector<float>& src, unsigned channels) {
            std::vector<float> dst(n * channels);
            for (std::size_t i = 0; i < n; ++i)
                std::copy_n(
                    src.data() + std::size_t(training->pixels[i]) * channels, channels,
                    dst.data() + i * channels);
            return tinytensor::Tensor::from_vector(
                dst, {n, std::size_t(channels)}, model.means.device());
        };
        auto world = means;
        for (float& v : world)
            v = std::copysign(std::expm1(std::abs(v)), v);
        training->latent = model;
        training->latent.means = tensor(world, 3);
        training->latent.log_scales = tensor(scales, 3);
        training->latent.quaternions = tensor(q, 4);
        training->latent.opacity_logits = tensor(opacity, 1);
        std::vector<float> full(n * bases * 3);
        for (std::size_t i = 0; i < n; ++i) {
            const auto pixel = training->pixels[i];
            std::copy_n(dc.data() + std::size_t(pixel) * 3, 3, full.data() + i * bases * 3);
            if (dims)
                std::copy_n(
                    ac.data() + std::size_t(order[pixel]) * dims, dims,
                    full.data() + i * bases * 3 + 3);
        }
        training->latent.sh =
            tinytensor::Tensor::from_vector(full, {n, std::size_t(bases), 3}, model.means.device());
        training->latent.normal_features = {};
        training->latent.filter_3d = {};
    }
    SogXtFiles files;
    struct PendingImage { std::string name; std::future<Bytes> result; };
    std::deque<PendingImage> pending;
    const auto collect = [&] {
        auto bytes = pending.front().result.get();
        files.emplace(pending.front().name, std::move(bytes));
        pending.pop_front();
    };
    const auto enqueue = [&](std::string name, Bytes rgba, unsigned w, unsigned h) {
        if (pending.size() == 2) collect();
        pending.push_back({name, std::async(std::launch::async,
            [rgba = std::move(rgba), w, h, name] {
                const auto start = std::chrono::steady_clock::now();
                auto bytes = write_image(rgba, w, h);
                core::Logger::instance().info("sog_xt_export image=", name,
                    " bytes=", bytes.size(),
                    " elapsed_ms=", std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
                return bytes;
            })});
    };
    Json metadata = {
        {"format", "SOG-XT"},
        {"version", 2},
        {"meta",
         {{"count", n},
          {"grid_side", side},
          {"sh_degree", degree},
          {"image_codec", "webp"},
          {"antialiased", model.filter_mode == SplatFilterMode::mip_2d}}},
        {"ops", Json::array()}};
    const auto store = [&](const std::string& field, const std::vector<float>& values,
                           unsigned channels, unsigned levels) {
        Quantizer quant(values, channels, levels, &active);
        if (field == "means") {
            quant.low = {*std::min_element(quant.low.begin(), quant.low.end())};
            quant.high = {*std::max_element(quant.high.begin(), quant.high.end())};
        }
        auto ranges = quant.json();
        if (quant.low.size() == 1) {
            ranges["min_values"] = quant.low[0];
            ranges["max_values"] = quant.high[0];
        }
        metadata["ops"].push_back(
            {{"input_fields", {field}}, {"transforms", {{{"simple_quantize", ranges}}}}});
        Bytes image(pixels * 4, 255), high(pixels * 4, 255);
        for (std::size_t i = 0; i < pixels; ++i)
            for (unsigned d = 0; d < channels; ++d) {
                const auto v =
                    active[i]
                        ? quant.encode(values[i * channels + d], quant.low.size() == 1 ? 0 : d)
                        : 0;
                image[i * 4 + d] = std::uint8_t(v & 255);
                high[i * 4 + d] = std::uint8_t(v >> 8);
            }
        if (levels > 256) {
            enqueue(field + "_bytes_0.webp", std::move(image), side, side);
            enqueue(field + "_bytes_1.webp", std::move(high), side, side);
        } else
            enqueue(field + ".webp", std::move(image), side, side);
    };
    store("means", means, 3, 65536);
    store("scales", scales, 3, 256);
    store("quaternions", q, 4, 100);
    store("opacities", opacity, 1, 256);
    store("f_dc", dc, 3, 256);
    Bytes mask(pixels * 4, 255);
    for (std::size_t i = 0; i < pixels; ++i)
        mask[i * 4] = active[i] ? 255 : 0;
    enqueue("active_mask.webp", std::move(mask), side, side);
    if (dims) {
        const unsigned centroid_side =
            std::clamp(unsigned(std::sqrt(double(n) / 8)) / 8 * 8, 8U, 256U);
        const unsigned budget = std::min<unsigned>(unsigned(n), centroid_side * centroid_side);
        SogPalette palette;
#if defined(TINYTENSOR_HAS_VULKAN)
        if (n > 64)
            palette = cluster_sog_vulkan(ac, unsigned(n), dims, budget);
        else
#endif
            palette = cpu_palette(ac, unsigned(n), dims, budget);
        stage("sh_clustering_with_overlapped_webp", .8F);
        std::vector<float>().swap(ac);
        const auto k = palette.centroids.size() / dims;
        const unsigned cs = grid_side(k);
        std::vector<float> centroid_features(std::size_t(cs) * cs * dims);
        std::copy(palette.centroids.begin(), palette.centroids.end(), centroid_features.begin());
        const auto layout = sog_xt_layout(centroid_features, cs, dims);
        std::vector<unsigned> inverse(std::size_t(cs) * cs);
        for (unsigned i = 0; i < layout.size(); ++i)
            inverse[layout[i]] = i;
        Quantizer quant(palette.centroids, dims, 100);
        metadata["meta"]["f_rest_centroids_side"] = cs;
        auto ranges = quant.json();
        // Reference untile order is channel-major (R[15], G[15], B[15]).
        std::vector<float> low(45), high(45);
        for (unsigned c = 0; c < 3; ++c)
            for (unsigned b = 0; b < bases - 1; ++b) {
                low[c * 15 + b] = quant.low[b * 3 + c];
                high[c * 15 + b] = quant.high[b * 3 + c];
            }
        ranges["min_values"] = low;
        ranges["max_values"] = high;
        metadata["ops"].push_back(
            {{"input_fields", {"f_rest_centroids"}},
             {"transforms", {{{"simple_quantize", ranges}}}}});
        // Always use the paper's 3x5 tiling; unused SH bands are zero-padded.
        Bytes centers(std::size_t(cs) * 3 * cs * 5 * 4, 255), labels(pixels * 4, 255);
        for (unsigned i = 0; i < layout.size(); ++i)
            for (unsigned b = 0; b < 15; ++b)
                for (unsigned c = 0; c < 3; ++c) {
                    const auto pixel =
                        std::size_t((b / 5) * cs + i / cs) * (cs * 5) + (b % 5) * cs + i % cs;
                    centers[pixel * 4 + c] =
                        layout[i] < k && b < bases - 1
                            ? std::uint8_t(quant.encode(
                                  palette.centroids[std::size_t(layout[i]) * dims + b * 3 + c],
                                  b * 3 + c))
                            : 0;
                }
        // 3 rows x 5 columns, matching the reference decoder.
        enqueue("f_rest_centroids.webp", std::move(centers), cs * 5, cs * 3);
        for (std::size_t i = 0; i < pixels; ++i) {
            const unsigned label = active[i] ? inverse[palette.labels[order[i]]] : 0;
            labels[i * 4] = std::uint8_t(label % cs);
            labels[i * 4 + 1] = std::uint8_t(label / cs);
        }
        if (training) {
            training->center_side = cs;
            std::vector<int> ids(n), counts(std::size_t(cs) * cs);
            for (std::size_t i = 0; i < n; ++i) {
                const auto p = training->pixels[i];
                ids[i] = int(unsigned(labels[p * 4 + 1]) * cs + labels[p * 4]);
                ++counts[ids[i]];
            }
            training->labels = tinytensor::Tensor::from_vector(ids, {n}, model.means.device());
            training->counts =
                tinytensor::Tensor::from_vector(counts, {counts.size()}, model.means.device());
            training->sh_low = tinytensor::Tensor::from_vector(
                quant.low, {std::size_t(dims)}, model.means.device());
            training->sh_high = tinytensor::Tensor::from_vector(
                quant.high, {std::size_t(dims)}, model.means.device());
        }
        enqueue("f_rest_labels.webp", std::move(labels), side, side);
    }
    while (!pending.empty()) collect();
    stage("codebook_layout_and_webp", .95F);
    // JSON is a YAML 1.2 subset: paper-style metadata without a YAML dependency.
    const auto text = metadata.dump();
    files["container_meta.yaml"] = Bytes(text.begin(), text.end());
    if (training) {
        if (!dims)
            throw std::invalid_argument("SOG-XT adaptation requires non-DC SH bands");
        std::vector<float> low, high;
        for (const auto& field : {"means", "scales", "quaternions", "opacities", "f_dc"}) {
            const unsigned channels = std::string(field) == "quaternions" ? 4
                                      : std::string(field) == "opacities" ? 1
                                                                          : 3;
            for (const auto& op : metadata["ops"])
                if (op["input_fields"][0] == field) {
                    Quantizer quant(op["transforms"][0]["simple_quantize"]);
                    for (unsigned d = 0; d < channels; ++d) {
                        low.push_back(quant.low[quant.low.size() == 1 ? 0 : d]);
                        high.push_back(quant.high[quant.high.size() == 1 ? 0 : d]);
                    }
                }
        }
        training->low = tinytensor::Tensor::from_vector(low, {low.size()}, model.means.device());
        training->high = tinytensor::Tensor::from_vector(high, {high.size()}, model.means.device());
        std::vector<int> reverse(pixels, -1), neighbors(n * 2, -1);
        for (std::size_t i = 0; i < n; ++i)
            reverse[training->pixels[i]] = int(i);
        for (std::size_t i = 0; i < n; ++i) {
            const auto p = training->pixels[i];
            if (p % side + 1 < side)
                neighbors[i * 2] = reverse[p + 1];
            if (p + side < pixels)
                neighbors[i * 2 + 1] = reverse[p + side];
        }
        training->neighbors =
            tinytensor::Tensor::from_vector(neighbors, {n, 2}, model.means.device());
        training->files = files;
    }
    stage("metadata", .97F);
    core::Logger::instance().info("sog_xt_export encode_total_ms=",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return files;
}

SogXtFiles encode_sog_xt_frozen(const SogXtTrainingCodec& codec, const GaussianModel& decoded) {
    auto files = codec.files;
    const auto& bytes = files.at("container_meta.yaml");
    auto metadata = Json::parse(bytes.begin(), bytes.end());
    const auto n = decoded.size();
    const auto pixels = std::size_t(codec.side) * codec.side;
    const auto sh = decoded.sh.to_vector();
    const auto stride = decoded.sh.shape()[1] * 3;
    std::vector<float> dc(n * 3);
    for (std::size_t i = 0; i < n; ++i)
        std::copy_n(sh.data() + i * stride, 3, dc.data() + i * 3);
    auto means = decoded.means.to_vector();
    for (float& v : means)
        v = std::copysign(std::log1p(std::abs(v)), v);
    const std::array<std::vector<float>, 5> values{
        means, decoded.log_scales.to_vector(), decoded.quaternions.to_vector(),
        decoded.opacity_logits.to_vector(), dc};
    unsigned field_index = 0;
    for (const auto& field : {"means", "scales", "quaternions", "opacities", "f_dc"}) {
        const unsigned channels = std::string(field) == "quaternions" ? 4
                                  : std::string(field) == "opacities" ? 1
                                                                      : 3;
        for (const auto& op : metadata["ops"])
            if (op["input_fields"][0] == field) {
                Quantizer quant(op["transforms"][0]["simple_quantize"]);
                Bytes image(pixels * 4, 255), high(pixels * 4, 255);
                for (std::size_t i = 0; i < pixels; ++i)
                    for (unsigned d = 0; d < channels; ++d) {
                        image[i * 4 + d] = 0;
                        high[i * 4 + d] = 0;
                    }
                for (std::size_t i = 0; i < n; ++i)
                    for (unsigned d = 0; d < channels; ++d) {
                        const auto code = quant.encode(
                            values[field_index][i * channels + d], quant.low.size() == 1 ? 0 : d);
                        image[std::size_t(codec.pixels[i]) * 4 + d] = std::uint8_t(code & 255);
                        high[std::size_t(codec.pixels[i]) * 4 + d] = std::uint8_t(code >> 8);
                    }
                files
                    [std::string(field) +
                     (channels == 3 && std::string(field) == "means" ? "_bytes_0.webp" : ".webp")] =
                        write_image(image, codec.side, codec.side);
                if (std::string(field) == "means")
                    files["means_bytes_1.webp"] = write_image(high, codec.side, codec.side);
            }
        ++field_index;
    }
    const auto centers = codec.centers.to_vector(), lo = codec.sh_low.to_vector(),
               hi = codec.sh_high.to_vector();
    Bytes image(std::size_t(codec.center_side) * codec.center_side * 15 * 4, 255);
    for (unsigned i = 0; i < codec.center_side * codec.center_side; ++i)
        for (unsigned b = 0; b < 15; ++b)
            for (unsigned c = 0; c < 3; ++c) {
                const auto p = std::size_t((b / 5) * codec.center_side + i / codec.center_side) *
                                   (codec.center_side * 5) +
                               (b % 5) * codec.center_side + i % codec.center_side;
                unsigned code = 0;
                const unsigned d = b * 3 + c;
                if (d < codec.dimensions && hi[d] > lo[d])
                    code = unsigned(std::lround(
                        std::clamp(
                            (double(centers[std::size_t(i) * codec.dimensions + d]) - lo[d]) /
                                (hi[d] - lo[d]),
                            0.0, 1.0) *
                        99));
                image[p * 4 + c] = std::uint8_t(code);
            }
    files["f_rest_centroids.webp"] =
        write_image(image, codec.center_side * 5, codec.center_side * 3);
    // The paper's reference decoder normalizes by the observed code extrema.
    // Fixed training ranges can lose an endpoint as parameters move. Rebase
    // codes and ranges algebraically (no new quantization) so both readers
    // reconstruct the same values, even for a fully occupied image grid.
    for (auto& op : metadata["ops"]) {
        const auto field = op["input_fields"][0].get<std::string>();
        auto& config = op["transforms"][0]["simple_quantize"];
        const unsigned levels = config["levels"].get<unsigned>();
        const bool wide = field == "means";
        auto plane = read_image(files, field + (wide ? "_bytes_0.webp" : ".webp"));
        Image upper;
        if (wide)
            upper = read_image(files, "means_bytes_1.webp");
        const unsigned channels = field == "quaternions" ? 4 : field == "opacities" ? 1 : 3;
        unsigned first = 65535, last = 0;
        for (std::size_t p = 0; p < plane.rgba.size() / 4; ++p)
            for (unsigned d = 0; d < channels; ++d) {
                const unsigned code =
                    plane.rgba[p * 4 + d] | (wide ? unsigned(upper.rgba[p * 4 + d]) << 8 : 0);
                first = std::min(first, code);
                last = std::max(last, code);
            }
        if (first == 0 && last == levels - 1)
            continue;
        Quantizer old(config);
        for (unsigned d = 0; d < old.low.size(); ++d) {
            const double a = old.low[d], range = double(old.high[d]) - a;
            old.low[d] = float(a + range * first / (levels - 1));
            old.high[d] = float(a + range * last / (levels - 1));
        }
        old.levels = std::max(2U, last - first + 1);
        config = old.json();
        for (std::size_t p = 0; p < plane.rgba.size() / 4; ++p)
            for (unsigned d = 0; d < channels; ++d) {
                const unsigned code =
                    (plane.rgba[p * 4 + d] | (wide ? unsigned(upper.rgba[p * 4 + d]) << 8 : 0)) -
                    first;
                plane.rgba[p * 4 + d] = std::uint8_t(code & 255);
                if (wide)
                    upper.rgba[p * 4 + d] = std::uint8_t(code >> 8);
            }
        files[field + (wide ? "_bytes_0.webp" : ".webp")] =
            write_image(plane.rgba, plane.width, plane.height);
        if (wide)
            files["means_bytes_1.webp"] = write_image(upper.rgba, upper.width, upper.height);
    }
    const auto text = metadata.dump();
    files["container_meta.yaml"] = Bytes(text.begin(), text.end());
    return files;
}

void refresh_sog_xt_pras(SogXtTrainingCodec& codec) {
    const auto n = codec.latent.size(), pixels = std::size_t(codec.side) * codec.side;
    auto q = codec.latent.quaternions.to_vector(), s = codec.latent.log_scales.to_vector();
    std::vector<float> pq(pixels * 4), ps(pixels * 3);
    Bytes active(pixels);
    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Quaternionf rotation(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
        if (rotation.norm() < 1e-12F)
            rotation = Eigen::Quaternionf::Identity();
        else
            rotation.normalize();
        const auto p = codec.pixels[i];
        active[p] = 1;
        pq[p * 4] = rotation.w();
        pq[p * 4 + 1] = rotation.x();
        pq[p * 4 + 2] = rotation.y();
        pq[p * 4 + 3] = rotation.z();
        std::copy_n(s.data() + i * 3, 3, ps.data() + std::size_t(p) * 3);
    }
    sog_xt_pras(pq, ps, active, codec.side);
    for (std::size_t i = 0; i < n; ++i) {
        std::copy_n(pq.data() + std::size_t(codec.pixels[i]) * 4, 4, q.data() + i * 4);
        std::copy_n(ps.data() + std::size_t(codec.pixels[i]) * 3, 3, s.data() + i * 3);
    }
    codec.latent.quaternions = tinytensor::Tensor::from_vector(
        q, codec.latent.quaternions.shape(), codec.latent.means.device());
    codec.latent.log_scales = tinytensor::Tensor::from_vector(
        s, codec.latent.log_scales.shape(), codec.latent.means.device());
    auto low = codec.low.to_vector(), high = codec.high.to_vector();
    const auto& bytes = codec.files.at("container_meta.yaml");
    auto meta = Json::parse(bytes.begin(), bytes.end());
    for (auto& op : meta["ops"]) {
        const auto field = op["input_fields"][0].get<std::string>();
        if (field != "scales" && field != "quaternions")
            continue;
        const unsigned start = field == "scales" ? 3 : 6, channels = field == "scales" ? 3 : 4;
        Quantizer quant(field == "scales" ? s : q, channels, field == "scales" ? 256 : 100);
        for (unsigned d = 0; d < channels; ++d) {
            low[start + d] = quant.low[d];
            high[start + d] = quant.high[d];
        }
        op["transforms"][0]["simple_quantize"] = quant.json();
    }
    codec.low =
        tinytensor::Tensor::from_vector(low, codec.low.shape(), codec.latent.means.device());
    codec.high =
        tinytensor::Tensor::from_vector(high, codec.high.shape(), codec.latent.means.device());
    const auto text = meta.dump();
    codec.files["container_meta.yaml"] = Bytes(text.begin(), text.end());
}

DecodedSogXt decode_sog_xt(const SogXtFiles& files) {
    const auto it = files.find("container_meta.yaml");
    if (it == files.end())
        throw std::runtime_error("SOG-XT has no container_meta.yaml");
    if (it->second.size() > 1024 * 1024)
        throw std::runtime_error("SOG-XT metadata is too large");
    const auto metadata = yaml_json(YAML::Load(std::string(it->second.begin(), it->second.end())));
    const bool reference = !metadata.contains("format");
    const unsigned version = metadata.value("version", 0U);
    if (!reference && (metadata.value("format", "") != "SOG-XT" || (version != 1 && version != 2)))
        throw std::runtime_error("Unsupported SOG-XT metadata");
    const auto& meta = metadata.at("meta");
    const auto side = meta.at("grid_side").get<unsigned>();
    auto count = meta.value("count", std::size_t{0});
    const auto degree = meta.value("sh_degree", 3U);
    if (!side || side > WEBP_MAX_DIMENSION || degree > 3 || count > std::size_t(side) * side ||
        meta.at("image_codec") != "webp")
        throw std::runtime_error("Invalid SOG-XT dimensions");
    const auto pixels = std::size_t(side) * side;
    const unsigned bases = (degree + 1) * (degree + 1), dims = (bases - 1) * 3;
    const auto image = [&](const std::string& field) {
        auto value = read_image(files, field + ".webp");
        if (value.width != side || value.height != side)
            throw std::runtime_error("Inconsistent SOG-XT image dimensions");
        return value;
    };
    const auto mask = image("active_mask");
    std::size_t actual = 0;
    for (std::size_t i = 0; i < pixels; ++i)
        actual += mask.rgba[i * 4] != 0;
    if (reference && !meta.contains("count"))
        count = actual;
    if (!count || actual != count)
        throw std::runtime_error("SOG-XT count disagrees with active mask");
    const auto quantizer = [&](const std::string& field, unsigned channels) {
        for (const auto& op : metadata.at("ops")) {
            if (op.at("input_fields") != Json::array({field}))
                continue;
            for (const auto& t : op.at("transforms"))
                if (t.contains("simple_quantize")) {
                    Quantizer q(t.at("simple_quantize"));
                    if (q.low.size() != channels && q.low.size() != 1)
                        throw std::runtime_error("Invalid SOG-XT quantizer channels");
                    return q;
                }
        }
        throw std::runtime_error("Missing SOG-XT quantizer: " + field);
    };
    const auto read = [&](const std::string& field, unsigned channels, unsigned expected_levels) {
        const auto quant = quantizer(field, channels);
        if (!reference && quant.levels > expected_levels)
            throw std::runtime_error("Unsupported SOG-XT quantization precision");
        const bool wide = field == "means";
        const auto low = image(field + (wide ? "_bytes_0" : ""));
        Image high;
        if (wide)
            high = image(field + "_bytes_1");
        unsigned min_code = 65535, max_code = 0;
        const auto code = [&](std::size_t i, unsigned d) {
            return unsigned(low.rgba[i * 4 + d]) | (wide ? unsigned(high.rgba[i * 4 + d]) << 8 : 0);
        };
        if (reference)
            for (std::size_t i = 0; i < pixels; ++i)
                for (unsigned d = 0; d < channels; ++d) {
                    min_code = std::min(min_code, code(i, d));
                    max_code = std::max(max_code, code(i, d));
                }
        std::vector<float> values;
        values.reserve(count * channels);
        for (std::size_t i = 0; i < pixels; ++i)
            if (mask.rgba[i * 4])
                for (unsigned d = 0; d < channels; ++d) {
                    const unsigned v = code(i, d), channel = quant.low.size() == 1 ? 0 : d;
                    values.push_back(
                        reference ? float(
                                        double(quant.low[channel]) +
                                        (double(quant.high[channel]) - quant.low[channel]) *
                                            (v - min_code) / std::max(1U, max_code - min_code))
                                  : quant.decode(v, channel));
                }
        return values;
    };
    DecodedSogXt out;
    out.count = count;
    out.degree = degree;
    out.antialiased = meta.value("antialiased", false);
    out.means = read("means", 3, 65536);
    for (float& v : out.means)
        v = std::copysign(std::expm1(std::abs(v)), v);
    out.log_scales = read("scales", 3, 256);
    out.rotations = read("quaternions", 4, 100);
    for (std::size_t i = 0; i < count; ++i) {
        double norm = 0;
        for (unsigned d = 0; d < 4; ++d)
            norm += double(out.rotations[i * 4 + d]) * out.rotations[i * 4 + d];
        if (norm < 1e-12)
            throw std::runtime_error("Invalid SOG-XT quaternion");
        for (unsigned d = 0; d < 4; ++d)
            out.rotations[i * 4 + d] /= float(std::sqrt(norm));
    }
    out.opacity_logits = read("opacities", 1, 256);
    const auto dc = read("f_dc", 3, 256);
    out.sh.resize(count * bases * 3);
    for (std::size_t i = 0; i < count; ++i)
        std::copy_n(dc.data() + i * 3, 3, out.sh.data() + i * bases * 3);
    if (dims) {
        const auto cs = meta.at("f_rest_centroids_side").get<unsigned>();
        if (!cs || cs > 256)
            throw std::runtime_error("Invalid SOG-XT palette dimensions");
        const auto labels = image("f_rest_labels");
        const auto centers = read_image(files, "f_rest_centroids.webp");
        if (centers.width != cs * 5 || centers.height != cs * 3)
            throw std::runtime_error("Invalid SOG-XT centroid image");
        const bool channel_major = reference || version >= 2;
        const auto quant = quantizer("f_rest_centroids", channel_major ? 45 : dims);
        if (!reference && quant.levels > 100)
            throw std::runtime_error("Unsupported SOG-XT SH precision");
        unsigned min_code = 255, max_code = 0;
        if (reference)
            for (std::size_t i = 0; i < centers.rgba.size() / 4; ++i)
                for (unsigned c = 0; c < 3; ++c) {
                    min_code = std::min(min_code, unsigned(centers.rgba[i * 4 + c]));
                    max_code = std::max(max_code, unsigned(centers.rgba[i * 4 + c]));
                }
        std::size_t row = 0;
        for (std::size_t i = 0; i < pixels; ++i)
            if (mask.rgba[i * 4]) {
                const unsigned u = labels.rgba[i * 4], v = labels.rgba[i * 4 + 1];
                if (u >= cs || v >= cs)
                    throw std::runtime_error("SOG-XT label exceeds palette");
                for (unsigned b = 0; b < bases - 1; ++b)
                    for (unsigned c = 0; c < 3; ++c) {
                        const auto pixel =
                            std::size_t((b / 5) * cs + v) * (cs * 5) + (b % 5) * cs + u;
                        const auto channel = channel_major ? c * 15 + b : b * 3 + c;
                        const unsigned code = centers.rgba[pixel * 4 + c];
                        out.sh[(row * bases + b + 1) * 3 + c] =
                            reference
                                ? float(
                                      double(quant.low[channel]) +
                                      (double(quant.high[channel]) - quant.low[channel]) *
                                          (code - min_code) / std::max(1U, max_code - min_code))
                                : quant.decode(code, channel);
                    }
                ++row;
            }
    }
    return out;
}
} // namespace photara::splat::detail
