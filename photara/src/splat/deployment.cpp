#include "splat/deployment.hpp"
#include "deployment_vulkan.hpp"
#include "splat/formats.hpp"
#include "splat/simplify.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace photara::splat {
namespace {
using V = Eigen::Vector3f;
void check(const ProcessingControl &c) {
    if (c.cancelled && c.cancelled())
        throw std::runtime_error("Splat processing cancelled");
}
void report(const ProcessingControl &c, float p) {
    check(c);
    if (c.progress)
        c.progress(p);
}
struct Staging {
    std::filesystem::path dest, work;
    explicit Staging(const std::filesystem::path &path) {
        if (path.empty())
            throw std::invalid_argument("Output folder is empty");
        dest = std::filesystem::absolute(path);
        if (std::filesystem::exists(dest))
            throw std::runtime_error("Choose a new output folder");
        std::filesystem::create_directories(dest.parent_path());
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0;; ++i) {
            work = dest.parent_path() /
                   (".photara-splat-" + std::to_string(stamp) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(work))
                break;
        }
    }
    ~Staging() {
        std::error_code ec;
        std::filesystem::remove_all(work, ec);
    }
    void publish() {
        if (std::filesystem::exists(dest))
            throw std::runtime_error("Output folder already exists");
        std::filesystem::rename(work, dest);
    }
};
struct Host {
    std::vector<float> means, logs, quats, opacity, sh;
    unsigned degree{};
    std::size_t n{}, bases{};
    explicit Host(const GaussianModel &m)
        : degree(m.sh_degree), n(m.size()), bases((degree + 1) * (degree + 1)) {
        means = m.means.to_vector();
        logs = m.log_scales.to_vector();
        quats = m.quaternions.to_vector();
        opacity = m.opacity_logits.to_vector();
        sh = m.sh.to_vector();
        if (!n || degree > 3 || means.size() != 3 * n || logs.size() != 3 * n ||
            quats.size() != 4 * n || opacity.size() != n || sh.size() != n * bases * 3)
            throw std::invalid_argument("Invalid Gaussian model shape");
        for (const auto *values : {&means, &logs, &quats, &opacity, &sh})
            for (float value : *values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("Model contains non-finite values");
        for (std::size_t i = 0; i < n; ++i) {
            Eigen::Quaternionf q(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2],
                                 quats[4 * i + 3]);
            if (q.norm() < 1e-12F)
                throw std::invalid_argument("Zero-norm Gaussian quaternion");
            for (unsigned k = 0; k < 3; ++k) {
                float scale = std::exp(logs[i * 3 + k]);
                if (!std::isfinite(scale) || scale <= 0)
                    throw std::invalid_argument("Invalid Gaussian scale");
            }
        }
    }
    V mean(std::size_t i) const { return V(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]); }
    Eigen::Matrix3f rotation(std::size_t i) const {
        return Eigen::Quaternionf(quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2],
                                  quats[i * 4 + 3])
            .normalized()
            .toRotationMatrix();
    }
    V scales(std::size_t i) const {
        return V(std::exp(logs[i * 3]), std::exp(logs[i * 3 + 1]), std::exp(logs[i * 3 + 2]));
    }
    GaussianModel gather(const std::vector<unsigned> &ids) const {
        auto tensor = [&](const std::vector<float> &src, std::size_t stride,
                          tinytensor::TensorShape shape) {
            std::vector<float> out(ids.size() * stride);
            for (std::size_t j = 0; j < ids.size(); ++j)
                std::copy_n(src.data() + ids[j] * stride, stride, out.data() + j * stride);
            return tinytensor::Tensor::from_vector(out, shape, tinytensor::Device::CPU);
        };
        GaussianModel m;
        m.sh_degree = degree;
        m.means = tensor(means, 3, {ids.size(), 3});
        m.log_scales = tensor(logs, 3, {ids.size(), 3});
        m.quaternions = tensor(quats, 4, {ids.size(), 4});
        m.opacity_logits = tensor(opacity, 1, {ids.size(), 1});
        m.sh = tensor(sh, bases * 3, {ids.size(), bases, 3});
        return m;
    }
};
struct Bounds {
    V low = V::Constant(std::numeric_limits<float>::infinity());
    V high = V::Constant(-std::numeric_limits<float>::infinity());
    void add(V p, V extent) {
        low = low.cwiseMin(p - extent);
        high = high.cwiseMax(p + extent);
    }
    void unite(const Bounds &b) {
        low = low.cwiseMin(b.low);
        high = high.cwiseMax(b.high);
    }
};
Bounds bounds(const Host &h, const std::vector<unsigned> &ids) {
    Bounds b;
    for (auto i : ids) {
        // sqrt(diag(R diag(s^2) R^T)) gives tight rotated 3-sigma AABB.
        const Eigen::Matrix3f rs = h.rotation(i) * h.scales(i).asDiagonal();
        V extent = 3.F * rs.cwiseProduct(rs).rowwise().sum().cwiseSqrt();
        if (!extent.allFinite())
            throw std::invalid_argument("Gaussian bounds overflow");
        b.add(h.mean(i), extent);
    }
    return b;
}
void vector_json(std::ostream &out, const V &v) {
    out << '[' << v.x() << ',' << v.y() << ',' << v.z() << ']';
}
void bound_json(std::ostream &out, const Bounds &b) {
    out << "{\"min\":";
    vector_json(out, b.low);
    out << ",\"max\":";
    vector_json(out, b.high);
    out << '}';
}
unsigned morton(V p, const Bounds &b) {
    V relative = (p - b.low).cwiseQuotient((b.high - b.low).cwiseMax(1e-8F));
    unsigned key = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
        unsigned q = static_cast<unsigned>(std::clamp(relative[axis], 0.F, 1.F) * 1023.F);
        for (unsigned bit = 0; bit < 10; ++bit)
            key |= ((q >> bit) & 1) << (3 * bit + axis);
    }
    return key;
}
struct LodNode {
    Bounds bound;
    int left{-1}, right{-1};
    std::vector<unsigned> ids;
    std::vector<std::pair<unsigned, unsigned>> lods;
};
} // namespace

void export_streamed_sog(const GaussianModel &model, const std::filesystem::path &folder,
                         const LodOptions &o) {
    if (o.levels < 2 || o.levels > 8 || !std::isfinite(o.keep_ratio) || o.keep_ratio <= 0 ||
        o.keep_ratio >= 1 || o.chunk_count < 1 || !std::isfinite(o.chunk_extent) ||
        o.chunk_extent <= 0)
        throw std::invalid_argument("Invalid LOD settings");
    Host h(model);
    if (h.n > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Model too large");
    Staging staging(folder);
    report(o, 0);
    std::vector<LodNode> tree(1);
    tree[0].ids.resize(h.n);
    std::iota(tree[0].ids.begin(), tree[0].ids.end(), 0u);
    for (std::size_t n = 0; n < tree.size(); ++n) {
        check(o);
        tree[n].bound = bounds(h, tree[n].ids);
        auto &node = tree[n];
        Eigen::Index axis;
        float extent = (node.bound.high - node.bound.low).maxCoeff(&axis);
        if (node.ids.size() > 1 && (node.ids.size() > o.chunk_count ||
                                    (extent > o.chunk_extent && node.ids.size() > o.chunk_min))) {
            auto ids = std::move(node.ids);
            auto middle = ids.begin() + ids.size() / 2;
            std::nth_element(ids.begin(), middle, ids.end(), [&](unsigned a, unsigned b) {
                return h.means[a * 3 + axis] < h.means[b * 3 + axis];
            });
            LodNode left, right;
            left.ids.assign(ids.begin(), middle);
            right.ids.assign(middle, ids.end());
            node.left = static_cast<int>(tree.size());
            node.right = node.left + 1;
            tree.push_back(std::move(left));
            tree.push_back(std::move(right));
        }
    }
    std::vector<std::string> filenames;
    std::vector<std::size_t> counts(o.levels, 0);
    std::size_t leaf_count = 0, done = 0;
    for (const auto &node : tree)
        if (node.left < 0)
            ++leaf_count;
    for (auto &node : tree)
        if (node.left < 0) {
            auto chunk = h.gather(node.ids);
            const auto original = chunk.size();
            for (unsigned level = 0; level < o.levels; ++level) {
                check(o);
                if (level && chunk.size() > 1) {
                    SimplifyOptions options;
                    options.keep_ratio = o.keep_ratio;
                    options.min_opacity = 0;
                    options.vulkan = o.vulkan;
                    options.cancelled = o.cancelled;
                    chunk = simplify_gaussians(chunk, options);
                }
                Host current(chunk);
                std::vector<unsigned> ids(current.n);
                std::iota(ids.begin(), ids.end(), 0u);
                auto b = bounds(current, ids);
                node.bound.unite(b);
                std::stable_sort(ids.begin(), ids.end(), [&](unsigned a, unsigned c) {
                    return morton(current.mean(a), b) < morton(current.mean(c), b);
                });
                auto ordered = current.gather(ids);
                const auto name = std::to_string(level) + "_" + std::to_string(done);
                node.lods.emplace_back(static_cast<unsigned>(filenames.size()),
                                       static_cast<unsigned>(ordered.size()));
                counts[level] += ordered.size();
                filenames.push_back(name + "/meta.json");
                save_sog_unbundled(ordered, staging.work / name);
                report(o, 0.05F + 0.9F * float(done * o.levels + level + 1) /
                                      float(leaf_count * o.levels));
            }
            ++done;
            node.ids.clear();
            (void)original;
        }
    for (std::size_t n = tree.size(); n-- > 0;)
        if (tree[n].left >= 0) {
            tree[n].bound = tree[tree[n].left].bound;
            tree[n].bound.unite(tree[tree[n].right].bound);
        }
    std::ofstream json(staging.work / "lod-meta.json");
    json.imbue(std::locale::classic());
    json.precision(9);
    json << "{\"version\":1,\"asset\":{\"generator\":\"Photara C++\"},\"count\":"
         << std::accumulate(counts.begin(), counts.end(), std::size_t{}) << ",\"counts\":[";
    for (unsigned i = 0; i < o.levels; ++i) {
        if (i)
            json << ',';
        json << counts[i];
    }
    json << "],\"lodLevels\":" << o.levels << ",\"filenames\":[";
    for (std::size_t i = 0; i < filenames.size(); ++i) {
        if (i)
            json << ',';
        json << '\"' << filenames[i] << '\"';
    }
    json << "],\"tree\":";
    std::function<void(int)> write_node = [&](int n) {
        const auto &node = tree[n];
        json << "{\"bound\":";
        bound_json(json, node.bound);
        if (node.left >= 0) {
            json << ",\"children\":[";
            write_node(node.left);
            json << ',';
            write_node(node.right);
            json << ']';
        } else {
            json << ",\"lods\":{";
            for (unsigned level = 0; level < o.levels; ++level) {
                if (level)
                    json << ',';
                json << '\"' << level << "\":{\"file\":" << node.lods[level].first
                     << ",\"offset\":0,\"count\":" << node.lods[level].second << '}';
            }
            json << '}';
        }
        json << '}';
    };
    write_node(0);
    json << '}';
    json.close();
    if (!json)
        throw std::runtime_error("Cannot write LOD metadata");
    check(o);
    staging.publish();
    if (o.progress)
        o.progress(1);
}

} // namespace photara::splat
