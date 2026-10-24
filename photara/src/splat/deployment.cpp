#include "splat/deployment.hpp"
#include "deployment_vulkan.hpp"
#include "io/mesh.hpp"
#include "splat/formats.hpp"
#include "splat/simplify.hpp"
#include "../mvs/marching_cubes_const.hpp"
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
    RenderMethod render_method{RenderMethod::gaussian_3dgs};
    std::size_t n{}, bases{};
    explicit Host(const GaussianModel &m)
        : degree(m.sh_degree), render_method(m.render_method), n(m.size()), bases((degree + 1) * (degree + 1)) {
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
        m.render_method = render_method;
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
std::size_t grid_index(const CollisionGrid &g, unsigned x, unsigned y, unsigned z) {
    return x + std::size_t(g.dimensions[0]) * (y + std::size_t(g.dimensions[1]) * z);
}
bool occupied(const CollisionGrid &g, int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 && unsigned(x) < g.dimensions[0] &&
           unsigned(y) < g.dimensions[1] && unsigned(z) < g.dimensions[2] &&
           g.solid[grid_index(g, x, y, z)];
}

void fill_grid(CollisionGrid &g, const CollisionOptions &o) {
    auto nx = g.dimensions[0], ny = g.dimensions[1], nz = g.dimensions[2];
    if (o.fill == CollisionOptions::Fill::ground) {
        for (unsigned z = 0; z < nz; ++z) {
            check(o);
            for (unsigned x = 0; x < nx; ++x)
                for (unsigned y = 0; y < ny; ++y)
                    if (occupied(g, x, y, z)) {
                        for (unsigned j = 0; j < y; ++j)
                            g.solid[grid_index(g, x, j, z)] = 1;
                        break;
                    }
        }
    } else if (o.fill == CollisionOptions::Fill::interior) {
        // Close enclosed cavities by boundary flood fill. Open shells are left open.
        std::vector<std::uint8_t> exterior(g.solid.size());
        std::vector<unsigned> queue;
        auto add = [&](unsigned x, unsigned y, unsigned z) {
            auto i = grid_index(g, x, y, z);
            if (!g.solid[i] && !exterior[i]) {
                exterior[i] = 1;
                queue.push_back(static_cast<unsigned>(i));
            }
        };
        for (unsigned z = 0; z < nz; ++z)
            for (unsigned y = 0; y < ny; ++y) {
                add(0, y, z);
                add(nx - 1, y, z);
            }
        for (unsigned z = 0; z < nz; ++z)
            for (unsigned x = 0; x < nx; ++x) {
                add(x, 0, z);
                add(x, ny - 1, z);
            }
        for (unsigned y = 0; y < ny; ++y)
            for (unsigned x = 0; x < nx; ++x) {
                add(x, y, 0);
                add(x, y, nz - 1);
            }
        for (std::size_t j = 0; j < queue.size(); ++j) {
            if (j % 65536 == 0)
                check(o);
            auto i = queue[j];
            unsigned x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
            if (x)
                add(x - 1, y, z);
            if (x + 1 < nx)
                add(x + 1, y, z);
            if (y)
                add(x, y - 1, z);
            if (y + 1 < ny)
                add(x, y + 1, z);
            if (z)
                add(x, y, z - 1);
            if (z + 1 < nz)
                add(x, y, z + 1);
        }
        for (std::size_t i = 0; i < g.solid.size(); ++i)
            if (!exterior[i])
                g.solid[i] = 1;
    }
}
void save_octree(const CollisionGrid &g, const std::filesystem::path &folder) {
    using BlockMap = std::map<std::array<unsigned, 3>, std::uint64_t>;
    BlockMap blocks;
    for (unsigned z = 0; z < g.dimensions[2]; ++z)
        for (unsigned y = 0; y < g.dimensions[1]; ++y)
            for (unsigned x = 0; x < g.dimensions[0]; ++x)
                if (occupied(g, x, y, z))
                    blocks[{x / 4, y / 4, z / 4}] |= std::uint64_t{1}
                                                     << ((x % 4) + 4 * (y % 4) + 16 * (z % 4));
    unsigned depth = 1;
    while ((1u << depth) < (std::max({g.dimensions[0], g.dimensions[1], g.dimensions[2]}) + 3) / 4)
        ++depth;
    std::vector<std::uint32_t> nodes, leaves;
    struct Region {
        std::array<unsigned, 3> at;
        unsigned size;
        std::vector<BlockMap::const_iterator> blocks;
    };
    std::vector<Region> queue;
    if (!blocks.empty()) {
        Region root{{0, 0, 0}, 1u << depth, {}};
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
            root.blocks.push_back(it);
        queue.push_back(std::move(root));
        nodes.push_back(0);
    }
    unsigned interior = 0, mixed = 0;
    for (std::size_t i = 0; i < queue.size(); ++i) {
        auto region = std::move(queue[i]);
        if (region.size == 1) {
            auto mask = region.blocks[0]->second;
            if (mask == ~std::uint64_t{0})
                nodes[i] = 0xFF000000u;
            else {
                nodes[i] = static_cast<std::uint32_t>(leaves.size() / 2);
                leaves.push_back(static_cast<std::uint32_t>(mask));
                leaves.push_back(static_cast<std::uint32_t>(mask >> 32));
                ++mixed;
            }
            continue;
        }
        std::array<Region, 8> children;
        auto half = region.size / 2;
        for (unsigned oct = 0; oct < 8; ++oct) {
            children[oct].size = half;
            for (unsigned axis = 0; axis < 3; ++axis)
                children[oct].at[axis] = region.at[axis] + ((oct >> axis) & 1) * half;
        }
        for (auto it : region.blocks) {
            unsigned oct = 0;
            for (unsigned axis = 0; axis < 3; ++axis)
                if (it->first[axis] >= region.at[axis] + half)
                    oct |= 1u << axis;
            children[oct].blocks.push_back(it);
        }
        if (nodes.size() >= 0x1000000u)
            throw std::runtime_error("Voxel octree exceeds format limit");
        unsigned mask = 0, first = static_cast<unsigned>(nodes.size());
        for (unsigned oct = 0; oct < 8; ++oct)
            if (!children[oct].blocks.empty()) {
                mask |= 1u << oct;
                queue.push_back(std::move(children[oct]));
                nodes.push_back(0);
            }
        nodes[i] = (mask << 24) | first;
        ++interior;
    }
    if (nodes.size() >= 0x1000000u || leaves.size() / 2 >= 0x1000000u)
        throw std::runtime_error("Voxel octree exceeds format limit");
    std::ofstream bin(folder / "scene.voxel.bin", std::ios::binary);
    for (const auto *words : {&nodes, &leaves})
        for (auto word : *words) {
            char bytes[4];
            for (unsigned b = 0; b < 4; ++b)
                bytes[b] = static_cast<char>(word >> (b * 8));
            bin.write(bytes, 4);
        }
    if (!bin)
        throw std::runtime_error("Cannot write voxel data");
    Bounds grid, scene;
    for (unsigned k = 0; k < 3; ++k) {
        grid.low[k] = g.origin[k];
        grid.high[k] = g.origin[k] + g.voxel_size * g.dimensions[k];
        scene.low[k] = g.scene_min[k];
        scene.high[k] = g.scene_max[k];
    }
    std::ofstream json(folder / "scene.voxel.json");
    json.imbue(std::locale::classic());
    json.precision(9);
    json << "{\"version\":\"" << (g.playcanvas_coordinates ? "1.1" : "1.0")
         << "\",\"asset\":{\"generator\":\"Photara C++\"},\"gridBounds\":";
    bound_json(json, grid);
    json << ",\"sceneBounds\":";
    bound_json(json, scene);
    json << ",\"voxelResolution\":" << g.voxel_size << ",\"leafSize\":4,\"treeDepth\":" << depth
         << ",\"numInteriorNodes\":" << interior << ",\"numMixedLeaves\":" << mixed
         << ",\"nodeCount\":" << nodes.size() << ",\"leafDataCount\":" << leaves.size() << '}';
    if (!json)
        throw std::runtime_error("Cannot write voxel metadata");
}
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
    const std::size_t total_units = std::max<std::size_t>(
        1, leaf_count * static_cast<std::size_t>(o.levels));
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
                    const std::size_t work_unit = done * o.levels + level;
                    options.progress = [&](float p) {
                        report(o, 0.05F + 0.9F *
                            (static_cast<float>(work_unit) + std::clamp(p, 0.F, 1.F)) /
                            static_cast<float>(total_units));
                    };
                    chunk = simplify_gaussians(chunk, options);
                }
                Host current(chunk);
                std::vector<unsigned> ids(current.n);
                std::iota(ids.begin(), ids.end(), 0u);
                auto b = bounds(current, ids);
                node.bound.unite(b);
                std::vector<unsigned> morton_keys(current.n);
                for (std::size_t i = 0; i < current.n; ++i)
                    morton_keys[i] = morton(current.mean(i), b);
                std::stable_sort(ids.begin(), ids.end(), [&](unsigned a, unsigned c) {
                    return morton_keys[a] < morton_keys[c];
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

CollisionGrid voxelize_gaussians(const GaussianModel &model, const CollisionOptions &o) {
    if (!std::isfinite(o.voxel_size) || o.voxel_size <= 0 || !std::isfinite(o.opacity) ||
        o.opacity <= 0 || o.opacity > 1 || o.max_voxels < 64)
        throw std::invalid_argument("Invalid collision settings");
    Host h(model);
    if (h.n > std::size_t(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("Collision model exceeds candidate index limit");
    std::vector<unsigned> ids(h.n);
    std::iota(ids.begin(), ids.end(), 0u);
    CollisionGrid g;
    g.voxel_size = o.voxel_size;
    g.playcanvas_coordinates = o.playcanvas_coordinates;
    auto b = bounds(h, ids);
    if (o.playcanvas_coordinates) {
        float lx = b.low.x(), ly = b.low.y();
        b.low.x() = -b.high.x();
        b.high.x() = -lx;
        b.low.y() = -b.high.y();
        b.high.y() = -ly;
    }
    std::size_t count = 1;
    for (unsigned k = 0; k < 3; ++k) {
        g.scene_min[k] = b.low[k];
        g.scene_max[k] = b.high[k];
        double block = 4.0 * o.voxel_size;
        double lo = std::floor(double(b.low[k]) / block) * block - block;
        double hi = std::ceil(double(b.high[k]) / block) * block + block;
        double dim = std::round((hi - lo) / o.voxel_size);
        if (!std::isfinite(dim) || dim > o.max_voxels || dim < 4)
            throw std::runtime_error("Collision grid too large; increase voxel size");
        g.origin[k] = static_cast<float>(lo);
        g.dimensions[k] = static_cast<unsigned>(dim);
        if (count > o.max_voxels / g.dimensions[k])
            throw std::runtime_error("Collision grid exceeds voxel budget; increase voxel size");
        count *= g.dimensions[k];
    }
    const auto nx = g.dimensions[0], ny = g.dimensions[1], nz = g.dimensions[2];
    unsigned bx = (nx + 7) / 8, by = (ny + 7) / 8, bz = (nz + 7) / 8;
    std::vector<std::vector<std::int32_t>> bins(std::size_t(bx) * by * bz);
    std::vector<float> values(h.n * 16, 0);
    Eigen::Matrix3f transform = Eigen::Matrix3f::Identity();
    if (o.playcanvas_coordinates) {
        transform(0, 0) = -1;
        transform(1, 1) = -1;
    }
    std::size_t references = 0;
    for (std::size_t i = 0; i < h.n; ++i) {
        if (i % 4096 == 0)
            check(o);
        V p = transform * h.mean(i);
        Eigen::Matrix3f r = transform * h.rotation(i);
        V s = h.scales(i);
        float *row = values.data() + i * 16;
        for (unsigned k = 0; k < 3; ++k)
            row[k] = p[k];
        row[3] = 1.F / (1.F + std::exp(-std::clamp(h.opacity[i], -80.F, 80.F)));
        if (row[3] < 1.F / 255.F)
            continue;
        Eigen::Matrix3f inverse = s.cwiseInverse().asDiagonal() * r.transpose();
        for (unsigned k = 0; k < 3; ++k)
            for (unsigned j = 0; j < 3; ++j)
                row[4 + k * 4 + j] = inverse(k, j);
        Eigen::Matrix3f rs = r * s.asDiagonal();
        V extent = 3.F * rs.cwiseProduct(rs).rowwise().sum().cwiseSqrt();
        unsigned low[3], high[3];
        const unsigned bd[] = {bx, by, bz};
        for (unsigned k = 0; k < 3; ++k) {
            low[k] = static_cast<unsigned>(
                std::clamp(std::floor((p[k] - extent[k] - g.origin[k]) / (8.F * o.voxel_size)), 0.F,
                           float(bd[k] - 1)));
            high[k] = static_cast<unsigned>(
                std::clamp(std::floor((p[k] + extent[k] - g.origin[k]) / (8.F * o.voxel_size)), 0.F,
                           float(bd[k] - 1)));
        }
        for (unsigned z = low[2]; z <= high[2]; ++z)
            for (unsigned y = low[1]; y <= high[1]; ++y)
                for (unsigned x = low[0]; x <= high[0]; ++x) {
                    if (++references > 64000000)
                        throw std::runtime_error(
                            "Collision candidate budget exceeded; increase voxel size");
                    bins[x + bx * (y + by * z)].push_back(static_cast<std::int32_t>(i));
                }
    }
    std::vector<std::int32_t> offsets{0}, indices;
    indices.reserve(references);
    for (const auto &bin : bins) {
        indices.insert(indices.end(), bin.begin(), bin.end());
        offsets.push_back(static_cast<std::int32_t>(indices.size()));
    }
    bins.clear();
    g.solid.resize(count);
    report(o, 0.15F);
    std::function<std::vector<float>(unsigned, unsigned)> gpu_query;
    if (o.vulkan) {
#if defined(TINYTENSOR_HAS_VULKAN)
        gpu_query = detail::prepare_voxel_field_vulkan(values, offsets, indices, g.dimensions,
                                                       g.origin, o.voxel_size, o.opacity);
#else
        throw std::runtime_error("Vulkan collision processing was not compiled");
#endif
    }
    constexpr unsigned batch = 262144;
    for (std::size_t start = 0; start < count; start += batch) {
        check(o);
        unsigned take = static_cast<unsigned>(std::min<std::size_t>(batch, count - start));
        std::vector<float> field;
        if (o.vulkan) {
            field = gpu_query(static_cast<unsigned>(start), take);
        } else {
            field.resize(take);
            for (unsigned j = 0; j < take; ++j) {
                unsigned index = static_cast<unsigned>(start) + j, x = index % nx,
                         y = (index / nx) % ny, z = index / (nx * ny),
                         bin = x / 8 + bx * (y / 8 + by * (z / 8));
                V p(g.origin[0] + (x + .5F) * o.voxel_size, g.origin[1] + (y + .5F) * o.voxel_size,
                    g.origin[2] + (z + .5F) * o.voxel_size);
                float t = 1;
                for (int ref = offsets[bin]; ref < offsets[bin + 1]; ++ref) {
                    const float *row = values.data() + indices[ref] * 16;
                    V d = p - V(row[0], row[1], row[2]);
                    V local(V(row[4], row[5], row[6]).dot(d), V(row[8], row[9], row[10]).dot(d),
                            V(row[12], row[13], row[14]).dot(d));
                    float q = local.squaredNorm();
                    if (q > 9)
                        continue;
                    float a = row[3] * std::exp(-.5F * q);
                    if (a < 1.F / 255.F)
                        continue;
                    t *= 1 - a;
                    if (1 - t >= o.opacity)
                        break;
                }
                field[j] = 1 - t;
            }
        }
        for (unsigned j = 0; j < take; ++j)
            g.solid[start + j] = field[j] >= o.opacity;
        report(o, .15F + .65F * float(start + take) / float(count));
    }
    fill_grid(g, o);
    report(o, .85F);
    return g;
}

namespace {
mvs::Mesh collision_mesh_smooth(const CollisionGrid &g) {
    if (g.solid.size() != std::size_t(g.dimensions[0]) * g.dimensions[1] * g.dimensions[2] ||
        g.voxel_size <= 0)
        throw std::invalid_argument("Invalid collision grid");
    if (std::any_of(g.dimensions.begin(), g.dimensions.end(), [](const unsigned dimension) {
            return dimension > static_cast<unsigned>(std::numeric_limits<int>::max() - 1);
        }))
        throw std::invalid_argument("Collision grid dimension exceeds marching-cubes limits");
    const int nx = static_cast<int>(g.dimensions[0]);
    const int ny = static_cast<int>(g.dimensions[1]);
    const int nz = static_cast<int>(g.dimensions[2]);
    const auto occupied = [&](const int x, const int y, const int z) {
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return false;
        return g.solid[static_cast<std::size_t>(x) +
                       std::size_t(nx) * (static_cast<std::size_t>(y) +
                       std::size_t(ny) * static_cast<std::size_t>(z))] != 0;
    };
    mvs::Mesh mesh;
    std::unordered_map<std::uint64_t, int> edge_vertices;
    const std::size_t surface_hint = std::min<std::size_t>(g.solid.size(), 1U << 20U);
    edge_vertices.reserve(surface_hint);
    auto vertex = [&](const int x, const int y, const int z, const int axis) {
        const std::uint64_t linear =
            (static_cast<std::uint64_t>(x + 1) * (static_cast<std::uint64_t>(ny) + 2U) +
             static_cast<std::uint64_t>(y + 1)) * (static_cast<std::uint64_t>(nz) + 2U) +
            static_cast<std::uint64_t>(z + 1);
        const std::uint64_t key = linear * 3U + static_cast<unsigned>(axis);
        if (const auto found = edge_vertices.find(key); found != edge_vertices.end())
            return found->second;
        if (mesh.vertices.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Collision mesh exceeds 32-bit index limit");
        Eigen::Vector3f p(
            g.origin[0] + static_cast<float>(x) * g.voxel_size,
            g.origin[1] + static_cast<float>(y) * g.voxel_size,
            g.origin[2] + static_cast<float>(z) * g.voxel_size);
        p[axis] += 0.5F * g.voxel_size;
        const int id = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back(p);
        edge_vertices.emplace(key, id);
        return id;
    };

    // Like splat-transform's smooth default, voxel occupancy is sampled as a
    // binary scalar field and the surface is extracted at the 0.5 isovalue.
    // The one-sample empty border closes surfaces that touch the grid bounds.
    for (int z = -1; z < nz; ++z)
        for (int y = -1; y < ny; ++y)
            for (int x = -1; x < nx; ++x) {
                int cube = 0;
                for (int corner = 0; corner < 8; ++corner)
                    if (occupied(x + shift[corner].x(), y + shift[corner].y(),
                                 z + shift[corner].z()))
                        cube |= 1 << corner;
                if (cube == 0 || cube == 255) continue;
                std::array<int, 12> edge_ids;
                edge_ids.fill(-1);
                for (int edge = 0; edge < 12; ++edge) {
                    if ((edge_table[cube] & (1 << edge)) == 0) continue;
                    edge_ids[static_cast<std::size_t>(edge)] = vertex(
                        x + edge_shift[edge].x(), y + edge_shift[edge].y(),
                        z + edge_shift[edge].z(), edge_shift[edge].w());
                }
                for (int tri = 0; tri_table[cube][tri] >= 0; tri += 3)
                    mesh.faces.emplace_back(
                        edge_ids[static_cast<std::size_t>(tri_table[cube][tri])],
                        edge_ids[static_cast<std::size_t>(tri_table[cube][tri + 1])],
                        edge_ids[static_cast<std::size_t>(tri_table[cube][tri + 2])]);
            }
    return mesh;
}

mvs::Mesh collision_mesh_faces(const CollisionGrid &g) {
    if (g.solid.size() != std::size_t(g.dimensions[0]) * g.dimensions[1] * g.dimensions[2] ||
        g.voxel_size <= 0)
        throw std::invalid_argument("Invalid collision grid");
    mvs::Mesh mesh;
    std::unordered_map<std::uint64_t, int> corners;
    auto vertex = [&](unsigned x, unsigned y, unsigned z) {
        std::uint64_t key =
            x + std::uint64_t(g.dimensions[0] + 1) * (y + std::uint64_t(g.dimensions[1] + 1) * z);
        auto found = corners.find(key);
        if (found != corners.end())
            return found->second;
        if (mesh.vertices.size() >= 50000000)
            throw std::runtime_error("Collision mesh vertex budget exceeded");
        int id = static_cast<int>(mesh.vertices.size());
        corners.emplace(key, id);
        mesh.vertices.emplace_back(g.origin[0] + x * g.voxel_size, g.origin[1] + y * g.voxel_size,
                                   g.origin[2] + z * g.voxel_size);
        return id;
    };
    // Faces along each axis; cyclic axes form a right-handed basis.
    for (unsigned z = 0; z < g.dimensions[2]; ++z)
        for (unsigned y = 0; y < g.dimensions[1]; ++y)
            for (unsigned x = 0; x < g.dimensions[0]; ++x)
                if (occupied(g, x, y, z)) {
                    for (unsigned axis = 0; axis < 3; ++axis)
                        for (int sign : {-1, 1}) {
                            int neighbor[] = {int(x), int(y), int(z)};
                            neighbor[axis] += sign;
                            if (occupied(g, neighbor[0], neighbor[1], neighbor[2]))
                                continue;
                            unsigned at[] = {x, y, z};
                            if (sign > 0)
                                ++at[axis];
                            unsigned u = (axis + 1) % 3, v = (axis + 2) % 3;
                            int ids[4];
                            for (unsigned k = 0; k < 4; ++k) {
                                unsigned p[] = {at[0], at[1], at[2]};
                                p[u] += (k == 1 || k == 2);
                                p[v] += (k >= 2);
                                ids[k] = vertex(p[0], p[1], p[2]);
                            }
                            if (sign > 0) {
                                mesh.faces.emplace_back(ids[0], ids[1], ids[2]);
                                mesh.faces.emplace_back(ids[0], ids[2], ids[3]);
                            } else {
                                mesh.faces.emplace_back(ids[0], ids[2], ids[1]);
                                mesh.faces.emplace_back(ids[0], ids[3], ids[2]);
                            }
                        }
                }
    return mesh;
}
} // namespace

mvs::Mesh collision_mesh(const CollisionGrid &g) {
    return collision_mesh_faces(g);
}

mvs::Mesh collision_mesh(const CollisionGrid &g, const CollisionOptions::MeshShape shape) {
    return shape == CollisionOptions::MeshShape::smooth
        ? collision_mesh_smooth(g)
        : collision_mesh_faces(g);
}

void export_collision(const GaussianModel &model, const std::filesystem::path &folder,
                      const CollisionOptions &o) {
    Staging staging(folder);
    auto grid = voxelize_gaussians(model, o);
    check(o);
    save_octree(grid, staging.work);
    auto mesh = collision_mesh(grid, o.mesh_shape);
    check(o);
    if (mesh.faces.empty())
        throw std::runtime_error("No collision surface; lower the opacity threshold or voxel size");
    io::save_mesh_glb(mesh, staging.work / "scene.collision.glb");
    check(o);
    staging.publish();
    if (o.progress)
        o.progress(1);
}
} // namespace photara::splat
