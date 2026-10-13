#include "../src/splat/deployment_vulkan.hpp"
#include "splat/deployment.hpp"
#include "splat/formats.hpp"
#include "splat/simplify.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <bit>
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace photara::splat;
using boost::property_tree::ptree;
void require(bool b, const char *message) {
    if (!b)
        throw std::runtime_error(message);
}
GaussianModel model(unsigned n) {
    std::vector<float> means(n * 3), logs(n * 3), q(n * 4), opacity(n, 0), sh(n * 3);
    for (unsigned i = 0; i < n; ++i) {
        means[i * 3] = 1.F + .11F * (i % 4);
        means[i * 3 + 1] = .5F + .09F * ((i / 4) % 4);
        means[i * 3 + 2] = .07F * (i / 16);
        if (n > 1) {
            means[i * 3] += .013F * std::sin(float(i) * 17.F);
            means[i * 3 + 1] += .009F * std::cos(float(i) * 13.F);
        }
        logs[i * 3] = std::log(.16F);
        logs[i * 3 + 1] = std::log(.06F);
        logs[i * 3 + 2] = std::log(.1F);
        auto rotation = Eigen::Quaternionf(Eigen::AngleAxisf(.7F, Eigen::Vector3f::UnitZ()));
        q[i * 4] = rotation.w();
        q[i * 4 + 1] = rotation.x();
        q[i * 4 + 2] = rotation.y();
        q[i * 4 + 3] = rotation.z();
        sh[i * 3] = .03F * i;
    }
    GaussianModel m;
    m.sh_degree = 0;
    m.means = tinytensor::Tensor::from_vector(means, {n, 3}, tinytensor::Device::CPU);
    m.log_scales = tinytensor::Tensor::from_vector(logs, {n, 3}, tinytensor::Device::CPU);
    m.quaternions = tinytensor::Tensor::from_vector(q, {n, 4}, tinytensor::Device::CPU);
    m.opacity_logits = tinytensor::Tensor::from_vector(opacity, {n, 1}, tinytensor::Device::CPU);
    m.sh = tinytensor::Tensor::from_vector(sh, {n, 1, 3}, tinytensor::Device::CPU);
    return m;
}
ptree read_json(const std::filesystem::path &p) {
    ptree t;
    boost::property_tree::read_json(p.string(), t);
    return t;
}
void verify_gpu_costs(const GaussianModel &m) {
#if defined(TINYTENSOR_HAS_VULKAN)
    const auto means = m.means.to_vector(), logs = m.log_scales.to_vector(),
               quats = m.quaternions.to_vector(), sh = m.sh.to_vector();
    std::vector<float> rows(m.size() * 16, 0);
    std::vector<Eigen::Matrix3f> covs;
    std::vector<std::int32_t> pairs;
    for (unsigned i = 0; i < m.size(); ++i) {
        Eigen::Quaternionf q(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
        Eigen::Vector3f s(std::exp(logs[i * 3]), std::exp(logs[i * 3 + 1]),
                          std::exp(logs[i * 3 + 2]));
        Eigen::Matrix3f rs = q.normalized().toRotationMatrix() * s.asDiagonal(),
                        c = rs * rs.transpose();
        covs.push_back(c);
        auto *r = rows.data() + 16 * i;
        std::copy_n(means.data() + i * 3, 3, r);
        r[3] = .5F * s.prod();
        r[4] = c(0, 0);
        r[5] = c(1, 1);
        r[6] = c(2, 2);
        r[7] = c(0, 1);
        r[8] = c(0, 2);
        r[9] = c(1, 2);
        r[10] = std::log(c.determinant());
        std::copy_n(sh.data() + i * 3, 3, r + 11);
        if (i) {
            pairs.push_back(i - 1);
            pairs.push_back(i);
        }
    }
    auto actual = photara::splat::detail::lod_costs_vulkan(rows, pairs);
    for (unsigned i = 1; i < m.size(); ++i) {
        const auto *a = rows.data() + 16 * (i - 1);
        const auto *b = rows.data() + 16 * i;
        Eigen::Vector3f pa(a[0], a[1], a[2]), pb(b[0], b[1], b[2]),
            mu = (a[3] * pa + b[3] * pb) / (a[3] + b[3]), da = pa - mu, db = pb - mu;
        Eigen::Matrix3f c =
            (a[3] * (covs[i - 1] + da * da.transpose()) + b[3] * (covs[i] + db * db.transpose())) /
            (a[3] + b[3]);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(c);
        Eigen::Vector3f ev = eig.eigenvalues().cwiseMax(1e-8F);
        Eigen::Matrix3f inverse =
            eig.eigenvectors() * ev.cwiseInverse().asDiagonal() * eig.eigenvectors().transpose();
        float ld = std::log(ev.prod());
        float ka = .5F * ((inverse * covs[i - 1]).trace() + da.dot(inverse * da) - 3 + ld - a[10]);
        float kb = .5F * ((inverse * covs[i]).trace() + db.dot(inverse * db) - 3 + ld - b[10]);
        float expected = a[3] * ka + b[3] * kb;
        for (unsigned k = 0; k < 3; ++k)
            expected += 8 * (a[11 + k] - b[11 + k]) * (a[11 + k] - b[11 + k]);
        if (std::abs(expected - actual[i - 1]) > 1e-5F + std::abs(expected) * 1e-4F) {
            std::cerr << "Merge cost mismatch: " << expected << " / " << actual[i - 1] << '\n';
            throw std::runtime_error("Vulkan merge cost differs from Eigen reference");
        }
    }
#else
    (void)m;
    throw std::runtime_error("Vulkan tests require the Vulkan build");
#endif
}
void verify_lod(const std::filesystem::path &folder, unsigned source_count) {
    auto meta = read_json(folder / "lod-meta.json");
    require(meta.get<unsigned>("version") == 1, "LOD version");
    std::vector<std::string> files;
    for (const auto &f : meta.get_child("filenames"))
        files.push_back(f.second.get_value<std::string>());
    unsigned leaf_count = 0, finest = 0;
    std::vector<bool> referenced(files.size());
    std::function<void(const ptree &)> visit = [&](const ptree &node) {
        auto children = node.get_child_optional("children");
        if (children) {
            require(children->size() == 2, "Binary tree child count");
            for (const auto &child : *children)
                visit(child.second);
            return;
        }
        ++leaf_count;
        unsigned previous = ~0u;
        for (const auto &entry : node.get_child("lods")) {
            auto file = entry.second.get<unsigned>("file"),
                 count = entry.second.get<unsigned>("count");
            require(file < files.size() && !referenced[file],
                    "Invalid or repeated chunk reference");
            referenced[file] = true;
            require(entry.second.get<unsigned>("offset") == 0, "Chunk offset");
            require(count <= previous && count > 0, "LOD count must decrease");
            previous = count;
            auto chunk =
                load_gaussians(folder / files[file], GaussianFormat::sog, tinytensor::Device::CPU);
            require(chunk.size() == count, "SOG chunk count disagrees");
            require(std::filesystem::exists((folder / files[file]).parent_path() / "means_l.webp"),
                    "Unbundled SOG textures missing");
            if (entry.first == "0")
                finest += count;
        }
    };
    visit(meta.get_child("tree"));
    require(leaf_count > 1 && finest == source_count, "LOD partition lost source splats");
    require(std::all_of(referenced.begin(), referenced.end(), [](bool b) { return b; }),
            "Unreferenced SOG chunk");
}
int main(int argc, char **argv) {
    try {
        bool gpu = argc > 1 && std::string(argv[1]) == "--vulkan";
        auto root = std::filesystem::temp_directory_path() /
                    ("photara-deployment-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        struct Cleanup {
            std::filesystem::path p;
            ~Cleanup() {
                std::error_code ec;
                std::filesystem::remove_all(p, ec);
            }
        } cleanup{root};
        auto source = model(32);
        if (gpu) verify_gpu_costs(source);
        LodOptions lod;
        lod.vulkan = gpu;
        lod.levels = 3;
        lod.chunk_count = 8;
        export_streamed_sog(source, root / "lod", lod);
        verify_lod(root / "lod", 32);
        if (gpu) {
            SimplifyOptions a;
            a.min_opacity = 0;
            a.keep_ratio = .5F;
            auto reference = simplify_gaussians(source, a);
            a.vulkan = true;
            auto actual = simplify_gaussians(source, a);
            require(reference.size() == actual.size() && actual.size() == 16, "GPU simplify count");
            auto x = reference.means.to_vector(), y = actual.means.to_vector();
            double sx = std::accumulate(x.begin(), x.end(), 0.0),
                   sy = std::accumulate(y.begin(), y.end(), 0.0);
            std::cout << "Simplify position sums: " << sx << " / " << sy << '\n';
            require(std::abs(sx - sy) < .05, "GPU simplify differs excessively");
        }
        bool rejected = false;
        try {
            export_streamed_sog(source, root / "lod", lod);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Existing output was overwritten");
        lod.cancelled = [] { return true; };
        rejected = false;
        try {
            export_streamed_sog(source, root / "cancelled", lod);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected && !std::filesystem::exists(root / "cancelled"),
                "Cancelled output published");
        std::cout << (gpu ? "Vulkan" : "CPU") << " deployment tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
