#include "splat/visualize.hpp"
#include "splat/trainer.hpp"
#include "core/camera_projection.hpp"
#include "io/image.hpp"
#include "../src/splat/adc_backends.hpp"
#include "../src/splat/geometry_vulkan.hpp"
#include "vulkan/backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace photara::splat;
using tinytensor::Tensor;
constexpr auto gpu = tinytensor::Device::Vulkan;
void require(bool b, const char* message) {
    if (!b)
        throw std::runtime_error(message);
}
Camera camera() {
    Camera c;
    c.width = 32;
    c.height = 24;
    c.fx = c.fy = 15;
    c.cx = 15.37F;
    c.cy = 11.63F;
    c.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    return c;
}
GaussianModel model(unsigned n) {
    GaussianModel m;
    m.sh_degree = 0;
    std::vector<float> means(n * 3), sh(n * 3), q(n * 4);
    for (unsigned i = 0; i < n; ++i) {
        means[i * 3] = .07F * float(int(i % 7) - 3);
        means[i * 3 + 1] = .09F * float(int(i % 5) - 2);
        means[i * 3 + 2] = 2 + .1F * (i % 3);
        if (i % 11 == 0)
            means[i * 3] = 50;
        if (i % 13 == 0)
            means[i * 3 + 2] = -1;
        q[4 * i] = 1;
        sh[3 * i] = .4F;
        sh[3 * i + 1] = -.3F;
        sh[3 * i + 2] = .2F;
    }
    m.means = Tensor::from_vector(means, {n, 3}, gpu);
    m.sh = Tensor::from_vector(sh, {n, 1, 3}, gpu);
    m.quaternions = Tensor::from_vector(q, {n, 4}, gpu);
    m.log_scales = Tensor::full({n, 3}, -2.F, gpu);
    m.opacity_logits = Tensor::full({n, 1}, -4.F, gpu);
    return m;
}
void points_test() {
    auto m = model(1025);
    auto positions = m.means.to_vector();
    VisualizeOptions o;
    o.mode = VisualizationMode::points;
    o.point_size_px = 1.4F;
    o.background = {.1F, .2F, .3F};
    auto c = camera();
    for (unsigned mode : {0u, 1u, 3u, 4u}) {
        c.orthographic = mode == 4;
        c.model = static_cast<photara::CameraModel>(mode == 4 ? 0 : mode);
        c.k1 = .03F;
        auto actual = visualize(m, c, o).to_vector();
        std::vector<unsigned> hits(c.width * c.height);
        for (unsigned i = 0; i < m.size(); ++i) {
            double x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2], u{},
                   v{};
            bool valid = true;
            if (mode == 3) {
                auto p = photara::project_equirectangular_camera(x, y, z, c.width, c.height);
                u = p.u;
                v = p.v;
                valid = p.valid;
            } else if (mode == 1) {
                auto p = photara::project_fisheye_camera(
                    x, y, z, c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.k4);
                u = p.u;
                v = p.v;
                valid = p.valid;
            } else {
                valid = z > o.rasterization.near_plane;
                u = c.fx * x / (mode == 4 ? 1 : z) + c.cx;
                v = c.fy * y / (mode == 4 ? 1 : z) + c.cy;
            }
            if (!valid)
                continue;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    if (dx * dx + dy * dy > o.point_size_px * o.point_size_px)
                        continue;
                    int xx = int(std::floor(u)) + dx, yy = int(std::floor(v)) + dy;
                    if (xx >= 0 && yy >= 0 && xx < int(c.width) && yy < int(c.height))
                        ++hits[yy * c.width + xx];
                }
        }
        for (unsigned channel = 0; channel < 3; ++channel)
            for (unsigned i = 0; i < hits.size(); ++i) {
                const float coeff[] = {.4F, -.3F, .2F};
                float expected =
                    hits[i] ? .5F + .28209479177387814F * coeff[channel] : o.background[channel];
                if (std::abs(actual[channel * hits.size() + i] - expected) >= 3e-5F) {
                    std::cerr << "points mode=" << mode << " pixel=" << i << " channel=" << channel
                              << " hits=" << hits[i] << " expected=" << expected
                              << " actual=" << actual[channel * hits.size() + i] << '\n';
                    throw std::runtime_error(
                        "Vulkan points pixels differ from CPU projection/compositing");
                }
            }
    }
    // The optical axis has known SH basis values for each degree.
    m = model(1);
    m.means = Tensor::from_vector({0.F, 0.F, 2.F}, {1, 3}, gpu);
    m.sh_degree = 3;
    std::vector<float> coeff(48);
    coeff[0] = .1F;
    coeff[6] = .1F;
    coeff[18] = .1F;
    coeff[36] = .1F;
    m.sh = Tensor::from_vector(coeff, {1, 16, 3}, gpu);
    c = camera();
    const float basis[] = {
        .28209479177387814F, .4886025119029199F, .6307831305050401F, .7463526651802308F};
    float expected = .5F;
    for (unsigned degree = 0; degree <= 3; ++degree) {
        o.active_sh_degree = degree;
        expected += .1F * basis[degree];
        auto image = visualize(m, c, o).to_vector();
        require(std::abs(image[11 * 32 + 15] - expected) < 1e-6F, "Points SH degree mismatch");
    }
    std::cout
        << "Vulkan points: pinhole, fisheye, panorama, orthographic, overlap and SH0..3 passed\n";
}
std::uint32_t hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}
float normal(unsigned i, unsigned seed, unsigned axis) {
    float u1 = (hash(i * 3 + axis + seed * 17) + 1.F) / 4294967297.F,
          u2 = (hash(i * 7 + axis + seed * 29) + 1.F) / 4294967297.F;
    return std::sqrt(-2 * std::log(std::max(u1, 1e-12F))) * std::cos(6.283185307179586F * u2);
}
void noise_test() {
    auto m = model(257);
    const auto before = m.means.to_vector();
    std::vector<float> scales(257 * 3), quats(257 * 4), opacity(257), vis(257, 1.F);
    std::vector<int> radii(257, 2);
    for (unsigned i = 0; i < 257; ++i) {
        scales[i * 3] = std::log(.1F);
        scales[i * 3 + 1] = std::log(.2F);
        scales[i * 3 + 2] = std::log(.3F);
        quats[i * 4] = i % 2 ? std::sqrt(.5F) : 1;
        quats[i * 4 + 3] = i % 2 ? std::sqrt(.5F) : 0;
        opacity[i] = i % 5 == 0 ? -20.F : -4.F;
        if (i % 7 == 0)
            vis[i] = 0;
        if (i % 11 == 0)
            radii[i] = 0;
    }
    m.log_scales = Tensor::from_vector(scales, {257, 3}, gpu);
    m.quaternions = Tensor::from_vector(quats, {257, 4}, gpu);
    m.opacity_logits = Tensor::from_vector(opacity, {257, 1}, gpu);
    auto v = Tensor::from_vector(vis, {257}, gpu);
    auto r = Tensor::from_vector(radii, {257}, gpu);
    detail::inject_adc_noise(m, v, 10.F, 0.F, 42, r, true);
    auto after = m.means.to_vector();
    for (unsigned i = 0; i < 257; ++i) {
        float a = 1 / (1 + std::exp(-opacity[i]));
        float rate = std::min(
            .05F * 10 * std::sqrt(2 * std::log(std::max(255 * a, 1.F))) * std::pow(1 - a, 150.F),
            1.F);
        float x = .1F * rate * normal(i, 42, 0), y = .2F * rate * normal(i, 42, 1),
              z = .3F * rate * normal(i, 42, 2);
        std::array<float, 3> delta =
            i % 2 ? std::array<float, 3>{-y, x, z} : std::array<float, 3>{x, y, z};
        for (unsigned d = 0; d < 3; ++d) {
            float expected = before[i * 3 + d] + (vis[i] > 0 && radii[i] > 0 ? delta[d] : 0);
            require(
                std::abs(after[i * 3 + d] - expected) < 3e-6F,
                "Revised ADC-IGS noise differs from CPU oracle");
        }
    }
    std::cout << "ADC-IGS revised noise: CPU oracle and visibility/radius gates passed\n";
}
void supervision_test() {
    auto c = camera();
    const unsigned n = c.width * c.height;
    RenderResult rendered;
    rendered.color = Tensor::zeros({3, c.height, c.width}, gpu);
    rendered.median_depth = Tensor::full({c.height, c.width}, 3.F, gpu);
    rendered.normal = Tensor::zeros({3, c.height, c.width}, gpu);
    TrainingView target;
    target.camera = c;
    target.has_mask = true;
    std::vector<float> mask(n, 1.F), depth(n, 2.F), normal(n * 3);
    for (unsigned i = 0; i < n; ++i) {
        mask[i] = i % 3 == 0 ? 0.F : i % 3 == 1 ? .25F : 1.F;
        normal[2 * n + i] = 1.F;
        if (i % 5 == 0)
            depth[i] = 0;
    }
    target.mask = Tensor::from_vector(mask, {c.height, c.width}, gpu);
    target.depth = Tensor::from_vector(depth, {c.height, c.width}, gpu);
    target.normal = Tensor::from_vector(normal, {3, c.height, c.width}, gpu);
    TrainingOptions o;
    o.use_mvs_depth = o.use_mvs_normals = true;
    o.use_mask = true;
    o.depth_weight = .2F;
    o.normal_weight = .3F;
    detail::LossGradients g;
    detail::add_mvs_loss_vulkan(rendered, target, o, g, true);
    auto gd = g.depth.to_vector(), gn = g.normal.to_vector();
    double dl = 0, nl = 0;
    for (unsigned i = 0; i < n; ++i) {
        float expected =
            mask[i] > 0 && depth[i] > 0
                ? .2F / n * .5F / (std::sqrt(.25F + o.geometry_epsilon * o.geometry_epsilon) * 2)
                : 0;
        require(std::abs(gd[i] - expected) < 1e-7F, "MVS depth gradient mismatch");
        require(
            std::abs(gn[2 * n + i] - (mask[i] > 0 ? -.3F / n : 0.F)) < 1e-7F,
            "MVS normal validity mask mismatch");
        if (mask[i] > 0) {
            nl += .3 / n;
            if (depth[i] > 0)
                dl += .2 / n * std::sqrt(.25 + o.geometry_epsilon * o.geometry_epsilon);
        }
    }
    require(std::abs(g.total - dl - nl) < 1e-5, "MVS scalar losses mismatch");
    std::cout << "MVS depth/normal supervision: CPU loss and gradient oracle passed\n";
}
void training_test() {
    const auto root = std::filesystem::temp_directory_path() / "photara-vulkan-adc-igs-regression";
    std::filesystem::create_directories(root);
    photara::mvs::MvsScene scene;
    auto image = root / "rgb.png";
    photara::io::save_rgb_png({32, 32, std::vector<std::uint8_t>(32 * 32 * 3, 128)}, image);
    for (unsigned i = 0; i < 3; ++i) {
        photara::mvs::MvsView v;
        v.id = i;
        v.path = image;
        v.width = v.height = v.src_width = v.src_height = 32;
        v.fx = v.fy = v.src_fx = v.src_fy = 20;
        v.cx = v.cy = v.src_cx = v.src_cy = 15.5F;
        v.pose.C.x() = .1 * i;
        scene.views.push_back(v);
    }
    for (float x : {-.2F, .2F})
        for (float y : {-.2F, .2F}) {
            photara::mvs::DensePoint p;
            p.position = {x, y, 2.F};
            p.normal = photara::mvs::Vec3f::UnitZ();
            p.color = photara::mvs::Vec3f::Constant(.5F);
            scene.dense_cloud.points.push_back(p);
        }
    TrainingOptions o;
    o.backend = TrainingBackend::vulkan;
    o.densification_strategy = DensificationStrategy::adc_igs;
    o.iterations = 8;
    o.sh_degree = 1;
    o.input_is_dense = false;
    o.use_ppisp = o.use_bilateral_grid = true;
    o.use_mvs_depth = o.use_mvs_normals = false;
    o.use_depth_normal_loss = o.use_normal_field = false;
    o.initialize_scale_from_knn = true;
    o.initial_scale = .1F;
    o.prune_opacity = 0;
    o.opacity_reset_every = 0;
    o.densification_cap = 16;
    o.refine_start_iter = 1;
    o.refine_stop_iter = o.grow_stop_iter = 7;
    o.refine_every = 2;
    o.refine_stop_num_iter = 0;
    o.densify_gradient_threshold = -1;
    o.densify_select_fraction = 1;
    o.maximum_scale_fraction = 1;
    o.densify_revised_noise = true;
    o.mean_noise_weight = 1.F;
    o.log_interval = 1;
    o.training_async_upload = false;
    unsigned reports = 0;
    auto result = Trainer(o).train(scene, [&](const TrainingProgress& p) {
        require(std::isfinite(p.loss), "ADC-IGS produced non-finite loss");
        ++reports;
        return true;
    });
    require(
        result.size() > 4 && result.size() <= 16 && reports >= 8,
        "ADC-IGS training did not refine/report");
    for (float x : result.means.to_vector())
        require(std::isfinite(x), "ADC-IGS produced invalid positions");
    // Busy transports must skip rasterization, and a rejected submission must
    // leave its camera pending instead of waiting for the next 100-step tick.
    auto preview_options = o;
    preview_options.iterations = 6;
    preview_options.preview_interval = 100;
    preview_options.preview_vis_file = root / "unused-vis";
    unsigned readiness_checks = 0, attempts = 0, preview_reports = 0;
    unsigned first_attempt = 0;
    bool retried_before_final = false;
    preview_options.preview_ready = [&] { return ++readiness_checks > 2; };
    auto preview_model = Trainer(preview_options).train(scene,
        [&](const TrainingProgress&) {
            ++preview_reports;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return true;
        }, {}, {}, [&](unsigned iteration, std::size_t, const Camera&, const Tensor&) {
            if (!first_attempt) first_attempt = iteration;
            if (++attempts <= 2) return false;
            retried_before_final |= iteration < preview_options.iterations;
            return true;
        });
    require(first_attempt >= 3 && attempts >= 3 && retried_before_final && preview_reports == 6,
            "Busy/rejected previews lost the pending camera or skipped optimizer steps");
    for (auto& view : scene.views) {
        view.foreground_mask.resize(32 * 32);
        for (unsigned i = 0; i < 32 * 32; ++i)
            view.foreground_mask[i] = i % 32 < 8 ? 0 : 255;
        view.depth_map.depth.assign(32 * 32, 2.F);
        view.depth_map.normal.assign(32 * 32, photara::mvs::Vec3f::UnitZ());
    }
    o.use_mvs_depth = o.use_mvs_normals = true;
    o.use_depth_normal_loss = o.use_normal_field = true;
    o.depth_normal_from_iter = 1;
    o.normal_field_from_iter = 1;
    o.multi_view_geo_weight = .02F;
    o.multi_view_ncc_weight = .1F;
    o.filter_mode = SplatFilterMode::mip_3d;
    o.filter_3d_update_interval = 2;
    o.opacity_reset_every = 2;
    o.densify_revised_noise = false;
    reports = 0;
    bool saw_depth = false, saw_alpha = false;
    o.alpha_mode = AlphaMode::transparent;
    o.use_mask = true;
    o.initial_opacity = .9F;
    o.input_is_dense = true;
    o.initialize_scale_from_knn = false;
    o.initial_scale = 1.F;
    o.constrain_scale_range = false;
    auto supervised = Trainer(o).train(scene, [&](const TrainingProgress& p) {
        require(std::isfinite(p.loss), "Supervised ADC-IGS produced non-finite loss");
        saw_depth = saw_depth || p.depth_loss > 0;
        saw_alpha = saw_alpha || p.alpha_loss > 0;
        ++reports;
        return true;
    });
    require(
        supervised.size() && reports >= 8 && saw_depth && saw_alpha,
        "Supervised ADC-IGS did not consume MVS targets/masks");
    o.use_mvs_depth = o.use_mvs_normals = false;
    o.use_depth_normal_loss = o.use_normal_field = false;
    o.multi_view_geo_weight = o.multi_view_ncc_weight = 0;
    o.filter_mode = SplatFilterMode::classic;
    o.render_method = RenderMethod::gaussian_3dgut;
    o.densify_revised_noise = true;
    reports = 0;
    auto gut = Trainer(o).train(scene, [&](const TrainingProgress& p) {
        require(std::isfinite(p.loss), "3DGUT ADC-IGS produced non-finite loss");
        ++reports;
        return true;
    });
    require(gut.size() && reports >= 8, "3DGUT ADC-IGS did not complete");
    std::filesystem::remove_all(root);
    std::cout << "ADC-IGS end-to-end: 3DGS, MVS/normal field/multi-view/Mip3D, opacity resets and "
                 "3DGUT passed\n";
}
int main() {
    int status = 0;
    std::thread worker([&] {
        try {
            points_test();
            noise_test();
            supervision_test();
            training_test();
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            status = 1;
        }
    });
    worker.join();
    tinytensor::vulkan::shutdown();
    return status;
}
