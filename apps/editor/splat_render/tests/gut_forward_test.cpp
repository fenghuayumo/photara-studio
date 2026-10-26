#include "splat_drender/vulkan_api.h"
#include "splat_render/renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace splat_drender::vulkan;
using Vec = std::array<double, 3>;
using Mat = std::array<Vec, 3>;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
double dot(Vec a, Vec b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec mul(Mat a, Vec b) { return {dot(a[0], b), dot(a[1], b), dot(a[2], b)}; }
Mat rotation(const float *q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)},
             {2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)},
             {2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)}}};
}
struct Scene {
    std::vector<float> means{-.12F, .08F, 2.F, .2F, -.1F, 2.6F};
    std::vector<float> scales{.36F, .28F, .20F, .4F, .3F, .25F};
    std::vector<float> rotations{.98F, .08F, -.12F, .09F, 1.F, 0.F, 0.F, 0.F};
    std::vector<float> opacity{.88F, .65F};
    std::vector<float> colors{.8F, .2F, .4F, .1F, .6F, .3F};
    std::array<float, 16> view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::array<float, 3> eye{};
    Scene() {
        double n = 0;
        for (int k = 0; k < 4; ++k)
            n += rotations[k] * rotations[k];
        for (int k = 0; k < 4; ++k)
            rotations[k] /= float(std::sqrt(n));
    }
    SplatGaussians gaussians() const {
        SplatGaussians g;
        g.means = means;
        g.scales = scales;
        g.rotations = rotations;
        g.opacities = opacity;
        g.colors = colors;
        return g;
    }
    SplatCamera camera(unsigned mode) const {
        SplatCamera c;
        c.width = 35;
        c.height = 29;
        c.mode = mode;
        c.fx = c.fy = 24;
        c.cx = 17;
        c.cy = 14;
        c.k1 = .05F;
        c.k2 = -.01F;
        if (mode == 3) {
            c.fx = c.width / (2 * 3.141592653589793F);
            c.fy = c.height / 3.141592653589793F;
            c.cx = c.width * .5F;
            c.cy = c.height * .5F;
        }
        c.world_to_camera = view;
        c.center = eye;
        return c;
    }
};

// Independent double-precision all-Gaussian ray oracle. No projected bounds,
// tile lists or shader helpers are involved.
SplatForwardOutput reference(const Scene &scene, const SplatCamera &c, const SplatSettings &s) {
    const unsigned pixels = c.width * c.height;
    SplatForwardOutput out;
    out.color.resize(3 * pixels);
    out.alpha.resize(pixels);
    out.median_depth.resize(pixels);
    out.normal.resize(3 * pixels);
    Mat W{{{scene.view[0], scene.view[1], scene.view[2]},
           {scene.view[4], scene.view[5], scene.view[6]},
           {scene.view[8], scene.view[9], scene.view[10]}}};
    Mat camera_rotation{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            camera_rotation[a][b] = W[b][a];
    std::vector<int> order(scene.opacity.size());
    std::iota(order.begin(),order.end(),0);
    auto distance = [&](int i) {
        Vec v{};
        for (int a = 0; a < 3; ++a)
            v[a] = scene.means[3 * i + a] - scene.eye[a];
        return dot(v, v);
    };
    std::stable_sort(order.begin(),order.end(),[&](int a,int b) { return distance(a)<distance(b); });
    for (unsigned y = 0; y < c.height; ++y)
        for (unsigned x = 0; x < c.width; ++x) {
            const double u = (double(x) - c.cx) / c.fx, v = (double(y) - c.cy) / c.fy;
            Vec ray{u, v, 1}, origin{scene.eye[0], scene.eye[1], scene.eye[2]};
            if (c.mode == 4) {
                ray = {0, 0, 1};
                const Vec offset = mul(W, {u, v, 0});
                for (int a = 0; a < 3; ++a)
                    origin[a] += offset[a];
            } else if (c.mode == 3)
                ray = {std::cos(v) * std::sin(u), std::sin(v), std::cos(v) * std::cos(u)};
            else if (c.mode == 1) {
                const double r = std::hypot(u, v);
                double lo = 0, hi = 1.57079632679;
                for (int k = 0; k < 40; ++k) {
                    const double t = (lo + hi) * .5, q = t * t;
                    const double td = t * (1 + q * (c.k1 + q * (c.k2 + q * (c.k3 + q * c.k4))));
                    if (td < r)
                        lo = t;
                    else
                        hi = t;
                }
                const double t = (lo + hi) * .5, scale = r > 1e-8 ? std::sin(t) / r : 1.;
                ray = {u * scale, v * scale, std::cos(t)};
            } else {
                const double n = std::sqrt(dot(ray, ray));
                for (auto &a : ray)
                    a /= n;
            }
            const double ray_z = c.mode == 3 ? 1. : ray[2];
            const Vec world_ray = mul(W, ray);
            double T = 1, depth = 0;
            Vec rgb{}, normal{};
            for (int i : order) {
                const Mat R = rotation(scene.rotations.data() + 4 * i);
                Vec relative{};
                for (int a = 0; a < 3; ++a)
                    relative[a] = origin[a] - scene.means[3 * i + a];
                Vec o = mul(R, relative), d = mul(R, world_ray);
                for (int a = 0; a < 3; ++a) {
                    o[a] /= scene.scales[3 * i + a] * s.scale_modifier;
                    d[a] /= scene.scales[3 * i + a] * s.scale_modifier;
                }
                const double peak = -dot(o, d) / dot(d, d);
                if (!(peak > 0))
                    continue;
                Vec closest{};
                for (int a = 0; a < 3; ++a)
                    closest[a] = o[a] + peak * d[a];
                const double alpha =
                    std::min(.99, double(scene.opacity[i]) * std::exp(-.5 * dot(closest, closest)));
                if (alpha < 1. / 255)
                    continue;
                const double next = T * (1 - alpha);
                if (next < 1e-4)
                    break;
                const double weight = alpha * T;
                int axis = 0;
                for (int a = 1; a < 3; ++a)
                    if (scene.scales[3 * i + a] < scene.scales[3 * i + axis])
                        axis = a;
                Vec n = R[axis];
                if (dot(n, world_ray) > 0)
                    for (auto &a : n)
                        a = -a;
                n = mul(camera_rotation, n);
                for (int a = 0; a < 3; ++a) {
                    rgb[a] += weight * scene.colors[3 * i + a];
                    normal[a] += weight * n[a];
                }
                if (T >= .5 && next < .5)
                    depth = peak * ray_z;
                T = next;
            }
            const unsigned p = y * c.width + x;
            out.alpha[p] = float(1 - T);
            out.median_depth[p] = float(depth);
            for (unsigned a = 0; a < 3; ++a) {
                out.color[a * pixels + p] = float(rgb[a] + T * s.background[a]);
                out.normal[a * pixels + p] = float(normal[a] / std::max(1 - T, 1e-8));
            }
        }
    return out;
}
void close(const std::vector<float> &actual, const std::vector<float> &expected, double tolerance,
           const char *name) {
    require(actual.size() == expected.size(), "attachment extent mismatch");
    double error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "nonfinite Vulkan output");
        error = std::max(error, std::abs(double(actual[i]) - expected[i]));
    }
    if (error > tolerance) {
        std::cerr << name << " max_error=" << error << '\n';
        throw std::runtime_error("Vulkan 3DGUT differs from ray oracle");
    }
}
void ray_tests(bool validation) {
    photara::vk::DeviceRequest request;
    request.validation = validation;
    request.synchronization_validation = validation;
    auto device = photara::vk::Device::create(request);
    ContextOptions options;
    options.external_device.instance = device.instance();
    options.external_device.physical_device = device.physical();
    options.external_device.device = device.handle();
    options.external_device.queue = device.queue();
    options.external_device.queue_family = device.queue_family();
    Context context(options);
    SplatRasterizer raster(context);
    for (unsigned mode : {0u, 1u, 3u, 4u})
        for (bool transformed : {false, true})
          for (bool dense : {false, true}) {
            Scene scene;
            if (mode == 3) {
                scene.means[2] = -2;
                scene.means[5] = -2.6F;
            } // panorama seam
            if (dense) {
                const Scene base = scene;
                constexpr unsigned n = 300;
                scene.means.resize(3*n); scene.scales.resize(3*n); scene.rotations.resize(4*n);
                scene.colors.resize(3*n); scene.opacity.assign(n,.006F);
                for (unsigned i=0;i<n;++i) {
                    for (unsigned c=0;c<3;++c) {
                        scene.means[3*i+c] = base.means[3*(i%2)+c];
                        scene.scales[3*i+c] = base.scales[3*(i%2)+c];
                        scene.colors[3*i+c] = base.colors[3*(i%2)+c];
                    }
                    scene.means[3*i] += float(int(i%9)-4)*.11F;
                    scene.means[3*i+1] += float(int(i%7)-3)*.09F;
                    scene.means[3*i+2] += (mode==3 ? -.001F : .001F)*i;
                    for (unsigned c=0;c<4;++c) scene.rotations[4*i+c]=base.rotations[4*(i%2)+c];
                }
            }
            if (transformed) {
                const float co = std::cos(.2F), si = std::sin(.2F);
                scene.eye = {.15F, -.05F, .1F};
                scene.view = {co,
                              0,
                              -si,
                              0,
                              0,
                              1,
                              0,
                              0,
                              si,
                              0,
                              co,
                              0,
                              -.15F * co - .1F * si,
                              .05F,
                              .15F * si - .1F * co,
                              1};
            }
            raster.upload_model(scene.gaussians());
            const auto camera = scene.camera(mode);
            SplatSettings settings;
            settings.primitive = SplatPrimitive::gaussian_3dgut;
            settings.background[0] = .13F;
            settings.background[1] = .07F;
            settings.background[2] = .11F;
            settings.scale_modifier = .9F;
            const auto expected = reference(scene, camera, settings);
            const auto actual = raster.render(camera, settings);
            close(actual.color, expected.color, 8e-5, "RGB");
            close(actual.alpha, expected.alpha, 8e-5, "alpha");
            close(actual.normal, expected.normal, 1e-4, "normal");
            close(actual.median_depth, expected.median_depth, 1e-4, "depth");
            // Exercise GPU-driven capacity reuse and compact RGBA output.
            settings.need_depth = false;
            std::vector<std::uint8_t> rgba;
            raster.render_rgba(camera, settings, rgba);
            require(rgba.size() == camera.width * camera.height * 4ull, "RGBA extent mismatch");
            const unsigned pixels = camera.width * camera.height;
            for (unsigned p = 0; p < pixels; ++p)
                for (unsigned a = 0; a < 3; ++a)
                    require(
                        std::abs(int(rgba[4 * p + a]) -
                                 int(std::clamp(expected.color[a * pixels + p], 0.F, 1.F) * 255 +
                                     .5F)) <= 1,
                        "RGBA render lost GUT response");
            if (dense) {
                const float original_scale = scene.scales[2];
                for (bool thin : {false,true}) {
                    scene.scales[2] = thin ? 1e-8F : original_scale;
                    raster.upload_model(scene.gaussians());
                    for (bool rings : {false,true}) {
                        settings.preview_rings = rings;
                        const auto original = raster.render(camera,settings);
                        raster.render_rgba(camera,settings,rgba);
                        for (unsigned p=0;p<pixels;++p) {
                            require(rgba[4*p+3]==255,"packed GUT alpha mismatch");
                            for (unsigned a=0;a<3;++a) {
                                if(std::abs(int(rgba[4*p+a])-int(std::clamp(original.color[a*pixels+p],0.F,1.F)*255.F+.5F))>1)
                                    std::cerr<<"pixel "<<p<<" channel "<<a<<" rgba="<<int(rgba[4*p+a])<<" float="<<original.color[a*pixels+p]<<std::endl;
                                require(std::abs(int(rgba[4*p+a])-int(std::clamp(original.color[a*pixels+p],0.F,1.F)*255.F+.5F))<=1,
                                        "dense/thin/rings GUT fast path differs from original");
                            }
                        }
                    }
                }
                settings.preview_rings = false;
                scene.scales[2] = original_scale;
                raster.upload_model(scene.gaussians());
            }
            settings.primitive = SplatPrimitive::gaussian_3dgs;
            (void)raster.render(camera, settings);
            settings.primitive = SplatPrimitive::gaussian_3dgut;
            const auto switch_first=raster.render(camera,settings);
            const auto switch_second=raster.render(camera,settings);
            close(switch_second.color,expected.color,8e-5,"second method switch");
            close(switch_first.color, expected.color, 8e-5, "method switch");
            std::vector<float> sh(scene.colors.size());
            for (std::size_t k = 0; k < sh.size(); ++k)
                sh[k] = (scene.colors[k] - .5F) / .28209479177387814F;
            auto sh_model = scene.gaussians();
            sh_model.colors = {};
            sh_model.sh = sh;
            sh_model.sh_bases = 1;
            raster.upload_model(sh_model);
            close(raster.render(camera, settings).color, expected.color, 8e-5, "SH colors");
            settings.pixel_snapshots = true;
            bool rejected = false;
            try {
                (void)raster.render(camera, settings);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            require(rejected, "Vulkan GUT backward was silently accepted");
            settings.pixel_snapshots = false;
            scene.opacity.assign(scene.opacity.size(),0.F);
            raster.update_means_and_opacities(scene.means, scene.opacity);
            const auto empty = raster.render(camera, settings);
            for (float alpha : empty.alpha)
                require(alpha == 0, "all-culled frame retained stale alpha");
        }
    // Large supports crossing z=0 and the old z=0.2 projection guard,
    // eye inside the ellipsoid, offscreen centers, and strongly tilted axes.
    for (unsigned fixture=0;fixture<16;++fixture) {
        Scene scene;
        scene.means={float(int(fixture%4)-2)*.35F,.1F,.21F+float(fixture/4)*.13F,
                     -.1F,.05F,.75F};
        scene.scales={.6F,.45F,.5F,.3F,.3F,.6F};
        if(fixture%3==0) scene.scales={.15F,.7F,.65F,.3F,.3F,.6F};
        auto camera=scene.camera(0);
        camera.fx=11.F+fixture;camera.fy=19.F;
        camera.cx=(fixture%2==0 ? -3.F : 17.F);camera.cy=14.F;
        raster.upload_model(scene.gaussians());
        SplatSettings settings;settings.primitive=SplatPrimitive::gaussian_3dgut;
        settings.background[0]=.13F;settings.background[1]=.07F;settings.background[2]=.11F;
        const auto expected=reference(scene,camera,settings);
        const auto actual=raster.render(camera,settings);
        close(actual.color,expected.color,1e-4,"near-plane RGB");
        close(actual.alpha,expected.alpha,1e-4,"near-plane alpha");
        close(actual.normal,expected.normal,1e-4,"near-plane normal");
        close(actual.median_depth,expected.median_depth,1e-4,"near-plane depth");
        for (float sigma : {.5F,2.828427F,8.F}) {
            settings.need_depth=false;settings.preview_rings=true;settings.preview_ring_scale=sigma;
            const auto filtered=raster.render(camera,settings);
            const char* value=std::getenv("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE");
            const std::string previous=value ? value : "";
#ifdef _WIN32
            _putenv_s("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE","1");
#else
            setenv("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE","1",1);
#endif
            const auto unfiltered=raster.render(camera,settings);
#ifdef _WIN32
            _putenv_s("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE",previous.c_str());
#else
            if(previous.empty()) unsetenv("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE");
            else setenv("SPLAT_DRENDER_GUT_CANDIDATE_REFERENCE",previous.c_str(),1);
#endif
            close(filtered.color,unfiltered.color,1e-5,"near-plane rings RGB");
            close(filtered.alpha,unfiltered.alpha,1e-5,"near-plane rings alpha");
        }
    }
    require(device.validation_error_count() == 0, "Vulkan ray renderer validation errors");
    std::cout << "Vulkan GUT CPU ray parity, 4 cameras, transforms, seam, RGBA, method switches "
                 "and empty frames passed\n";
}

void viewport_tests(bool validation) {
    photara::vk::DeviceRequest request;
    request.want.graphics = true;
    request.validation = validation;
    request.synchronization_validation = validation;
    auto device = photara::vk::Device::create(request);
    splat_render::Renderer renderer;
    renderer.attach(device);
    const float means[]{.4F, 0, 2.F}, logs[]{std::log(.4F), std::log(.25F), std::log(.3F)};
    const float q[]{1, 0, 0, 0}, opacity[]{4.F}, sh[]{.7F, .2F, .1F};
    splat_render::GaussianCloud cloud;
    cloud.count = 1;
    cloud.means = means;
    cloud.log_scales = logs;
    cloud.quaternions = q;
    cloud.opacity_logits = opacity;
    cloud.sh = sh;
    splat_render::Camera camera;
    camera.width = 35;
    camera.height = 29;
    camera.fx = camera.fy = 24;
    camera.cx = 17;
    camera.cy = 14;
    camera.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    splat_render::DisplayOptions display;
    splat_render::FrameTarget target;
    auto draw = [&](splat_render::Shading shading) {
        require(renderer.draw(camera, target, shading, display), renderer.failure().c_str());
        std::vector<std::uint8_t> rgb;
        std::uint32_t w = 0, h = 0;
        require(renderer.download_rgb(rgb, w, h), "viewport download failed");
        require(w == 35 && h == 29, "viewport extent mismatch");
        return rgb;
    };
    require(renderer.upload(cloud, "same-model"), "GS upload failed");
    const auto gs = draw(splat_render::Shading::gaussian);
    cloud.render_method = splat_render::RenderMethod::gaussian_3dgut;
    require(renderer.upload(cloud, "same-model"), "GUT upload failed");
    const auto gut = draw(splat_render::Shading::gaussian);
    require(gs != gut, "same-key method switch retained the EWA image");
    const int cached = target.slot;
    require(draw(splat_render::Shading::gaussian) == gut && target.slot == cached,
            "GUT frame cache mismatch");
    const auto depth = draw(splat_render::Shading::depth),
               normal = draw(splat_render::Shading::normal);
    require(depth != gut && normal != gut, "GUT diagnostics used RGB");
    const auto rings = draw(splat_render::Shading::rings);
    require(rings != gut, "GUT rings used RGB");
    require(draw(splat_render::Shading::gaussian) == gut,
            "switching diagnostic shading lost GUT method");
    display.selection_depth_enabled = true;
    display.selection_depth_near = 3;
    display.selection_depth_far = 4;
    require(draw(splat_render::Shading::gaussian) != gut, "GUT selection cue missing");
    display.selection_depth_enabled = false;
    const float centers[]{.1F, .1F, 2.F, .9F};
    require(renderer.update_centers(centers, 1), "GUT center edit failed");
    require(draw(splat_render::Shading::gaussian) != gut, "GUT edit left stale pixels");
    renderer.clear_model();
    require(renderer.upload(cloud, "same-model"), "GUT reload failed");
    require(draw(splat_render::Shading::gaussian) == gut, "GUT reload lost method");
    cloud.render_method = splat_render::RenderMethod::gaussian_3dgs;
    require(renderer.upload(cloud, "same-model"), "GS switch back failed");
    require(draw(splat_render::Shading::gaussian) == gs, "GS switch back did not restore EWA");
    renderer.reset();
    require(device.validation_error_count() == 0, "Viewport Vulkan validation errors");
    std::cout
        << "Viewport GUT method dispatch, cache, diagnostics, rings, edits and reload passed\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        const bool validation = argc == 2 && std::string_view(argv[1]) == "--validation";
        ray_tests(validation);
        viewport_tests(validation);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
