#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#include "splat/rasterizer.hpp"
#include "io/image.hpp"
#include "sfm/asfm.hpp"
#include "mvs/densify.hpp"
#include "core/camera_projection.hpp"
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Re-render models from different binaries through one image loader and
// rasterizer. Positional arguments keep this experiment tool independent of
// the reconstruction CLI and its changing defaults.
int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7 && argc != 9)
            throw std::runtime_error("Usage: splat_quality_eval DATASET_OR_ASFM IMAGES MODEL OUTPUT_DIR SAMPLE_EVERY [--visibility-audit|--prune-unobserved|--sampled-novel|--projection-audit|--haze-audit] [--views-file FILE]");
        const bool visibility_audit = argc >= 7 && std::string(argv[6]) == "--visibility-audit";
        const bool prune_unobserved = argc >= 7 && std::string(argv[6]) == "--prune-unobserved";
        const bool sampled_novel = argc >= 7 && std::string(argv[6]) == "--sampled-novel";
        const bool projection_audit = argc >= 7 && std::string(argv[6]) == "--projection-audit";
        const bool haze_audit = argc >= 7 && std::string(argv[6]) == "--haze-audit";
        if (argc >= 7 && !visibility_audit && !prune_unobserved && !sampled_novel && !projection_audit && !haze_audit) throw std::runtime_error("Unknown evaluation mode");
        if (argc == 9 && (std::string(argv[7]) != "--views-file" || visibility_audit || prune_unobserved))
            throw std::runtime_error("--views-file requires a rendering audit mode");
        const unsigned split = std::stoul(argv[5]);
        if (split < 2) throw std::runtime_error("SPLIT_EVERY must be >= 2");
        photara::splat::DatasetLoadRequest request;
        request.source = argv[1];
        request.image_directory = argv[2];
        photara::splat::DatasetLoadResult dataset;
        if (request.source.extension() == ".bin" && std::filesystem::is_regular_file(request.source)) {
            photara::sfm::AsfmOptions asfm;
            asfm.path_base = request.image_directory;
            dataset.scene = photara::mvs::build_mvs_scene(
                photara::sfm::load_asfm(request.source, asfm), photara::mvs::DensifyOptions{});
        } else {
            dataset = photara::splat::load_splat_dataset(request);
        }
        std::vector<std::size_t> evaluation_views;
        if (argc == 9) {
            std::ifstream names_file(argv[8]);
            if (!names_file) throw std::runtime_error("Cannot read view list");
            std::set<std::string> names;
            std::string name;
            while (std::getline(names_file, name)) {
                if (!name.empty() && name.back() == '\r') name.pop_back();
                if (!name.empty()) names.insert(name);
            }
            for (std::size_t i = 0; i < dataset.scene.views.size(); ++i)
                if (names.erase(dataset.scene.views[i].path.filename().string()) != 0) evaluation_views.push_back(i);
            if (!names.empty() || evaluation_views.empty()) throw std::runtime_error("View list contains unregistered or missing images");
        } else {
            for (std::size_t i = 0; i < dataset.scene.views.size(); i += split) evaluation_views.push_back(i);
        }
        auto model = photara::splat::load_gaussians(argv[3]);
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output);
        photara::splat::TrainingOptions options;
        options.use_mask = projection_audit || haze_audit;
        options.use_source_resolution = true;
        options.ignore_undistortion_border = true;
        options.progressive_resolution = false;
        if (haze_audit) {
            const auto means = model.means.to_vector();
            const auto scales = model.log_scales.to_vector();
            const auto opacity = model.opacity_logits.to_vector();
            std::ofstream audit(output / "haze.csv");
            if (!audit) throw std::runtime_error("Cannot create haze.csv");
            audit << "view,variant,excluded,psnr,foreground_psnr,ssim\n";
            for (const std::size_t i : evaluation_views) {
                const auto& view = dataset.scene.views[i];
                for (unsigned variant = 0; variant < 5; ++variant) {
                    auto altered = model;
                    auto logits = opacity;
                    std::size_t excluded = 0;
                    for (std::size_t g = 0; g < model.size(); ++g) {
                        const float distance = (Eigen::Vector3f(means[3*g], means[3*g+1], means[3*g+2]) - view.pose.C.cast<float>()).norm();
                        const float largest = std::exp(std::max({scales[3*g],scales[3*g+1],scales[3*g+2]}));
                        const float smallest = std::exp(std::min({scales[3*g],scales[3*g+1],scales[3*g+2]}));
                        const float alpha = 1.F / (1.F + std::exp(-opacity[g]));
                        const float angular_size = largest / std::max(distance,1e-8F);
                        // Per-camera causal diagnostics, not a global pruning recipe:
                        // legitimate nearby surfaces can also satisfy these predicates.
                        const bool reject = (variant == 1 && alpha < 0.05F)
                            || (variant == 2 && angular_size > 0.05F && alpha < 0.5F)
                            || (variant == 3 && distance < 3.F * largest)
                            || (variant == 4 && largest > 100.F * smallest && angular_size > 0.05F);
                        if (reject) { logits[g] = -30.F; ++excluded; }
                    }
                    if (variant != 0)
                        altered.opacity_logits = tinytensor::Tensor::from_vector(logits, model.opacity_logits.shape(), model.opacity_logits.device());
                    const auto metrics = photara::splat::render_evaluation_png(altered, view,
                        output / ("haze_" + std::to_string(i) + "_variant_" + std::to_string(variant) + ".png"), options);
                    audit << i << ',' << variant << ',' << excluded << ',' << metrics.psnr << ',' << metrics.foreground_psnr << ',' << metrics.ssim << '\n';
                    audit.flush();
                }
            }
            return 0;
        }
        std::ofstream projection_csv;
        if (projection_audit) {
            projection_csv.open(output / "projection.csv");
            if (!projection_csv) throw std::runtime_error("Cannot create projection.csv");
            projection_csv << "view,yaw,perspective_psnr,warped_native_psnr,direct_vs_warped_psnr\n";
        }
        std::ofstream view_csv(output / "views.csv");
        if (!view_csv) throw std::runtime_error("Cannot create views.csv");
        view_csv << "view,path,camera_model,width,height\n";
        if (visibility_audit || prune_unobserved) {
            std::vector<unsigned> observations(model.size(), 0);
            for (std::size_t i = 0; i < dataset.scene.views.size(); ++i) {
                if (i % split == 0) continue;
                const auto camera = photara::splat::make_training_view(dataset.scene.views[i], options).camera;
                photara::splat::RasterizeOptions raster;
                raster.active_sh_degree = model.sh_degree;
                raster.require_depth = false;
                const auto visible = photara::splat::Rasterizer().forward(model, camera, raster).visibility.to_vector();
                for (std::size_t g = 0; g < model.size(); ++g) observations[g] += visible[g] > 0.F;
            }
            std::vector<std::size_t> counts(dataset.scene.views.size() + 1, 0);
            std::vector<double> mass(counts.size(), 0.0);
            const auto logits = model.opacity_logits.to_vector();
            for (std::size_t g = 0; g < model.size(); ++g) {
                ++counts[observations[g]];
                mass[observations[g]] += 1.0 / (1.0 + std::exp(-logits[g]));
            }
            std::ofstream audit(output / "visibility.csv");
            audit << "training_views,count,opacity_sum\n";
            for (std::size_t i = 0; i < counts.size(); ++i)
                audit << i << ',' << counts[i] << ',' << mass[i] << '\n';
            std::cout << "invisible=" << counts[0] << " opacity_sum=" << mass[0] << '\n';
            if (prune_unobserved) {
                std::vector<int> retained;
                for (std::size_t g = 0; g < observations.size(); ++g)
                    if (observations[g] != 0) retained.push_back(static_cast<int>(g));
                if (retained.empty()) throw std::runtime_error("No observed Gaussians; refusing empty export");
                const auto indices = tinytensor::Tensor::from_vector(
                    retained, {retained.size()}, tinytensor::Device::CUDA);
                for (auto* tensor : {&model.means, &model.log_scales, &model.quaternions,
                         &model.opacity_logits, &model.sh, &model.normal_features, &model.filter_3d})
                    if (tensor->is_valid()) *tensor = tensor->index_select(0, indices);
                photara::splat::save_gaussians_ply(model, output / "model_splat.ply");
            }
            return 0;
        }
        std::ofstream csv(output / "metrics.csv");
        if (!csv) throw std::runtime_error("Cannot create metrics.csv");
        csv << "view,psnr,ssim,full_psnr,full_ssim,valid_fraction,foreground_psnr\n";
        double psnr = 0, ssim = 0, full_psnr = 0, full_ssim = 0;
        std::size_t count = 0;
        for (const std::size_t i : evaluation_views) {
            const auto target = photara::splat::make_training_view(dataset.scene.views[i], options);
            view_csv << i << ",\"" << dataset.scene.views[i].path.string() << "\"," << int(target.camera.model)
                     << ',' << target.camera.width << ',' << target.camera.height << '\n';
            const auto pixels = target.rgb.to_vector();
            photara::io::RgbImage image;
            image.width = target.camera.width;
            image.height = target.camera.height;
            const std::size_t area = image.width * image.height;
            image.pixels.resize(3 * area);
            for (std::size_t p = 0; p < area; ++p)
                for (std::size_t c = 0; c < 3; ++c)
                    image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                        std::lround(255.F * std::clamp(pixels[c * area + p], 0.F, 1.F)));
            photara::io::save_rgb_png(image, output / ("target_" + std::to_string(i) + ".png"));
            if (projection_audit && (photara::uses_native_splat_projection(target.camera.model) ||
                                     target.camera.model == photara::CameraModel::pinhole)) {
                photara::splat::RasterizeOptions raster;
                raster.active_sh_degree = model.sh_degree;
                raster.require_depth = false;
                raster.record_backward_state = false;
                const auto native = photara::splat::Rasterizer().forward(model, target.camera, raster).color.to_vector();
                const auto validity = target.has_mask ? target.mask.to_vector() : std::vector<float>{};
                const bool panorama = target.camera.model == photara::CameraModel::equirectangular;
                for (const int yaw : panorama ? std::vector<int>{0, 90, 180, 270} : std::vector<int>{0}) {
                    auto pin = target.camera;
                    pin.model = photara::CameraModel::pinhole;
                    pin.width = 640; pin.height = 360;
                    pin.fx = pin.fy = 320.F; pin.cx = 320.F; pin.cy = 180.F;
                    pin.k1 = pin.k2 = pin.k3 = pin.k4 = 0.F;
                    const Eigen::Matrix3f turn = Eigen::AngleAxisf(
                        -yaw * 3.14159265358979323846F / 180.F, Eigen::Vector3f::UnitY()).toRotationMatrix();
                    Eigen::Map<Eigen::Matrix4f> transform(pin.world_to_camera.data());
                    transform.block<3, 4>(0, 0) = (turn * transform.block<3, 4>(0, 0)).eval();
                    const auto direct = photara::splat::Rasterizer().forward(model, pin, raster).color.to_vector();
                    const std::size_t pin_area = pin.width * pin.height;
                    photara::io::RgbImage reference{pin.width, pin.height, std::vector<std::uint8_t>(3 * pin_area)};
                    auto direct_image = reference, warped_image = reference;
                    double direct_error = 0, warped_error = 0, pair_error = 0, valid_weight = 0;
                    for (unsigned y = 0; y < pin.height; ++y) for (unsigned x = 0; x < pin.width; ++x) {
                        const Eigen::Vector3f ray = turn.transpose() * Eigen::Vector3f((x - pin.cx) / pin.fx, (y - pin.cy) / pin.fy, 1.F);
                        const auto& cam = target.camera;
                        const auto uv = panorama ? photara::project_equirectangular_camera(ray.x(), ray.y(), ray.z(), cam.width, cam.height)
                            : cam.model == photara::CameraModel::pinhole
                                ? photara::CameraPixel{cam.fx * ray.x() / ray.z() + cam.cx, cam.fy * ray.y() / ray.z() + cam.cy, true}
                                : photara::project_fisheye_camera(ray.x(), ray.y(), ray.z(), cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.k3, cam.k4);
                        const int x0 = static_cast<int>(std::floor(uv.u)), y0 = static_cast<int>(std::floor(uv.v));
                        const float dx = static_cast<float>(uv.u - x0), dy = static_cast<float>(uv.v - y0);
                        const auto sample = [&](const std::vector<float>& data, int xx, int yy, std::size_t channel) {
                            xx = panorama ? (xx % int(cam.width) + int(cam.width)) % int(cam.width) : std::clamp(xx, 0, int(cam.width) - 1);
                            yy = std::clamp(yy, 0, int(cam.height) - 1);
                            return data[channel * area + std::size_t(yy) * cam.width + xx];
                        };
                        const auto interpolate = [&](const std::vector<float>& data, std::size_t channel) {
                            return (1-dx)*(1-dy)*sample(data,x0,y0,channel) + dx*(1-dy)*sample(data,x0+1,y0,channel)
                                + (1-dx)*dy*sample(data,x0,y0+1,channel) + dx*dy*sample(data,x0+1,y0+1,channel);
                        };
                        const float weight = validity.empty() ? 1.F : std::clamp(interpolate(validity,0),0.F,1.F);
                        valid_weight += weight;
                        const std::size_t p = std::size_t(y) * pin.width + x;
                        for (std::size_t c = 0; c < 3; ++c) {
                            const float truth = std::clamp(interpolate(pixels,c),0.F,1.F);
                            const float warp = std::clamp(interpolate(native,c),0.F,1.F);
                            const float value = std::clamp(direct[c * pin_area + p],0.F,1.F);
                            reference.pixels[3*p+c] = static_cast<std::uint8_t>(std::lround(255*truth));
                            direct_image.pixels[3*p+c] = static_cast<std::uint8_t>(std::lround(255*value));
                            warped_image.pixels[3*p+c] = static_cast<std::uint8_t>(std::lround(255*warp));
                            direct_error += weight*(value-truth)*(value-truth);
                            warped_error += weight*(warp-truth)*(warp-truth);
                            pair_error += weight*(value-warp)*(value-warp);
                        }
                    }
                    const auto score = [&](double error) { return valid_weight > 0 ? -10 * std::log10(std::max(error / (3 * valid_weight), 1e-12)) : std::numeric_limits<double>::quiet_NaN(); };
                    const auto suffix = std::to_string(i) + "_yaw_" + std::to_string(yaw) + ".png";
                    photara::io::save_rgb_png(reference, output / ("perspective_target_" + suffix));
                    photara::io::save_rgb_png(direct_image, output / ("perspective_direct_" + suffix));
                    photara::io::save_rgb_png(warped_image, output / ("perspective_warped_" + suffix));
                    projection_csv << i << ',' << yaw << ',' << score(direct_error) << ',' << score(warped_error) << ',' << score(pair_error) << '\n';
                }
            }
            // Extrapolation diagnostics: move along the optical axis using
            // a scene-derived distance, independent of the trained model.
            // Keep all metric views; optionally save a fixed subset of novel
            // poses for large datasets. The selection never depends on quality.
            const bool save_novel = !projection_audit && (!sampled_novel || (i / split) % 8 == 0 ||
                i + split >= dataset.scene.views.size());
            std::vector<float> depths;
            if (save_novel)
            for (const auto& point : dataset.scene.sparse_points) {
                if (std::find(point.view_ids.begin(), point.view_ids.end(), i) == point.view_ids.end())
                    continue;
                const auto pc = dataset.scene.views[i].pose.R.cast<float>() *
                    (point.position - dataset.scene.views[i].pose.C.cast<float>());
                if (pc.z() > 0.F && std::isfinite(pc.z())) depths.push_back(pc.z());
            }
            if (!depths.empty()) {
                std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
                const float depth = depths[depths.size() / 2];
                for (const int percent : {-40, -20, 20, 40}) {
                    auto camera = target.camera;
                    Eigen::Map<Eigen::Matrix4f> transform(camera.world_to_camera.data());
                    const Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
                    Eigen::Vector3f position(camera.position.data());
                    position += rotation.row(2).transpose() * (depth * percent / 100.F);
                    for (int axis = 0; axis < 3; ++axis) camera.position[axis] = position[axis];
                    transform.block<3, 1>(0, 3) = -rotation * position;
                    photara::splat::RasterizeOptions raster;
                    raster.active_sh_degree = model.sh_degree;
                    raster.require_depth = false;
                    const auto color = photara::splat::Rasterizer().forward(model, camera, raster).color.to_vector();
                    for (std::size_t p = 0; p < area; ++p)
                        for (std::size_t c = 0; c < 3; ++c)
                            image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                                std::lround(255.F * std::clamp(color[c * area + p], 0.F, 1.F)));
                    photara::io::save_rgb_png(image, output / ("dolly_" + std::to_string(percent) +
                        "_" + std::to_string(i) + ".png"));
                }
            }
            // Unmeasured novel-camera diagnostic for floaters. Every model
            // receives the same midpoint pose, independent of its geometry.
            if (save_novel && i + split < dataset.scene.views.size()) {
                auto camera = target.camera;
                const auto peer = photara::splat::make_training_view(
                    dataset.scene.views[i + split], options).camera;
                const Eigen::Map<const Eigen::Matrix4f> m0(camera.world_to_camera.data());
                const Eigen::Map<const Eigen::Matrix4f> m1(peer.world_to_camera.data());
                const Eigen::Quaternionf q0(m0.block<3, 3>(0, 0));
                const Eigen::Quaternionf q1(m1.block<3, 3>(0, 0));
                const Eigen::Matrix3f rotation = q0.slerp(0.5F, q1).normalized().toRotationMatrix();
                Eigen::Vector3f position;
                for (int axis = 0; axis < 3; ++axis)
                    position[axis] = camera.position[axis] =
                        0.5F * (camera.position[axis] + peer.position[axis]);
                Eigen::Map<Eigen::Matrix4f> transform(camera.world_to_camera.data());
                transform.setIdentity();
                transform.block<3, 3>(0, 0) = rotation;
                transform.block<3, 1>(0, 3) = -rotation * position;
                photara::splat::RasterizeOptions raster;
                raster.active_sh_degree = model.sh_degree;
                raster.require_depth = false;
                const auto novel = photara::splat::Rasterizer().forward(model, camera, raster).color.to_vector();
                for (std::size_t p = 0; p < area; ++p)
                    for (std::size_t c = 0; c < 3; ++c)
                        image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                            std::lround(255.F * std::clamp(novel[c * area + p], 0.F, 1.F)));
                photara::io::save_rgb_png(image, output / ("novel_" + std::to_string(i) + ".png"));
            }
            const auto metrics = photara::splat::render_evaluation_png(
                model, dataset.scene.views[i],
                output / ("view_" + std::to_string(i) + ".png"), options);
            auto full_options = options;
            full_options.ignore_undistortion_border = false;
            const auto full = photara::splat::render_evaluation_png(
                model, dataset.scene.views[i],
                output / ("view_" + std::to_string(i) + ".png"), full_options);
            double valid_fraction = 1.0;
            if (target.has_mask) {
                const auto mask = target.mask.to_vector();
                valid_fraction = std::accumulate(mask.begin(), mask.end(), 0.0) / area;
            }
            csv << i << ',' << metrics.psnr << ',' << metrics.ssim << ','
                << full.psnr << ',' << full.ssim << ',' << valid_fraction << ',' << metrics.foreground_psnr << '\n';
            psnr += metrics.psnr;
            ssim += metrics.ssim;
            full_psnr += full.psnr;
            full_ssim += full.ssim;
            ++count;
        }
        if (count == 0) throw std::runtime_error("No evaluation cameras");
        std::cout << "views=" << count << " valid_psnr=" << psnr / count
                  << " valid_ssim=" << ssim / count << " full_psnr=" << full_psnr / count
                  << " full_ssim=" << full_ssim / count << " gaussians=" << model.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
