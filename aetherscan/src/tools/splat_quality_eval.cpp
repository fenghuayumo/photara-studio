#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#include "splat/rasterizer.hpp"
#include "io/image.hpp"
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Re-render models from different binaries through one image loader and
// rasterizer. Positional arguments keep this experiment tool independent of
// the reconstruction CLI and its changing defaults.
int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::runtime_error("Usage: splat_quality_eval DATASET IMAGES MODEL OUTPUT_DIR SPLIT_EVERY");
        const unsigned split = std::stoul(argv[5]);
        if (split < 2) throw std::runtime_error("SPLIT_EVERY must be >= 2");
        aetherscan::splat::DatasetLoadRequest request;
        request.source = argv[1];
        request.image_directory = argv[2];
        const auto dataset = aetherscan::splat::load_splat_dataset(request);
        const auto model = aetherscan::splat::load_gaussians(argv[3]);
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output);
        std::ofstream csv(output / "metrics.csv");
        if (!csv) throw std::runtime_error("Cannot create metrics.csv");
        csv << "view,psnr,ssim,full_psnr,full_ssim,valid_fraction\n";
        aetherscan::splat::TrainingOptions options;
        options.use_mask = false;
        options.use_source_resolution = true;
        options.ignore_undistortion_border = true;
        options.progressive_resolution = false;
        double psnr = 0, ssim = 0, full_psnr = 0, full_ssim = 0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < dataset.scene.views.size(); i += split) {
            const auto target = aetherscan::splat::make_training_view(dataset.scene.views[i], options);
            const auto pixels = target.rgb.to_vector();
            aetherscan::io::RgbImage image;
            image.width = target.camera.width;
            image.height = target.camera.height;
            const std::size_t area = image.width * image.height;
            image.pixels.resize(3 * area);
            for (std::size_t p = 0; p < area; ++p)
                for (std::size_t c = 0; c < 3; ++c)
                    image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                        std::lround(255.F * std::clamp(pixels[c * area + p], 0.F, 1.F)));
            aetherscan::io::save_rgb_png(image, output / ("target_" + std::to_string(i) + ".png"));
            // Unmeasured novel-camera diagnostic for floaters. Every model
            // receives the same midpoint pose, independent of its geometry.
            if (i + split < dataset.scene.views.size()) {
                auto camera = target.camera;
                const auto peer = aetherscan::splat::make_training_view(
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
                aetherscan::splat::RasterizeOptions raster;
                raster.active_sh_degree = model.sh_degree;
                raster.require_depth = false;
                const auto novel = aetherscan::splat::Rasterizer().forward(model, camera, raster).color.to_vector();
                for (std::size_t p = 0; p < area; ++p)
                    for (std::size_t c = 0; c < 3; ++c)
                        image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                            std::lround(255.F * std::clamp(novel[c * area + p], 0.F, 1.F)));
                aetherscan::io::save_rgb_png(image, output / ("novel_" + std::to_string(i) + ".png"));
            }
            const auto metrics = aetherscan::splat::render_evaluation_png(
                model, dataset.scene.views[i],
                output / ("view_" + std::to_string(i) + ".png"), options);
            auto full_options = options;
            full_options.ignore_undistortion_border = false;
            const auto full = aetherscan::splat::render_evaluation_png(
                model, dataset.scene.views[i],
                output / ("view_" + std::to_string(i) + ".png"), full_options);
            double valid_fraction = 1.0;
            if (target.mask_is_validity) {
                const auto mask = target.mask.to_vector();
                valid_fraction = std::accumulate(mask.begin(), mask.end(), 0.0) / area;
            }
            csv << i << ',' << metrics.psnr << ',' << metrics.ssim << ','
                << full.psnr << ',' << full.ssim << ',' << valid_fraction << '\n';
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
