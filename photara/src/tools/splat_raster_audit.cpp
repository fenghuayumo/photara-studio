#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#include "mvs/export.hpp"
#include "io/image.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8)
            throw std::runtime_error("Usage: splat_raster_audit COLMAP IMAGES MODEL OUTPUT PROFILE FILTER [--mesh]");
        using namespace photara;
        using namespace splat;
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output);
        TrainingOptions options;
        options.use_source_resolution = true;
        options.max_image_dimension = 0;
        options.progressive_resolution = false;
        options.ignore_undistortion_border = true;
        options.use_mask = false;
        const std::string profile(argv[5]), filter(argv[6]);
        if (profile == "mip-splatting") options.rasterization = {true, .999F, .01F, .5F, true};
        else if (profile == "z") options.rasterization.sort_by_camera_z = true;
        else if (profile == "alpha") options.rasterization.alpha_clip = .999F;
        else if (profile == "near") options.rasterization.near_plane = .01F;
        else if (profile == "pixel") options.rasterization.pixel_center_offset = .5F;
        else if (profile != "legacy") throw std::runtime_error("Unknown profile");
        if (filter == "classic") options.filter_mode = SplatFilterMode::classic;
        else if (filter == "mip2d") options.filter_mode = SplatFilterMode::mip_2d;
        else if (filter == "mip3d") options.filter_mode = SplatFilterMode::mip_3d;
        else if (filter == "mip") options.filter_mode = SplatFilterMode::mip;
        else if (filter != "legacy") throw std::runtime_error("Unknown filter");
        DatasetLoadRequest request;
        request.source = argv[1]; request.image_directory = argv[2];
        auto scene = load_splat_dataset(request).scene;
        auto model = load_gaussians(argv[3]);
        RasterizeOptions raster;
        configure_rasterization(raster, options);
        raster.active_sh_degree = model.sh_degree;
        raster.record_backward_state = false;
        raster.require_depth = true;
        Rasterizer renderer;
        Json report{{"profile", profile}, {"filter", filter}, {"gaussians", model.size()},
                    {"views", Json::array()}, {"timing", Json::array()}};
        const auto write_floats = [&](const std::filesystem::path& path, const auto& tensor) {
            const auto data = tensor.to_vector();
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()), data.size()*sizeof(float));
            if (!file) throw std::runtime_error("Cannot write " + path.string());
        };
        // Fixed split shared with Trainer: every eighth registered image is held out.
        for (std::size_t i = 0; i < scene.views.size(); i += 8) {
            const auto target = make_training_view(scene.views[i], options);
            const auto stem = std::to_string(i);
            const auto metrics = render_evaluation_png(model, scene.views[i], output/ ("render_"+stem+".png"), options);
            const auto sample_camera = raster_sample_camera(target.camera, raster);
            auto rendered = renderer.forward(model, target.camera, raster);
            write_floats(output/("depth_"+stem+".f32"), rendered.median_depth);
            write_floats(output/("alpha_"+stem+".f32"), rendered.alpha);
            write_floats(output/("normal_"+stem+".f32"), rendered.normal);
            if (target.has_mask) write_floats(output/("valid_"+stem+".f32"), target.mask);
            io::RgbImage truth;
            truth.width = target.camera.width; truth.height = target.camera.height;
            const auto rgb = target.rgb.to_vector();
            const auto area = std::size_t(truth.width)*truth.height;
            truth.pixels.resize(3*area);
            for (std::size_t p=0; p<area; ++p)
                for (std::size_t c=0; c<3; ++c)
                    truth.pixels[3*p+c] = static_cast<std::uint8_t>(std::lround(255.F*std::clamp(rgb[c*area+p],0.F,1.F)));
            io::save_rgb_png(truth, output/("target_"+stem+".png"));
            Json view{{"index", i}, {"name", scene.views[i].path.filename().string()},
                      {"width", truth.width}, {"height", truth.height},
                      {"fx", sample_camera.fx}, {"fy", sample_camera.fy},
                      {"cx", sample_camera.cx}, {"cy", sample_camera.cy},
                      {"world_to_camera_column_major", sample_camera.world_to_camera},
                      {"position", sample_camera.position}, {"has_mask", target.has_mask},
                      {"psnr", metrics.psnr}, {"valid_psnr", metrics.foreground_psnr}, {"ssim", metrics.ssim}};
            report["views"].push_back(view);
            // Synchronized device-resident forward latency, excluding readback,
            // target loading, PNG encoding and backward snapshot bookkeeping.
            if (report["timing"].size() < 6) {
                rendered.context.impl.reset();
                for (bool geometry : {false, true}) {
                    auto timed = raster; timed.require_depth = geometry;
                    std::vector<double> times;
                    for (int repeat=-2; repeat<5; ++repeat) {
                        cudaDeviceSynchronize();
                        const auto start=Clock::now();
                        auto frame=renderer.forward(model, target.camera, timed);
                        if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("CUDA forward failed");
                        const auto ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                        if (repeat>=0) times.push_back(ms);
                    }
                    std::sort(times.begin(),times.end());
                    report["timing"].push_back({{"view",i},{"geometry",geometry},{"median_ms",times[times.size()/2]}, {"samples_ms",times}});
                }
            }
        }
        std::ofstream(output/"audit.json") << report.dump(2) << '\n';
        if (argc == 8) {
            if (std::string(argv[7]) != "--mesh") throw std::runtime_error("Unknown audit argument");
            SplatMeshOptions mesh_options;
            mesh_options.fusion.mesh_method = mvs::MeshMethod::tsdf;
            mesh_options.fusion.mesh_tsdf_voxel_scale = 1.F;
            mesh_options.fusion.mesh_tsdf_smooth_iters = 2;
            const auto start=Clock::now();
            auto mesh=extract_splat_mesh(model,scene,options,mesh_options);
            mvs::save_mesh_ply(mesh.mesh,output/"mesh.ply");
            report["mesh"]={{"vertices",mesh.mesh.vertices.size()},{"faces",mesh.mesh.faces.size()},
                            {"valid_depth_pixels",mesh.valid_depth_pixels},
                            {"seconds",std::chrono::duration<double>(Clock::now()-start).count()}};
        }
        std::ofstream(output/"audit.json") << report.dump(2) << '\n';
        std::cout << "audit completed views=" << report["views"].size() << " gaussians=" << model.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
