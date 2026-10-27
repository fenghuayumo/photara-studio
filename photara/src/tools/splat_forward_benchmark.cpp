#include "splat/formats.hpp"
#include "splat/rasterizer.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

// Fixed model, fixed cameras, no dataset/image reads in the timing loop.
// A short audit is useful during training; this deliberately longer benchmark
// is run serially after training and CPU image analysis have stopped.
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: splat_forward_benchmark MODEL CAMERA_AUDIT_JSON OUTPUT_JSON");
        using namespace photara::splat;
        using Json=nlohmann::json;
        auto model=load_gaussians(argv[1]);
        const auto audit=Json::parse(std::ifstream(argv[2]));
        struct Configuration { const char* name; RasterizationPolicy policy; SplatFilterMode filter; };
        std::vector<Configuration> configurations{
            {"classic",{},SplatFilterMode::classic},
            {"alpha_999",{.999F,.2F,0.F,true},SplatFilterMode::classic},
            {"near_001",{.99F,.01F,0.F,true},SplatFilterMode::classic},
            {"pixel_half",{.99F,.2F,.5F,true},SplatFilterMode::classic},
            {"mip_splatting_classic",{.999F,.01F,.5F,true},SplatFilterMode::classic},
            {"mip_splatting_mip2d",{.999F,.01F,.5F,true},SplatFilterMode::mip_2d}};
        std::mt19937 random(20261026);
        Rasterizer renderer;
        Json results={{"model",argv[1]},{"gaussians",model.size()},{"measurements",Json::array()}};
        const auto synchronize=[] {
            if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("CUDA synchronization failed");
        };
        for (int round=0; round<2; ++round) {
            std::shuffle(configurations.begin(),configurations.end(),random);
            for (const auto& config : configurations) {
                TrainingOptions training;
                training.rasterization=config.policy; training.filter_mode=config.filter;
                RasterizeOptions raster;
                configure_rasterization(raster,training);
                raster.active_sh_degree=model.sh_degree; raster.record_backward_state=false;
                for (std::size_t v=0; v<std::min<std::size_t>(3,audit["views"].size()); ++v) {
                    const auto& view=audit["views"][v];
                    Camera camera;
                    camera.world_to_camera=view["world_to_camera_column_major"].get<std::array<float,16>>();
                    camera.position=view["position"].get<std::array<float,3>>();
                    camera.width=view["width"]; camera.height=view["height"];
                    camera.fx=view["fx"]; camera.fy=view["fy"];
                    // Audit stores sample-space intrinsics. The benchmark's
                    // Source is a standard-profile audit (offset zero).
                    camera.cx=view["cx"]; camera.cy=view["cy"];
                    if (audit["profile"]!="standard") throw std::runtime_error("Use standard audit cameras for a common convention");
                    for (bool geometry : {false,true}) {
                        raster.require_depth=geometry;
                        std::vector<double> times;
                        int instances=0;
                        for (int repeat=-20; repeat<50; ++repeat) {
                            synchronize();
                            const auto start=std::chrono::steady_clock::now();
                            auto frame=renderer.forward(model,camera,raster);
                            synchronize();
                            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                            if (repeat>=0) times.push_back(ms);
                            instances=frame.rendered_instances;
                        }
                        auto sorted=times; std::sort(sorted.begin(),sorted.end());
                        results["measurements"].push_back({{"round",round},{"variant",config.name},
                            {"view",view["index"]},{"geometry",geometry},{"instances",instances},
                            {"median_ms",sorted[25]},{"p10_ms",sorted[5]},{"p90_ms",sorted[45]},
                            {"samples_ms",times}});
                    }
                }
                std::cout << "round=" << round << " variant=" << config.name << '\n';
            }
        }
        std::ofstream(argv[3]) << results.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
