#include "splat/formats.hpp"
#include "splat/rasterizer.hpp"
#include "io/image.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: splat_multiscale_audit MODEL NATIVE_AUDIT_JSON OUTPUT");
        using namespace photara::splat;
        using Json=nlohmann::json;
        auto model=load_gaussians(argv[1]);
        const auto audit=Json::parse(std::ifstream(argv[2]));
        const auto profile=audit["profile"].get<std::string>();
        const auto filter=audit["filter"].get<std::string>();
        TrainingOptions training;
        if (profile=="mip-splatting") training.rasterization={true,.999F,.01F,.5F,true};
        else if (profile=="z") training.rasterization.sort_by_camera_z=true;
        else if (profile=="alpha") training.rasterization.alpha_clip=.999F;
        else if (profile=="near") training.rasterization.near_plane=.01F;
        else if (profile=="pixel") training.rasterization.pixel_center_offset=.5F;
        else if (profile!="legacy") throw std::runtime_error("Unknown raster profile");
        if (filter=="classic") training.filter_mode=SplatFilterMode::classic;
        else if (filter=="mip2d") training.filter_mode=SplatFilterMode::mip_2d;
        else if (filter=="mip3d") training.filter_mode=SplatFilterMode::mip_3d;
        else if (filter=="mip") training.filter_mode=SplatFilterMode::mip;
        else if (filter!="legacy") throw std::runtime_error("Unknown filter mode");
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        RasterizeOptions raster;
        configure_rasterization(raster,training);
        raster.active_sh_degree=model.sh_degree; raster.require_depth=false;
        raster.record_backward_state=false;
        Rasterizer renderer;
        Json results={{"native_audit",std::filesystem::absolute(argv[2]).string()},
                      {"views",Json::array()}};
        for (const auto& view : audit["views"]) {
            for (float scale : {1.F,.5F,.25F}) {
                Camera camera;
                camera.world_to_camera=view["world_to_camera_column_major"].get<std::array<float,16>>();
                camera.position=view["position"].get<std::array<float,3>>();
                const auto w=view["width"].get<unsigned>(), h=view["height"].get<unsigned>();
                camera.width=static_cast<unsigned>(std::lround(w*scale));
                camera.height=static_cast<unsigned>(std::lround(h*scale));
                const float sx=float(camera.width)/w, sy=float(camera.height)/h;
                camera.fx=view["fx"].get<float>()*sx;
                camera.fy=view["fy"].get<float>()*sy;
                // Resize in the native audit's integer sample space, then
                // convert back to the rasterizer's requested sample convention.
                // This avoids mixing a half-pixel shift with a scale change.
                camera.cx=(view["cx"].get<float>()+.5F)*sx-.5F+raster.rasterization.pixel_center_offset;
                camera.cy=(view["cy"].get<float>()+.5F)*sy-.5F+raster.rasterization.pixel_center_offset;
                auto frame=renderer.forward(model,camera,raster);
                const auto color=frame.color.to_vector();
                photara::io::RgbImage image;
                image.width=camera.width; image.height=camera.height;
                const auto pixels=std::size_t(image.width)*image.height;
                image.pixels.resize(3*pixels);
                for (std::size_t p=0;p<pixels;++p)
                    for (std::size_t c=0;c<3;++c)
                        image.pixels[3*p+c]=static_cast<std::uint8_t>(std::lround(255.F*std::clamp(color[c*pixels+p],0.F,1.F)));
                const auto index=view["index"].get<unsigned>();
                const auto name="render_"+std::to_string(index)+"_"+std::to_string(image.width)+".png";
                photara::io::save_rgb_png(image,output/name);
                results["views"].push_back({{"index",index},{"width",image.width},{"height",image.height},{"file",name}});
            }
        }
        std::ofstream(output/"multiscale.json") << results.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
