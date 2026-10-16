#include "prior_preview_data.hpp"
#include "splat_render/include/splat_render/display_color.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
void check(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
template<class F> void fails(F action) {
    try {action();} catch(const std::exception&) {return;}
    throw std::runtime_error("Expected invalid preview failure");
}
}
int main(int argc,char** argv) {
    if(argc==3) {
        try {
            const auto data=editor::load_prior_preview(std::filesystem::u8path(argv[1]));
            const auto output=std::filesystem::u8path(argv[2]);
            std::filesystem::create_directories(output);
            photara::io::save_rgb_png(data.photo,output/"photo.png");
            if(!data.depth.pixels.empty()) photara::io::save_rgb_png(data.depth,output/"depth.png");
            if(!data.normal.pixels.empty()) photara::io::save_rgb_png(data.normal,output/"normal.png");
            std::cout<<"Preview validated: "<<data.prior.camera.width<<'x'<<data.prior.camera.height
                <<" depth range="<<data.depth_near<<".."<<data.depth_far<<'\n';
            return 0;
        } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
    }
    const auto root=std::filesystem::temp_directory_path()/
        ("photara-prior-preview-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        check(splat_render::depth_color(0.F,false)==std::array<float,3>{.5F,0,0} &&
            splat_render::depth_color(.5F,false)==std::array<float,3>{.5F,1,.5F} &&
            splat_render::depth_color(1.F,false)==std::array<float,3>{0,0,.5F},
            "Depth palette must retain the original AI-prior rainbow");
        check(splat_render::normal_color(0,0,-2)==std::array<float,3>{.5F,.5F,0} &&
            splat_render::normal_color(0,0,2)==std::array<float,3>{.5F,.5F,1},
            "Normal colors must preserve camera-space direction and normalize magnitude");
        std::filesystem::create_directories(root);
        const auto image=root/"source.png";
        photara::io::save_rgb_png({2,2,{255,0,0, 0,255,0, 0,0,255, 255,255,255}},image);
        photara::splat::TrainingPrior prior;
        prior.camera.width=prior.camera.height=2;
        prior.camera.fx=prior.camera.fy=10; prior.camera.cx=prior.camera.cy=.5F;
        prior.depth={0,1,2,3};
        prior.normals={0,1,0,0, 0,0,1,0, 0,0,0,-1};
        const auto both=root/"view_10.prior", normal=root/"view_2.prior", depth=root/"view_25.prior";
        photara::splat::save_training_prior(both,image,prior);
        check(editor::find_prior_preview(root,image)==both,"Prior did not match its source photo");
        check(editor::find_prior_preview(root/"missing",image).empty(),"Missing cache matched a photo");
        check(editor::find_prior_preview(root,{}).empty(),"Empty source matched a prior");
        const auto other_directory=root/"other";
        std::filesystem::create_directory(other_directory);
        const auto other_image=other_directory/image.filename();
        photara::io::save_rgb_png({1,1,{0,0,0}},other_image);
        check(editor::find_prior_preview(root,other_image).empty(),"Prior matched a different photo with the same basename");
        std::filesystem::remove(other_image);
        std::filesystem::remove(other_directory);
        auto data=editor::load_prior_preview(both);
        check(data.source==image && data.photo.width==2 && data.depth.width==2 && data.normal.width==2,"Preview dimensions/source mismatch");
        check(data.depth.pixels[0]==0 && data.normal.pixels[0]==0,"Invalid pixels were not black");
        check(data.normal.pixels[3]==255 && data.normal.pixels[4]==128 && data.normal.pixels[5]==128,"Camera normal color mapping mismatch");
        check(data.normal.pixels[11]==0,"Negative Z normal must retain its sign");
        check(data.depth_near==1 && data.depth_far>data.depth_near,"Invalid depth display range");
        const auto pixel=editor::sample_prior_preview(data,.75F,.25F);
        check(pixel && pixel->x==1 && pixel->y==0 && pixel->depth==1.F && pixel->depth_valid &&
            pixel->normal && *pixel->normal==std::array<float,3>{1,0,0} && pixel->normal_valid,
            "Hover did not read the displayed source pixel");
        const auto invalid_pixel=editor::sample_prior_preview(data,.1F,.1F);
        check(invalid_pixel && invalid_pixel->depth==0.F && !invalid_pixel->depth_valid &&
            invalid_pixel->normal && !invalid_pixel->normal_valid,"Invalid pixels were reported as valid");
        check(!editor::sample_prior_preview(data,-.1F,.5F) && !editor::sample_prior_preview(data,1.F,.5F) &&
            !editor::sample_prior_preview({},.5F,.5F),"Hover outside the image sampled a pixel");
        editor::PriorPreviewData reduced;
        reduced.prior.camera.width=2048; reduced.prior.camera.height=2;
        reduced.prior.depth.resize(4096);
        for(unsigned y=0;y<2;++y) for(unsigned x=0;x<2048;++x) reduced.prior.depth[y*2048+x]=x+10*y;
        const auto reduced_pixel=editor::sample_prior_preview(reduced,0.F,0.F);
        check(reduced_pixel && reduced_pixel->x==1 && reduced_pixel->y==1 && reduced_pixel->depth==11.F,
            "Hover did not match the downsampled preview texel center");
        const auto manual=editor::colorize_prior_depth(data.prior,1.F,3.F);
        for (std::size_t p=1;p<4;++p) {
            const auto expected=splat_render::depth_color((data.prior.depth[p]-1.F)/2.F,false);
            for(int c=0;c<3;++c)
                check(manual.pixels[3*p+c]==static_cast<unsigned char>(std::lround(expected[c]*255.F)),
                    "Prior and rasterized depth palettes differ");
        }
        const auto gray=editor::colorize_prior_depth(data.prior,1.F,3.F,true);
        check(gray.pixels[3]==255 && gray.pixels[6]==128 && gray.pixels[9]==0,
            "Prior grayscale range differs from rasterized depth");
        auto full_camera=prior.camera;
        full_camera.width=2048; full_camera.height=1024;
        full_camera.fx=1000;full_camera.fy=800;full_camera.cx=900;full_camera.cy=400;
        full_camera.world_to_camera={1,0,0,0, 0,1,0,0, 0,0,1,0, -2,-3,-4,1};
        const auto preview_camera=editor::prior_preview_camera(full_camera);
        check(preview_camera.width==1024 && preview_camera.height==512 && preview_camera.fx==500 &&
            preview_camera.fy==400 && preview_camera.cx==450 && preview_camera.cy==200,
            "Comparison camera did not preserve prior intrinsics at preview resolution");
        check(preview_camera.world_to_camera==full_camera.world_to_camera &&
            preview_camera.position==std::array<float,3>{2,3,4},"Comparison camera changed the SfM pose");
        check(manual.pixels[3]>manual.pixels[5] && manual.pixels[11]>manual.pixels[9],"Depth near/far colors reversed");
        fails([&]{editor::colorize_prior_depth(data.prior,1.F,1.F);});
        {
            std::ifstream input(both,std::ios::binary);
            const std::vector<char> bytes{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
            std::ofstream output(root/"legacy.prior",std::ios::binary);
            output.write("PHPRIOR1",8);output.write(bytes.data()+12,bytes.size()-12);output.close();
            const auto legacy=editor::load_prior_preview(root/"legacy.prior");
            check(legacy.depth.pixels==data.depth.pixels && legacy.normal.pixels==data.normal.pixels,"Legacy prior preview mismatch");
        }
        prior.depth.clear(); photara::splat::save_training_prior(normal,image,prior);
        check(editor::find_prior_preview(root,image)==normal,"Source lookup must not assume the photo index matches the cache view ID");
        data=editor::load_prior_preview(normal);
        check(data.depth.pixels.empty() && !data.normal.pixels.empty(),"Normal-only preview failed");
        const auto normal_pixel=editor::sample_prior_preview(data,.75F,.25F);
        check(normal_pixel && !normal_pixel->depth && normal_pixel->normal_valid,"Normal-only hover failed");
        prior.depth={2,2,2,2};prior.normals.clear();photara::splat::save_training_prior(depth,image,prior);
        data=editor::load_prior_preview(depth);
        check(data.normal.pixels.empty() && data.depth_far>data.depth_near,"Depth-only constant-map preview failed");
        const auto depth_pixel=editor::sample_prior_preview(data,.75F,.25F);
        check(depth_pixel && depth_pixel->depth_valid && !depth_pixel->normal,"Depth-only hover failed");
        prior.depth={0,0,0,0};photara::splat::save_training_prior(depth,image,prior);
        data=editor::load_prior_preview(depth);
        check(data.depth.pixels[0]==0,"All-invalid depth preview failed");
        std::ofstream(root/"view_invalid.prior")<<"incomplete";
        std::ofstream(root/"manifest.txt")<<"metadata";
        const auto files=editor::list_prior_previews(root);
        check(files.size()==3 && files[0]==normal && files[1]==both,"Prior views not numerically sorted");
        check(editor::list_prior_previews(root/"missing").empty(),"Missing cache did not return empty list");
        fails([&]{editor::load_prior_preview(root/"view_invalid.prior");});
        std::filesystem::resize_file(both,20);
        fails([&]{editor::load_prior_preview(both);});
        std::filesystem::remove(normal);
        std::filesystem::remove(depth);
        check(editor::find_prior_preview(root,image).empty(),"Truncated cache matched a source photo");
        photara::splat::save_training_prior(normal,image,prior);
        std::ofstream(image,std::ios::app|std::ios::binary)<<'x';
        fails([&]{editor::load_prior_preview(normal);});
        std::cout<<"Prior preview tests passed\n";
        for(const auto& entry:std::filesystem::directory_iterator(root)) std::filesystem::remove(entry.path());
        std::filesystem::remove(root);
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<" (fixtures: "<<root<<")\n";return 1;}
}
