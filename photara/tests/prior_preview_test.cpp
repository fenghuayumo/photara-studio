#include "prior_preview_data.hpp"
#include <chrono>
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
        check(data.depth.pixels[0]==24 && data.normal.pixels[0]==24,"Invalid pixels were not dark");
        check(data.normal.pixels[3]==255 && data.normal.pixels[4]==128 && data.normal.pixels[5]==128,"Camera normal color mapping mismatch");
        check(data.normal.pixels[11]==0,"Negative Z normal must retain its sign");
        check(data.depth_near==1 && data.depth_far>data.depth_near,"Invalid depth display range");
        const auto manual=editor::colorize_prior_depth(data.prior,1.F,3.F);
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
        prior.depth={2,2,2,2};prior.normals.clear();photara::splat::save_training_prior(depth,image,prior);
        data=editor::load_prior_preview(depth);
        check(data.normal.pixels.empty() && data.depth_far>data.depth_near,"Depth-only constant-map preview failed");
        prior.depth={0,0,0,0};photara::splat::save_training_prior(depth,image,prior);
        data=editor::load_prior_preview(depth);
        check(data.depth.pixels[0]==24,"All-invalid depth preview failed");
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
