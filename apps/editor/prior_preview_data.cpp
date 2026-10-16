#include "prior_preview_data.hpp"
#include "splat_render/include/splat_render/display_color.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace editor {
namespace {
template<class T> void read(std::istream& input,T& value) {
    input.read(reinterpret_cast<char*>(&value),sizeof(value));
    if(!input) throw std::runtime_error("Truncated training prior");
}
std::uint32_t view_id(const std::filesystem::path& file) {
    const auto stem=file.stem().string();
    if(!stem.starts_with("view_") || stem.size()<=5) throw std::invalid_argument("Invalid prior filename");
    const auto number=stem.substr(5);
    if(number.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("Invalid prior filename");
    const auto value=std::stoull(number);
    if(value>UINT32_MAX) throw std::invalid_argument("Invalid prior view ID");
    return static_cast<std::uint32_t>(value);
}
std::uint8_t byte(float value) {
    return static_cast<std::uint8_t>(std::lround(std::clamp(value,0.F,1.F)*255.F));
}
}
std::vector<std::filesystem::path> list_prior_previews(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    if(!std::filesystem::exists(directory)) return files;
    for(const auto& entry:std::filesystem::directory_iterator(directory)) {
        if(!entry.is_regular_file() || entry.path().extension()!=".prior") continue;
        try { (void)view_id(entry.path()); files.push_back(entry.path()); }
        catch(const std::exception&) {}
    }
    std::sort(files.begin(),files.end(),[](const auto& a,const auto& b){return view_id(a)<view_id(b);});
    return files;
}
std::filesystem::path find_prior_preview(
    const std::filesystem::path& directory, const std::filesystem::path& source) {
    if (directory.empty() || source.empty()) return {};
    for (const auto& file : list_prior_previews(directory)) {
        try {
            std::ifstream input(file, std::ios::binary);
            char magic[8]{}; input.read(magic, 8);
            const std::string version(magic, 8);
            if (version != "PHPRIOR1" && version != "PHPRIOR2") continue;
            if (version == "PHPRIOR2") {
                std::uint32_t maps{}; read(input, maps);
                if (!maps || (maps & ~3U)) continue;
            }
            photara::splat::Camera camera;
            read(input, camera.width); read(input, camera.height);
            read(input, camera.fx); read(input, camera.fy); read(input, camera.cx); read(input, camera.cy);
            for (float& value : camera.world_to_camera) read(input, value);
            std::uint32_t length{}; read(input, length);
            if (!length || length > 32768) continue;
            std::string name(length, '\0'); input.read(name.data(), length);
            if (!input || name.find('\0') != std::string::npos) continue;
            std::error_code error;
            if (std::filesystem::equivalent(std::filesystem::u8path(name), source, error) && !error)
                return file;
        } catch (const std::exception&) {
            // Preparation may still be replacing a cache file.
        }
    }
    return {};
}
photara::splat::Camera prior_preview_camera(const photara::splat::Camera& source) {
    if (!source.width || !source.height) throw std::invalid_argument("Invalid prior camera dimensions");
    auto camera = source;
    const float factor = std::min(1.F, 1024.F / std::max(source.width, source.height));
    camera.width = std::max(1U, static_cast<unsigned>(source.width * factor));
    camera.height = std::max(1U, static_cast<unsigned>(source.height * factor));
    const float sx = static_cast<float>(camera.width) / source.width;
    const float sy = static_cast<float>(camera.height) / source.height;
    camera.fx *= sx; camera.fy *= sy; camera.cx *= sx; camera.cy *= sy;
    const auto& m = camera.world_to_camera;
    camera.position = {-(m[0]*m[12]+m[1]*m[13]+m[2]*m[14]),
                       -(m[4]*m[12]+m[5]*m[13]+m[6]*m[14]),
                       -(m[8]*m[12]+m[9]*m[13]+m[10]*m[14])};
    return camera;
}
std::optional<PriorPreviewPixel> sample_prior_preview(const PriorPreviewData& data, float u, float v) {
    const auto& prior = data.prior;
    if (!std::isfinite(u) || !std::isfinite(v) || u < 0.F || u >= 1.F || v < 0.F || v >= 1.F ||
        !prior.camera.width || !prior.camera.height) return {};
    const auto preview = prior_preview_camera(prior.camera);
    const auto px = std::min(preview.width-1, static_cast<unsigned>(u*preview.width));
    const auto py = std::min(preview.height-1, static_cast<unsigned>(v*preview.height));
    PriorPreviewPixel pixel;
    // Match the source texel used by the downsampled preview, including zoom/pan.
    pixel.x = std::min(prior.camera.width-1, static_cast<unsigned>((px+.5)*prior.camera.width/preview.width));
    pixel.y = std::min(prior.camera.height-1, static_cast<unsigned>((py+.5)*prior.camera.height/preview.height));
    const auto count = static_cast<std::size_t>(prior.camera.width)*prior.camera.height;
    const auto index = static_cast<std::size_t>(pixel.y)*prior.camera.width+pixel.x;
    if (prior.depth.size() == count) {
        pixel.depth = prior.depth[index];
        pixel.depth_valid = std::isfinite(*pixel.depth) && *pixel.depth > 0.F;
    }
    if (prior.normals.size() == count*3) {
        const std::array<float,3> n{prior.normals[index],prior.normals[count+index],prior.normals[2*count+index]};
        pixel.normal = n;
        pixel.normal_valid = std::isfinite(n[0]+n[1]+n[2]) && n[0]*n[0]+n[1]*n[1]+n[2]*n[2] > .25F;
    }
    return pixel;
}
photara::io::RgbImage colorize_prior_depth(const photara::splat::TrainingPrior& prior,float near_depth,float far_depth,bool grayscale) {
    if(prior.depth.empty()) return {};
    if(!std::isfinite(near_depth) || !std::isfinite(far_depth) || near_depth<0 || far_depth<=near_depth)
        throw std::invalid_argument("Invalid depth display range");
    const auto& camera=prior.camera;
    const float factor=std::min(1.F,1024.F/std::max(camera.width,camera.height));
    const auto width=std::max(1U,static_cast<unsigned>(camera.width*factor));
    const auto height=std::max(1U,static_cast<unsigned>(camera.height*factor));
    photara::io::RgbImage image{width,height,std::vector<std::uint8_t>(3ull*width*height)};
    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
        const auto sx=std::min(camera.width-1,static_cast<unsigned>((x+.5)*camera.width/width));
        const auto sy=std::min(camera.height-1,static_cast<unsigned>((y+.5)*camera.height/height));
        const float depth=prior.depth[static_cast<std::size_t>(sy)*camera.width+sx];
        auto* rgb=image.pixels.data()+3ull*(y*width+x);
        rgb[0]=rgb[1]=rgb[2]=0;
        if(depth>0) {
            const auto color=splat_render::depth_color((depth-near_depth)/(far_depth-near_depth),grayscale);
            for(int c=0;c<3;++c) rgb[c]=byte(color[c]);
        }
    }
    return image;
}
PriorPreviewData load_prior_preview(const std::filesystem::path& file) {
    // Read only the small self-description here, then use the training loader
    // for map sizes, camera, source identity and finite/unit-vector validation.
    std::ifstream input(file,std::ios::binary);
    char magic[8]{}; input.read(magic,8);
    const std::string version(magic,8);
    if(version!="PHPRIOR1" && version!="PHPRIOR2") throw std::runtime_error("Missing or invalid training prior");
    std::uint32_t maps=3;
    if(version=="PHPRIOR2") read(input,maps);
    if(!maps || (maps&~3U)) throw std::runtime_error("Invalid training prior map flags");
    photara::splat::Camera camera;
    read(input,camera.width); read(input,camera.height);
    read(input,camera.fx); read(input,camera.fy); read(input,camera.cx); read(input,camera.cy);
    for(float& value:camera.world_to_camera) read(input,value);
    std::uint32_t length{}; read(input,length);
    if(!length || length>32768) throw std::runtime_error("Invalid training prior source name");
    std::string name(length,'\0'); input.read(name.data(),length);
    if(!input || name.find('\0')!=std::string::npos) throw std::runtime_error("Invalid training prior source name");
    PriorPreviewData result;
    result.source=std::filesystem::u8path(name);
    result.prior=photara::splat::load_training_prior(file,result.source,camera,(maps&1U)!=0,(maps&2U)!=0);
    const auto& prior=result.prior;
    const std::size_t count=static_cast<std::size_t>(camera.width)*camera.height;
    std::vector<float> depths;
    for(float value:prior.depth) if(value>0) depths.push_back(value);
    if(!depths.empty()) {
        const auto percentile=[&](float p) {
            const auto index=static_cast<std::size_t>((depths.size()-1)*p);
            std::nth_element(depths.begin(),depths.begin()+index,depths.end());
            return depths[index];
        };
        result.depth_near=percentile(.02F); result.depth_far=percentile(.98F);
        if(result.depth_far<=result.depth_near)
            result.depth_far=result.depth_near+std::max(1e-4F,result.depth_near*.01F);
    }
    else if(!prior.depth.empty()) result.depth_far=1.F;
    result.depth=colorize_prior_depth(prior,result.depth_near,result.depth_far);
    const float factor=std::min(1.F,1024.F/std::max(camera.width,camera.height));
    const auto width=std::max(1U,static_cast<unsigned>(camera.width*factor));
    const auto height=std::max(1U,static_cast<unsigned>(camera.height*factor));
    const auto blank=[&] { return photara::io::RgbImage{width,height,std::vector<std::uint8_t>(3ull*width*height)}; };
    if(!prior.normals.empty()) result.normal=blank();
    for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
        const auto sx=std::min(camera.width-1,static_cast<unsigned>((x+.5)*camera.width/width));
        const auto sy=std::min(camera.height-1,static_cast<unsigned>((y+.5)*camera.height/height));
        const auto src=static_cast<std::size_t>(sy)*camera.width+sx, dst=3ull*(y*width+x);
        if(!prior.normals.empty()) {
            auto* rgb=result.normal.pixels.data()+dst;
            const float nx=prior.normals[src],ny=prior.normals[count+src],nz=prior.normals[2*count+src];
            if(nx*nx+ny*ny+nz*nz>.25F) {
                const auto color=splat_render::normal_color(nx,ny,nz);
                for(int c=0;c<3;++c) rgb[c]=byte(color[c]);
            } else {rgb[0]=rgb[1]=rgb[2]=0;}
        }
    }
    auto photo=photara::io::load_rgb_with_minimum_size(result.source,width,height);
    result.photo=blank();
    photara::io::resize_bilinear(photo.pixels.data(),photo.width,photo.height,3,result.photo.pixels.data(),width,height);
    return result;
}
} // namespace editor
