#include "splat/prior.hpp"
#include "native_prior.hpp"
#include "splat/trainer.hpp"
#include "training_data_loader.hpp"
#include "cuda_ops.hpp"
#include "checkpoint.hpp"
#if defined(TINYTENSOR_HAS_VULKAN)
#include "geometry_vulkan.hpp"
#include "vulkan/ops.hpp"
#endif
#include "io/image.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace photara::splat;
using tinytensor::Tensor;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void close(float a, float b) { if (!std::isfinite(a) || std::abs(a-b) > 2e-5F) {
    std::cerr << "actual=" << a << " expected=" << b << '\n';
    throw std::runtime_error("Prior loss/gradient mismatch");
} }
template<class F> void fails(F action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected invalid prior failure");
}

void native_priors() {
    const auto directory=std::filesystem::temp_directory_path()/
        ("photara_native_prior_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto source=directory/"image.png";
    photara::io::RgbImage image{64,32,std::vector<std::uint8_t>(64*32*3,128)};
    photara::io::save_rgb_png(image,source);
    for(auto model:{photara::CameraModel::opencv_fisheye,photara::CameraModel::equirectangular}) {
        Camera c; c.width=64;c.height=32;c.model=model;c.fx=c.fy=15;c.cx=32;c.cy=16;
        c.k1=.02F;c.k2=-.001F;c.k3=.0001F;c.k4=-.000001F;
        c.world_to_camera={0,0,-1,0, 0,1,0,0, 1,0,0,0, .1F,.2F,.3F,1};
        if(model==photara::CameraModel::equirectangular) {
            c.fx=float(64/(2*photara::k_pi));c.fy=float(32/photara::k_pi);c.k1=c.k2=c.k3=c.k4=0;
        }
        const auto fake=[](const photara::io::RgbImage& img,const Camera& tile) {
            photara::inference::GeometryPrior p;p.width=img.width;p.height=img.height;
            const auto count=std::size_t(p.width)*p.height;
            p.depth.resize(count);p.normals.resize(count);p.valid.assign(count,1);
            for(unsigned y=0;y<p.height;++y)for(unsigned x=0;x<p.width;++x) {
                const auto i=std::size_t(y)*p.width+x;
                const Eigen::Vector3d ray=Eigen::Vector3d{(x-tile.cx)/tile.fx,(y-tile.cy)/tile.fy,1}.normalized();
                p.depth[i]=float(ray.z()); // radius one, scaled to two scene units below.
                for(unsigned ch=0;ch<3;++ch)p.normals[i][ch]=float(-ray[ch]);
            }
            return p;
        };
        for(auto maps:{std::pair{true,false},std::pair{false,true},std::pair{true,true}}) {
            unsigned alignments=0;
            const auto prior=detail::predict_native_prior(image,{},c,maps.first,maps.second,fake,
                [&](const Camera&,const auto&){++alignments;return 2.;},128);
            check((alignments>0)==maps.first,"Normal-only prediction required scale support");
            check(prior.depth.empty()!=maps.first && prior.normals.empty()!=maps.second,"Independent native maps lost");
            for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x) {
                auto ray=detail::prior_source_ray(c,x,y);const auto i=std::size_t(y)*c.width+x;
                if(ray.isZero()) {if(maps.first)close(prior.depth[i],0);continue;}
                if(maps.first && prior.depth[i]>0) check(std::abs(prior.depth[i]-2*(model==photara::CameraModel::equirectangular?1:ray.z()))<.04,
                    "Native prior depth convention/coverage mismatch");
                if(maps.second && (prior.normals[i]!=0 || prior.normals[64*32+i]!=0 || prior.normals[2*64*32+i]!=0)) for(unsigned ch=0;ch<3;++ch)
                    check(std::abs(prior.normals[ch*64*32+i]+ray[ch])<.03,"Native normal rotation/coverage mismatch");
            }
            if(maps.first) check(std::count_if(prior.depth.begin(),prior.depth.end(),[](float x){return x>0;})>900,"Native prior coverage is too sparse");
            const auto path=directory/"native.prior";save_training_prior(path,source,prior);
            const auto roundtrip=load_training_prior(path,source,c,maps.first,maps.second);
            check(roundtrip.depth==prior.depth && roundtrip.normals==prior.normals,"Native prior cache roundtrip");
            auto wrong=c;wrong.k1+=.01F;fails([&]{(void)load_training_prior(path,source,wrong,maps.first,maps.second);});
            wrong=c;wrong.model=photara::CameraModel::pinhole;fails([&]{(void)load_training_prior(path,source,wrong,maps.first,maps.second);});
            auto small=c;small.width/=2;small.height/=2;small.fx*=.5F;small.fy*=.5F;
            small.cx=model==photara::CameraModel::equirectangular?c.cx*.5F:(c.cx+.5F)*.5F-.5F;
            small.cy=model==photara::CameraModel::equirectangular?c.cy*.5F:(c.cy+.5F)*.5F-.5F;
            const auto resized=resize_training_prior(prior,small);
            check(resized.camera.model==model,"Native prior resize changed projection");
        }
        photara::io::GrayImage zero{64,32,std::vector<std::uint8_t>(64*32)};
        fails([&]{(void)detail::predict_native_prior(image,zero,c,true,true,fake,[](const auto&,const auto&){return 2.;},32);});
        fails([&]{(void)detail::predict_native_prior(image,{},c,true,false,fake,[](const auto&,const auto&){return 0.;},32);});
    }
    // Delete only the known files created by this fixture.
    std::filesystem::remove(directory/"native.prior");std::filesystem::remove(source);std::filesystem::remove(directory);
}
void losses(tinytensor::Device device, photara::CameraModel camera_model=photara::CameraModel::pinhole) {
    constexpr std::size_t n = 64;
    std::vector<float> d(n,4.5F), td(n,4.F), normal(3*n), target(3*n), mask(n,1.F);
    for (std::size_t i=0;i<n;++i) { normal[2*n+i]=-.9F; target[i]=-.6F; target[2*n+i]=-.8F; }
    mask[0]=0; mask[3]=.5F; td[1]=0; target[2]=target[2*n+2]=0;
    RenderResult rendered;
    rendered.color=Tensor::zeros({3,8,8},device); rendered.alpha=Tensor::zeros({8,8},device);
    rendered.median_depth=Tensor::from_vector(d,{8,8},device);
    rendered.normal=Tensor::from_vector(normal,{3,8,8},device);
    TrainingView view;
    view.camera.model=camera_model;
    view.camera.width=view.camera.height=8; view.camera.fx=view.camera.fy=32;
    view.rgb=Tensor::zeros({3,8,8},device); view.depth=Tensor::from_vector(td,{8,8},device);
    view.normal=Tensor::from_vector(target,{3,8,8},device); view.mask=Tensor::from_vector(mask,{8,8},device);
    view.has_mask=true;
    for (const auto weights : {std::pair{0.F,0.F},std::pair{.05F,0.F},std::pair{0.F,.02F},std::pair{.05F,.02F}}) {
        std::cout << "Checking " << (device==tinytensor::Device::CUDA?"CUDA":"Vulkan")
            << " lambda_d=" << weights.first << " lambda_n=" << weights.second << '\n';
        TrainingOptions options; options.photometric_weight=0; options.lambda_d=weights.first;
        options.lambda_n=weights.second; options.use_mask=true;
        auto loss=detail::compute_training_loss(rendered,view,options,true,false,weights.first>0||weights.second>0);
        if (weights.first==0 && weights.second==0) {
            check(!loss.depth.is_valid() && !loss.normal.is_valid(),"Zero priors allocated geometry gradients");
            continue;
        }
        const auto gd=loss.depth.to_vector(), gn=loss.normal.to_vector();
        float expected_depth=0,expected_normal=0;
        for (std::size_t i=0;i<n;++i) {
            float depth_gradient=0;
            if (td[i]>0) {
                const float r=(d[i]-td[i])/td[i], robust=std::sqrt(r*r+options.geometry_epsilon*options.geometry_epsilon);
                depth_gradient=weights.first*mask[i]/n*r/(robust*td[i]);
                expected_depth+=weights.first*mask[i]/n*robust;
            }
            close(gd[i],depth_gradient);
            if (i!=2) {
                expected_normal+=weights.second*mask[i]/n*(1.F-.9F*.8F);
                close(gn[i],weights.second*mask[i]/n*.6F);
                close(gn[2*n+i],weights.second*mask[i]/n*.8F);
            } else { close(gn[i],0); close(gn[2*n+i],0); }
            close(gn[n+i],0);
        }
        close(loss.depth_value,expected_depth); close(loss.normal_value,expected_normal);
        close(loss.total,expected_depth+expected_normal);
        // Independent finite difference for one rendered depth, with fixed target.
        const auto objective=[&](double value) { const double r=(value-4)/4; return weights.first/n*std::sqrt(r*r+1e-6); };
        close(gd[4],static_cast<float>((objective(4.5001)-objective(4.4999))/.0002));
    }
    // Warmup must skip prior maps and geometry attachments entirely, even
    // with nonzero configured weights and DN disabled.
    TrainingOptions warmup; warmup.photometric_weight=0;
    warmup.lambda_d=.05F; warmup.lambda_n=.02F;
    rendered.median_depth={}; rendered.normal={};
    view.depth={}; view.normal={};
    const auto inactive=detail::compute_training_loss(rendered,view,warmup,true,false,false,false);
    close(inactive.depth_value,0); close(inactive.normal_value,0); close(inactive.total,0);
    check(!inactive.depth.is_valid() && !inactive.normal.is_valid(),
        "Inactive priors allocated geometry gradients during warmup");
    std::cout << "Prior loss/gradient checks passed on " << (device==tinytensor::Device::CUDA?"CUDA":"Vulkan") << '\n';
}
void parameter_gradients(tinytensor::Device device, photara::CameraModel camera_model=photara::CameraModel::pinhole) {
    // Audit the complete AI loss -> rasterizer -> Gaussian parameter chain.
    // A rotated camera and overlapping anisotropic splats exercise camera
    // coordinates, normal blending, median depth and opacity derivatives.
    Camera camera;
    camera.width=48; camera.height=32;
    camera.fx=38; camera.fy=36; camera.cx=23.2F; camera.cy=15.1F;
    camera.model=camera_model;
    if(camera_model==photara::CameraModel::opencv_fisheye) {camera.k1=.02F;camera.k2=-.001F;}
    if(camera_model==photara::CameraModel::equirectangular) {
        camera.fx=float(camera.width/(2*photara::k_pi));camera.fy=float(camera.height/photara::k_pi);
        camera.cx=camera.width*.5F;camera.cy=camera.height*.5F;
    }
    const float c=std::cos(.23F), s=std::sin(.23F);
    camera.world_to_camera={c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1};
    const std::size_t pixels=camera.width*camera.height, pixel=16*camera.width+(camera_model==photara::CameraModel::equirectangular?26:30);
    const std::vector<float> parameters={
        -.10F,.04F,2.F, .05F,-.02F,2.12F,
        std::log(.22F),std::log(.16F),std::log(.09F),
        std::log(.20F),std::log(.14F),std::log(.08F),
        .94F,.13F,-.18F,.21F, .97F,-.1F,.17F,.05F,
        1.7F,1.2F, .4F,.2F,.3F, .1F,.5F,.2F};
    const auto make_model=[&](const std::vector<float>& p) {
        const auto upload=[&](int first,int count,std::initializer_list<std::size_t> shape) {
            return Tensor::from_vector(std::vector<float>(p.begin()+first,p.begin()+first+count),shape,device);
        };
        GaussianModel m;
        m.means=upload(0,6,{2,3}); m.log_scales=upload(6,6,{2,3});
        m.quaternions=upload(12,8,{2,4}); m.opacity_logits=upload(20,2,{2,1});
        m.sh=upload(22,6,{2,1,3}); m.sh_degree=0;
        return m;
    };
    Rasterizer rasterizer;
    RasterizeOptions raster_options; raster_options.require_depth=true; raster_options.kernel_size=0;
    auto model=make_model(parameters);
    const auto rendered=rasterizer.forward(model,camera,raster_options);
    const float target_depth=rendered.median_depth.to_vector()[pixel]*1.08F;
    check(target_depth>0,"Parameter audit requires valid median depth");
    const float target_z=-std::sqrt(.87F);
    std::vector<float> td(pixels),tn(3*pixels),mask(pixels);
    td[pixel]=target_depth; tn[pixel]=-.3F; tn[pixels+pixel]=.2F; tn[2*pixels+pixel]=target_z;
    mask[pixel]=1;
    TrainingView view; view.camera=camera; view.has_mask=true;
    view.rgb=Tensor::zeros({3,camera.height,camera.width},device);
    view.mask=Tensor::from_vector(mask,{camera.height,camera.width},device);
    view.depth=Tensor::from_vector(td,{camera.height,camera.width},device);
    view.normal=Tensor::from_vector(tn,{3,camera.height,camera.width},device);
    for (int mode=0;mode<3;++mode) {
        TrainingOptions options; options.photometric_weight=0; options.use_mask=true;
        options.lambda_d=mode==1?0.F:1.F; options.lambda_n=mode==0?0.F:1.F;
        const auto loss=detail::compute_training_loss(rendered,view,options,true,false,true);
        const auto gradient=rasterizer.backward(model,rendered,loss.color,loss.alpha,loss.depth,loss.normal);
        std::vector<float> analytic;
        for (const auto* tensor : {&gradient.means,&gradient.log_scales,&gradient.quaternions,&gradient.opacity_logits,&gradient.sh}) {
            const auto values=tensor->to_vector();
            for (float value:values) analytic.push_back(value*pixels);
        }
        const auto objective=[&](const std::vector<float>& p) {
            const auto out=rasterizer.forward(make_model(p),camera,raster_options);
            double value=0;
            if(options.lambda_d>0) {
                const double depth=out.median_depth.to_vector()[pixel];
                check(depth>0,"Parameter perturbation changed depth validity");
                const double r=(depth-target_depth)/target_depth;
                value+=std::sqrt(r*r+double(options.geometry_epsilon)*options.geometry_epsilon);
            }
            if(options.lambda_n>0) {
                const auto n=out.normal.to_vector();
                value+=1.-(-.3*n[pixel]+.2*n[pixels+pixel]+double(target_z)*n[2*pixels+pixel]);
            }
            return value;
        };
        double error2=0,numeric2=0,dot=0;
        for(std::size_t i=0;i<parameters.size();++i) {
            auto plus=parameters,minus=parameters;
            const float h=5e-4F;
            plus[i]+=h;minus[i]-=h;
            const double numeric=(objective(plus)-objective(minus))/(double(plus[i])-minus[i]);
            const double error=analytic[i]-numeric;
            error2+=error*error; numeric2+=numeric*numeric; dot+=analytic[i]*numeric;
            if(!std::isfinite(analytic[i]) || std::abs(error)>.006+.04*std::abs(numeric)) {
                std::cerr<<"Prior parameter FD mode="<<mode<<" parameter="<<i
                    <<" analytic="<<analytic[i]<<" numeric="<<numeric<<'\n';
                throw std::runtime_error("AI prior parameter gradient mismatch");
            }
        }
        const double relative_error=std::sqrt(error2/std::max(numeric2,1e-20));
        std::cout<<"Prior parameter FD backend="<<(device==tinytensor::Device::CUDA?"CUDA":"Vulkan")
            <<" mode="<<mode<<" relative_l2="<<relative_error<<'\n';
        check(relative_error<.02,"Prior parameter gradient relative error exceeded two percent");
        check(dot>0,"Prior gradient is not a descent direction");
        auto next=parameters;
        for(std::size_t i=0;i<next.size();++i) next[i]-=1e-3F*analytic[i];
        check(objective(next)<objective(parameters),"Prior-only parameter step failed to reduce loss");
    }
}
void prefetch_image_parity() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("photara-prior-prefetch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto path = directory / "source.png";
    struct Cleanup {
        std::filesystem::path path, directory;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); std::filesystem::remove(directory, ec); }
    } cleanup{path, directory};
    photara::io::RgbImage source{128, 96, std::vector<std::uint8_t>(128 * 96 * 3)};
    for (std::size_t i = 0; i < source.pixels.size(); ++i) source.pixels[i] = std::uint8_t(i * 37);
    photara::io::save_rgb_png(source, path);
    photara::mvs::MvsView view;
    view.path = path; view.width = view.src_width = 128; view.height = view.src_height = 96;
    view.fx = view.fy = view.src_fx = view.src_fy = 100.F;
    view.cx = view.src_cx = 63.5F; view.cy = view.src_cy = 47.5F;
    view.k1 = -.04F; view.k2 = .003F;
    view.foreground_mask.resize(128 * 96);
    for (std::size_t i = 0; i < view.foreground_mask.size(); ++i) view.foreground_mask[i] = std::uint8_t(i % 256);
    TrainingOptions options;
    options.use_source_resolution = false; options.max_image_dimension = 0;
    options.use_mask = true; options.undistort_to_pinhole = true;
    const auto foreground = training_data::load_training_image(view, options);
    const auto background = training_data::load_training_image(view, options, false);
    check(foreground.rgb.width * foreground.rgb.height >= 4096, "Prefetch test misses parallel threshold");
    check(foreground.rgb.pixels == background.rgb.pixels && foreground.mask.pixels == background.mask.pixels,
        "Background packing changed RGB or mask pixels");
}

void packed_priors(tinytensor::Device device = tinytensor::Device::CUDA) {
    const auto directory = std::filesystem::temp_directory_path() /
        ("photara-packed-prior-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto source = directory / "image.png", path = training_prior_path(directory, 0);
    photara::io::save_rgb_png({33, 17, std::vector<std::uint8_t>(33 * 17 * 3, 127)}, source);
    TrainingPrior prior;
    photara::mvs::MvsView view;
    view.path = source; view.id = 0;
    view.width = view.src_width = 33; view.height = view.src_height = 17;
    view.fx = view.fy = view.src_fx = view.src_fy = 25;
    TrainingOptions options;
    prior.camera = training_data::training_camera(view, options);
    const std::size_t count = 33 * 17;
    prior.normals.resize(3 * count);
    prior.depth.assign(count, 2.F);
    for (std::size_t i = 1; i < count; ++i) {
        const double z = -1. + 2. * double(i - 1) / double(count - 2);
        const double radius = std::sqrt(std::max(0., 1. - z * z)), angle = i * 2.399963229728653;
        prior.normals[i] = float(radius * std::cos(angle));
        prior.normals[count + i] = float(radius * std::sin(angle));
        prior.normals[2 * count + i] = float(z);
    }
    save_training_prior(path, source, prior);
    const auto packed = load_packed_training_prior(path, source, prior.camera, prior.camera, true, true);
    check(packed.depth == prior.depth && packed.normals.size() == count, "Packed prior shape/depth");
    auto words = Tensor::from_vector(packed.normals, {17, 33}, device);
    Tensor normal_image;
    if (device == tinytensor::Device::CUDA)
        normal_image = detail::decode_packed_training_normals(words, 33, 17);
#if defined(TINYTENSOR_HAS_VULKAN)
    else {
        normal_image = Tensor::empty({3, 17, 33}, device);
        tinytensor::vulkan::unpack_training_normals(words, normal_image);
    }
#endif
    const auto decoded = normal_image.to_vector();
    for (std::size_t i = 0; i < count; ++i) {
        double error = 0, length = 0;
        for (std::size_t ch = 0; ch < 3; ++ch) {
            const auto j = ch * count + i;
            error += std::pow(double(decoded[j]) - prior.normals[j], 2);
            length += double(decoded[j]) * decoded[j];
        }
        check(error < 1e-8 && (i == 0 ? length == 0 : std::abs(length - 1) < 1e-6),
              "Packed normal exceeds error bound or changed invalid/unit semantics");
    }
    const auto reused = load_packed_training_prior(path, source, prior.camera, prior.camera, true, true);
    check(reused.normals == packed.normals && reused.depth == packed.depth, "Packed cache reuse changed data");
    const auto encoded_path = std::filesystem::path(path.string() + ".cuda-oct16-33x17-3");
    check(std::filesystem::file_size(encoded_path) < std::filesystem::file_size(path), "Packed cache did not shrink");
    {
        options.lambda_d = options.lambda_n = .01F;
        options.backend = device == tinytensor::Device::CUDA ? TrainingBackend::cuda : TrainingBackend::vulkan;
        options.prior_directory = directory;
        options.adaptive_training_cache = false;
        options.training_device_cache_bytes = 1024 * 1024;
        options.training_view_cache_bytes = 1024 * 1024;
        options.training_prefetch_views = 1;
        std::vector<photara::mvs::MvsView> views{view, view};
        training_data::TrainingDataLoader loader(views, options);
        check(loader.get(0).normal.to_vector() == decoded, "Packed resident normal decode changed");
        check(loader.get(0).normal.to_vector() == decoded && loader.stats().device_hits > 0,
              "Packed device cache did not retain the encoded normals");
        if (device == tinytensor::Device::CUDA) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        do {
            loader.prefetch(1);
            if (loader.stats().async_upload_issued) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (std::chrono::steady_clock::now() < deadline);
        check(loader.stats().async_upload_issued > 0, "Packed normal async upload not issued");
        check(loader.get(1).normal.to_vector() == decoded && loader.stats().async_upload_hits > 0 &&
              loader.stats().async_upload_failures == 0, "Packed async normal upload changed supervision");
        } else {
            loader.prefetch(1);
            check(loader.get(1).normal.to_vector() == decoded, "Vulkan packed prefetch changed normals");
            loader.set_resolution_scale(.5F);
            const auto half = loader.get(1);
            check(half.normal.numel() == 3 * half.camera.width * half.camera.height,
                  "Vulkan packed prior resolution transition changed shape");
        }
    }
    { std::ofstream file(encoded_path, std::ios::binary | std::ios::trunc); file << "truncated"; }
    check(load_packed_training_prior(path, source, prior.camera, prior.camera, true, true).normals == packed.normals,
          "Incomplete derived cache was not rebuilt");
    auto wrong = prior.camera; wrong.world_to_camera[12] += 1;
    fails([&] { (void)load_packed_training_prior(path, source, wrong, wrong, true, true); });
    { std::ofstream file(source, std::ios::app); file << "changed"; }
    fails([&] { (void)load_packed_training_prior(path, source, prior.camera, prior.camera, true, true); });
    std::filesystem::remove_all(directory);
}

void cache() {
    const auto directory=std::filesystem::temp_directory_path()/("photara-prior-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup(){
        std::error_code ec;
        if (path.is_absolute() && path.parent_path()==std::filesystem::temp_directory_path() &&
            path.filename().string().starts_with("photara-prior-test-")) std::filesystem::remove_all(path,ec);
    } } cleanup{directory};
    const auto image=directory/"rgb.png";
    photara::io::save_rgb_png({8,8,std::vector<std::uint8_t>(8*8*3,127)},image);
    photara::mvs::MvsView view;
    view.id=4; view.path=image; view.width=view.height=8; view.fx=view.fy=32; view.cx=view.cy=3.5F;
    TrainingOptions options; options.adaptive_training_cache=false; options.training_async_upload=false;
    options.training_device_cache_bytes=0; options.training_view_cache_bytes=0;
    TrainingPrior prior; prior.camera=training_data::training_camera(view,options);
    prior.depth.assign(64,4.F); prior.depth[0]=0;
    prior.normals.assign(192,0); for (int i=0;i<64;++i) prior.normals[128+i]=-1;
    save_training_prior(training_prior_path(directory,4),image,prior);
    photara::mvs::MvsScene scene; scene.views={view};
    const auto baseline_fingerprint=detail::training_fingerprint(scene,options);
    options.prior_directory=directory;
    check(detail::training_fingerprint(scene,options)==baseline_fingerprint,"Disabled prior changed checkpoint compatibility");
    options.lambda_d=.01F;
    const auto supervised_fingerprint=detail::training_fingerprint(scene,options);
    check(supervised_fingerprint!=baseline_fingerprint,"Checkpoint ignored prior supervision");
    options.lambda_n=.01F;
    check(detail::training_fingerprint(scene,options)!=supervised_fingerprint,"Checkpoint ignored normal prior weight");
    options.lambda_d=options.lambda_n=0;
    const auto loaded=load_training_prior(training_prior_path(directory,4),image,prior.camera);
    check(loaded.depth==prior.depth && loaded.normals==prior.normals,"Prior cache roundtrip");
    auto changed=prior.camera; changed.fx*=2;
    fails([&]{(void)load_training_prior(training_prior_path(directory,4),image,changed);});
    auto half=training_data::training_camera(view,options,.5F);
    auto resized=resize_training_prior(prior,half);
    check(resized.depth.size()==16 && resized.depth[0]==4,"Progressive prior resize changed depth units");
    auto malformed=prior; malformed.normals[128]=2;
    fails([&]{malformed.validate();});
    auto empty=prior; empty.depth.clear(); empty.normals.clear();
    fails([&]{empty.validate();});
    // Single-map caches are smaller and may only satisfy the requested loss.
    const auto both_bytes=std::filesystem::file_size(training_prior_path(directory,4));
    for (bool depth_only : {true,false}) {
        auto single=prior;
        if (depth_only) single.normals.clear(); else single.depth.clear();
        save_training_prior(training_prior_path(directory,4),image,single);
        check(std::filesystem::file_size(training_prior_path(directory,4)) < both_bytes,"Unused map stored");
        auto one=load_training_prior(training_prior_path(directory,4),image,prior.camera,depth_only,!depth_only);
        check(one.depth==single.depth && one.normals==single.normals,"Single-map roundtrip");
        auto small=resize_training_prior(one,half);
        check(small.depth.size()==(depth_only?16:0) && small.normals.size()==(depth_only?0:48),"Single-map resize");
        fails([&]{(void)load_training_prior(training_prior_path(directory,4),image,prior.camera,!depth_only,depth_only);});
        options.lambda_d=depth_only?.01F:0.F; options.lambda_n=depth_only?0.F:.01F;
        training_data::TrainingDataLoader loader(scene.views,options);
        auto target=loader.get(0);
        check((depth_only?target.depth:target.normal).to_vector()==(depth_only?prior.depth:prior.normals),"Single-map loader");
    }
    save_training_prior(training_prior_path(directory,4),image,prior);
    auto selected=load_training_prior(training_prior_path(directory,4),image,prior.camera,true,false);
    check(selected.normals.empty() && selected.depth==prior.depth,"Unused map loaded");
    // Legacy PHPRIOR1 caches always contained both maps, without the flags word.
    { std::ifstream input(training_prior_path(directory,4),std::ios::binary);
      std::vector<char> bytes((std::istreambuf_iterator<char>(input)),{});
      bytes[7]='1'; bytes.erase(bytes.begin()+8,bytes.begin()+12);
      const auto old=directory/"legacy.prior";
      { std::ofstream output(old,std::ios::binary); output.write(bytes.data(),bytes.size()); }
      auto legacy=load_training_prior(old,image,prior.camera,false,true);
      check(legacy.depth.empty() && legacy.normals==prior.normals,"Legacy cache compatibility"); }
    options.lambda_d=options.lambda_n=0;
    std::vector<photara::mvs::MvsView> views{view};
    options.prior_directory=directory/"missing";
    { training_data::TrainingDataLoader loader(views,options); auto target=loader.get(0);
      check(target.depth.numel()==1 && target.normal.numel()==1,"Disabled prior read cache"); }
    options.lambda_d=.05F; options.lambda_n=.02F; options.prior_directory=directory;
    { training_data::TrainingDataLoader loader(views,options); auto target=loader.get(0);
      check(target.depth.to_vector()==prior.depth,"Prior not loaded into training target");
      check(!target.has_mask,"Prior validity incorrectly masked RGB"); }
    { training_data::TrainingDataLoader loader(views,options,.5F); auto target=loader.get(0);
      check(target.depth.numel()==16 && target.normal.numel()==48,"Progressive prior target shape"); }
    {
        auto warmup = options;
        warmup.iterations = 2; warmup.depth_normal_from_iter = 3;
        warmup.enable_densification = false; warmup.use_mask = false;
        warmup.log_interval = 1; warmup.preview_interval = 0;
        warmup.prior_directory = directory / "not-loaded-during-warmup";
        photara::mvs::DensePoint point;
        point.position = {0, 0, 4}; point.normal = {0, 0, -1}; point.color = {.5F, .5F, .5F};
        scene.dense_cloud.points = {point};
        (void)Trainer(warmup).train(scene, [](const TrainingProgress& p) {
            check(p.depth_loss == 0 && p.normal_loss == 0, "Warmup applied prior supervision");
            return true;
        });
        warmup.iterations = 3; warmup.depth_normal_from_iter = 2;
        warmup.prior_directory = directory;
        bool activated = false;
        (void)Trainer(warmup).train(scene, [&](const TrainingProgress& p) {
            if (p.iteration == 1)
                check(p.depth_loss == 0 && p.normal_loss == 0, "Prior enabled before activation");
            if (p.iteration >= 2 && p.depth_loss + p.normal_loss > 0) activated = true;
            return true;
        });
        check(activated, "Deferred prior data was not activated at the loss boundary");
    }
    // A held-out image deliberately has no prior. Evaluation must still work
    // with the same options as the supervised training run.
    options.prior_directory=directory/"missing-held-out-cache";
    GaussianModel model;
    const auto device=tinytensor::Device::CUDA;
    model.means=Tensor::from_vector(std::vector<float>{0,0,4},{1,3},device);
    model.log_scales=Tensor::from_vector(std::vector<float>{-2,-2,-2},{1,3},device);
    model.quaternions=Tensor::from_vector(std::vector<float>{1,0,0,0},{1,4},device);
    model.opacity_logits=Tensor::from_vector(std::vector<float>{1},{1,1},device);
    model.sh=Tensor::zeros({1,1,3},device); model.sh_degree=0;
    (void)render_evaluation_png(model,view,directory/"evaluation.png",options);
    check(std::filesystem::exists(directory/"evaluation.png"),"Held-out evaluation used a prior");
    std::ofstream(image,std::ios::app|std::ios::binary)<<'x';
    fails([&]{(void)load_training_prior(training_prior_path(directory,4),image,prior.camera);});
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            prefetch_image_parity(); native_priors(); cache(); packed_priors(); losses(tinytensor::Device::CUDA);
            losses(tinytensor::Device::CUDA,photara::CameraModel::opencv_fisheye);
            losses(tinytensor::Device::CUDA,photara::CameraModel::equirectangular); parameter_gradients(tinytensor::Device::CUDA);
            parameter_gradients(tinytensor::Device::CUDA,photara::CameraModel::opencv_fisheye);
            parameter_gradients(tinytensor::Device::CUDA,photara::CameraModel::equirectangular);
        } else if (argc == 2 && std::string_view(argv[1]) == "--vulkan") {
#if defined(TINYTENSOR_HAS_VULKAN)
            packed_priors(tinytensor::Device::Vulkan);
            losses(tinytensor::Device::Vulkan,photara::CameraModel::opencv_fisheye);
            losses(tinytensor::Device::Vulkan,photara::CameraModel::equirectangular);
            losses(tinytensor::Device::Vulkan); parameter_gradients(tinytensor::Device::Vulkan);
            parameter_gradients(tinytensor::Device::Vulkan,photara::CameraModel::opencv_fisheye);
            parameter_gradients(tinytensor::Device::Vulkan,photara::CameraModel::equirectangular);
#else
            throw std::runtime_error("Vulkan is unavailable in this build");
#endif
        } else {
            throw std::invalid_argument("Usage: photara_splat_prior_test [--vulkan]");
        }
        std::cout << "AI prior tests passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
