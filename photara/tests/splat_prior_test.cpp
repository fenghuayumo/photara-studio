#include "splat/prior.hpp"
#include "splat/trainer.hpp"
#include "training_data_loader.hpp"
#include "cuda_ops.hpp"
#include "checkpoint.hpp"
#if defined(TINYTENSOR_HAS_VULKAN)
#include "geometry_vulkan.hpp"
#endif
#include "io/image.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

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
void losses(tinytensor::Device device) {
    constexpr std::size_t n = 64;
    std::vector<float> d(n,4.5F), td(n,4.F), normal(3*n), target(3*n), mask(n,1.F);
    for (std::size_t i=0;i<n;++i) { normal[2*n+i]=-.9F; target[i]=-.6F; target[2*n+i]=-.8F; }
    mask[0]=0; mask[3]=.5F; td[1]=0; target[2]=target[2*n+2]=0;
    RenderResult rendered;
    rendered.color=Tensor::zeros({3,8,8},device); rendered.alpha=Tensor::zeros({8,8},device);
    rendered.median_depth=Tensor::from_vector(d,{8,8},device);
    rendered.normal=Tensor::from_vector(normal,{3,8,8},device);
    TrainingView view;
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
            cache(); losses(tinytensor::Device::CUDA);
        } else if (argc == 2 && std::string_view(argv[1]) == "--vulkan") {
#if defined(TINYTENSOR_HAS_VULKAN)
            losses(tinytensor::Device::Vulkan);
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
