#include "sfm/gpu_geometry.hpp"
#include "sfm/tracks.hpp"
#include <future>
#include <iomanip>
#include <iostream>
#include <random>

using namespace photara::sfm;
namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
Scene fixture(unsigned count,photara::CameraModel model) {
    Scene s; s.thread_count=4;
    PinholeCamera c; c.width=1280; c.height=720; c.fx=c.fy=800; c.cx=640; c.cy=360;
    c.model=model; c.k1=.01; c.k2=-.001; c.p1=.0001; c.p2=-.0002;
    s.cameras.push_back(c); s.images.resize(4);
    for(unsigned j=0;j<4;++j) {
        auto& image=s.images[j]; image.id=j; image.camera_id=0; image.registered=j<3;
        image.pose.C={.45*j,.02*j,0};
        image.pose.R=Eigen::AngleAxisd(.01*j,Vec3::UnitY()).toRotationMatrix();
    }
    std::mt19937 rng(42); std::uniform_real_distribution<double> xy(-1.,1.),z(3.,9.);
    for(unsigned i=0;i<count;++i) {
        Track track; Vec3 p(xy(rng),xy(rng),z(rng)); track.position=p;
        unsigned views=i%2+2;
        for(unsigned j=0;j<4;++j) {
            auto pixel=c.project(s.images[j].pose.transform_world_to_camera(p));
            photara::features::Keypoint key{}; key.x=static_cast<float>(pixel.x()); key.y=static_cast<float>(pixel.y());
            if(i%11==0&&j==1) key.x+=30;
            s.images[j].features.keypoints.push_back(key);
            if(j<views||j==3) track.observations.push_back({j,i});
        }
        track.num_inliers=static_cast<std::uint8_t>(views);
        s.tracks.push_back(track);
    }
    return s;
}
void compare(const Scene& a,const Scene& b) {
    require(a.tracks.size()==b.tracks.size(),"track count parity");
    for(std::size_t i=0;i<a.tracks.size();++i) {
        const auto& x=a.tracks[i]; const auto& y=b.tracks[i];
        require(x.num_inliers==y.num_inliers,"inlier parity");
        if(x.is_triangulated()) {
            if ((x.position-y.position).norm()>=1e-6) {
                std::cerr<<std::setprecision(17)<<"position mismatch track "<<i<<" diff="
                    <<(x.position-y.position).norm()<<" cpu="<<x.position.transpose()
                    <<" gpu="<<y.position.transpose()<<" cpu_inliers="<<int(x.num_inliers)
                    <<" gpu_inliers="<<int(y.num_inliers)<<" obs="<<x.observations.size()<<"\n";
                throw std::runtime_error("position parity");
            }
        }
        require(x.observations.size()==y.observations.size(),"observation count parity");
        for(std::size_t j=0;j<x.observations.size();++j)
            require(x.observations[j].image_id==y.observations[j].image_id&&x.observations[j].feature_id==y.observations[j].feature_id,"observation order parity");
    }
}
void run(photara::CameraModel model) {
    Scene source=fixture(5000,model),cpu=source,gpu=source;
    TriangulationOptions options; options.split_tracks=false; options.min_angle_deg=.1f;
    set_geometry_backend(GeometryBackend::cpu);
    triangulate_tracks(cpu,false,options);
    set_geometry_backend(GeometryBackend::cuda);
    auto handled=try_triangulate_tracks_cuda(gpu,nullptr,false,options);
    require(std::count(handled.begin(),handled.end(),1)>1000,"CUDA short tracks actually executed");
    for(std::size_t i=0;i<gpu.tracks.size();++i) if(!handled[i]) triangulate_track(gpu.tracks[i],gpu,options);
    compare(cpu,gpu);
    auto before=cpu;
    set_geometry_backend(GeometryBackend::cpu); auto a=filter_tracks(cpu,3,.1f,0,0);
    set_geometry_backend(GeometryBackend::cuda); auto b=filter_tracks(gpu,3,.1f,0,0);
    compare(cpu,gpu);
    require(std::abs(a.first-b.first)<1e-6&&std::abs(a.second-b.second)<1e-4,"filter statistics parity");
    // Selected IDs include duplicates/out-of-range; public filtering canonicalizes them.
    std::vector<Index> selected{4,5,5,23,999999}; cpu=before; gpu=before;
    set_geometry_backend(GeometryBackend::cpu); filter_tracks(cpu,selected,3,.1f);
    set_geometry_backend(GeometryBackend::cuda); filter_tracks(gpu,selected,3,.1f); compare(cpu,gpu);
    // The panorama branch is intentionally retained on CPU.
    gpu.cameras[0].set_equirectangular_intrinsics();
    auto panorama=try_triangulate_tracks_cuda(gpu,nullptr,false,options);
    require(std::count(panorama.begin(),panorama.end(),1)==0,"panorama CPU fallback");
}
void linear_tracks() {
    Scene source=fixture(5000,photara::CameraModel::pinhole),cpu=source,gpu=source;
    TriangulationOptions options; options.split_tracks=false; options.use_lo_ransac=false;
    set_geometry_backend(GeometryBackend::cpu);
    triangulate_tracks(cpu,false,options);
    set_geometry_backend(GeometryBackend::cuda);
    auto handled=try_triangulate_tracks_cuda(gpu,nullptr,false,options);
    require(std::count(handled.begin(),handled.end(),1)>4500,
        "CUDA linear triangulation covers four-observation tracks");
    for(std::size_t i=0;i<gpu.tracks.size();++i) if(!handled[i]) triangulate_track(gpu.tracks[i],gpu,options);
    compare(cpu,gpu);
    // Automatic keeps the CPU path because the measured CUDA stages were not
    // faster; only the explicit CUDA override may execute these kernels.
    set_geometry_backend(GeometryBackend::automatic);
    auto none=try_triangulate_tracks_cuda(source,nullptr,false,options);
    require(none.empty(),"automatic backend keeps geometry on the CPU");
}
void pairs() {
    Scene scene=fixture(1024,photara::CameraModel::pinhole);
    std::vector<Vec2> a,b;
    for(unsigned i=0;i<1024;++i) {
        auto x=scene.images[0].features.keypoints[i],y=scene.images[1].features.keypoints[i];
        a.emplace_back(x.x,x.y); b.emplace_back(y.x,y.y);
    }
    RelativePoseOptions opts; opts.min_ray_angle_deg=.1;
    auto estimate=[&] { return estimate_relative_pose(a,b,scene.cameras[0],scene.cameras[0],opts); };
    set_geometry_backend(GeometryBackend::cpu); auto cpu=estimate();
    require(cpu.success,"CPU pair fit");
    set_geometry_backend(GeometryBackend::cuda);
    std::vector<std::future<RelativePoseResult>> tasks;
    for(int j=0;j<8;++j) tasks.push_back(std::async(std::launch::async,estimate));
    for(auto& task:tasks) {
        auto gpu=task.get(); require(gpu.success,"CUDA pair fit");
        require(gpu.inlier_mask==cpu.inlier_mask,"pair verification mask parity");
        require(std::abs(gpu.mean_ray_angle-cpu.mean_ray_angle)<1e-5,"pair angle parity");
    }
    std::vector<char> mask(a.size(),1); float mean=0; unsigned count=0;
    require(try_filter_matches_cuda(a,b,scene.cameras[0],scene.cameras[0],scene.images[1].pose,
        opts,mask,mean,nullptr,count),"CUDA known-pose verification actually executed");
    require(count>800,"known-pose consensus");
}
}
int main() {
    if(!geometry_cuda_available()) { std::cout<<"CUDA unavailable\n"; return 77; }
    try {
        run(photara::CameraModel::pinhole); run(photara::CameraModel::opencv_fisheye);
        linear_tracks(); pairs();
        set_geometry_backend(GeometryBackend::automatic);
        std::cout<<"GPU geometry parity passed\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
