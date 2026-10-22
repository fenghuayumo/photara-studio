#include "sfm/gpu_geometry.hpp"
#include "geometry_cuda.hpp"
#include "core/logging.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

namespace photara::sfm {
namespace {
std::atomic<int> preference{-1};
// Automatic deliberately keeps the CPU implementation: on Berlin/London the
// CUDA stages were slower than the 32-core CPU path (see SFM_ARCHITECTURE.md).
// The env/API CUDA override remains the explicit way to exercise these kernels.
bool enabled() {
    return geometry_backend()==GeometryBackend::cuda && geometry_cuda_available();
}
gpu::V3 pack(const Vec3& p) { return {p.x(),p.y(),p.z()}; }
gpu::View pack(const Pose3D& pose,const PinholeCamera& camera) {
    gpu::View v{};
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) v.R[r*3+c]=pose.R(r,c);
    for(int i=0;i<3;++i) v.C[i]=pose.C(i);
    v.fx=camera.fx; v.fy=camera.fy; v.cx=camera.cx; v.cy=camera.cy;
    v.k1=camera.k1; v.k2=camera.k2; v.p1=camera.p1; v.p2=camera.p2;
    v.model=static_cast<unsigned>(camera.model); return v;
}

#if defined(PHOTARA_HAS_CUDA)
struct PairRequest {
    gpu::PairJob job;
    std::vector<gpu::RayPair> rays;
    std::promise<std::vector<gpu::PairResult>> promise;
};
// One CUDA owner thread batches calls from independent CPU PoseLib workers.
// The worker never calls into the CPU pool, including during shutdown.
class PairQueue {
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::shared_ptr<PairRequest>> pending_;
    bool stopping_{};
    std::thread worker_{[this] { run(); }};
    void run() {
        for(;;) {
            std::vector<std::shared_ptr<PairRequest>> batch;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock,[&]{return stopping_||!pending_.empty();});
                if(stopping_&&pending_.empty()) return;
                condition_.wait_for(lock,std::chrono::microseconds(200),[&]{return stopping_||pending_.size()>=16;});
                while(!pending_.empty()&&batch.size()<32) {
                    batch.push_back(std::move(pending_.front())); pending_.pop_front();
                }
            }
            try {
                std::vector<gpu::PairJob> jobs;
                std::vector<gpu::RayPair> rays;
                for(std::size_t j=0;j<batch.size();++j) {
                    jobs.push_back(batch[j]->job);
                    for(auto ray:batch[j]->rays) { ray.job=static_cast<unsigned>(j); rays.push_back(ray); }
                }
                core::StageScope stage("sfm.verify_filter",core::LogLevel::debug,"cuda");
                std::vector<gpu::PairResult> results;
                gpu::cuda_filter_pairs(rays,jobs,results);
                std::size_t offset=0;
                for(auto& request:batch) {
                    auto next=offset+request->rays.size();
                    request->promise.set_value({results.begin()+offset,results.begin()+next}); offset=next;
                }
            } catch(...) {
                for(auto& request:batch) request->promise.set_exception(std::current_exception());
            }
        }
    }
public:
    ~PairQueue() {
        { std::lock_guard lock(mutex_); stopping_=true; }
        condition_.notify_one(); worker_.join();
    }
    std::vector<gpu::PairResult> submit(std::shared_ptr<PairRequest> request) {
        auto future=request->promise.get_future();
        { std::lock_guard lock(mutex_); pending_.push_back(std::move(request)); }
        condition_.notify_one(); return future.get();
    }
};
PairQueue& pair_queue() { static PairQueue instance; return instance; }
#endif

// A batch keeps one View per referenced image (not one per observation) and
// stores a view index in each observation. This cuts the host-to-device traffic
// by roughly four times for multi-view tracks.
struct PackedTracks {
    explicit PackedTracks(const Scene& scene) :
        view_slots(scene.images.size(),-1),camera_min_cosine(scene.cameras.size(),2.0) {}
    std::vector<int> view_slots;
    // The reprojection threshold is camera dependent and shared by every track;
    // computing its cosine once per camera avoids millions of trig calls.
    std::vector<double> camera_min_cosine;
    std::vector<gpu::View> views;
    std::vector<gpu::Observation> observations;
    std::vector<gpu::TrackJob> jobs;
    std::vector<Index> track_ids;
    std::vector<unsigned> obs_indices;
};
bool append(PackedTracks& batch,const Scene& scene,Index id,bool triangulate,
            const TriangulationOptions& options,float threshold) {
    const auto& track=scene.tracks[id];
    if(!track.is_valid()) return false;
    if(!triangulate&&!track.position.allFinite()) return false;
    const std::size_t observation_begin=batch.observations.size();
    const std::size_t index_begin=batch.obs_indices.size();
    const auto rollback=[&] {
        batch.observations.resize(observation_begin);
        batch.obs_indices.resize(index_begin);
    };
    for(unsigned i=0;i<track.observations.size();++i) {
        const auto obs=track.observations[i];
        if(obs.image_id>=scene.images.size()) continue;
        const auto& image=scene.images[obs.image_id];
        if(!image.registered||obs.feature_id>=image.features.keypoints.size()) continue;
        const auto& camera=scene.camera_of(image);
        // Full-sphere tangent residuals retain the existing CPU implementation.
        if(camera.model!=CameraModel::pinhole&&camera.model!=CameraModel::opencv_fisheye) {
            rollback(); return false;
        }
        const auto& kp=image.features.keypoints[obs.feature_id];
        const Vec3 ray=camera.unproject_normalized({kp.x,kp.y});
        if(!ray.allFinite()) { rollback(); return false; }
        int& slot=batch.view_slots[obs.image_id];
        if(slot<0) {
            batch.views.push_back(pack(image.pose,camera));
            slot=static_cast<int>(batch.views.size()-1);
        }
        double minimum_cosine=0.0;
        if(!triangulate) {
            const Index camera_id=image.camera_id;
            double& cached=batch.camera_min_cosine[camera_id];
            if(cached>1.0) cached=std::cos(camera.pixel_error_to_angular(threshold));
            minimum_cosine=cached;
        }
        batch.observations.push_back({pack(ray),kp.x,kp.y,minimum_cosine,
            static_cast<unsigned>(slot)});
        batch.obs_indices.push_back(i);
    }
    const std::size_t count=batch.observations.size()-observation_begin;
    if(count<2||count>255) { rollback(); return false; }
    if(triangulate) {
        if(count<options.min_inliers) { rollback(); return false; }
        // CPU LO-RANSAC remains tuned for the sequential support cache and was
        // faster than a per-track CUDA replay in the Berlin/London benchmarks.
        // Only the deterministic linear-consensus path is scheduled here.
        if(options.use_lo_ransac&&count>=options.min_observations_for_ransac) {
            rollback(); return false;
        }
    }
    batch.jobs.push_back({static_cast<unsigned>(observation_begin),
        static_cast<unsigned>(count),pack(track.position)});
    batch.track_ids.push_back(id);
    return true;
}
} // namespace

void set_geometry_backend(GeometryBackend backend) { preference.store(static_cast<int>(backend)); }
GeometryBackend geometry_backend() {
    int value=preference.load();
    if(value>=0) return static_cast<GeometryBackend>(value);
    const char* env=std::getenv("PHOTARA_SFM_GEOMETRY_BACKEND");
    if(!env||std::string_view(env)=="automatic") return GeometryBackend::automatic;
    if(std::string_view(env)=="cpu") return GeometryBackend::cpu;
    if(std::string_view(env)=="cuda") return GeometryBackend::cuda;
    throw std::invalid_argument("PHOTARA_SFM_GEOMETRY_BACKEND must be automatic, cpu or cuda");
}
bool geometry_cuda_available() {
#if defined(PHOTARA_HAS_CUDA)
    static const bool available=gpu::cuda_available();
    return available;
#else
    return false;
#endif
}

bool try_filter_matches_cuda(const std::vector<Vec2>& p1,const std::vector<Vec2>& p2,
    const PinholeCamera& c1,const PinholeCamera& c2,const Pose3D& pose,
    const RelativePoseOptions& options,std::vector<char>& mask,float& mean,
    std::vector<Vec2>* pixels,unsigned& count) {
#if defined(PHOTARA_HAS_CUDA)
    if(!enabled()||p1.size()!=p2.size()||mask.size()!=p1.size()||
        c1.is_equirectangular()||c2.is_equirectangular()) return false;
    auto request=std::make_shared<PairRequest>();
    auto& job=request->job; job.view=pack(pose,c2);
    job.cos_reproj=std::cos(.5*(c1.pixel_error_to_angular(options.max_reproj_error_px)+c2.pixel_error_to_angular(options.max_reproj_error_px)));
    job.min_cos_angle=std::cos(options.min_ray_angle_deg*3.14159265358979323846/180.);
    job.cos_epipole=std::cos(.5*(c1.pixel_error_to_angular(options.epipole_filter_px)+c2.pixel_error_to_angular(options.epipole_filter_px)));
    if(pose.C.norm()>1e-12) job.epipole1=pack(pose.C.normalized());
    Vec3 t=pose.translation(); if(t.norm()>1e-12) job.epipole2=pack(t.normalized());
    job.epipole=options.epipole_filter_px>0; job.reprojection=options.max_reproj_error_px>0; job.angle=options.min_ray_angle_deg>0;
    std::vector<std::size_t> indices;
    for(std::size_t i=0;i<mask.size();++i) if(mask[i]) {
        request->rays.push_back({pack(c1.unproject_normalized(p1[i])),pack(c2.unproject_normalized(p2[i])),0}); indices.push_back(i);
    }
    if(indices.empty()) return false;
    try {
        const auto results=pair_queue().submit(request);
        if(std::any_of(results.begin(),results.end(),[](auto r){return r.uncertain!=0;})) return false;
        count=0; double weights=0,cosine=0; if(pixels) pixels->clear();
        for(std::size_t i=0;i<indices.size();++i) {
            const auto& r=results[i]; mask[indices[i]]=static_cast<char>(r.accepted);
            if(!r.accepted) continue;
            ++count; weights+=r.weight; cosine+=r.cosine*r.weight;
            if(pixels) pixels->push_back(p1[indices[i]]);
        }
        mean=count&&weights>0?static_cast<float>(std::acos(std::clamp(cosine/weights,-1.,1.))):0.f;
        return true;
    } catch(const std::exception& e) {
        core::Logger::instance().warning("CUDA verification filter falling back to CPU: ",e.what());
    }
#endif
    return false;
}

std::vector<unsigned char> try_triangulate_tracks_cuda(Scene& scene,const std::vector<Index>* selected,
    bool outliers_only,const TriangulationOptions& options) {
    std::vector<unsigned char> handled;
#if defined(PHOTARA_HAS_CUDA)
    std::size_t n=selected?selected->size():scene.tracks.size();
    if(!enabled()) return handled;
    handled.resize(n);
    gpu::TriangulationConfig config{options.reproj_threshold_px,options.min_angle_deg,
        options.min_inliers,options.refine_iterations,options.refine_nonlinear};
    core::StageScope stage("sfm.triangulate_tracks_gpu",core::LogLevel::debug,"cuda");
    try {
        for(std::size_t start=0;start<n;start+=16384) {
            PackedTracks batch(scene); std::vector<std::size_t> slots;
            for(std::size_t i=start;i<std::min(n,start+16384);++i) {
                Index id=selected?(*selected)[i]:static_cast<Index>(i);
                if(id>=scene.tracks.size()||(outliers_only&&scene.tracks[id].is_triangulated())) continue;
                if(append(batch,scene,id,true,options,options.reproj_threshold_px)) slots.push_back(i);
            }
            std::vector<gpu::TrackResult> results; std::vector<unsigned char> inliers;
            gpu::cuda_triangulate(batch.views,batch.observations,batch.jobs,config,results,inliers);
            for(std::size_t j=0;j<results.size();++j) {
                const auto& r=results[j]; if(r.fallback) continue;
                const auto& job=batch.jobs[j]; auto& track=scene.tracks[batch.track_ids[j]];
                std::vector<unsigned char> used(track.observations.size()); std::vector<Observation> ordered;
                for(unsigned k=0;k<job.count;++k) if(inliers[job.begin+k]) {
                    unsigned index=batch.obs_indices[job.begin+k]; ordered.push_back(track.observations[index]); used[index]=1;
                }
                for(std::size_t k=0;k<used.size();++k) if(!used[k]) ordered.push_back(track.observations[k]);
                track.observations=std::move(ordered); track.position={r.position.x,r.position.y,r.position.z};
                track.num_inliers=static_cast<std::uint8_t>(r.inliers); handled[slots[j]]=1;
            }
        }
    } catch(const std::exception& e) {
        core::Logger::instance().warning("CUDA triangulation falling back to CPU: ",e.what());
    }
    core::Logger::instance().debug("GPU track triangulation: handled=",std::count(handled.begin(),handled.end(),1)," candidates=",n);
#endif
    return handled;
}

std::vector<GpuTrackFilterStats> try_filter_tracks_cuda(Scene& scene,const std::vector<Index>* selected,
    float threshold,float min_angle) {
    std::vector<GpuTrackFilterStats> stats;
#if defined(PHOTARA_HAS_CUDA)
    std::size_t n=selected?selected->size():scene.tracks.size();
    if(!enabled()) return stats;
    stats.resize(n); TriangulationOptions options;
    core::StageScope stage("sfm.filter_tracks_batch",core::LogLevel::debug,"cuda");
    try {
        for(std::size_t start=0;start<n;start+=4096) {
            PackedTracks batch(scene); std::vector<std::size_t> slots;
            for(std::size_t i=start;i<std::min(n,start+4096);++i) {
                Index id=selected?(*selected)[i]:static_cast<Index>(i);
                if(id<scene.tracks.size()&&
                    append(batch,scene,id,false,options,threshold)) slots.push_back(i);
            }
            std::vector<gpu::TrackResult> results; std::vector<unsigned char> inliers;
            gpu::cuda_filter_tracks(batch.views,batch.observations,batch.jobs,
                threshold,min_angle,results,inliers);
            for(std::size_t j=0;j<results.size();++j) {
                const auto& r=results[j]; if(r.fallback) continue;
                auto& track=scene.tracks[batch.track_ids[j]]; const auto& job=batch.jobs[j]; unsigned kept=0;
                for(unsigned k=0;k<job.count;++k) if(inliers[job.begin+k]) {
                    std::swap(track.observations[kept],track.observations[batch.obs_indices[job.begin+k]]); ++kept;
                }
                track.split_generation=0; track.num_inliers=static_cast<std::uint8_t>(r.inliers);
                stats[slots[j]]={r.inliers?r.sum_error:0,r.inliers?r.sum_cosine:0,r.inliers?r.distance/r.inliers:0,
                    r.inliers,r.inliers>=2,true};
            }
        }
    } catch(const std::exception& e) {
        core::Logger::instance().warning("CUDA track filter falling back to CPU: ",e.what());
    }
#endif
    return stats;
}
} // namespace photara::sfm
